"""Two controlled QLoRA experiments with persistent checkpoints and manifests."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic
from .evaluation import fingerprint


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def require_qlora_runtime() -> None:
    """Fail with one actionable message before allocating a large model."""
    missing = [
        name for name in ("bitsandbytes", "peft", "transformers", "accelerate")
        if _package_version(name) == "not-installed"
    ]
    if missing:
        raise RuntimeError(f"Missing QLoRA dependencies: {', '.join(missing)}")
    from packaging.version import Version
    minimum = {"transformers": "5.10.1", "peft": "0.19.0"}
    outdated = [
        f"{name} {_package_version(name)} < {required}"
        for name, required in minimum.items()
        if Version(_package_version(name)) < Version(required)
    ]
    if outdated:
        raise RuntimeError(
            "Gemma 4 QLoRA runtime is outdated: "
            + "; ".join(outdated)
            + ". Run Section 0 dependencies and restart the Colab runtime."
        )
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")


@lru_cache(maxsize=1)
def _optional_hf_token() -> str | None:
    """Read the Colab secret once so long experiments do not re-query the UI."""
    import os
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except (ImportError, KeyError, AttributeError):
        return None
    except Exception as error:
        if type(error).__name__ in {
            "SecretNotFoundError", "NotebookAccessError", "TimeoutException"
        }:
            return None
        raise


def load_qlora_processor(model_id: str):
    """Load only the processor for exact pre-training token diagnostics."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, token=_optional_hf_token())


def load_quantized_base(model_id: str, *, training: bool):
    """Load Gemma in NF4 and optionally prepare it for adapter training."""
    require_qlora_runtime()
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )
    token = _optional_hf_token()
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
        token=token,
    )
    if training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.eval()
    return processor, model


def release_model(*objects) -> None:
    """Release one rank before loading the next large model."""
    for value in objects:
        del value
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _chat_text(processor, messages, *, generation_prompt: bool) -> str:
    """Render Gemma chat text; tokenization is delegated to the processor."""
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation_prompt,
        enable_thinking=False,
    ).strip()


def _text_ids(processor, text: str) -> list[int]:
    """Normalize processor output across Transformers versions."""
    encoded = processor(text=text, return_tensors=None)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _assistant_boundary(processor, row: dict[str, Any]) -> tuple[list[int], int]:
    """Locate the assistant span despite small chat-template suffix differences.

    Some Gemma templates render an empty generation turn with control tokens
    that differ slightly from a completed assistant turn. The shared prefix is
    therefore the reliable masking boundary; a strict proximity gate rejects
    any mismatch that occurs inside the actual user prompt.
    """
    prompt_text = _chat_text(processor, row["messages"][:-1], generation_prompt=True)
    full_text = _chat_text(processor, row["messages"], generation_prompt=False)
    prompt_ids = _text_ids(processor, prompt_text)
    full_ids = _text_ids(processor, full_text)
    boundary = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        boundary += 1

    allowed_template_suffix = 32
    if boundary < len(prompt_ids) - allowed_template_suffix:
        raise RuntimeError(
            f"Chat-template mismatch occurs inside the prompt: {row['example_id']} "
            f"(shared {boundary}/{len(prompt_ids)} tokens)"
        )
    if boundary >= len(full_ids):
        raise RuntimeError(f"Assistant target is empty: {row['example_id']}")
    return full_ids, boundary


def token_diagnostics(examples, processor, max_sequence_length: int):
    """Measure exact Gemma sequence lengths before training and reject truncation."""
    lengths = []
    target_lengths = []
    for row in examples:
        full_ids, boundary = _assistant_boundary(processor, row)
        lengths.append(len(full_ids))
        target_lengths.append(len(full_ids) - boundary)
    over_limit = sum(length > max_sequence_length for length in lengths)
    return {
        "examples": len(lengths),
        "minimum_tokens": min(lengths),
        "median_tokens": int(sorted(lengths)[len(lengths) // 2]),
        "maximum_tokens": max(lengths),
        "maximum_target_tokens": max(target_lengths),
        "over_limit": over_limit,
        "max_sequence_length": max_sequence_length,
    }


class QLoRAChatDataset:
    """Expose raw chat examples; the multimodal processor runs in the collator."""

    def __init__(self, examples, processor, max_sequence_length: int):
        self.examples = examples
        self.processor = processor
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CompletionCollator:
    """Process full chats and mask prompt/padding tokens from supervised loss."""

    def __init__(self, processor, max_sequence_length: int):
        self.processor = processor
        self.max_sequence_length = max_sequence_length

    def __call__(self, features):
        texts = [_chat_text(self.processor, row["messages"], generation_prompt=False) for row in features]
        boundaries = [_assistant_boundary(self.processor, row)[1] for row in features]
        batch = self.processor(text=texts, return_tensors="pt", padding=True)
        if batch["input_ids"].shape[1] > self.max_sequence_length:
            raise RuntimeError("A QLoRA batch exceeds max_sequence_length; truncation is disabled")

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        left_padding = self.processor.tokenizer.padding_side == "left"
        for index, boundary in enumerate(boundaries):
            padding = int((batch["attention_mask"][index] == 0).sum()) if left_padding else 0
            labels[index, padding:padding + boundary] = -100
        batch["labels"] = labels
        return dict(batch)


def _experiment_identity(name, settings, contract, dataset_manifest):
    return {
        "schema_version": "qlora_training_identity_v1",
        "name": name,
        "settings": settings,
        "shared_training": {
            key: contract[key]
            for key in (
                "base_model_id", "seed", "max_sequence_length", "num_train_epochs",
                "learning_rate", "per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "logging_steps", "warmup_ratio",
                "lr_scheduler_type", "weight_decay", "max_grad_norm",
                "save_total_limit", "lora_dropout",
            )
        },
        "dataset_identity": dataset_manifest["identity_fingerprint"],
    }


def _final_adapter_valid(adapter_dir: Path, manifest_path: Path, expected: str) -> bool:
    if not manifest_path.is_file() or not (adapter_dir / "adapter_config.json").is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("identity_fingerprint") == expected


def train_or_load_adapter(
    *, name: str, settings: dict[str, Any], contract: dict[str, Any],
    train_examples, validation_examples, dataset_manifest: dict[str, Any],
    adapters_root: Path, checkpoints_root: Path, reports_root: Path,
):
    """LOAD a matching adapter or BUILD/RESUME one controlled two-epoch run."""
    from transformers.trainer_utils import get_last_checkpoint

    identity = _experiment_identity(name, settings, contract, dataset_manifest)
    identity_fingerprint = fingerprint(identity)
    adapter_dir = Path(adapters_root) / name
    checkpoint_dir = Path(checkpoints_root) / name
    report_dir = Path(reports_root) / name
    manifest_path = report_dir / "training_manifest.json"
    history_path = report_dir / "trainer_history.json"
    for path in (adapter_dir, checkpoint_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    if _final_adapter_valid(adapter_dir, manifest_path, identity_fingerprint):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"[LOAD] {name}: compatible final adapter")
        return {"action": "LOAD", "adapter_dir": adapter_dir, "manifest": manifest, "history": history}

    processor, model = load_quantized_base(contract["base_model_id"], training=True)
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments

    lora = LoraConfig(
        r=settings["rank"],
        lora_alpha=settings["alpha"],
        lora_dropout=contract["lora_dropout"],
        # PEFT >= 0.19 provides Gemma 4 defaults scoped to language layers.
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    train_dataset = QLoRAChatDataset(train_examples, processor, contract["max_sequence_length"])
    validation_dataset = QLoRAChatDataset(validation_examples, processor, contract["max_sequence_length"])
    diagnostics = {
        "train": token_diagnostics(train_examples, processor, contract["max_sequence_length"]),
        "validation": token_diagnostics(validation_examples, processor, contract["max_sequence_length"]),
    }
    if diagnostics["train"]["over_limit"] or diagnostics["validation"]["over_limit"]:
        raise RuntimeError("QLoRA token diagnostics failed; reduce context without truncating Gold evidence")

    use_bf16 = __import__("torch").cuda.is_bf16_supported()
    updates_per_epoch = math.ceil(
        len(train_dataset)
        / (
            contract["per_device_train_batch_size"]
            * contract["gradient_accumulation_steps"]
        )
    )
    warmup_steps = math.ceil(
        updates_per_epoch
        * contract["num_train_epochs"]
        * contract["warmup_ratio"]
    )
    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=contract["num_train_epochs"],
        learning_rate=contract["learning_rate"],
        per_device_train_batch_size=contract["per_device_train_batch_size"],
        per_device_eval_batch_size=contract["per_device_eval_batch_size"],
        gradient_accumulation_steps=contract["gradient_accumulation_steps"],
        logging_steps=contract["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Transformers 5 removed warmup_ratio; the equivalent fixed number of
        # optimizer warm-up steps also makes the experiment manifest explicit.
        warmup_steps=warmup_steps,
        lr_scheduler_type=contract["lr_scheduler_type"],
        weight_decay=contract["weight_decay"],
        max_grad_norm=contract["max_grad_norm"],
        save_total_limit=contract["save_total_limit"],
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=contract["seed"],
        data_seed=contract["seed"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionCollator(processor, contract["max_sequence_length"]),
    )
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    action = "RESUME" if last_checkpoint else "BUILD"
    print(f"[{action}] {name}: r={settings['rank']}, alpha={settings['alpha']}, 2 epochs")
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    history = trainer.state.log_history
    write_json_atomic(history, history_path)
    manifest = {
        "schema_version": "qlora_training_manifest_v1",
        "identity": identity,
        "identity_fingerprint": identity_fingerprint,
        "adapter_dir": str(adapter_dir),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percentage": 100.0 * trainable / total,
        "train_metrics": result.metrics,
        "token_diagnostics": diagnostics,
        "versions": {
            name: _package_version(name)
            for name in ("transformers", "peft", "bitsandbytes", "accelerate")
        },
    }
    write_json_atomic(manifest, manifest_path)
    output = {"action": action, "adapter_dir": adapter_dir, "manifest": manifest, "history": history}
    # Drop the large cyclic Trainer/model graph before allocating the next rank.
    del trainer, model, processor
    release_model()
    return output


def training_history_rows(results: dict[str, dict[str, Any]]):
    """Normalize Trainer logs for compact notebook plots."""
    rows = []
    for experiment, result in results.items():
        for item in result["history"]:
            if "loss" in item:
                rows.append({
                    "experiment": experiment, "series": "Train loss",
                    "step": item.get("step"), "epoch": item.get("epoch"), "loss": item["loss"],
                })
            if "eval_loss" in item:
                rows.append({
                    "experiment": experiment, "series": "Validation loss",
                    "step": item.get("step"), "epoch": item.get("epoch"), "loss": item["eval_loss"],
                })
    return rows


def load_adapter_generator(adapter_dir: Path, generation_contract, system_prompt):
    """Create a GroundedGenerator whose only changed component is one QLoRA adapter."""
    from .generation import GroundedGenerator

    class AdapterGroundedGenerator(GroundedGenerator):
        def load_model(self):
            if self._model is not None:
                return self._processor, self._model
            from peft import PeftModel
            processor, base = load_quantized_base(self.contract["model_id"], training=False)
            self._processor = processor
            self._model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
            return self._processor, self._model

    return AdapterGroundedGenerator(contract=generation_contract, system_prompt=system_prompt)
