"""Grounded Gemma generation and stable baseline-demo persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_json_atomic


MessageBuilder = Callable[[str, str], tuple[list[dict[str, str]], str]]
_CITATION_PATTERN = re.compile(r"\[(tckt-\d{4})\]")


def load_generation_contract(path: Path) -> dict[str, Any]:
    """Load and validate the versioned baseline-generation contract."""
    with Path(path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    required = {
        "schema_version",
        "artifact_version",
        "model_id",
        "prompt_version",
        "prompt_file",
        "default_top_k",
        "max_input_tokens",
        "max_new_tokens",
        "do_sample",
        "require_cuda",
        "context_fields",
        "excluded_context_fields",
    }
    if set(contract) != required:
        raise RuntimeError("Generation contract schema is invalid")
    for field in ("default_top_k", "max_input_tokens", "max_new_tokens"):
        value = contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Generation contract requires positive {field}")
    if contract["do_sample"] is not False:
        raise RuntimeError("The baseline generation contract must be deterministic")
    return contract


def load_system_prompt(configs_root: Path, contract: dict[str, Any]) -> tuple[str, str]:
    """Load the baseline system prompt and return its text and SHA-256 identity."""
    prompt_path = Path(configs_root) / contract["prompt_file"]
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError("The baseline system prompt is empty")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return prompt, prompt_sha256


def validate_citations(answer: str, retrieved_ids: list[str]) -> dict[str, Any]:
    """Validate that bracketed ticket citations belong to retrieved evidence."""
    cited_ids = list(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
    retrieved_set = set(retrieved_ids)
    valid = [ticket_id for ticket_id in cited_ids if ticket_id in retrieved_set]
    invalid = [ticket_id for ticket_id in cited_ids if ticket_id not in retrieved_set]
    return {
        "has_citations": bool(cited_ids),
        "has_valid_citation": bool(valid),
        "all_citations_valid": bool(cited_ids) and not invalid,
        "cited_ticket_ids": cited_ids,
        "valid_citations": valid,
        "invalid_citations": invalid,
    }


def build_generation_identity(
    *,
    question: str,
    retrieved_ticket_ids: list[str],
    corpus_fingerprint: str,
    prompt_sha256: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Define when a saved baseline answer may be loaded rather than regenerated."""
    identity = {
        "artifact_version": contract["artifact_version"],
        "model_id": contract["model_id"],
        "prompt_version": contract["prompt_version"],
        "prompt_sha256": prompt_sha256,
        "question": question,
        "top_k": contract["default_top_k"],
        "retrieved_ticket_ids": retrieved_ticket_ids,
        "corpus_fingerprint": corpus_fingerprint,
        "max_input_tokens": contract["max_input_tokens"],
        "max_new_tokens": contract["max_new_tokens"],
        "do_sample": contract["do_sample"],
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    identity["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return identity


def ensure_generation_artifact(
    artifact_path: Path,
    identity: dict[str, Any],
    build_result: Callable[[], dict[str, Any]],
    result_validator: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, Any], str]:
    """LOAD a matching baseline answer or BUILD and atomically persist it once."""
    artifact_path = Path(artifact_path)
    if artifact_path.is_file():
        try:
            with artifact_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            saved_result = saved.get("result")
            valid_result = isinstance(saved_result, dict) and (
                result_validator is None or result_validator(saved_result)
            )
            if saved.get("identity") == identity and valid_result:
                return saved_result, "LOAD"
        except (OSError, json.JSONDecodeError):
            pass

    result = build_result()
    if result_validator is not None and not result_validator(result):
        raise RuntimeError("Generated result failed validation and was not persisted")
    write_json_atomic(
        {
            "schema_version": "grounded_generation_artifact_v1",
            "identity": identity,
            "result": result,
        },
        artifact_path,
    )
    return result, "BUILD"


class GroundedGenerator:
    """Generate natural-language answers grounded only in retrieved Jira tickets."""

    def __init__(
        self,
        *,
        contract: dict[str, Any],
        system_prompt: str,
    ) -> None:
        self.contract = contract
        self.system_prompt = system_prompt
        self._processor = None
        self._model = None

    @staticmethod
    def _optional_hf_token() -> str | None:
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        try:
            from google.colab import userdata

            return userdata.get("HF_TOKEN")
        except (ImportError, KeyError, AttributeError):
            return None
        except Exception as error:
            if type(error).__name__ in {"SecretNotFoundError", "NotebookAccessError"}:
                return None
            raise

    def load_model(self):
        """Load Gemma once on any CUDA GPU, preferring bfloat16 when supported."""
        if self._model is not None:
            return self._processor, self._model

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if self.contract["require_cuda"] and not torch.cuda.is_available():
            raise RuntimeError("Gemma generation requires a CUDA GPU for this project")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        token = self._optional_hf_token()
        model_id = self.contract["model_id"]
        print(f"[INFO] Loading {model_id} on {torch.cuda.get_device_name(0)} ({dtype})...")

        self._processor = AutoProcessor.from_pretrained(model_id, token=token)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            token=token,
        )
        self._model.eval()
        return self._processor, self._model

    def build_messages(self, question: str, context_text: str):
        """Build the baseline Prompt Engineering message contract."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context Evidence:\n{context_text}\n\nQuestion: {question}",
            },
        ]
        return messages, self.system_prompt

    @staticmethod
    def _validate_inputs(question: str, retrieval_results: list[dict[str, Any]]) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        if not retrieval_results:
            raise ValueError("Retrieval results cannot be empty")
        required = {
            "rank",
            "score",
            "document_id",
            "content",
            "metadata",
            "evaluation",
            "system",
        }
        if any(not required.issubset(result) for result in retrieval_results):
            raise ValueError("A retrieval result does not follow the public schema")
        return question.strip()

    def _context_block(self, result: dict[str, Any], description: str) -> str:
        """Render only approved evidence fields; ranks, scores and Gold labels stay out."""
        return (
            f"Ticket ID: {result['document_id']}\n"
            f"Summary: {result['content']['summary']}\n"
            f"Component: {result['metadata']['component']}\n"
            f"Work Type: {result['metadata']['work_type']}\n"
            f"Status: {result['metadata']['status']}\n"
            f"Priority: {result['metadata']['priority']}\n"
            f"Description:\n{description}\n"
            "---\n"
        )

    def build_context(
        self,
        question: str,
        retrieval_results: list[dict[str, Any]],
        processor: Any,
        message_builder: MessageBuilder,
    ) -> tuple[str, dict[str, Any]]:
        """Fit complete sources into one token budget, truncating descriptions last."""

        def token_count(blocks: list[str]) -> int:
            context = "\n".join(blocks)
            messages, _ = message_builder(question, context)
            tokens = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return len(tokens)

        included_ids: list[str] = []
        truncated_ids: list[str] = []
        omitted_ids: list[str] = []
        blocks: list[str] = []

        for result in retrieval_results:
            ticket_id = result["document_id"]
            description = result["content"]["description"]
            full_block = self._context_block(result, description)
            if token_count(blocks + [full_block]) <= self.contract["max_input_tokens"]:
                blocks.append(full_block)
                included_ids.append(ticket_id)
                continue

            empty_block = self._context_block(result, "[TRUNCATED]")
            if token_count(blocks + [empty_block]) > self.contract["max_input_tokens"]:
                omitted_ids.append(ticket_id)
                continue

            description_tokens = processor.tokenizer.encode(
                description,
                add_special_tokens=False,
            )
            low, high = 0, len(description_tokens)
            best_description = "[TRUNCATED]"
            while low <= high:
                midpoint = (low + high) // 2
                partial = processor.tokenizer.decode(description_tokens[:midpoint])
                candidate = self._context_block(
                    result,
                    partial + "... [TRUNCATED]",
                )
                if token_count(blocks + [candidate]) <= self.contract["max_input_tokens"]:
                    best_description = partial + "... [TRUNCATED]"
                    low = midpoint + 1
                else:
                    high = midpoint - 1

            blocks.append(self._context_block(result, best_description))
            included_ids.append(ticket_id)
            truncated_ids.append(ticket_id)

        if not blocks:
            raise RuntimeError("The token budget cannot fit one retrieved source")
        context_text = "\n".join(blocks)
        metadata = {
            "included_ticket_ids": included_ids,
            "truncated_ticket_ids": truncated_ids,
            "omitted_ticket_ids": omitted_ids,
            "input_token_count": token_count(blocks),
            "context_fields": list(self.contract["context_fields"]),
            "excluded_context_fields": list(
                self.contract["excluded_context_fields"]
            ),
        }
        return context_text, metadata

    def generate(
        self,
        question: str,
        retrieval_results: list[dict[str, Any]],
        *,
        message_builder: MessageBuilder | None = None,
        max_new_tokens: int | None = None,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        """Generate one deterministic answer from retrieved evidence."""
        question = self._validate_inputs(question, retrieval_results)
        message_builder = message_builder or self.build_messages
        if max_new_tokens is None:
            max_new_tokens = self.contract["max_new_tokens"]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        import torch

        processor, model = self.load_model()
        context_text, context_metadata = self.build_context(
            question,
            retrieval_results,
            processor,
            message_builder,
        )
        messages, _ = message_builder(question, context_text)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[-1]
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.contract["do_sample"],
            )
        latency = time.perf_counter() - started
        generated_tokens = outputs[0][input_length:]
        answer = processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
        if not answer:
            raise RuntimeError("Gemma generated an empty response")

        return {
            "question": question,
            "answer": answer,
            "retrieved_ticket_ids": [
                result["document_id"] for result in retrieval_results
            ],
            "retrieval_results": retrieval_results,
            "context_metadata": context_metadata,
            "generation_metadata": {
                "model_id": self.contract["model_id"],
                "prompt_version": prompt_version or self.contract["prompt_version"],
                "top_k": len(retrieval_results),
                "device": str(model.device),
                "dtype": str(model.dtype),
                "max_input_tokens": self.contract["max_input_tokens"],
                "input_token_count": context_metadata["input_token_count"],
                "generated_token_count": len(generated_tokens),
                "max_new_tokens": max_new_tokens,
                "generation_latency_seconds": round(latency, 3),
            },
        }

    def release_model(self) -> None:
        """Release generator memory before another large model is loaded."""
        import gc

        self._processor = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def validate_grounded_result(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate structural grounding without claiming factual answer quality."""
    retrieved_ids = result.get("retrieved_ticket_ids", [])
    citations = validate_citations(result.get("answer", ""), retrieved_ids)
    context_metadata = result.get("context_metadata", {})
    checks = {
        "non_empty_answer": bool(result.get("answer", "").strip()),
        "valid_citation_present": citations["has_valid_citation"],
        "all_citations_from_context": citations["all_citations_valid"],
        "token_budget_respected": (
            context_metadata.get("input_token_count", contract["max_input_tokens"] + 1)
            <= contract["max_input_tokens"]
        ),
        "context_schema_enforced": (
            context_metadata.get("context_fields") == contract["context_fields"]
            and context_metadata.get("excluded_context_fields")
            == contract["excluded_context_fields"]
        ),
        "model_contract_matched": (
            result.get("generation_metadata", {}).get("model_id")
            == contract["model_id"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "citations": citations,
    }
