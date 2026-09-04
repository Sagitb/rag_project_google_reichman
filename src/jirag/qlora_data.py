"""Leakage-safe supervised examples for grounded QLoRA answer generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import calculate_sha256, write_json_atomic
from .evaluation import fingerprint, load_jsonl, write_jsonl


SPLITS = ("train", "validation")
DESCRIPTION_HEADINGS = (
    "CONTEXT",
    "ISSUE OR REQUEST",
    "EXPECTED BEHAVIOR",
    "ACTUAL BEHAVIOR",
    "INVESTIGATION AND FINDINGS",
    "RESOLUTION",
    "VALIDATION",
)


def load_qlora_contract(path: Path) -> dict[str, Any]:
    """Load the single versioned contract governing data and both experiments."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != "qlora_experiment_v1":
        raise RuntimeError("Unsupported QLoRA experiment contract")
    if contract["context_ticket_count"] < 2:
        raise RuntimeError("QLoRA contexts require one source and at least one distractor")
    if not 0 < contract["supplement_fraction"] < 0.1:
        raise RuntimeError("The behavior-supplement fraction must remain small")
    if contract["num_train_epochs"] != 2:
        raise RuntimeError("The approved QLoRA experiment uses exactly two epochs")
    experiments = contract.get("experiments", {})
    expected = {"qlora_r8": (8, 16), "qlora_r16": (16, 32)}
    actual = {
        name: (settings.get("rank"), settings.get("alpha"))
        for name, settings in experiments.items()
    }
    if actual != expected:
        raise RuntimeError("Expected the approved r=8/16 experiments with constant alpha/r")
    return contract


def _description_sections(description: str) -> dict[str, str]:
    """Parse the seven approved Jira description sections without generating facts."""
    pattern = re.compile(
        r"(?m)^(" + "|".join(re.escape(value) for value in DESCRIPTION_HEADINGS) + r")\s*$"
    )
    matches = list(pattern.finditer(description))
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        sections[match.group(1)] = description[match.end():end].strip()
    if tuple(sections) != DESCRIPTION_HEADINGS or any(not value for value in sections.values()):
        raise RuntimeError("A QLoRA source does not contain the seven approved sections")
    return sections


def _context_block(document: dict[str, Any]) -> str:
    """Render exactly the evidence fields used by the frozen RAG generator."""
    return (
        f"Ticket ID: {document['document_id']}\n"
        f"Summary: {document['content']['summary']}\n"
        f"Component: {document['metadata']['component']}\n"
        f"Work Type: {document['metadata']['work_type']}\n"
        f"Status: {document['metadata']['status']}\n"
        f"Priority: {document['metadata']['priority']}\n"
        f"Description:\n{document['content']['description']}\n---"
    )


def _question(document: dict[str, Any]) -> str:
    summary = document["content"]["summary"]
    solution_type = document["evaluation"]["solution_type"]
    templates = {
        "solution-verified": (
            "Was the issue '{summary}' resolved, how was it resolved, and how was the fix validated?",
            "What verified resolution is documented for '{summary}'?",
            "Describe the completed fix and validation evidence for '{summary}'.",
        ),
        "solution-partial": (
            "What was improved for '{summary}', and what limitation or remaining work is documented?",
            "What partial resolution and current evidence are recorded for '{summary}'?",
            "Describe what has been addressed and what is still incomplete for '{summary}'.",
        ),
        "solution-workaround": (
            "What workaround is documented for '{summary}', and what prevents it from being a permanent fix?",
            "How can users currently mitigate '{summary}', and what limitation remains?",
            "Describe the temporary handling and current evidence for '{summary}'.",
        ),
        "solution-unresolved": (
            "What is currently known about '{summary}', and what remains unresolved?",
            "What evidence is available for '{summary}' even though no verified fix exists?",
            "Summarize the unresolved state and current validation findings for '{summary}'.",
        ),
    }
    if solution_type not in templates:
        raise RuntimeError(f"Unsupported solution type: {solution_type}")
    choices = templates[solution_type]
    choice = int(hashlib.sha256(document["document_id"].encode()).hexdigest(), 16) % len(choices)
    return choices[choice].format(summary=summary)


def _target(document: dict[str, Any]) -> str:
    sections = _description_sections(document["content"]["description"])
    solution_type = document["evaluation"]["solution_type"]
    opening = {
        "solution-verified": "The issue has a verified resolution.",
        "solution-partial": "The issue has only a partial resolution.",
        "solution-workaround": "A workaround is available, but it is not a permanent resolution.",
        "solution-unresolved": "The issue remains unresolved.",
    }[solution_type]
    return (
        f"{opening} {sections['RESOLUTION']} "
        f"Validation or current evidence: {sections['VALIDATION']} "
        f"[{document['document_id']}]"
    )


def _stable_order(documents: list[dict[str, Any]], key: str, seed: int):
    return sorted(
        documents,
        key=lambda item: hashlib.sha256(
            f"{seed}:{key}:{item['document_id']}".encode("utf-8")
        ).hexdigest(),
    )


def _nearest_same_split(
    source_id: str,
    split: str,
    count: int,
    documents_by_id: dict[str, dict[str, Any]],
    embeddings: np.ndarray,
    position_by_id: dict[str, int],
) -> list[str]:
    """Select hard semantic distractors while enforcing the source split."""
    source_position = position_by_id[source_id]
    scores = embeddings @ embeddings[source_position]
    candidates = [
        document_id
        for document_id, position in position_by_id.items()
        if document_id != source_id
        and documents_by_id[document_id]["system"]["split"] == split
    ]
    candidates.sort(key=lambda document_id: (-float(scores[position_by_id[document_id]]), document_id))
    if len(candidates) < count:
        raise RuntimeError(f"Not enough {split} distractors for {source_id}")
    return candidates[:count]


def _messages(prompt: str, question: str, context_documents, target: str):
    context = "\n\n".join(_context_block(document) for document in context_documents)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context Evidence:\n{context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": target},
    ]


def _single_example(
    source: dict[str, Any],
    distractors: list[dict[str, Any]],
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    context = _stable_order([source, *distractors], source["document_id"], seed)
    return {
        "example_id": f"{source['system']['split']}-single-{source['document_id']}",
        "example_type": "single_source",
        "messages": _messages(prompt, _question(source), context, _target(source)),
        "gold_ticket_ids": [source["document_id"]],
        "context_ticket_ids": [document["document_id"] for document in context],
        "source_solution_types": [source["evaluation"]["solution_type"]],
    }


def _multi_examples(
    documents: list[dict[str, Any]],
    maximum: int,
    prompt: str,
    seed: int,
    distractor_count: int,
    distractors_for: dict[str, list[str]],
    documents_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build explicitly comparative two-source questions from related real tickets."""
    grouped = defaultdict(list)
    for document in documents:
        grouped[document["evaluation"]["family"]].append(document)
    pairs = []
    for family in sorted(grouped):
        ordered = _stable_order(grouped[family], family, seed)
        for index in range(0, len(ordered) - 1, 2):
            first, second = ordered[index:index + 2]
            if first["evaluation"]["solution_type"] == second["evaluation"]["solution_type"]:
                continue
            pairs.append((first, second))
    pairs = pairs[:maximum]
    rows = []
    for position, (first, second) in enumerate(pairs, 1):
        forbidden = {first["document_id"], second["document_id"]}
        distractor_ids = []
        for candidate in distractors_for[first["document_id"]] + distractors_for[second["document_id"]]:
            if candidate not in forbidden and candidate not in distractor_ids:
                distractor_ids.append(candidate)
            if len(distractor_ids) == distractor_count:
                break
        context = [first, second, *[documents_by_id[value] for value in distractor_ids]]
        context = _stable_order(context, f"multi-{position}", seed)
        question = (
            "Compare the documented outcomes of these two related incidents: "
            f"'{first['content']['summary']}' and '{second['content']['summary']}'. "
            "What was done in each case?"
        )
        target = f"First incident: {_target(first)} Second incident: {_target(second)}"
        split = first["system"]["split"]
        rows.append({
            "example_id": f"{split}-multi-{position:03d}",
            "example_type": "multi_source",
            "messages": _messages(prompt, question, context, target),
            "gold_ticket_ids": [first["document_id"], second["document_id"]],
            "context_ticket_ids": [document["document_id"] for document in context],
            "source_solution_types": [
                first["evaluation"]["solution_type"], second["evaluation"]["solution_type"]
            ],
        })
    return rows


def _no_answer_examples(
    documents: list[dict[str, Any]],
    maximum: int,
    prompt: str,
    seed: int,
    distractors_for: dict[str, list[str]],
    documents_by_id: dict[str, dict[str, Any]],
    context_count: int,
) -> list[dict[str, Any]]:
    """Ask for an exact absent attribute and require an explicit grounded refusal."""
    eligible = [
        document for document in documents
        if not re.search(r"(?:\$|\bUSD\b|\bdollars?\b)", document["content"]["description"], re.I)
    ]
    selected = _stable_order(eligible, "no-answer", seed)[:maximum]
    eligible_ids = {document["document_id"] for document in eligible}
    rows = []
    for position, source in enumerate(selected, 1):
        ids = [source["document_id"]]
        for candidate in distractors_for[source["document_id"]]:
            if candidate in eligible_ids and candidate not in ids:
                ids.append(candidate)
        if len(ids) < context_count:
            fallback = _stable_order(eligible, f"no-answer-fallback-{position}", seed)
            for candidate in fallback:
                if candidate["document_id"] not in ids:
                    ids.append(candidate["document_id"])
                if len(ids) == context_count:
                    break
        if len(ids) < context_count:
            raise RuntimeError("Not enough cost-free evidence for a no-answer context")
        context = [documents_by_id[value] for value in ids[:context_count]]
        context = _stable_order(context, f"no-answer-{position}", seed)
        question = (
            "What was the exact cost in US dollars of resolving the incident "
            f"'{source['content']['summary']}'?"
        )
        target = "The supplied Jira evidence does not state the exact monetary cost, so it cannot be determined."
        split = source["system"]["split"]
        rows.append({
            "example_id": f"{split}-no-answer-{position:03d}",
            "example_type": "no_answer",
            "messages": _messages(prompt, question, context, target),
            "gold_ticket_ids": [],
            "context_ticket_ids": [document["document_id"] for document in context],
            "source_solution_types": [],
        })
    return rows


def _validate_examples(rows, split, source_ids, context_count):
    expected_types = {"single_source", "multi_source", "no_answer"}
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate {split} QLoRA example IDs")
    for row in rows:
        if row["example_type"] not in expected_types:
            raise RuntimeError("Unexpected QLoRA example type")
        if not 2 <= len(row["context_ticket_ids"]) <= context_count:
            raise RuntimeError("QLoRA context has an invalid ticket count")
        if not set(row["context_ticket_ids"]) <= source_ids:
            raise RuntimeError(f"Cross-split context leakage in {row['example_id']}")
        if not set(row["gold_ticket_ids"]) <= set(row["context_ticket_ids"]):
            raise RuntimeError(f"Gold source absent from {row['example_id']}")
        answer = row["messages"][-1]["content"]
        citations = set(re.findall(r"\[(tckt-\d{4})\]", answer))
        if citations != set(row["gold_ticket_ids"]):
            raise RuntimeError(f"Target citations do not match Gold in {row['example_id']}")
        serialized_input = json.dumps(row["messages"][:-1], ensure_ascii=False)
        if any(label in serialized_input for label in ("solution-verified", "solution-partial", "solution-workaround", "solution-unresolved")):
            raise RuntimeError(f"Protected label leaked into {row['example_id']}")


def _build_examples(documents_by_split, embeddings, mapping, contract, prompt):
    documents = [document for split in documents_by_split.values() for document in split]
    documents_by_id = {document["document_id"]: document for document in documents}
    position_by_id = {row["document_id"]: int(row["vector_id"]) for row in mapping}
    if set(documents_by_id) != set(position_by_id):
        raise RuntimeError("Vector mapping and RAG documents are not aligned")
    distractor_count = contract["context_ticket_count"] - 1
    distractors_for = {}
    for split in SPLITS:
        for document in documents_by_split[split]:
            distractors_for[document["document_id"]] = _nearest_same_split(
                document["document_id"], split, distractor_count,
                documents_by_id, embeddings, position_by_id,
            )

    output = {}
    for split in SPLITS:
        split_documents = documents_by_split[split]
        singles = [
            _single_example(
                source,
                [documents_by_id[value] for value in distractors_for[source["document_id"]]],
                prompt,
                contract["seed"],
            )
            for source in split_documents
        ]
        supplement_count = max(1, math.floor(len(split_documents) * contract["supplement_fraction"]))
        multis = _multi_examples(
            split_documents, supplement_count, prompt, contract["seed"],
            contract["context_ticket_count"] - 2, distractors_for, documents_by_id,
        )
        no_answers = _no_answer_examples(
            split_documents, supplement_count, prompt, contract["seed"],
            distractors_for, documents_by_id, contract["context_ticket_count"],
        )
        output[split] = singles + multis + no_answers
        _validate_examples(
            output[split], split,
            {document["document_id"] for document in split_documents},
            contract["context_ticket_count"],
        )
    return output


def _identity(documents_by_split, embeddings, mapping, contract, prompt):
    allowed_ids = {
        document["document_id"]
        for split in SPLITS
        for document in documents_by_split[split]
    }
    selected_mapping = [row for row in mapping if row["document_id"] in allowed_ids]
    selected_positions = [int(row["vector_id"]) for row in selected_mapping]
    return {
        "schema_version": "qlora_dataset_identity_v1",
        "contract": contract,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "document_fingerprints": {
            split: fingerprint(documents_by_split[split]) for split in SPLITS
        },
        "embeddings_sha256": hashlib.sha256(
            np.asarray(embeddings)[selected_positions].tobytes()
        ).hexdigest(),
        "mapping_fingerprint": fingerprint(selected_mapping),
    }


def _summary(examples):
    return {
        split: {
            "total": len(rows),
            **{
                kind: sum(row["example_type"] == kind for row in rows)
                for kind in ("single_source", "multi_source", "no_answer")
            },
        }
        for split, rows in examples.items()
    }


def ensure_qlora_datasets(
    *, documents_by_split, embeddings, mapping, contract_path: Path,
    configs_root: Path, output_dir: Path,
):
    """LOAD compatible examples or BUILD, validate and atomically publish them."""
    contract = load_qlora_contract(contract_path)
    prompt = (Path(configs_root) / contract["system_prompt_file"]).read_text(encoding="utf-8").strip()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: output_dir / f"qlora_{split}_{contract['dataset_version']}.jsonl"
        for split in SPLITS
    }
    manifest_path = output_dir / f"qlora_dataset_manifest_{contract['dataset_version']}.json"
    identity = _identity(documents_by_split, embeddings, mapping, contract, prompt)
    identity_fingerprint = fingerprint(identity)

    if manifest_path.is_file() and all(path.is_file() for path in paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("identity_fingerprint") == identity_fingerprint
            and manifest.get("file_sha256")
            == {split: calculate_sha256(path) for split, path in paths.items()}
        ):
            examples = {split: load_jsonl(path) for split, path in paths.items()}
            for split in SPLITS:
                _validate_examples(
                    examples[split], split,
                    {document["document_id"] for document in documents_by_split[split]},
                    contract["context_ticket_count"],
                )
            return {"action": "LOAD", "examples": examples, "manifest": manifest, "contract": contract}

    examples = _build_examples(documents_by_split, np.asarray(embeddings), mapping, contract, prompt)
    staged = {split: path.with_suffix(".tmp.jsonl") for split, path in paths.items()}
    for split in SPLITS:
        write_jsonl(staged[split], examples[split])
        load_jsonl(staged[split])
    for split in SPLITS:
        staged[split].replace(paths[split])
    manifest = {
        "schema_version": "qlora_dataset_manifest_v1",
        "dataset_version": contract["dataset_version"],
        "identity_fingerprint": identity_fingerprint,
        "summary": _summary(examples),
        "file_sha256": {split: calculate_sha256(path) for split, path in paths.items()},
        "integrity": {
            "train_validation_disjoint": not (
                {value for row in examples["train"] for value in row["context_ticket_ids"]}
                & {value for row in examples["validation"] for value in row["context_ticket_ids"]}
            ),
            "test_used": False,
            "gold_sources_present": True,
            "target_citations_exact": True,
            "protected_labels_excluded_from_input": True,
        },
    }
    required_true = (
        "train_validation_disjoint",
        "gold_sources_present",
        "target_citations_exact",
        "protected_labels_excluded_from_input",
    )
    if (
        not all(manifest["integrity"][key] for key in required_true)
        or manifest["integrity"]["test_used"] is not False
    ):
        raise RuntimeError("QLoRA dataset integrity failed")
    write_json_atomic(manifest, manifest_path)
    return {"action": "BUILD", "examples": examples, "manifest": manifest, "contract": contract}
