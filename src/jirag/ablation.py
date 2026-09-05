"""Controlled generation ablations for separating RAG and QLoRA effects."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from .evaluation import _abstained, _language_match


_CITATION = re.compile(r"\[(tckt-\d{4})\]")


class NoRAGGenerator:
    """Use an existing Base or adapter generator without retrieved evidence."""

    def __init__(self, generator, message_builder, prompt_version: str):
        self.generator = generator
        self.message_builder = message_builder
        self.prompt_version = prompt_version

    def generate(self, question: str, retrieval_results, **_) -> dict[str, Any]:
        """Generate deterministically while deliberately supplying empty Context."""
        import torch

        processor, model = self.generator.load_model()
        messages, _ = self.message_builder(question, "")
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
                max_new_tokens=self.generator.contract["max_new_tokens"],
                do_sample=False,
            )
        answer = processor.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()
        if not answer:
            raise RuntimeError("No-RAG ablation generated an empty response")
        return {
            "question": question,
            "answer": answer,
            "retrieved_ticket_ids": [],
            "retrieval_results": [],
            "context_metadata": {
                "ablation": "no_rag",
                "evidence_supplied": False,
                "included_ticket_ids": [],
            },
            "generation_metadata": {
                "model_id": self.generator.contract["model_id"],
                "prompt_version": self.prompt_version,
                "do_sample": False,
                "latency_seconds": time.perf_counter() - started,
            },
        }

    def release_model(self):
        self.generator.release_model()


def score_no_rag_generation(records, generated, point_encoder, threshold: float):
    """Score no-evidence answers without pretending citation precision is defined."""
    from rouge_score import rouge_scorer

    record_by_id = {row["query_id"]: row for row in records}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []
    for result in generated:
        record = record_by_id[result["query_id"]]
        answer = result["answer"]
        citations = list(dict.fromkeys(_CITATION.findall(answer)))
        point_scores = []
        if record["expected_answer_points"]:
            answer_vector = point_encoder.encode(
                ["passage: " + answer], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            point_vectors = point_encoder.encode(
                ["query: " + point for point in record["expected_answer_points"]],
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            point_scores = (point_vectors @ answer_vector.T).reshape(-1).astype(float).tolist()
        reference = " ".join(record["expected_answer_points"])
        rows.append({
            "query_id": record["query_id"],
            "answerability": record["answerability"],
            "language": record["language"],
            "answer": answer,
            "cited_ticket_ids": citations,
            "abstained": float(_abstained(answer)),
            "language_match": float(_language_match(answer, record["language"])),
            "point_coverage": None if not point_scores else float(
                np.mean(np.asarray(point_scores) >= threshold)
            ),
            "all_points_covered": None if not point_scores else float(
                all(score >= threshold for score in point_scores)
            ),
            "rouge_l": None if not reference else scorer.score(reference, answer)["rougeL"].fmeasure,
        })

    answerable = [row for row in rows if row["answerability"] == "answerable"]
    no_answer = [row for row in rows if row["answerability"] == "no_answer"]
    mean = lambda group, key: float(np.mean([row[key] for row in group if row[key] is not None]))
    metrics = {
        "expected_point_coverage": mean(answerable, "point_coverage"),
        "all_points_covered": mean(answerable, "all_points_covered"),
        "gold_citation_recall": 0.0,
        "citation_precision": None,
        "invalid_citation_rate": float(np.mean([bool(row["cited_ticket_ids"]) for row in rows])),
        "no_answer_abstention": mean(no_answer, "abstained"),
        "answerable_abstention": mean(answerable, "abstained"),
        "overall_abstention": mean(rows, "abstained"),
        "language_match": mean(rows, "language_match"),
        "rouge_l": mean(answerable, "rouge_l"),
    }
    return rows, metrics


def four_way_metric_rows(metrics_by_system: dict[str, dict[str, Any]]):
    """Create one presentation table for the Base/QLoRA × No-RAG/RAG design."""
    labels = {
        "expected_point_coverage": "Expected-point coverage",
        "all_points_covered": "All points covered",
        "gold_citation_recall": "Gold citation recall",
        "invalid_citation_rate": "Invalid/invented citation rate",
        "no_answer_abstention": "No-answer abstention",
        "answerable_abstention": "Answerable-question abstention",
        "rouge_l": "ROUGE-L",
    }
    return [
        {"system": system, "metric": label, "value": metrics.get(key)}
        for system, metrics in metrics_by_system.items()
        for key, label in labels.items()
    ]
