"""Behavioral comparison and Validation-only selection for QLoRA adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from .evaluation import score_generation


def score_qlora_generation(records, generated, point_encoder, threshold: float):
    """Reuse the frozen metrics and add citation F1, invalid rate and ROUGE-L."""
    from rouge_score import rouge_scorer

    rows, nested = score_generation(records, generated, point_encoder, threshold)
    record_by_id = {record["query_id"]: record for record in records}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_values = []
    for row in rows:
        record = record_by_id[row["query_id"]]
        if record["answerability"] != "answerable":
            row["rouge_l"] = None
            continue
        reference = " ".join(record["expected_answer_points"])
        row["rouge_l"] = scorer.score(reference, row["answer"])["rougeL"].fmeasure
        rouge_values.append(row["rouge_l"])

    answerable = [row for row in rows if row["answerability"] == "answerable"]
    multi = [
        row for row in answerable
        if record_by_id[row["query_id"]]["query_type"] == "multi_ticket"
    ]
    recall = nested["answerable"]["gold_citation_recall"]
    precision = nested["answerable"]["gold_citation_precision"]
    citation_f1 = 0.0 if not recall or not precision else 2 * recall * precision / (recall + precision)
    flat = {
        "citation_presence": nested["answerable"]["citation_presence"],
        "valid_context_citations": nested["answerable"]["all_citations_from_context"],
        "invalid_citation_rate": 1.0 - nested["answerable"]["all_citations_from_context"],
        "gold_citation_recall": recall,
        "gold_citation_precision": precision,
        "citation_f1": citation_f1,
        "expected_point_coverage": nested["answerable"]["point_coverage"],
        "all_points_covered": nested["answerable"]["all_points_covered"],
        "language_match": nested["answerable"]["language_match"],
        "no_answer_abstention": nested["no_answer"]["abstention_rate"],
        "rouge_l": float(np.mean(rouge_values)),
        "multi_source_citation_recall": (
            float(np.mean([row["gold_citation_recall"] for row in multi])) if multi else None
        ),
        "answerable_questions": len(answerable),
        "no_answer_questions": len(rows) - len(answerable),
    }
    return rows, flat


def select_adapter(validation_metrics: dict[str, dict[str, float]], ranks: dict[str, int]):
    """Select on Validation only, protecting grounding before fluency proxies."""
    if set(validation_metrics) != set(ranks):
        raise RuntimeError("Adapter metrics and rank declarations differ")

    def selection_key(name):
        metrics = validation_metrics[name]
        return (
            -metrics["invalid_citation_rate"],
            metrics["no_answer_abstention"],
            metrics["citation_f1"],
            metrics["expected_point_coverage"],
            -ranks[name],
        )

    selected = max(validation_metrics, key=selection_key)
    return {
        "selected_adapter": selected,
        "selected_rank": ranks[selected],
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_order": [
            "minimum invalid citation rate",
            "maximum no-answer abstention",
            "maximum Gold citation F1",
            "maximum expected-point coverage",
            "smaller rank when otherwise tied",
        ],
        "selection_tuple": list(selection_key(selected)),
    }


def comparison_rows(metrics_by_system: dict[str, dict[str, Any]]):
    """Create a presentation-ready long table without hiding metric definitions."""
    labels = {
        "citation_f1": "Gold citation F1",
        "gold_citation_recall": "Gold citation recall",
        "gold_citation_precision": "Gold citation precision",
        "expected_point_coverage": "Expected-point coverage",
        "all_points_covered": "All points covered",
        "invalid_citation_rate": "Invalid citation rate",
        "no_answer_abstention": "No-answer abstention",
        "multi_source_citation_recall": "Multi-source citation recall",
        "rouge_l": "ROUGE-L",
    }
    return [
        {"system": system, "metric": label, "value": metrics[key]}
        for system, metrics in metrics_by_system.items()
        for key, label in labels.items()
    ]


def failure_analysis(base_rows, tuned_rows, records):
    """Expose regressions and remaining errors for honest qualitative review."""
    base = {row["query_id"]: row for row in base_rows}
    tuned = {row["query_id"]: row for row in tuned_rows}
    record_by_id = {row["query_id"]: row for row in records}
    findings = []
    for query_id in sorted(tuned):
        old, new = base[query_id], tuned[query_id]
        reasons = []
        for key, label in (
            ("gold_citation_recall", "lower Gold citation recall"),
            ("gold_citation_precision", "lower Gold citation precision"),
            ("point_coverage", "lower expected-point coverage"),
        ):
            if old[key] is not None and new[key] is not None and new[key] < old[key]:
                reasons.append(label)
        if not new["all_citations_from_context"]:
            reasons.append("citation outside supplied context")
        if record_by_id[query_id]["answerability"] == "no_answer" and not new["abstained"]:
            reasons.append("failed to abstain")
        if reasons:
            findings.append({
                "query_id": query_id,
                "question": record_by_id[query_id]["question"],
                "reasons": "; ".join(reasons),
                "base_answer": old["answer"],
                "tuned_answer": new["answer"],
            })
    return findings
