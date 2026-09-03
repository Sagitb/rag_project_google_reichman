"""Frozen benchmark, retrieval comparison and grounded-generation evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import write_json_atomic

_CITATION = re.compile(r"\[(tckt-\d{4})\]")


def load_evaluation_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != "unified_rag_evaluation_v1":
        raise RuntimeError("Unsupported evaluation contract")
    if contract["reranker_candidate_k"] < contract["retrieval_output_k"]:
        raise RuntimeError("Reranker candidate_k must cover the final Top-k")
    return contract


def load_gold_benchmark(path: Path, documents: list[dict[str, Any]]):
    """Load the reviewed benchmark and prove that Gold IDs belong to their split."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    by_id = {document["document_id"]: document for document in documents}
    if len(records) != 44 or len({r["query_id"] for r in records}) != 44:
        raise RuntimeError("The frozen benchmark must contain 44 unique questions")
    for record in records:
        if record["answerability"] == "answerable" and not record["gold_ticket_ids"]:
            raise RuntimeError(f"Missing Gold source: {record['query_id']}")
        if record["answerability"] == "no_answer" and record["gold_ticket_ids"]:
            raise RuntimeError(f"No-answer query has Gold sources: {record['query_id']}")
        for ticket_id in record["gold_ticket_ids"]:
            if ticket_id not in by_id or by_id[ticket_id]["system"]["split"] != record["evaluation_split"]:
                raise RuntimeError(f"Gold split leakage: {record['query_id']} / {ticket_id}")
    counts = {
        split: sum(r["evaluation_split"] == split for r in records)
        for split in ("validation", "test")
    }
    if counts != {"validation": 24, "test": 20}:
        raise RuntimeError(f"Unexpected benchmark split counts: {counts}")
    expected = {
        "validation": {"single_ticket": 20, "multi_ticket": 2, "no_answer": 2},
        "test": {"single_ticket": 16, "multi_ticket": 2, "no_answer": 2},
    }
    for split, design in expected.items():
        subset = [r for r in records if r["evaluation_split"] == split]
        actual = {
            "single_ticket": sum(r["query_type"] == "single_ticket" for r in subset),
            "multi_ticket": sum(r["query_type"] == "multi_ticket" for r in subset),
            "no_answer": sum(r["answerability"] == "no_answer" for r in subset),
        }
        if actual != design:
            raise RuntimeError(f"Unexpected {split} benchmark design: {actual}")
    if sum(r["language"] == "he" for r in records) != 8:
        raise RuntimeError("The frozen Hebrew robustness slice must contain 8 questions")
    return records


class BGEReranker:
    """Lazily score query-document pairs with the multilingual BGE cross-encoder."""

    def __init__(self, model_id: str, batch_size: int = 16, max_length: int = 512):
        self.model_id, self.batch_size, self.max_length = model_id, batch_size, max_length
        self.tokenizer = self.model = self.device = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[INFO] Loading reranker: {self.model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, dtype=dtype)
        self.model.to(self.device).eval()

    def search(self, query: str, candidates: list[dict[str, Any]], top_k: int):
        import torch
        self.load()
        pairs = [(query, item["content"]["embedding_text"]) for item in candidates]
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            encoded = self.tokenizer(
                pairs[start:start + self.batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                batch = self.model(**encoded).logits.reshape(-1).float().cpu().tolist()
            scores.extend(batch)
        ranked = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]["document_id"]))
        results = []
        for rank, (candidate, score) in enumerate(ranked[:top_k], 1):
            item = dict(candidate); item["rank"] = rank; item["score"] = float(score)
            results.append(item)
        return results

    def release(self):
        import gc
        self.tokenizer = self.model = self.device = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError:
            pass


def evaluate_retrieval(records, searchers: dict[str, Callable], output_k: int = 5):
    """Compare retrievers on answerable Gold queries; no-answer belongs to generation."""
    rows = []
    for system, search in searchers.items():
        for record in records:
            if record["answerability"] != "answerable":
                continue
            started = time.perf_counter(); results = search(record["question"])
            ids = [item["document_id"] for item in results[:output_k]]
            ranks = {ticket_id: (ids.index(ticket_id) + 1 if ticket_id in ids else None) for ticket_id in record["gold_ticket_ids"]}
            found = [rank for rank in ranks.values() if rank is not None]
            rows.append({
                "system": system, "query_id": record["query_id"], "language": record["language"],
                "query_type": record["query_type"], "gold_ticket_ids": record["gold_ticket_ids"],
                "retrieved_ticket_ids": ids, "gold_ranks": ranks,
                "hit_at_1": float(bool(found) and min(found) <= 1),
                "hit_at_5": float(bool(found)), "recall_at_5": len(found) / len(ranks),
                "complete_at_5": float(len(found) == len(ranks)),
                "reciprocal_rank_at_5": 0.0 if not found else 1.0 / min(found),
                "latency_seconds": time.perf_counter() - started,
            })
    metrics = {}
    for system in searchers:
        subset = [r for r in rows if r["system"] == system]
        metrics[system] = {key: float(np.mean([r[key] for r in subset])) for key in (
            "hit_at_1", "hit_at_5", "recall_at_5", "complete_at_5", "reciprocal_rank_at_5", "latency_seconds"
        )}
        metrics[system]["query_count"] = len(subset)
    return rows, metrics


def select_retriever(metrics: dict[str, dict[str, float]], policy: list[str]):
    key_map = {"complete_at_5":"complete_at_5", "recall_at_5":"recall_at_5", "mrr_at_5":"reciprocal_rank_at_5"}
    baseline = tuple(metrics["semantic_baseline"][key_map[k]] for k in policy)
    improved = tuple(metrics["semantic_reranked"][key_map[k]] for k in policy)
    selected = "semantic_reranked" if improved > baseline else "semantic_baseline"
    return {"selected_system": selected, "policy": policy, "baseline_tuple": baseline, "improved_tuple": improved}


def _abstained(answer: str) -> bool:
    text = answer.casefold()
    phrases = (
        "cannot determine", "not enough information", "insufficient information",
        "no evidence", "not supported by the provided", "sources do not contain",
        "איני יכול", "לא ניתן לקבוע", "אין מספיק מידע", "לא נמצא מידע",
        "אין מידע במקורות", "אין ראיות", "הטיקטים אינם מכילים",
    )
    return any(phrase in text for phrase in phrases)


def _language_match(answer: str, language: str) -> bool:
    hebrew = len(re.findall(r"[\u0590-\u05ff]", answer))
    latin = len(re.findall(r"[A-Za-z]", answer))
    return hebrew >= 10 if language == "he" else latin >= 10 and latin >= hebrew


def score_generation(
    records, generated: list[dict[str, Any]], point_encoder, threshold: float,
    query_prefix: str = "query: ", passage_prefix: str = "passage: ",
):
    """Score citations exactly and answer-point coverage with a documented semantic proxy."""
    record_by_id = {r["query_id"]: r for r in records}; rows = []
    for result in generated:
        record = record_by_id[result["query_id"]]; answer = result["answer"]
        retrieved = set(result["retrieved_ticket_ids"]); cited = list(dict.fromkeys(_CITATION.findall(answer)))
        gold = set(record["gold_ticket_ids"]); valid = set(cited) & retrieved
        point_scores = []
        if record["expected_answer_points"]:
            answer_vector = point_encoder.encode(
                [passage_prefix + answer], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            point_vectors = point_encoder.encode(
                [query_prefix + point for point in record["expected_answer_points"]],
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            )
            point_scores = (point_vectors @ answer_vector.T).reshape(-1).astype(float).tolist()
        rows.append({
            "query_id": record["query_id"], "answerability": record["answerability"], "language": record["language"],
            "answer": answer, "retrieved_ticket_ids": result["retrieved_ticket_ids"], "cited_ticket_ids": cited,
            "citation_presence": float(bool(cited)), "all_citations_from_context": float(bool(cited) and set(cited) <= retrieved),
            "gold_citation_recall": None if not gold else len(valid & gold) / len(gold),
            "gold_citation_precision": None if not cited else len(valid & gold) / len(cited),
            "point_scores": point_scores, "point_coverage": None if not point_scores else float(np.mean(np.array(point_scores) >= threshold)),
            "all_points_covered": None if not point_scores else float(all(score >= threshold for score in point_scores)),
            "language_match": float(_language_match(answer, record["language"])), "abstained": float(_abstained(answer)),
        })
    answerable = [r for r in rows if r["answerability"] == "answerable"]
    no_answer = [r for r in rows if r["answerability"] == "no_answer"]
    mean = lambda group,key: float(np.mean([r[key] for r in group if r[key] is not None]))
    metrics = {"answerable": {key: mean(answerable,key) for key in (
        "citation_presence","all_citations_from_context","gold_citation_recall","gold_citation_precision","point_coverage","all_points_covered","language_match")},
        "no_answer":{"abstention_rate":mean(no_answer,"abstained"),"language_match":mean(no_answer,"language_match")}}
    return rows, metrics


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def load_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_retrieval_contexts(path: Path, records, search: Callable, identity: dict[str, Any]):
    """LOAD matching retrieved contexts or BUILD them once for later generation."""
    path = Path(path); manifest_path = path.with_suffix(".manifest.json")
    expected = fingerprint(identity)
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = load_jsonl(path)
        if manifest.get("identity") == expected and len(rows) == len(records):
            return rows, "LOAD"
    rows = []
    for record in records:
        results = search(record["question"])
        rows.append({"query_id": record["query_id"], "retrieval_results": results})
    write_jsonl(path, rows)
    write_json_atomic({"identity": expected, "row_count": len(rows)}, manifest_path)
    return rows, "BUILD"


def ensure_generated_answers(
    path: Path,
    records,
    contexts,
    generator,
    message_builder,
    prompt_version: str,
    identity: dict[str, Any],
):
    """Resume deterministic generation query by query and reuse a completed artifact."""
    path = Path(path); manifest_path = path.with_suffix(".manifest.json")
    expected = fingerprint(identity); record_by_id = {r["query_id"]: r for r in records}
    completed = {}
    if path.is_file():
        completed = {r["query_id"]: r for r in load_jsonl(path)}
    if manifest_path.is_file() and len(completed) == len(records):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") == expected:
            return [completed[r["query_id"]] for r in records], "LOAD"
    for context in contexts:
        query_id = context["query_id"]
        if query_id in completed:
            continue
        result = generator.generate(
            record_by_id[query_id]["question"], context["retrieval_results"],
            message_builder=message_builder, prompt_version=prompt_version,
        )
        result["query_id"] = query_id; completed[query_id] = result
        write_jsonl(path, [completed[r["query_id"]] for r in records if r["query_id"] in completed])
    ordered = [completed[r["query_id"]] for r in records]
    write_json_atomic({"identity": expected, "row_count": len(ordered)}, manifest_path)
    return ordered, "BUILD"
