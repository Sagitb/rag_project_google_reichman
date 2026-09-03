"""Reproducible embedding-model selection for semantic RAG retrieval."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import calculate_sha256, write_json_atomic
from .documents import load_jsonl


SELECTION_MANIFEST_SCHEMA = "embedding_selection_manifest_v2"
METRIC_NAMES = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")


def stable_fingerprint(value: Any) -> str:
    """Fingerprint JSON-compatible experiment inputs deterministically."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_embedding_selection_contract(path: Path) -> dict[str, Any]:
    """Load and validate the versioned model-selection protocol."""
    contract = _load_json(path)
    sample = contract.get("sample", {})
    if sample.get("group_fields") != ["family", "solution_type"]:
        raise RuntimeError("Embedding-selection grouping contract is invalid")
    if sample.get("documents_per_group") != 2:
        raise RuntimeError("Embedding-selection sample size per group is invalid")
    if sample.get("expected_groups") * sample.get("documents_per_group") != sample.get(
        "expected_documents"
    ):
        raise RuntimeError("Embedding-selection sample totals are inconsistent")
    if contract.get("ranking_top_k", 0) < 5:
        raise RuntimeError("Embedding-selection ranking depth must support Hit@5")
    if len(contract.get("models", {})) < 2:
        raise RuntimeError("At least two embedding candidates are required")
    for model_key, model in contract["models"].items():
        required = {
            "model_name",
            "query_prefix",
            "document_prefix",
            "role",
            "trust_remote_code",
        }
        if set(model) != required:
            raise RuntimeError(f"{model_key}: embedding-model contract is invalid")
    return contract


def load_embedding_queries(path: Path) -> list[dict[str, Any]]:
    """Load the manually reviewed query/gold set tracked in Git."""
    queries = _load_json(path)
    if not isinstance(queries, list):
        raise RuntimeError("Embedding query source must be a JSON array")
    return queries


def select_balanced_train_documents(
    train_documents: list[dict[str, Any]],
    contract: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select two deterministic Train documents from every family/solution group."""
    rows = []
    document_lookup = {}
    for document in train_documents:
        if document["system"]["split"] != "train":
            raise RuntimeError("Non-Train document entered embedding model selection")
        document_id = document["document_id"]
        if document_id in document_lookup:
            raise RuntimeError(f"Duplicate Train document ID: {document_id}")
        document_lookup[document_id] = document
        rows.append(
            {
                "document_id": document_id,
                "family": document["evaluation"]["family"],
                "solution_type": document["evaluation"]["solution_type"],
            }
        )

    source = pd.DataFrame(rows).sort_values("document_id").reset_index(drop=True)
    sample = contract["sample"]
    group_fields = sample["group_fields"]
    observed_groups = source[group_fields].drop_duplicates().shape[0]
    if observed_groups != sample["expected_groups"]:
        raise RuntimeError(
            f"Expected {sample['expected_groups']} Train groups, found {observed_groups}"
        )

    selected = (
        source.groupby(group_fields, group_keys=False, sort=True)
        .sample(n=sample["documents_per_group"], random_state=seed)
        .sort_values([*group_fields, "document_id"])
        .reset_index(drop=True)
    )
    if len(selected) != sample["expected_documents"]:
        raise RuntimeError("Balanced embedding-selection sample has an invalid size")
    if not selected.groupby(group_fields).size().eq(sample["documents_per_group"]).all():
        raise RuntimeError("Embedding-selection groups are not equally represented")

    selected_documents = [
        document_lookup[document_id] for document_id in selected["document_id"]
    ]
    return selected, selected_documents


def validate_embedding_queries(
    queries: list[dict[str, Any]],
    pilot_documents: list[dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    """Block stale, leaked or incomplete query/gold definitions."""
    required_fields = {
        "query_id",
        "language",
        "query",
        "gold_ticket_ids",
        "gold_rationale",
        "query_type",
    }
    expected_counts = contract["query_counts"]
    if len(queries) != expected_counts["total"]:
        raise RuntimeError("Embedding query count differs from the contract")
    query_ids = [query.get("query_id") for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("Embedding query IDs are not unique")

    language_counts = pd.Series([query.get("language") for query in queries]).value_counts()
    for language in ("en", "he"):
        if int(language_counts.get(language, 0)) != expected_counts[language]:
            raise RuntimeError(f"Embedding query language count mismatch: {language}")

    pilot_ids = {document["document_id"] for document in pilot_documents}
    pilot_summaries = {
        document["content"]["summary"].strip().casefold()
        for document in pilot_documents
    }
    for query in queries:
        if set(query) != required_fields:
            raise RuntimeError(f"{query.get('query_id')}: query schema mismatch")
        text_fields = ("query_id", "language", "query", "gold_rationale", "query_type")
        if any(not isinstance(query[field], str) or not query[field].strip() for field in text_fields):
            raise RuntimeError(f"{query['query_id']}: empty query field")
        gold_ids = query["gold_ticket_ids"]
        if not isinstance(gold_ids, list) or not gold_ids or not set(gold_ids).issubset(pilot_ids):
            raise RuntimeError(f"{query['query_id']}: gold ID is absent from the pilot")
        normalized_query = query["query"].strip().casefold()
        if normalized_query in pilot_summaries:
            raise RuntimeError(f"{query['query_id']}: query copies a ticket summary")
        if any(document_id.casefold() in normalized_query for document_id in pilot_ids):
            raise RuntimeError(f"{query['query_id']}: ticket ID leaked into query text")


def _token_diagnostics(
    candidates: dict[str, dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Confirm that ticket-level documents fit each candidate context window."""
    from transformers import AutoTokenizer

    rows = []
    for model_key, model_config in candidates.items():
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_name"],
            trust_remote_code=model_config["trust_remote_code"],
        )
        lengths = np.asarray(
            [
                len(
                    tokenizer.encode(
                        model_config["document_prefix"]
                        + document["search"]["embedding_text"],
                        add_special_tokens=True,
                        truncation=False,
                    )
                )
                for document in documents
            ]
        )
        model_limit = tokenizer.model_max_length
        if not isinstance(model_limit, int) or model_limit > 1_000_000:
            model_limit = 512
        truncated = lengths > model_limit
        rows.append(
            {
                "model": model_key,
                "maximum_tokens": int(lengths.max()),
                "model_limit": int(model_limit),
                "truncated_documents": int(truncated.sum()),
            }
        )
    if any(row["truncated_documents"] for row in rows):
        raise RuntimeError("A pilot document exceeds an embedding-model token limit")
    return rows


def _metric_block(ranks: list[int]) -> dict[str, Any]:
    values = np.asarray(ranks, dtype=int)
    return {
        "query_count": int(len(values)),
        "hit_at_1": float(((values > 0) & (values <= 1)).mean()),
        "hit_at_3": float(((values > 0) & (values <= 3)).mean()),
        "hit_at_5": float(((values > 0) & (values <= 5)).mean()),
        "mrr": float(np.mean([1.0 / rank if rank > 0 else 0.0 for rank in values])),
    }


def _evaluate_model(
    model_key: str,
    model_config: dict[str, Any],
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Embed one shared corpus/query set and return auditable retrieval rankings."""
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(
        model_config["model_name"],
        device=device,
        trust_remote_code=model_config["trust_remote_code"],
    )
    load_seconds = time.perf_counter() - started
    document_texts = [
        model_config["document_prefix"] + document["search"]["embedding_text"]
        for document in documents
    ]
    query_texts = [model_config["query_prefix"] + query["query"] for query in queries]

    started = time.perf_counter()
    document_embeddings = model.encode(
        document_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    document_seconds = time.perf_counter() - started
    started = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_seconds = time.perf_counter() - started

    if document_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise RuntimeError(f"{model_key}: embedding output is not a matrix")
    if document_embeddings.shape[0] != len(documents):
        raise RuntimeError(f"{model_key}: document embedding count mismatch")
    if query_embeddings.shape[0] != len(queries):
        raise RuntimeError(f"{model_key}: query embedding count mismatch")
    if document_embeddings.shape[1] != query_embeddings.shape[1]:
        raise RuntimeError(f"{model_key}: document/query dimensions differ")
    if not np.isfinite(document_embeddings).all() or not np.isfinite(query_embeddings).all():
        raise RuntimeError(f"{model_key}: non-finite embedding values detected")

    similarities = query_embeddings @ document_embeddings.T
    rankings = []
    ranks_by_language = {"overall": [], "en": [], "he": []}
    for query_index, query in enumerate(queries):
        indices = np.argsort(-similarities[query_index], kind="stable")[:top_k]
        retrieved = [
            {
                "rank": rank,
                "ticket_id": documents[index]["document_id"],
                "summary": documents[index]["content"]["summary"],
                "score": float(similarities[query_index, index]),
            }
            for rank, index in enumerate(indices, start=1)
        ]
        gold_ids = set(query["gold_ticket_ids"])
        first_gold_rank = next(
            (item["rank"] for item in retrieved if item["ticket_id"] in gold_ids),
            0,
        )
        ranks_by_language["overall"].append(first_gold_rank)
        ranks_by_language[query["language"]].append(first_gold_rank)
        rankings.append(
            {
                "model": model_key,
                "query_id": query["query_id"],
                "language": query["language"],
                "query": query["query"],
                "gold_ticket_ids": query["gold_ticket_ids"],
                "first_gold_rank": first_gold_rank,
                "top_10": retrieved,
            }
        )

    metrics = {
        language: _metric_block(ranks)
        for language, ranks in ranks_by_language.items()
    }
    metrics["resources"] = {
        "load_seconds": float(load_seconds),
        "document_embedding_seconds": float(document_seconds),
        "query_embedding_seconds": float(query_seconds),
        "embedding_dimension": int(document_embeddings.shape[1]),
    }
    del model, document_embeddings, query_embeddings, similarities
    return metrics, rankings


def select_embedding_model(metrics: dict[str, Any]) -> dict[str, Any]:
    """Select by English retrieval first and Hebrew only after an English tie."""
    scored = []
    for model_key, values in metrics.items():
        english = values["en"]
        hebrew = values["he"]
        english_score = tuple(
            english[name] for name in ("hit_at_5", "mrr", "hit_at_3", "hit_at_1")
        )
        hebrew_score = tuple(
            hebrew[name] for name in ("hit_at_5", "mrr", "hit_at_3", "hit_at_1")
        )
        scored.append((english_score, hebrew_score, model_key))
    scored.sort(reverse=True)
    best_english, best_hebrew, best_model = scored[0]

    if best_english == (0.0, 0.0, 0.0, 0.0):
        raise RuntimeError("No embedding candidate retrieved an English gold ticket")
    fully_tied = [
        model_key
        for english_score, hebrew_score, model_key in scored
        if english_score == best_english and hebrew_score == best_hebrew
    ]
    if len(fully_tied) > 1:
        raise RuntimeError(f"Embedding candidates remain fully tied: {fully_tied}")

    english_tied = [
        model_key for english_score, _, model_key in scored if english_score == best_english
    ]
    used_hebrew = len(english_tied) > 1
    reason = (
        "English retrieval was tied; Hebrew-to-English retrieval resolved the tie."
        if used_hebrew
        else "The model achieved the strongest English retrieval score."
    )
    return {
        "selected_model": best_model,
        "status": "selected",
        "reason": reason,
        "confidence": "moderate",
        "english_score": list(best_english),
        "hebrew_tiebreaker": list(best_hebrew),
        "used_hebrew_tiebreaker": used_hebrew,
    }


def _save_comparison_figure(metrics: dict[str, Any], output_path: Path) -> None:
    """Persist one decision-focused chart for the primary English metrics."""
    model_names = list(metrics)
    positions = np.arange(len(model_names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - width / 2,
        [metrics[name]["en"]["hit_at_1"] for name in model_names],
        width,
        label="English Hit@1",
    )
    ax.bar(
        positions + width / 2,
        [metrics[name]["en"]["mrr"] for name in model_names],
        width,
        label="English MRR",
    )
    ax.set_title("Embedding Model Selection — English Retrieval Quality")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(positions, [name.replace("_", "\n") for name in model_names])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)


def _validate_results(
    metrics: dict[str, Any],
    rankings: list[dict[str, Any]],
    recommendation: dict[str, Any],
    contract: dict[str, Any],
    queries: list[dict[str, Any]],
) -> None:
    """Validate complete model/query coverage and metric/recommendation structure."""
    model_keys = set(contract["models"])
    query_ids = {query["query_id"] for query in queries}
    if set(metrics) != model_keys:
        raise RuntimeError("Embedding metrics do not cover every candidate model")
    for model_key in model_keys:
        for language in ("overall", "en", "he"):
            if not set(METRIC_NAMES).issubset(metrics[model_key][language]):
                raise RuntimeError(f"{model_key}: incomplete {language} metrics")
    expected_pairs = {(model, query_id) for model in model_keys for query_id in query_ids}
    actual_pairs = {(row.get("model"), row.get("query_id")) for row in rankings}
    if actual_pairs != expected_pairs or len(rankings) != len(expected_pairs):
        raise RuntimeError("Saved rankings do not cover every model/query pair")
    if recommendation.get("selected_model") not in model_keys:
        raise RuntimeError("Embedding recommendation is not a candidate model")


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: calculate_sha256(paths[name])
        for name in ("metrics", "rankings", "figure")
    }


def _manifest_compatible(
    manifest: dict[str, Any] | None,
    expected: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> bool:
    if manifest is None or not all(path.is_file() for path in artifact_paths.values()):
        return False
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    return manifest.get("artifact_sha256") == _artifact_hashes(artifact_paths)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return _load_json(path) if Path(path).is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_embedding_model_selection(
    train_documents: list[dict[str, Any]],
    source_train_fingerprint: str,
    config: dict[str, Any],
    contract_path: Path,
    queries_path: Path,
    artifact_paths: dict[str, Path],
    device: str,
) -> dict[str, Any]:
    """BUILD the embedding comparison once or LOAD fully compatible results."""
    artifact_paths = {name: Path(path) for name, path in artifact_paths.items()}
    contract_path, queries_path = Path(contract_path), Path(queries_path)
    contract = load_embedding_selection_contract(contract_path)
    if config.get("embedding_selection_version") != contract["experiment_version"]:
        raise RuntimeError("Embedding-selection config and contract versions differ")
    queries = load_embedding_queries(queries_path)
    selection, pilot_documents = select_balanced_train_documents(
        train_documents,
        contract,
        int(config["random_seed"]),
    )
    validate_embedding_queries(queries, pilot_documents, contract)

    expected_manifest = {
        "schema_version": SELECTION_MANIFEST_SCHEMA,
        "experiment_version": contract["experiment_version"],
        "protocol_version": contract["protocol_version"],
        "seed": int(config["random_seed"]),
        "source_train_rag_fingerprint": source_train_fingerprint,
        "pilot_fingerprint": stable_fingerprint(selection["document_id"].tolist()),
        "query_fingerprint": stable_fingerprint(queries),
        "contract_sha256": calculate_sha256(contract_path),
        "queries_sha256": calculate_sha256(queries_path),
    }
    manifest = _read_manifest(artifact_paths["manifest"])
    if _manifest_compatible(manifest, expected_manifest, artifact_paths):
        metrics = _load_json(artifact_paths["metrics"])
        rankings = load_jsonl(artifact_paths["rankings"])
        recommendation = manifest["recommendation"]
        _validate_results(metrics, rankings, recommendation, contract, queries)
        return {
            "action": "LOAD",
            "selection": selection,
            "pilot_documents": pilot_documents,
            "queries": queries,
            "metrics": metrics,
            "rankings": rankings,
            "recommendation": recommendation,
            "manifest": manifest,
            "token_diagnostics": pd.DataFrame(manifest["token_diagnostics"]),
            "contract": contract,
            "paths": artifact_paths,
        }

    artifact_root = artifact_paths["manifest"].parent
    artifact_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".embedding_selection_", dir=artifact_root))
    staged_paths = {name: staging_root / path.name for name, path in artifact_paths.items()}
    try:
        token_diagnostics = _token_diagnostics(contract["models"], pilot_documents)
        metrics = {}
        rankings = []
        for model_key, model_config in contract["models"].items():
            print(f"[INFO] Evaluating embedding model: {model_key}")
            model_metrics, model_rankings = _evaluate_model(
                model_key,
                model_config,
                pilot_documents,
                queries,
                contract["ranking_top_k"],
                device,
            )
            metrics[model_key] = model_metrics
            rankings.extend(model_rankings)
        recommendation = select_embedding_model(metrics)
        _validate_results(metrics, rankings, recommendation, contract, queries)

        write_json_atomic(metrics, staged_paths["metrics"])
        with staged_paths["rankings"].open("w", encoding="utf-8") as handle:
            for ranking in rankings:
                handle.write(json.dumps(ranking, ensure_ascii=False) + "\n")
        _save_comparison_figure(metrics, staged_paths["figure"])

        manifest = {
            **expected_manifest,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "pilot_document_ids": selection["document_id"].tolist(),
            "query_counts": contract["query_counts"],
            "candidate_models": contract["models"],
            "token_diagnostics": token_diagnostics,
            "recommendation": recommendation,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sentence_transformers": _package_version("sentence-transformers"),
                "transformers": _package_version("transformers"),
            },
            "artifact_sha256": _artifact_hashes(staged_paths),
        }
        write_json_atomic(manifest, staged_paths["manifest"])

        reloaded_metrics = _load_json(staged_paths["metrics"])
        reloaded_rankings = load_jsonl(staged_paths["rankings"])
        _validate_results(
            reloaded_metrics,
            reloaded_rankings,
            recommendation,
            contract,
            queries,
        )
        if staged_paths["figure"].stat().st_size == 0:
            raise RuntimeError("Embedding comparison figure is empty")

        for name in ("metrics", "rankings", "figure"):
            staged_paths[name].replace(artifact_paths[name])
        staged_paths["manifest"].replace(artifact_paths["manifest"])
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "action": "BUILD",
        "selection": selection,
        "pilot_documents": pilot_documents,
        "queries": queries,
        "metrics": metrics,
        "rankings": rankings,
        "recommendation": recommendation,
        "manifest": manifest,
        "token_diagnostics": pd.DataFrame(token_diagnostics),
        "contract": contract,
        "paths": artifact_paths,
    }


def representative_failures(
    rankings: list[dict[str, Any]],
    selected_model: str,
    limit: int = 5,
) -> pd.DataFrame:
    """Return a small, readable sample of selected-model ranking mistakes."""
    rows = []
    for ranking in rankings:
        if ranking["model"] != selected_model or ranking["first_gold_rank"] == 1:
            continue
        rows.append(
            {
                "Query": ranking["query_id"],
                "Language": ranking["language"],
                "Gold rank": ranking["first_gold_rank"] or "Outside Top-10",
                "Top result": ranking["top_10"][0]["ticket_id"],
                "Gold ticket": ", ".join(ranking["gold_ticket_ids"]),
            }
        )
    return pd.DataFrame(rows[:limit])
