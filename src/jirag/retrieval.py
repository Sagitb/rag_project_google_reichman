"""Retrieval services over the persistent jiRAG FAISS vector store."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_TICKET_ID_PATTERN = re.compile(
    r"^(?:tckt\s*[-#:]?\s*|ticket(?:\s+number)?\s*[-#:]?\s*|טיקט\s*[-#:]?\s*)?(\d{1,4})$",
    flags=re.IGNORECASE,
)


def load_retrieval_contract(path: Path) -> dict[str, Any]:
    """Load the stable retrieval interface and validate its small public contract."""
    with Path(path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    required = {
        "contract_version",
        "default_top_k",
        "maximum_top_k",
        "open_statuses",
        "operational_filter_fields",
        "internal_evaluation_fields",
    }
    if set(contract) != required:
        raise RuntimeError("Retrieval contract schema is invalid")
    if not 0 < contract["default_top_k"] <= contract["maximum_top_k"]:
        raise RuntimeError("Retrieval top-k limits are inconsistent")
    if not contract["open_statuses"]:
        raise RuntimeError("Retrieval contract must define at least one open status")
    return contract


def validate_positive_integer(value: int, name: str) -> int:
    """Require a positive integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_ticket_id(ticket_id: str) -> str:
    """Normalize supported explicit ID forms such as 'ticket 101' to tckt-0101."""
    if not isinstance(ticket_id, str):
        raise TypeError("ticket_id must be a string")
    cleaned = ticket_id.strip()
    if not cleaned:
        raise ValueError("ticket_id cannot be empty or whitespace")

    match = _TICKET_ID_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError(f"Unsupported ticket ID format: {ticket_id}")
    return f"tckt-{int(match.group(1)):04d}"


class RetrievalEngine:
    """Expose semantic, exact, structured and hybrid access to one aligned corpus."""

    def __init__(
        self,
        *,
        index: Any,
        embeddings: np.ndarray,
        mapping: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        model_config: dict[str, Any],
        vector_contract: dict[str, Any],
        retrieval_contract: dict[str, Any],
        device: str,
    ) -> None:
        self.index = index
        self.embeddings = embeddings
        self.mapping = mapping
        self.documents = documents
        self.model_config = model_config
        self.vector_contract = vector_contract
        self.contract = retrieval_contract
        self.device = device
        self._query_encoder = None

        self.document_by_id = {
            document["document_id"]: document for document in documents
        }
        self.document_position_by_id = {
            document["document_id"]: position
            for position, document in enumerate(documents)
        }
        self.open_statuses = frozenset(retrieval_contract["open_statuses"])
        self.supported_filter_fields = frozenset(
            retrieval_contract["operational_filter_fields"]
            + retrieval_contract["internal_evaluation_fields"]
        )
        self._validate_resources()

    def _validate_resources(self) -> None:
        """Perform the lightweight boundary check needed after Section 6 validation."""
        count = len(self.documents)
        dimension = self.vector_contract["embedding_dimension"]
        if self.index.ntotal != count or len(self.mapping) != count:
            raise RuntimeError("Retrieval resources have inconsistent document counts")
        if self.index.d != dimension or self.embeddings.shape != (count, dimension):
            raise RuntimeError("Retrieval resources have inconsistent dimensions")
        if self.embeddings.dtype != np.float32:
            raise RuntimeError("Retrieval embeddings must use float32")

        for position, (entry, document) in enumerate(zip(self.mapping, self.documents)):
            if entry.get("vector_id") != position:
                raise RuntimeError(f"Invalid vector mapping at position {position}")
            if entry.get("document_id") != document.get("document_id"):
                raise RuntimeError(f"Document mapping mismatch at position {position}")

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query cannot be empty or whitespace")
        return cleaned

    def get_query_encoder(self):
        """Load the selected E5 encoder on first semantic use and then reuse it."""
        if self._query_encoder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self.vector_contract["model_name"]
            print(f"[INFO] Loading query encoder: {model_name} on {self.device}...")
            self._query_encoder = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=self.model_config.get("trust_remote_code", False),
            )
        return self._query_encoder

    def encode_query(self, query: str) -> np.ndarray:
        """Create one normalized E5 query vector under the stored model contract."""
        cleaned = self._validate_query(query)
        query_vector = self.get_query_encoder().encode(
            [self.vector_contract["query_prefix"] + cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        expected_shape = (1, self.vector_contract["embedding_dimension"])
        if query_vector.shape != expected_shape or not np.isfinite(query_vector).all():
            raise RuntimeError("Query encoder returned an invalid vector")
        if not np.allclose(np.linalg.norm(query_vector, axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Query vector is not L2-normalized")
        return query_vector

    def release_query_encoder(self) -> None:
        """Release the embedding model before loading a memory-intensive generator."""
        import gc

        self._query_encoder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _document_field(self, document: dict[str, Any], field: str) -> Any:
        if field == "document_id":
            return document["document_id"]
        if field in {"component", "status", "priority", "work_type"}:
            return document["metadata"][field]
        if field in {"family", "solution_type"}:
            return document["evaluation"][field]
        if field == "split":
            return document["system"]["split"]
        if field == "is_open":
            return document["metadata"]["status"] in self.open_statuses
        raise ValueError(f"Unsupported field: {field}")

    def _validate_filters(self, filters: dict[str, Any] | None) -> dict[str, Any]:
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")
        for field, criterion in filters.items():
            if field not in self.supported_filter_fields:
                raise ValueError(f"Unsupported filter field: {field}")
            if field == "is_open":
                if type(criterion) is not bool:
                    raise ValueError("is_open must be exactly True or False")
            elif isinstance(criterion, (list, tuple, set)):
                if not criterion or any(
                    value is None or not str(value).strip() for value in criterion
                ):
                    raise ValueError(f"Invalid filter collection for {field}")
            elif criterion is None or not str(criterion).strip():
                raise ValueError(f"Filter value for {field} cannot be empty")
        return filters

    def _matches(self, document: dict[str, Any], filters: dict[str, Any]) -> bool:
        for field, criterion in filters.items():
            document_value = self._document_field(document, field)
            if field == "is_open":
                if document_value is not criterion:
                    return False
                continue
            normalized = str(document_value).strip().casefold()
            if isinstance(criterion, (list, tuple, set)):
                allowed = {str(value).strip().casefold() for value in criterion}
                if normalized not in allowed:
                    return False
            elif normalized != str(criterion).strip().casefold():
                return False
        return True

    @staticmethod
    def format_result(
        document: dict[str, Any], rank: int, score: float | None = None
    ) -> dict[str, Any]:
        """Return an isolated result using the schema consumed by generation stages."""
        document_copy = copy.deepcopy(document)
        return {
            "rank": int(rank),
            "score": float(score) if score is not None else None,
            "document_id": document_copy["document_id"],
            "content": document_copy["content"],
            "metadata": document_copy["metadata"],
            "evaluation": document_copy["evaluation"],
            "system": document_copy["system"],
        }

    def validate_ranked_results(
        self,
        results: list[dict[str, Any]],
        expected_filters: dict[str, Any] | None = None,
    ) -> bool:
        """Validate ranking mechanics without claiming semantic relevance."""
        if not results:
            raise AssertionError("Expected at least one ranked result")
        ranks = [result["rank"] for result in results]
        scores = [result["score"] for result in results]
        ids = [result["document_id"] for result in results]
        if ranks != list(range(1, len(results) + 1)):
            raise AssertionError("Ranks are not consecutive")
        if not all(isinstance(score, float) and np.isfinite(score) for score in scores):
            raise AssertionError("Results contain an invalid score")
        if any(scores[i] < scores[i + 1] - 1e-8 for i in range(len(scores) - 1)):
            raise AssertionError("Scores are not descending")
        if len(ids) != len(set(ids)) or any(item not in self.document_by_id for item in ids):
            raise AssertionError("Results contain duplicate or unknown document IDs")
        if expected_filters:
            filters = self._validate_filters(expected_filters)
            if any(not self._matches(self.document_by_id[item], filters) for item in ids):
                raise AssertionError("A ranked result violates its metadata filters")
        return True

    def semantic_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Search the complete FAISS corpus by semantic similarity."""
        cleaned = self._validate_query(query)
        if top_k is None:
            top_k = self.contract["default_top_k"]
        validate_positive_integer(top_k, "top_k")
        result_count = min(top_k, self.contract["maximum_top_k"], len(self.documents))
        scores, positions = self.index.search(self.encode_query(cleaned), result_count)

        results = []
        for rank, (score, raw_position) in enumerate(
            zip(scores[0], positions[0]), start=1
        ):
            position = int(raw_position)
            if not 0 <= position < len(self.mapping):
                raise RuntimeError(f"FAISS returned invalid position: {position}")
            document_id = self.mapping[position]["document_id"]
            results.append(
                self.format_result(self.document_by_id[document_id], rank, score)
            )
        self.validate_ranked_results(results)
        return results

    def lookup_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Resolve an explicit ticket reference without semantic search."""
        normalized_id = normalize_ticket_id(ticket_id)
        if normalized_id not in self.document_by_id:
            raise KeyError(f"Ticket ID not found: {normalized_id}")
        return self.format_result(self.document_by_id[normalized_id], rank=1)

    def filter_tickets(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Apply exact structured filters without creating a query embedding."""
        validated = self._validate_filters(filters)
        if limit is not None:
            validate_positive_integer(limit, "limit")
        matches = [
            document for document in self.documents if self._matches(document, validated)
        ]
        if limit is not None:
            matches = matches[:limit]
        return [
            self.format_result(document, rank)
            for rank, document in enumerate(matches, start=1)
        ]

    def aggregate_tickets(
        self,
        filters: dict[str, Any] | None = None,
        group_by: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Calculate exact counts over canonical metadata."""
        validated = self._validate_filters(filters)
        if group_by is None:
            grouping_fields: list[str] = []
        elif isinstance(group_by, str):
            grouping_fields = [group_by]
        elif isinstance(group_by, (list, tuple)):
            grouping_fields = list(group_by)
        else:
            raise TypeError("group_by must be a field name, list, tuple, or None")
        if len(grouping_fields) != len(set(grouping_fields)):
            raise ValueError("group_by cannot contain duplicate fields")
        if any(field not in self.supported_filter_fields for field in grouping_fields):
            raise ValueError("group_by contains an unsupported field")

        eligible = [
            document for document in self.documents if self._matches(document, validated)
        ]
        grouped_counts: dict[str, int] = {}
        for document in eligible:
            values = []
            for field in grouping_fields:
                value = self._document_field(document, field)
                if field == "is_open":
                    values.append("Open" if value else "Closed")
                else:
                    values.append(str(value))
            key = " | ".join(values)
            if key:
                grouped_counts[key] = grouped_counts.get(key, 0) + 1
        grouped_counts = dict(sorted(grouped_counts.items(), key=lambda item: item[0].casefold()))
        if grouped_counts and sum(grouped_counts.values()) != len(eligible):
            raise RuntimeError("Grouped counts do not sum to the filtered total")
        return {
            "filters": copy.deepcopy(validated),
            "total": len(eligible),
            "group_by": grouping_fields or None,
            "grouped_counts": grouped_counts,
            "sample_document_ids": [item["document_id"] for item in eligible[:5]],
        }

    def hybrid_search(
        self, query: str, filters: dict[str, Any], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rank a metadata-filtered subset by exact cosine-equivalent similarity."""
        cleaned = self._validate_query(query)
        validated = self._validate_filters(filters)
        if top_k is None:
            top_k = self.contract["default_top_k"]
        validate_positive_integer(top_k, "top_k")
        top_k = min(top_k, self.contract["maximum_top_k"])

        positions = [
            position
            for position, document in enumerate(self.documents)
            if self._matches(document, validated)
        ]
        if not positions:
            return []
        query_vector = self.encode_query(cleaned).reshape(-1)
        scores = self.embeddings[positions] @ query_vector
        candidates = sorted(
            zip(scores, positions), key=lambda item: (-float(item[0]), item[1])
        )[:top_k]
        results = [
            self.format_result(self.documents[position], rank, score)
            for rank, (score, position) in enumerate(candidates, start=1)
        ]
        self.validate_ranked_results(results, expected_filters=validated)
        return results


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a compact display table while preserving full result objects."""
    columns = [
        "rank",
        "score",
        "document_id",
        "summary",
        "component",
        "status",
        "priority",
        "work_type",
        "family",
        "solution_type",
        "split",
    ]
    rows = [
        {
            "rank": result["rank"],
            "score": round(result["score"], 4) if result["score"] is not None else None,
            "document_id": result["document_id"],
            "summary": result["content"]["summary"],
            "component": result["metadata"]["component"],
            "status": result["metadata"]["status"],
            "priority": result["metadata"]["priority"],
            "work_type": result["metadata"]["work_type"],
            "family": result["evaluation"]["family"],
            "solution_type": result["evaluation"]["solution_type"],
            "split": result["system"]["split"],
        }
        for result in results
    ]
    return pd.DataFrame(rows, columns=columns)
