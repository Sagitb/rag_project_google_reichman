"""Persistent FAISS vector-store construction and integrity validation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import calculate_sha256, write_json_atomic
from .embedding_selection import stable_fingerprint


SPLIT_NAMES = ("train", "validation", "test")
PAYLOAD_NAMES = ("index", "embeddings", "mapping")


def load_vector_store_contract(path: Path) -> dict[str, Any]:
    """Load and validate the versioned FAISS storage contract."""
    with Path(path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    required = {
        "contract_version",
        "manifest_schema",
        "vector_store_version",
        "selected_model_key",
        "model_name",
        "document_prefix",
        "query_prefix",
        "embedding_dimension",
        "embedding_dtype",
        "normalized",
        "index_type",
        "similarity",
        "expected_document_count",
        "expected_split_counts",
    }
    if set(contract) != required:
        raise RuntimeError("Vector-store contract schema is invalid")
    if contract["index_type"] != "IndexFlatIP":
        raise RuntimeError("This project requires exact IndexFlatIP retrieval")
    if contract["embedding_dtype"] != "float32" or not contract["normalized"]:
        raise RuntimeError("Vector-store embeddings must be normalized float32")
    if sum(contract["expected_split_counts"].values()) != contract[
        "expected_document_count"
    ]:
        raise RuntimeError("Vector-store document and split counts are inconsistent")
    return contract


def build_full_corpus(
    documents_by_split: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    """Assemble a deterministic all-ticket knowledge corpus and its fingerprints."""
    if set(documents_by_split) != set(SPLIT_NAMES):
        raise RuntimeError("Vector-store input must contain all three dataset splits")
    corpus = [
        document
        for split_name in SPLIT_NAMES
        for document in documents_by_split[split_name]
    ]
    corpus.sort(key=lambda document: document["document_id"])
    document_ids = [document["document_id"] for document in corpus]
    if len(corpus) != contract["expected_document_count"]:
        raise RuntimeError("Vector-store corpus size differs from its contract")
    if len(document_ids) != len(set(document_ids)) or document_ids != sorted(document_ids):
        raise RuntimeError("Vector-store corpus IDs are not unique and sorted")

    split_counts = {
        split_name: sum(
            document["system"]["split"] == split_name for document in corpus
        )
        for split_name in SPLIT_NAMES
    }
    if split_counts != contract["expected_split_counts"]:
        raise RuntimeError("Vector-store corpus split counts differ from the contract")

    fingerprint_payload = [
        {
            "document_id": document["document_id"],
            "split": document["system"]["split"],
            "embedding_text": document["search"]["embedding_text"],
        }
        for document in corpus
    ]
    legacy_payload = [
        (
            document["document_id"],
            document["system"]["split"],
            document["search"]["embedding_text"],
        )
        for document in corpus
    ]
    return corpus, stable_fingerprint(fingerprint_payload), stable_fingerprint(legacy_payload)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: calculate_sha256(paths[name]) for name in PAYLOAD_NAMES}


def _expected_manifest_fields(
    contract: dict[str, Any],
    contract_sha256: str,
    corpus_fingerprint: str,
    rag_document_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return every field that defines vector-store identity and lineage."""
    return {
        "schema_version": contract["manifest_schema"],
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256,
        "vector_store_version": contract["vector_store_version"],
        "model_key": contract["selected_model_key"],
        "model_name": contract["model_name"],
        "document_prefix": contract["document_prefix"],
        "query_prefix": contract["query_prefix"],
        "normalized": contract["normalized"],
        "embedding_dimension": contract["embedding_dimension"],
        "dtype": contract["embedding_dtype"],
        "index_type": contract["index_type"],
        "similarity": contract["similarity"],
        "document_count": contract["expected_document_count"],
        "split_counts": contract["expected_split_counts"],
        "corpus_fingerprint": corpus_fingerprint,
        "source_rag_document_fingerprints": rag_document_manifest[
            "output_fingerprints"
        ],
        "source_split_fingerprints": rag_document_manifest["source_fingerprints"],
    }


def _build_manifest(
    expected_fields: dict[str, Any],
    paths: dict[str, Path],
    device: str,
    created_at_utc: str | None = None,
    legacy_upgrade: bool = False,
) -> dict[str, Any]:
    """Create a complete manifest after all three payload files exist."""
    return {
        **expected_fields,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "manifest_updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_device": device,
        "artifact_files": {name: paths[name].name for name in PAYLOAD_NAMES},
        "artifact_sha256": _artifact_hashes(paths),
        "legacy_manifest_upgraded": legacy_upgrade,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": _package_version("faiss-cpu"),
            "sentence_transformers": _package_version("sentence-transformers"),
        },
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _current_artifacts_compatible(
    manifest: dict[str, Any] | None,
    paths: dict[str, Path],
    expected_fields: dict[str, Any],
) -> bool:
    if manifest is None or not all(paths[name].is_file() for name in (*PAYLOAD_NAMES, "manifest")):
        return False
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        return False
    return manifest.get("artifact_sha256") == _artifact_hashes(paths)


def _legacy_artifacts_candidate(
    manifest: dict[str, Any] | None,
    paths: dict[str, Path],
    contract: dict[str, Any],
    corpus_fingerprints: set[str],
    rag_document_manifest: dict[str, Any],
) -> bool:
    """Recognize the validated earlier Section 6 format for a one-time upgrade."""
    if manifest is None or not all(paths[name].is_file() for name in (*PAYLOAD_NAMES, "manifest")):
        return False
    if manifest.get("model_key") != contract["selected_model_key"]:
        return False
    if manifest.get("embedding_dimension") != contract["embedding_dimension"]:
        return False
    if manifest.get("corpus_fingerprint") not in corpus_fingerprints:
        return False
    stored_version = manifest.get("vector_store_version", manifest.get("store_version"))
    if stored_version not in (None, contract["vector_store_version"]):
        return False
    for key, expected in (
        ("source_rag_document_fingerprints", rag_document_manifest["output_fingerprints"]),
        ("source_split_fingerprints", rag_document_manifest["source_fingerprints"]),
    ):
        if key in manifest and manifest[key] != expected:
            return False
    stored_hashes = manifest.get("artifact_sha256")
    return stored_hashes is None or stored_hashes == _artifact_hashes(paths)


def _load_payloads(paths: dict[str, Path]) -> tuple[Any, np.ndarray, list[dict[str, Any]]]:
    import faiss

    index = faiss.read_index(str(paths["index"]))
    embeddings = np.load(paths["embeddings"], allow_pickle=False)
    with paths["mapping"].open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return index, embeddings, mapping


def _validate_payloads(
    index: Any,
    embeddings: np.ndarray,
    mapping: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, bool]:
    """Validate shape, normalization and exhaustive vector-to-ticket alignment."""
    expected_count = contract["expected_document_count"]
    expected_dimension = contract["embedding_dimension"]
    if type(index).__name__ != contract["index_type"]:
        raise RuntimeError(f"Unexpected FAISS index type: {type(index).__name__}")
    if index.ntotal != expected_count or index.d != expected_dimension:
        raise RuntimeError("FAISS index dimensions differ from the contract")
    if embeddings.shape != (expected_count, expected_dimension):
        raise RuntimeError("Persisted embedding matrix has an invalid shape")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise RuntimeError("Persisted embeddings are not finite float32 values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Persisted embeddings are not L2-normalized")
    if len(mapping) != expected_count:
        raise RuntimeError("Vector-to-document mapping has an invalid size")

    required_mapping_fields = {"vector_id", "document_id", "split"}
    for vector_id, (entry, document) in enumerate(zip(mapping, corpus)):
        expected_entry = {
            "vector_id": vector_id,
            "document_id": document["document_id"],
            "split": document["system"]["split"],
        }
        if set(entry) != required_mapping_fields or entry != expected_entry:
            raise RuntimeError(f"Mapping alignment failed at vector {vector_id}")

    reconstructed = np.vstack(
        [index.reconstruct(vector_id) for vector_id in range(index.ntotal)]
    ).astype(np.float32, copy=False)
    if not np.allclose(reconstructed, embeddings, atol=1e-6, rtol=1e-5):
        maximum_error = float(np.max(np.abs(reconstructed - embeddings)))
        raise RuntimeError(f"FAISS reconstruction mismatch: {maximum_error:.3e}")
    return {
        "corpus_contract_passed": True,
        "embedding_matrix_passed": True,
        "mapping_alignment_passed": True,
        "faiss_reconstruction_passed": True,
        "artifact_hashes_passed": True,
    }


def _build_staged_store(
    corpus: list[dict[str, Any]],
    model_config: dict[str, Any],
    contract: dict[str, Any],
    expected_manifest_fields: dict[str, Any],
    final_paths: dict[str, Path],
    device: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Build and fully validate a temporary store before manifest-last promotion."""
    import faiss
    from sentence_transformers import SentenceTransformer

    store_root = final_paths["manifest"].parent
    store_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".vector_store_", dir=store_root))
    staged_paths = {name: staging_root / path.name for name, path in final_paths.items()}
    try:
        print(
            f"[INFO] Encoding {len(corpus)} documents with "
            f"{model_config['model_name']} on {device}."
        )
        model = SentenceTransformer(
            model_config["model_name"],
            device=device,
            trust_remote_code=model_config["trust_remote_code"],
        )
        texts = [
            model_config["document_prefix"] + document["search"]["embedding_text"]
            for document in corpus
        ]
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape != (
            contract["expected_document_count"],
            contract["embedding_dimension"],
        ):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise RuntimeError("Generated embeddings contain non-finite values")

        index = faiss.IndexFlatIP(contract["embedding_dimension"])
        index.add(embeddings)
        mapping = [
            {
                "vector_id": vector_id,
                "document_id": document["document_id"],
                "split": document["system"]["split"],
            }
            for vector_id, document in enumerate(corpus)
        ]
        faiss.write_index(index, str(staged_paths["index"]))
        np.save(staged_paths["embeddings"], embeddings)
        with staged_paths["mapping"].open("w", encoding="utf-8") as handle:
            json.dump(mapping, handle, ensure_ascii=False, indent=2)
        staged_manifest = _build_manifest(
            expected_manifest_fields,
            staged_paths,
            device,
        )
        write_json_atomic(staged_manifest, staged_paths["manifest"])

        reloaded = _load_payloads(staged_paths)
        _validate_payloads(*reloaded, corpus, contract)
        if staged_manifest["artifact_sha256"] != _artifact_hashes(staged_paths):
            raise RuntimeError("Staged vector-store hashes changed during validation")

        for name in PAYLOAD_NAMES:
            staged_paths[name].replace(final_paths[name])
        staged_paths["manifest"].replace(final_paths["manifest"])
        return staged_manifest, staged_paths
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def ensure_vector_store(
    documents_by_split: dict[str, list[dict[str, Any]]],
    rag_document_manifest: dict[str, Any],
    selected_model_key: str,
    model_config: dict[str, Any],
    config: dict[str, Any],
    contract_path: Path,
    artifact_paths: dict[str, Path],
    device: str,
) -> dict[str, Any]:
    """BUILD the full FAISS store once or LOAD and validate compatible artifacts."""
    artifact_paths = {name: Path(path) for name, path in artifact_paths.items()}
    contract_path = Path(contract_path)
    contract = load_vector_store_contract(contract_path)
    if config.get("vector_store_version") != contract["vector_store_version"]:
        raise RuntimeError("Vector-store config and contract versions differ")
    if selected_model_key != contract["selected_model_key"]:
        raise RuntimeError("Section 5 selected a model outside the Vector DB contract")
    model_expectations = {
        "model_name": contract["model_name"],
        "document_prefix": contract["document_prefix"],
        "query_prefix": contract["query_prefix"],
    }
    if any(model_config.get(key) != value for key, value in model_expectations.items()):
        raise RuntimeError("Selected embedding model differs from the Vector DB contract")

    corpus, corpus_fingerprint, legacy_fingerprint = build_full_corpus(
        documents_by_split,
        contract,
    )
    expected_fields = _expected_manifest_fields(
        contract,
        calculate_sha256(contract_path),
        corpus_fingerprint,
        rag_document_manifest,
    )
    manifest = _read_manifest(artifact_paths["manifest"])
    manifest_upgraded = False

    if _current_artifacts_compatible(manifest, artifact_paths, expected_fields):
        action = "LOAD"
    elif _legacy_artifacts_candidate(
        manifest,
        artifact_paths,
        contract,
        {corpus_fingerprint, legacy_fingerprint},
        rag_document_manifest,
    ):
        index, embeddings, mapping = _load_payloads(artifact_paths)
        integrity = _validate_payloads(index, embeddings, mapping, corpus, contract)
        manifest = _build_manifest(
            expected_fields,
            artifact_paths,
            manifest.get("build_device", "unknown_legacy"),
            created_at_utc=manifest.get("created_at_utc", manifest.get("timestamp_utc")),
            legacy_upgrade=True,
        )
        write_json_atomic(manifest, artifact_paths["manifest"])
        action = "LOAD"
        manifest_upgraded = True
        return {
            "action": action,
            "index": index,
            "embeddings": embeddings,
            "mapping": mapping,
            "manifest": manifest,
            "corpus": corpus,
            "corpus_fingerprint": corpus_fingerprint,
            "integrity": integrity,
            "contract": contract,
            "manifest_upgraded": manifest_upgraded,
            "paths": artifact_paths,
        }
    else:
        manifest, _ = _build_staged_store(
            corpus,
            model_config,
            contract,
            expected_fields,
            artifact_paths,
            device,
        )
        action = "BUILD"

    index, embeddings, mapping = _load_payloads(artifact_paths)
    integrity = _validate_payloads(index, embeddings, mapping, corpus, contract)
    if manifest["artifact_sha256"] != _artifact_hashes(artifact_paths):
        raise RuntimeError("Active vector-store hashes differ from the manifest")
    return {
        "action": action,
        "index": index,
        "embeddings": embeddings,
        "mapping": mapping,
        "manifest": manifest,
        "corpus": corpus,
        "corpus_fingerprint": corpus_fingerprint,
        "integrity": integrity,
        "contract": contract,
        "manifest_upgraded": manifest_upgraded,
        "paths": artifact_paths,
    }
