"""Canonical RAG-document construction, validation and persistent loading."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any

import pandas as pd

from .artifacts import calculate_sha256, write_json_atomic


RAG_MANIFEST_SCHEMA = "rag_document_manifest_v2"
SPLIT_NAMES = ("train", "validation", "test")


def load_rag_document_contract(contract_path: Path) -> dict[str, Any]:
    """Load and validate the versioned mapping from tickets to RAG documents."""
    with Path(contract_path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    _validate_contract(contract)
    return contract


def _validate_contract(contract: dict[str, Any]) -> None:
    """Reject ambiguous schemas or embedding templates that could leak labels."""
    expected_top_level = {
        "document_id",
        "search",
        "content",
        "metadata",
        "evaluation",
        "system",
    }
    if set(contract.get("top_level_fields", [])) != expected_top_level:
        raise RuntimeError("RAG contract top-level fields are invalid")

    placeholders = [
        field_name
        for _, field_name, _, _ in Formatter().parse(contract["embedding_template"])
        if field_name is not None
    ]
    if placeholders != contract["embedding_content_fields"]:
        raise RuntimeError("Embedding template fields differ from the RAG contract")
    if not set(placeholders).issubset(contract["content_fields"]):
        raise RuntimeError("Embedding template references a non-content field")

    embedding_sources = {
        contract["content_fields"][field_name]
        for field_name in contract["embedding_content_fields"]
    }
    if embedding_sources != set(contract["embedding_source_fields"]):
        raise RuntimeError("Embedding source-field declaration is inconsistent")
    if embedding_sources & set(contract["excluded_from_embedding"]):
        raise RuntimeError("A protected field is included in the embedding source")


def build_embedding_text_from_content(
    content: dict[str, str],
    contract: dict[str, Any],
) -> str:
    """Construct the only text representation allowed into the vector index."""
    values = {
        field_name: content[field_name]
        for field_name in contract["embedding_content_fields"]
    }
    return contract["embedding_template"].format(**values)


def build_rag_document(
    source_record: dict[str, Any],
    split_name: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Map one approved split row to the canonical six-part RAG schema."""
    content = {
        target: str(source_record[source]).strip()
        for target, source in contract["content_fields"].items()
    }
    metadata = {
        target: str(source_record[source]).strip()
        for target, source in contract["metadata_fields"].items()
    }
    evaluation = {
        target: str(source_record[source]).strip()
        for target, source in contract["evaluation_fields"].items()
    }
    return {
        "document_id": str(source_record["ticket_id"]).strip(),
        "search": {
            "embedding_text": build_embedding_text_from_content(content, contract),
        },
        "content": content,
        "metadata": metadata,
        "evaluation": evaluation,
        "system": {
            "split": split_name,
            "document_version": contract["document_version"],
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL artifact while rejecting blank or non-object records."""
    documents = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"{Path(path).name}: blank JSONL line {line_number}")
            document = json.loads(line)
            if not isinstance(document, dict):
                raise RuntimeError(f"{Path(path).name}: line {line_number} is not an object")
            documents.append(document)
    return documents


def stable_json_fingerprint(value: Any) -> str:
    """Match the stable content fingerprint used by downstream experiments."""
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_document(
    document: dict[str, Any],
    split_name: str,
    contract: dict[str, Any],
) -> None:
    """Validate structure, leaf values, split identity and embedding isolation."""
    if set(document) != set(contract["top_level_fields"]):
        raise RuntimeError("RAG document top-level schema mismatch")
    if not isinstance(document["document_id"], str) or not document["document_id"].strip():
        raise RuntimeError("RAG document has an invalid document_id")

    nested_fields = {
        "search": contract["search_fields"],
        "content": list(contract["content_fields"]),
        "metadata": list(contract["metadata_fields"]),
        "evaluation": list(contract["evaluation_fields"]),
        "system": contract["system_fields"],
    }
    for block, expected_fields in nested_fields.items():
        value = document.get(block)
        if not isinstance(value, dict) or set(value) != set(expected_fields):
            raise RuntimeError(f"RAG document block '{block}' has an invalid schema")
        for field_name, field_value in value.items():
            if not isinstance(field_value, str) or not field_value.strip():
                raise RuntimeError(f"RAG document field '{block}.{field_name}' is empty")

    expected_search = build_embedding_text_from_content(document["content"], contract)
    if document["search"]["embedding_text"] != expected_search:
        raise RuntimeError("embedding_text was not derived from approved content fields")
    if document["system"]["split"] != split_name:
        raise RuntimeError("RAG document split marker is incorrect")
    if document["system"]["document_version"] != contract["document_version"]:
        raise RuntimeError("RAG document version is incorrect")


def _expected_documents(
    split_frames: dict[str, pd.DataFrame],
    contract: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic expected documents from the frozen split rows."""
    return {
        split_name: [
            build_rag_document(record, split_name, contract)
            for record in split_frames[split_name].to_dict(orient="records")
        ]
        for split_name in SPLIT_NAMES
    }


def _validate_collections(
    documents: dict[str, list[dict[str, Any]]],
    split_frames: dict[str, pd.DataFrame],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Compare every saved document with its exact expected source representation."""
    if set(documents) != set(SPLIT_NAMES) or set(split_frames) != set(SPLIT_NAMES):
        raise RuntimeError("RAG document collections do not match the three split names")

    expected = _expected_documents(split_frames, contract)
    all_document_ids = []
    for split_name in SPLIT_NAMES:
        actual_documents = documents[split_name]
        if len(actual_documents) != len(expected[split_name]):
            raise RuntimeError(f"{split_name}: RAG document count differs from source rows")
        for actual, expected_document in zip(actual_documents, expected[split_name]):
            _validate_document(actual, split_name, contract)
            if actual != expected_document:
                raise RuntimeError(
                    f"{split_name}/{actual['document_id']}: document differs from source row"
                )

        document_ids = [document["document_id"] for document in actual_documents]
        if len(document_ids) != len(set(document_ids)):
            raise RuntimeError(f"{split_name}: duplicate document IDs detected")
        all_document_ids.extend(document_ids)

    if len(all_document_ids) != len(set(all_document_ids)):
        raise RuntimeError("Cross-split RAG document overlap detected")
    return {
        "counts_match_source": True,
        "documents_match_source_rows": True,
        "unique_ids_within_splits": True,
        "pairwise_disjoint": True,
        "embedding_uses_approved_fields_only": True,
    }


def _write_jsonl_temporary(
    documents: list[dict[str, Any]],
    final_path: Path,
) -> Path:
    """Write one complete temporary JSONL file for reload validation."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    return temporary_path


def _build_manifest(
    documents: dict[str, list[dict[str, Any]]],
    artifact_paths: dict[str, Path],
    contract: dict[str, Any],
    contract_sha256: str,
    split_manifest: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    """Create the committed identity and lineage record for RAG documents."""
    return {
        "schema_version": RAG_MANIFEST_SCHEMA,
        "document_version": contract["document_version"],
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256,
        "split_version": split_manifest["split_version"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_fields": contract["embedding_source_fields"],
        "metadata_fields": list(contract["metadata_fields"]),
        "evaluation_fields": list(contract["evaluation_fields"]),
        "counts": {name: len(documents[name]) for name in SPLIT_NAMES},
        "source_fingerprints": split_manifest["files"],
        "output_fingerprints": {
            name: stable_json_fingerprint(documents[name])
            for name in SPLIT_NAMES
        },
        "files": {
            name: {
                "filename": artifact_paths[name].name,
                "sha256": calculate_sha256(artifact_paths[name]),
            }
            for name in SPLIT_NAMES
        },
        "integrity": integrity,
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """Load an existing manifest when it is readable JSON."""
    if not Path(path).is_file():
        return None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_is_compatible(
    manifest: dict[str, Any] | None,
    artifact_paths: dict[str, Path],
    contract: dict[str, Any],
    contract_sha256: str,
    split_manifest: dict[str, Any],
) -> bool:
    """Check the document contract, split lineage and physical JSONL hashes."""
    if manifest is None or not all(artifact_paths[name].is_file() for name in SPLIT_NAMES):
        return False
    expected = {
        "schema_version": RAG_MANIFEST_SCHEMA,
        "document_version": contract["document_version"],
        "contract_sha256": contract_sha256,
        "split_version": split_manifest["split_version"],
        "source_fingerprints": split_manifest["files"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    for name in SPLIT_NAMES:
        if manifest.get("files", {}).get(name, {}).get("sha256") != calculate_sha256(
            artifact_paths[name]
        ):
            return False
        if not manifest.get("output_fingerprints", {}).get(name):
            return False
    return True


def _load_collections(artifact_paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Load all three RAG-document JSONL files."""
    return {name: load_jsonl(artifact_paths[name]) for name in SPLIT_NAMES}


def ensure_rag_documents(
    split_frames: dict[str, pd.DataFrame],
    split_manifest: dict[str, Any],
    contract_path: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """BUILD canonical documents once or LOAD fully compatible saved artifacts."""
    artifact_paths = {name: Path(path) for name, path in artifact_paths.items()}
    contract_path = Path(contract_path)
    contract = load_rag_document_contract(contract_path)
    contract_sha256 = calculate_sha256(contract_path)
    manifest = _read_manifest(artifact_paths["manifest"])

    if (
        manifest
        and manifest.get("schema_version") == RAG_MANIFEST_SCHEMA
        and manifest.get("document_version") == contract["document_version"]
        and manifest.get("contract_sha256") != contract_sha256
    ):
        raise RuntimeError("RAG document contract changed; bump rag_doc_version")

    if _manifest_is_compatible(
        manifest,
        artifact_paths,
        contract,
        contract_sha256,
        split_manifest,
    ):
        documents = _load_collections(artifact_paths)
        integrity = _validate_collections(documents, split_frames, contract)
        for name in SPLIT_NAMES:
            if stable_json_fingerprint(documents[name]) != manifest["output_fingerprints"][name]:
                raise RuntimeError(f"{name}: RAG document content fingerprint mismatch")
        return {
            "action": "LOAD",
            "documents": documents,
            "manifest": manifest,
            "contract": contract,
            "integrity": integrity,
            "manifest_upgraded": False,
        }

    complete_jsonl = all(artifact_paths[name].is_file() for name in SPLIT_NAMES)
    if complete_jsonl:
        try:
            documents = _load_collections(artifact_paths)
            integrity = _validate_collections(documents, split_frames, contract)
        except (OSError, json.JSONDecodeError, RuntimeError):
            documents = None
        if documents is not None:
            upgraded_manifest = _build_manifest(
                documents,
                artifact_paths,
                contract,
                contract_sha256,
                split_manifest,
                integrity,
            )
            write_json_atomic(upgraded_manifest, artifact_paths["manifest"])
            return {
                "action": "LOAD",
                "documents": documents,
                "manifest": upgraded_manifest,
                "contract": contract,
                "integrity": integrity,
                "manifest_upgraded": True,
            }

    documents = _expected_documents(split_frames, contract)
    integrity = _validate_collections(documents, split_frames, contract)
    temporary_paths = {
        name: _write_jsonl_temporary(documents[name], artifact_paths[name])
        for name in SPLIT_NAMES
    }
    reloaded = {name: load_jsonl(path) for name, path in temporary_paths.items()}
    _validate_collections(reloaded, split_frames, contract)
    if reloaded != documents:
        raise RuntimeError("Temporary RAG-document reload differs from memory")

    for name in SPLIT_NAMES:
        temporary_paths[name].replace(artifact_paths[name])
    manifest = _build_manifest(
        documents,
        artifact_paths,
        contract,
        contract_sha256,
        split_manifest,
        integrity,
    )
    write_json_atomic(manifest, artifact_paths["manifest"])
    return {
        "action": "BUILD",
        "documents": documents,
        "manifest": manifest,
        "contract": contract,
        "integrity": integrity,
        "manifest_upgraded": False,
    }
