"""Small Jira Cloud reader and idempotent live-vector synchronization layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

import numpy as np

from .artifacts import calculate_sha256, write_json_atomic


SECRET_NAMES = (
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
)
ARTIFACT_FILES = {
    "documents": "jira_documents.jsonl",
    "embeddings": "jira_embeddings.npy",
    "mapping": "jira_mapping.json",
    "index": "jira_index.faiss",
    "manifest": "jira_sync_manifest.json",
    "report": "jira_last_sync_report.json",
}


@dataclass(frozen=True)
class JiraSettings:
    """Runtime-only Jira credentials; values are never persisted by this module."""

    base_url: str
    email: str
    api_token: str
    project_key: str

    def __post_init__(self) -> None:
        if not all((self.base_url, self.email, self.api_token, self.project_key)):
            raise ValueError("All Jira settings must be non-empty")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.project_key):
            raise ValueError("JIRA_PROJECT_KEY has an invalid format")
        if not self.base_url.startswith("https://"):
            raise ValueError("JIRA_BASE_URL must use HTTPS")

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


def load_jira_settings(
    secret_getter: Callable[[str], str | None],
) -> tuple[JiraSettings | None, list[str]]:
    """Load four Colab Secrets without exposing their values."""
    values: dict[str, str] = {}
    missing = []
    for name in SECRET_NAMES:
        try:
            value = secret_getter(name)
        except Exception:
            value = None
        if value is None or not str(value).strip():
            missing.append(name)
        else:
            values[name] = str(value).strip()
    if missing:
        return None, missing
    return JiraSettings(
        base_url=values["JIRA_BASE_URL"],
        email=values["JIRA_EMAIL"],
        api_token=values["JIRA_API_TOKEN"],
        project_key=values["JIRA_PROJECT_KEY"].upper(),
    ), []


def load_jira_contract(path: Path) -> dict[str, Any]:
    """Load the small, versioned boundary between Jira and jiRAG."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "contract_version", "artifact_version", "document_version",
        "internal_id_start", "maximum_issues", "demo_label", "jql_template", "fields",
        "recognized_solution_labels", "family_label_prefix", "fallback_family",
    }
    if set(contract) != required:
        raise RuntimeError("Jira synchronization contract schema is invalid")
    if contract["internal_id_start"] < 1001 or contract["maximum_issues"] < 1:
        raise RuntimeError("Jira synchronization limits are invalid")
    return contract


class JiraCloudClient:
    """Minimal Jira Cloud REST v3 client for one project."""

    def __init__(self, settings: JiraSettings, session: Any | None = None) -> None:
        if session is None:
            import requests

            session = requests.Session()
        self.settings = settings
        self.session = session
        self.session.auth = (settings.email, settings.api_token)
        self.session.headers.update({"Accept": "application/json"})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(
            method,
            f"{self.settings.normalized_base_url}{path}",
            timeout=30,
            **kwargs,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Jira returned a non-object JSON response")
        return payload

    def verify_connection(self) -> dict[str, str]:
        """Return only safe identity fields for the notebook preflight."""
        user = self._request("GET", "/rest/api/3/myself")
        project = self._request(
            "GET", f"/rest/api/3/project/{self.settings.project_key}"
        )
        return {
            "site": self.settings.normalized_base_url,
            "user": str(user.get("displayName", "Unknown")),
            "project_name": str(project.get("name", "Unknown")),
            "project_key": str(project.get("key", self.settings.project_key)),
        }

    def fetch_project_issues(self, contract: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch up to the approved POC limit through enhanced JQL search."""
        jql = contract["jql_template"].format(
            project_key=self.settings.project_key,
            demo_label=contract["demo_label"],
        )
        issues: list[dict[str, Any]] = []
        next_page_token = None
        while len(issues) < contract["maximum_issues"]:
            body: dict[str, Any] = {
                "jql": jql,
                "fields": contract["fields"],
                "maxResults": min(50, contract["maximum_issues"] - len(issues)),
            }
            if next_page_token:
                body["nextPageToken"] = next_page_token
            page = self._request(
                "POST",
                "/rest/api/3/search/jql",
                headers={"Content-Type": "application/json"},
                json=body,
            )
            page_issues = page.get("issues", [])
            if not isinstance(page_issues, list):
                raise RuntimeError("Jira search response has an invalid issues field")
            issues.extend(page_issues)
            next_page_token = page.get("nextPageToken")
            if not next_page_token or not page_issues:
                break
        return issues[: contract["maximum_issues"]]

    def ensure_demo_issue(
        self,
        *,
        source_ticket: dict[str, str],
        contract: dict[str, Any],
        source_id: str = "tckt-0186",
    ) -> dict[str, Any]:
        """Create one traceable demo issue once, leaving its Jira workflow untouched."""
        normalized_source_id = source_id.strip().casefold()
        if not re.fullmatch(r"tckt-\d{4}", normalized_source_id):
            raise ValueError("source_id must use the tckt-0000 format")
        source_label = f"source-{normalized_source_id}"
        jql = (
            f'project = "{self.settings.project_key}" '
            f'AND labels = "{source_label}" ORDER BY key ASC'
        )
        existing = self._request(
            "POST",
            "/rest/api/3/search/jql",
            headers={"Content-Type": "application/json"},
            json={
                "jql": jql,
                "fields": contract["fields"],
                "maxResults": 2,
            },
        ).get("issues", [])
        if existing:
            return {"action": "FOUND", "issue": existing[0], "source_label": source_label}

        labels = [contract["demo_label"], source_label]
        for key in ("family", "solution_type"):
            value = str(source_ticket.get(key) or "").strip()
            if value:
                labels.append(value)
        description = str(source_ticket.get("description") or "").strip()
        source_status = str(source_ticket.get("status") or "Unknown").strip()
        description += (
            "\n\nDEMO IMPORT METADATA\n"
            f"Source dataset ID: {normalized_source_id}\n"
            f"Source dataset status: {source_status}\n"
            "The Jira workflow status is intentionally not changed automatically."
        )
        create_fields: dict[str, Any] = {
            "project": {"key": self.settings.project_key},
            "summary": str(source_ticket.get("summary") or normalized_source_id).strip(),
            "description": {
                "type": "doc",
                "version": 1,
                "content": [{
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description}],
                }],
            },
            "issuetype": {"name": str(source_ticket.get("work_type") or "Task").strip()},
            "labels": labels,
        }
        priority = str(source_ticket.get("priority") or "").strip()
        if priority:
            create_fields["priority"] = {"name": priority}
        created = self._request(
            "POST",
            "/rest/api/3/issue",
            headers={"Content-Type": "application/json"},
            json={"fields": create_fields},
        )
        issue_key = str(created.get("key") or "").strip()
        if not issue_key:
            raise RuntimeError("Jira created the demo issue without returning its key")
        issue = self._request(
            "GET",
            f"/rest/api/3/issue/{issue_key}",
            params={"fields": ",".join(contract["fields"])},
        )
        return {"action": "CREATED", "issue": issue, "source_label": source_label}


def adf_to_text(value: Any) -> str:
    """Flatten Jira's Atlassian Document Format into readable plain text."""
    if value is None:
        return "No description provided."
    if isinstance(value, str):
        return value.strip() or "No description provided."
    fragments: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        if node_type == "text":
            fragments.append(str(node.get("text", "")))
        elif node_type == "hardBreak":
            fragments.append("\n")
        for child in node.get("content", []):
            visit(child)
        if node_type in {"paragraph", "heading", "listItem", "codeBlock"}:
            fragments.append("\n")

    visit(value)
    text = "".join(fragments)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "No description provided."


def _content_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _solution_type(fields: dict[str, Any], contract: dict[str, Any]) -> str:
    labels = {str(label).strip().casefold() for label in fields.get("labels") or []}
    recognized = [
        label for label in contract["recognized_solution_labels"]
        if label.casefold() in labels
    ]
    if recognized:
        return recognized[0]
    status_category = (fields.get("status") or {}).get("statusCategory") or {}
    is_done = str(status_category.get("key", "")).casefold() == "done"
    return "solution-partial" if is_done else "solution-unresolved"


def _family(fields: dict[str, Any], contract: dict[str, Any]) -> str:
    labels = [str(label).strip() for label in fields.get("labels") or []]
    return next(
        (label for label in labels if label.casefold().startswith(contract["family_label_prefix"])),
        contract["fallback_family"],
    )


def normalize_jira_issue(
    issue: dict[str, Any],
    document_id: str,
    settings: JiraSettings,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Convert one Jira REST issue into the schema consumed by jiRAG."""
    jira_key = str(issue.get("key", "")).strip()
    fields = issue.get("fields") or {}
    if not jira_key or not isinstance(fields, dict):
        raise RuntimeError("Jira issue is missing key or fields")
    summary = str(fields.get("summary") or "Untitled Jira issue").strip()
    description = adf_to_text(fields.get("description"))
    components = fields.get("components") or []
    component = ", ".join(
        str(item.get("name", "")).strip()
        for item in components if isinstance(item, dict) and item.get("name")
    ) or "Unspecified"
    status = str((fields.get("status") or {}).get("name") or "Unknown").strip()
    priority = str((fields.get("priority") or {}).get("name") or "Unspecified").strip()
    work_type = str((fields.get("issuetype") or {}).get("name") or "Issue").strip()
    embedding_text = (
        f"Summary: {summary}\nComponent: {component}\nDescription: {description}"
    )
    return {
        "document_id": document_id,
        "search": {"embedding_text": embedding_text},
        "content": {
            "summary": summary,
            "component": component,
            "description": description,
        },
        "metadata": {
            "component": component,
            "status": status,
            "priority": priority,
            "work_type": work_type,
            "jira_key": jira_key,
            "jira_url": f"{settings.normalized_base_url}/browse/{jira_key}",
            "jira_updated": str(fields.get("updated") or "Unknown"),
        },
        "evaluation": {
            "family": _family(fields, contract),
            "solution_type": _solution_type(fields, contract),
        },
        "system": {
            "split": "live_jira",
            "document_version": contract["document_version"],
        },
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_documents(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            documents.append(json.loads(line))
    return documents


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {name: root / filename for name, filename in ARTIFACT_FILES.items()}


def _save_live_store(
    root: Path,
    documents: list[dict[str, Any]],
    embeddings: np.ndarray,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Path]:
    import faiss

    root.mkdir(parents=True, exist_ok=True)
    final_paths = _artifact_paths(root)
    mapping = [
        {
            "vector_id": position,
            "document_id": document["document_id"],
            "jira_key": document["metadata"]["jira_key"],
        }
        for position, document in enumerate(documents)
    ]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    if len(embeddings):
        index.add(embeddings)

    with tempfile.TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary)
        staged = _artifact_paths(stage)
        staged["documents"].write_text(
            "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents),
            encoding="utf-8",
        )
        with staged["embeddings"].open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        staged["mapping"].write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        faiss.write_index(index, str(staged["index"]))
        staged["report"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["artifact_sha256"] = {
            name: calculate_sha256(staged[name])
            for name in ("documents", "embeddings", "mapping", "index")
        }
        staged["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name in ARTIFACT_FILES:
            os.replace(staged[name], final_paths[name])
    return final_paths


def sync_jira_live_index(
    *,
    client: JiraCloudClient,
    issues: list[dict[str, Any]],
    encoder: Any,
    model_name: str,
    document_prefix: str,
    embedding_dimension: int,
    contract: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Upsert Jira issues, encode changed documents only and persist one small live index."""
    import faiss

    root = Path(artifact_root)
    paths = _artifact_paths(root)
    prior_manifest = _read_json(paths["manifest"], {})
    prior_documents = _read_documents(paths["documents"])
    prior_by_key = {
        doc["metadata"]["jira_key"]: doc for doc in prior_documents
        if doc.get("metadata", {}).get("jira_key")
    }
    prior_embeddings = (
        np.load(paths["embeddings"], allow_pickle=False)
        if paths["embeddings"].is_file() else np.empty((0, embedding_dimension), dtype=np.float32)
    )
    prior_ready = bool(
        prior_manifest
        and paths["index"].is_file()
        and paths["mapping"].is_file()
        and len(prior_documents) == len(prior_embeddings)
        and prior_embeddings.ndim == 2
        and prior_embeddings.shape[1] == embedding_dimension
    )
    prior_vector_by_key = {
        doc["metadata"]["jira_key"]: prior_embeddings[position]
        for position, doc in enumerate(prior_documents)
        if position < len(prior_embeddings)
    }
    prior_ids = prior_manifest.get("jira_key_to_document_id", {})
    next_id = max(
        [contract["internal_id_start"] - 1]
        + [int(value.split("-")[-1]) for value in prior_ids.values() if re.fullmatch(r"tckt-\d+", value)]
    ) + 1

    raw_by_key = {str(issue.get("key", "")).strip(): issue for issue in issues}
    current_keys = sorted(key for key in raw_by_key if key)
    id_map = dict(prior_ids)
    new_keys = [key for key in current_keys if key not in id_map]
    for key in new_keys:
        id_map[key] = f"tckt-{next_id:04d}"
        next_id += 1

    documents = [
        normalize_jira_issue(raw_by_key[key], id_map[key], client.settings, contract)
        for key in current_keys
    ]
    document_by_key = {doc["metadata"]["jira_key"]: doc for doc in documents}
    current_hashes = {key: _content_hash(document_by_key[key]) for key in current_keys}
    prior_hashes = prior_manifest.get("content_hashes", {})
    model_changed = bool(
        prior_manifest
        and (
            not prior_ready
            or prior_manifest.get("model_name") != model_name
            or prior_manifest.get("document_prefix") != document_prefix
            or prior_manifest.get("embedding_dimension") != embedding_dimension
            or prior_manifest.get("contract_version") != contract["contract_version"]
            or prior_manifest.get("project_key") != client.settings.project_key
            or prior_manifest.get("site") != client.settings.normalized_base_url
        )
    )
    updated_keys = [
        key for key in current_keys
        if key in prior_hashes and current_hashes[key] != prior_hashes[key]
    ]
    unchanged_keys = [
        key for key in current_keys
        if key not in new_keys and key not in updated_keys and not model_changed
    ]
    encode_keys = current_keys if model_changed else sorted(new_keys + updated_keys)

    encoded_by_key: dict[str, np.ndarray] = {}
    if encode_keys:
        texts = [
            document_prefix + document_by_key[key]["search"]["embedding_text"]
            for key in encode_keys
        ]
        vectors = encoder.encode(
            texts,
            batch_size=min(16, len(texts)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if vectors.shape != (len(encode_keys), embedding_dimension):
            raise RuntimeError("Jira embedding output has an unexpected shape")
        encoded_by_key = dict(zip(encode_keys, vectors))

    embeddings = np.vstack([
        encoded_by_key[key] if key in encoded_by_key else prior_vector_by_key[key]
        for key in current_keys
    ]).astype(np.float32) if current_keys else np.empty((0, embedding_dimension), dtype=np.float32)
    if len(embeddings) and not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5, rtol=1e-5
    ):
        raise RuntimeError("Jira embeddings are not normalized")

    removed_keys = sorted(set(prior_by_key) - set(current_keys))
    now = datetime.now(timezone.utc).isoformat()
    action = "LOAD" if prior_manifest and not encode_keys and not removed_keys else "SYNC"
    report = {
        "action": action,
        "fetched": len(issues),
        "new": new_keys,
        "updated": updated_keys,
        "unchanged": unchanged_keys,
        "removed": removed_keys,
        "encoded_count": len(encode_keys),
        "synced_at_utc": now,
    }
    if action == "LOAD":
        write_json_atomic(report, paths["report"])
        index = faiss.read_index(str(paths["index"]))
        if index.ntotal != len(prior_documents):
            raise RuntimeError("Loaded Jira live index is misaligned")
        return {
            "action": action,
            "report": report,
            "documents": prior_documents,
            "embeddings": prior_embeddings,
            "index": index,
            "mapping": _read_json(paths["mapping"], []),
            "manifest": prior_manifest,
            "paths": paths,
        }
    manifest = {
        "schema_version": contract["artifact_version"],
        "contract_version": contract["contract_version"],
        "model_name": model_name,
        "document_prefix": document_prefix,
        "embedding_dimension": embedding_dimension,
        "project_key": client.settings.project_key,
        "site": client.settings.normalized_base_url,
        "document_count": len(documents),
        "jira_key_to_document_id": {key: id_map[key] for key in current_keys},
        "content_hashes": current_hashes,
        "synced_at_utc": now,
    }
    _save_live_store(root, documents, embeddings, manifest, report)
    index = faiss.read_index(str(paths["index"]))
    reloaded_embeddings = np.load(paths["embeddings"], allow_pickle=False)
    reloaded_documents = _read_documents(paths["documents"])
    if index.ntotal != len(reloaded_documents) or len(reloaded_embeddings) != len(reloaded_documents):
        raise RuntimeError("Reloaded Jira live-index artifacts are misaligned")
    return {
        "action": action,
        "report": report,
        "documents": reloaded_documents,
        "embeddings": reloaded_embeddings,
        "index": index,
        "mapping": _read_json(paths["mapping"], []),
        "manifest": _read_json(paths["manifest"], {}),
        "paths": paths,
    }


def search_live_jira(
    query: str,
    *,
    sync_result: dict[str, Any],
    encoder: Any,
    query_prefix: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search only the small Jira overlay while preserving the public retrieval schema."""
    if not query.strip() or top_k < 1 or not sync_result["documents"]:
        return []
    query_vector = encoder.encode(
        [query_prefix + query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    count = min(top_k, len(sync_result["documents"]))
    scores, positions = sync_result["index"].search(query_vector, count)
    results = []
    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        document = sync_result["documents"][int(position)]
        results.append({
            "rank": rank,
            "score": float(score),
            "document_id": document["document_id"],
            "content": document["content"],
            "metadata": document["metadata"],
            "evaluation": document["evaluation"],
            "system": document["system"],
        })
    return results
