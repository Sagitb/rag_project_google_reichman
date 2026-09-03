"""Small, shared helpers for fingerprints and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

try:
    import torch
except ImportError:  # Allows metadata utilities to run before a local torch install.
    torch = None


def calculate_sha256(file_path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 fingerprint of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: Any, file_path: Path) -> Path:
    """Write JSON through a temporary file so partial artifacts are not promoted."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(f"{file_path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    temporary_path.replace(file_path)
    return file_path


def save_config_snapshot(config: dict, manifests_dir: Path) -> Path:
    """Persist the effective runtime configuration at a deterministic path."""
    output_path = Path(manifests_dir) / "project_config_latest.json"
    return write_json_atomic(config, output_path)


def write_run_metadata(
    stage_name: str,
    config: dict,
    logs_dir: Path,
    extra_metadata: dict | None = None,
) -> Path:
    """Record the latest reproducibility metadata for a completed project stage."""
    metadata = {
        "stage_name": stage_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": config["project_version"],
        "dataset_version": config["dataset_version"],
        "prompt_version": config["prompt_version"],
        "split_version": config["split_version"],
        "index_version": config["index_version"],
        "model_version": config["model_version"],
        "random_seed": config["random_seed"],
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__ if torch is not None else None,
            "cuda_available": bool(
                torch is not None and torch.cuda.is_available()
            ),
        },
        "extra": extra_metadata or {},
    }
    output_path = Path(logs_dir) / f"run_{stage_name}_latest.json"
    return write_json_atomic(metadata, output_path)
