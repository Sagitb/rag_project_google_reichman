"""Repository and persistent-artifact path management."""

from __future__ import annotations

from pathlib import Path


def build_project_paths(repo_root: Path, artifact_root: Path) -> dict[str, Path]:
    """Build one path registry while keeping Git sources separate from artifacts."""
    repo_root = Path(repo_root).resolve()
    artifact_root = Path(artifact_root).resolve()

    return {
        "root": artifact_root,
        "repo_root": repo_root,
        "src": repo_root / "src",
        "configs": repo_root / "configs",
        "data_raw": repo_root / "data" / "raw",
        "data_processed": artifact_root / "data" / "processed",
        "data_splits": artifact_root / "data" / "splits",
        "data_eval": artifact_root / "data" / "evaluation",
        "data_manifests": artifact_root / "data" / "manifests",
        "vector_store": artifact_root / "vector_store",
        "models": artifact_root / "models",
        "checkpoints": artifact_root / "checkpoints",
        "reports_qa": artifact_root / "reports" / "data_qa",
        "reports_eval": artifact_root / "reports" / "evaluation",
        "reports_generation": artifact_root / "reports" / "generation",
        "reports_figs": artifact_root / "reports" / "figures",
        "logs": artifact_root / "logs",
        "exports": artifact_root / "exports",
    }


def create_artifact_directories(paths: dict[str, Path]) -> None:
    """Create only generated-output directories; never mutate Git source paths."""
    source_keys = {"repo_root", "src", "configs", "data_raw"}
    for name, path in paths.items():
        if name not in source_keys:
            path.mkdir(parents=True, exist_ok=True)
