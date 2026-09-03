"""Deterministic, leakage-safe BUILD/LOAD handling for dataset splits."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from .artifacts import calculate_sha256, write_json_atomic
from .data_pipeline import calculate_canonical_fingerprint


SPLIT_MANIFEST_SCHEMA = "split_manifest_v2"
SPLIT_ALGORITHM_VERSION = "sklearn_two_stage_stratified_v1"
SPLIT_NAMES = ("train", "validation", "test")
SOLUTION_ORDER = (
    "solution-verified",
    "solution-partial",
    "solution-workaround",
    "solution-unresolved",
)


def _config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Select the settings that define split membership."""
    return {
        "split_version": config["split_version"],
        "random_seed": config["random_seed"],
        "ratios": {
            "train": config["train_ratio"],
            "validation": config["validation_ratio"],
            "test": config["test_ratio"],
        },
        "stratification_columns": config["stratification_columns"],
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    """Fingerprint JSON-compatible configuration deterministically."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stratification_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Create the combined family/solution key used by both split stages."""
    return frame[columns].astype(str).agg("|".join, axis=1)


def _expected_sizes(row_count: int, ratios: dict[str, float]) -> dict[str, int]:
    """Derive the exact two-stage split sizes from the configured ratios."""
    train_count = int(row_count * ratios["train"])
    remaining_count = row_count - train_count
    validation_share = ratios["validation"] / (
        ratios["validation"] + ratios["test"]
    )
    validation_count = int(remaining_count * validation_share)
    return {
        "train": train_count,
        "validation": validation_count,
        "test": remaining_count - validation_count,
    }


def _build_split_frames(
    tickets_df: pd.DataFrame,
    config_payload: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Apply the approved two-stage stratified split and stable ID ordering."""
    working = tickets_df.copy()
    working["_stratification_key"] = _stratification_key(
        working,
        config_payload["stratification_columns"],
    )
    ratios = config_payload["ratios"]
    train_work, remaining_work = train_test_split(
        working,
        train_size=ratios["train"],
        random_state=config_payload["random_seed"],
        stratify=working["_stratification_key"],
    )
    validation_share = ratios["validation"] / (
        ratios["validation"] + ratios["test"]
    )
    validation_work, test_work = train_test_split(
        remaining_work,
        train_size=validation_share,
        random_state=config_payload["random_seed"],
        stratify=remaining_work["_stratification_key"],
    )

    def finalize(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.drop(columns="_stratification_key")
            .sort_values("ticket_id")
            .reset_index(drop=True)
        )

    return {
        "train": finalize(train_work),
        "validation": finalize(validation_work),
        "test": finalize(test_work),
    }


def _write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Persist a complete split CSV before promoting its stable path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    frame.to_csv(temporary_path, index=False, encoding="utf-8", lineterminator="\n")
    temporary_path.replace(output_path)


def _save_figure_atomic(fig: plt.Figure, output_path: Path) -> None:
    """Persist a complete PNG before promoting its stable path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary_path, dpi=150, bbox_inches="tight", format="png")
    temporary_path.replace(output_path)


def _calculate_evidence(
    tickets_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    """Calculate compact split evidence while validating every ticket and stratum."""
    errors = []
    ratios = config_payload["ratios"]
    stratification_columns = config_payload["stratification_columns"]
    expected_sizes = _expected_sizes(len(tickets_df), ratios)
    expected_schema = tuple(tickets_df.columns)
    source_ids = set(tickets_df["ticket_id"])
    expected_strata = set(_stratification_key(tickets_df, stratification_columns))
    split_id_sets = {}

    for name in SPLIT_NAMES:
        frame = split_frames[name]
        split_id_sets[name] = set(frame["ticket_id"])
        if len(frame) != expected_sizes[name]:
            errors.append(f"{name}: expected {expected_sizes[name]} rows, found {len(frame)}")
        if tuple(frame.columns) != expected_schema:
            errors.append(f"{name}: schema differs from the approved source")
        if not frame["ticket_id"].is_unique:
            errors.append(f"{name}: duplicate ticket IDs detected")
        if set(_stratification_key(frame, stratification_columns)) != expected_strata:
            errors.append(f"{name}: not all expected strata are represented")

        expected_content = (
            tickets_df.loc[tickets_df["ticket_id"].isin(split_id_sets[name])]
            .sort_values("ticket_id")
            .reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(frame, expected_content, check_dtype=False)
        except AssertionError:
            errors.append(f"{name}: content differs from the approved source rows")

    overlap_count = sum(
        len(split_id_sets[left] & split_id_sets[right])
        for left, right in combinations(SPLIT_NAMES, 2)
    )
    covered_ids = set().union(*split_id_sets.values())
    if overlap_count:
        errors.append(f"Pairwise split overlap contains {overlap_count} IDs")
    if covered_ids != source_ids:
        errors.append("Split IDs do not provide complete source coverage")
    if errors:
        raise RuntimeError("Split validation failed:\n- " + "\n- ".join(errors))

    all_frames = {"full_dataset": tickets_df, **split_frames}
    available_solutions = set(tickets_df["solution_type"])
    solution_order = [solution for solution in SOLUTION_ORDER if solution in available_solutions]
    solution_percentages = {
        name: {
            solution: float(frame["solution_type"].eq(solution).mean() * 100)
            for solution in solution_order
        }
        for name, frame in all_frames.items()
    }
    source_solution = pd.Series(solution_percentages["full_dataset"])
    maximum_solution_deviation = {
        name: float(
            (pd.Series(solution_percentages[name]) - source_solution).abs().max()
        )
        for name in SPLIT_NAMES
    }

    source_strata = _stratification_key(tickets_df, stratification_columns).value_counts()
    maximum_stratum_error = {}
    for name, frame in split_frames.items():
        observed = _stratification_key(frame, stratification_columns).value_counts()
        ideal = source_strata * ratios[name]
        maximum_stratum_error[name] = float(
            observed.reindex(source_strata.index, fill_value=0).sub(ideal).abs().max()
        )

    overview = []
    for name, frame in all_frames.items():
        overview.append(
            {
                "split": name,
                "tickets": int(len(frame)),
                "families": int(frame["family"].nunique()),
                "solution_types": int(frame["solution_type"].nunique()),
                "strata": int(_stratification_key(frame, stratification_columns).nunique()),
            }
        )
    integrity = {
        "pairwise_overlap_ids": overlap_count,
        "covered_source_ids": len(covered_ids),
        "source_ids": len(source_ids),
        "schema_preserved": True,
        "all_strata_present_in_every_split": True,
        "test_frozen": True,
    }
    diagnostics = {
        "maximum_solution_percentage_point_deviation": maximum_solution_deviation,
        "maximum_stratum_allocation_error_tickets": maximum_stratum_error,
    }
    return {
        "overview": overview,
        "integrity": integrity,
        "solution_percentages": solution_percentages,
        "diagnostics": diagnostics,
    }


def _build_balance_figure(
    solution_percentages: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Show whether solution-type proportions remain stable across splits."""
    display_order = ["full_dataset", *SPLIT_NAMES]
    table = pd.DataFrame.from_dict(solution_percentages, orient="index").reindex(display_order)
    table.index = [name.replace("_", " ").title() for name in table.index]
    table.columns = table.columns.str.removeprefix("solution-")

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 5))
    annotations = table.map(lambda value: f"{value:.1f}%")
    sns.heatmap(table, annot=annotations, fmt="", cmap="Blues", ax=ax)
    ax.set_title("Solution-Type Balance across Dataset Splits")
    ax.set_xlabel("Solution type")
    ax.set_ylabel("")
    fig.tight_layout()
    _save_figure_atomic(fig, output_path)
    plt.close(fig)


def _read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Read an existing manifest when it is available and valid JSON."""
    if not Path(manifest_path).is_file():
        return None
    try:
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _protect_frozen_split(
    artifact_paths: dict[str, Path],
    existing_manifest: dict[str, Any] | None,
    canonical_sha256: str,
    config_payload: dict[str, Any],
) -> None:
    """Prevent silent replacement of a frozen test set after its inputs change."""
    core_names = (*SPLIT_NAMES, "ids", "manifest")
    if not any(artifact_paths[name].exists() for name in core_names):
        return
    if existing_manifest is None:
        raise RuntimeError(
            "Existing split artifacts have no readable manifest; remove them or bump split_version"
        )
    if existing_manifest.get("source_canonical_sha256") != canonical_sha256:
        raise RuntimeError("Dataset changed after Test was frozen; bump split_version")

    comparable_fields = {
        "split_version": config_payload["split_version"],
        "random_seed": config_payload["random_seed"],
        "ratios": config_payload["ratios"],
        "stratification_columns": config_payload["stratification_columns"],
    }
    for field, expected in comparable_fields.items():
        if existing_manifest.get(field) != expected:
            raise RuntimeError(
                f"Split setting '{field}' changed after Test was frozen; bump split_version"
            )
    if (
        existing_manifest.get("schema_version") == SPLIT_MANIFEST_SCHEMA
        and existing_manifest.get("algorithm_version") != SPLIT_ALGORITHM_VERSION
    ):
        raise RuntimeError("Split algorithm changed; bump split_version")


def _artifacts_are_compatible(
    artifact_paths: dict[str, Path],
    manifest: dict[str, Any] | None,
    canonical_sha256: str,
    config_sha256: str,
) -> bool:
    """Accept saved splits only when source, settings and every file hash match."""
    if manifest is None or not all(path.is_file() for path in artifact_paths.values()):
        return False
    expected_fields = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "source_canonical_sha256": canonical_sha256,
        "split_config_sha256": config_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        return False

    for name in SPLIT_NAMES:
        if manifest.get("files", {}).get(name, {}).get("sha256") != calculate_sha256(
            artifact_paths[name]
        ):
            return False
    if manifest.get("split_ids_file", {}).get("sha256") != calculate_sha256(
        artifact_paths["ids"]
    ):
        return False
    return manifest.get("balance_figure", {}).get("sha256") == calculate_sha256(
        artifact_paths["figure"]
    )


def _load_split_frames(artifact_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load the three persisted split CSVs using stable names."""
    return {
        name: pd.read_csv(artifact_paths[name], encoding="utf-8")
        for name in SPLIT_NAMES
    }


def ensure_dataset_splits(
    tickets_df: pd.DataFrame,
    canonical_sha256: str,
    config: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """BUILD deterministic splits once or LOAD the compatible frozen artifacts."""
    artifact_paths = {name: Path(path) for name, path in artifact_paths.items()}
    if calculate_canonical_fingerprint(tickets_df) != canonical_sha256:
        raise RuntimeError("Split input differs from the dataset approved in Section 1")

    config_payload = _config_payload(config)
    ratios = config_payload["ratios"]
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Train, validation and test ratios must sum to 1.0")
    if any(column not in tickets_df.columns for column in config_payload["stratification_columns"]):
        raise ValueError("A configured stratification column is missing")

    config_sha256 = _payload_sha256(config_payload)
    manifest = _read_manifest(artifact_paths["manifest"])
    if _artifacts_are_compatible(
        artifact_paths,
        manifest,
        canonical_sha256,
        config_sha256,
    ):
        split_frames = _load_split_frames(artifact_paths)
        evidence = _calculate_evidence(tickets_df, split_frames, config_payload)
        with artifact_paths["ids"].open("r", encoding="utf-8") as handle:
            split_ids = json.load(handle)
        for name in SPLIT_NAMES:
            if split_ids["ids"][name] != split_frames[name]["ticket_id"].tolist():
                raise RuntimeError(f"{name}: frozen ID list differs from its split CSV")
        return {
            "action": "LOAD",
            "frames": split_frames,
            "manifest": manifest,
            "evidence": evidence,
            "paths": artifact_paths,
        }

    _protect_frozen_split(
        artifact_paths,
        manifest,
        canonical_sha256,
        config_payload,
    )
    split_frames = _build_split_frames(tickets_df, config_payload)
    evidence = _calculate_evidence(tickets_df, split_frames, config_payload)
    for name in SPLIT_NAMES:
        _write_csv_atomic(split_frames[name], artifact_paths[name])

    split_ids = {
        "split_version": config_payload["split_version"],
        "random_seed": config_payload["random_seed"],
        "source_canonical_sha256": canonical_sha256,
        "ids": {
            name: split_frames[name]["ticket_id"].tolist()
            for name in SPLIT_NAMES
        },
    }
    write_json_atomic(split_ids, artifact_paths["ids"])
    _build_balance_figure(evidence["solution_percentages"], artifact_paths["figure"])

    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "split_version": config_payload["split_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": config_payload["random_seed"],
        "source_dataset_version": config["dataset_version"],
        "source_canonical_sha256": canonical_sha256,
        "ratios": ratios,
        "stratification_columns": config_payload["stratification_columns"],
        "split_config_sha256": config_sha256,
        "counts": {name: len(frame) for name, frame in split_frames.items()},
        "unique_strata": {
            row["split"]: row["strata"]
            for row in evidence["overview"]
            if row["split"] in SPLIT_NAMES
        },
        "files": {
            name: {
                "filename": artifact_paths[name].name,
                "sha256": calculate_sha256(artifact_paths[name]),
            }
            for name in SPLIT_NAMES
        },
        "split_ids_file": {
            "filename": artifact_paths["ids"].name,
            "sha256": calculate_sha256(artifact_paths["ids"]),
        },
        "balance_figure": {
            "filename": artifact_paths["figure"].name,
            "sha256": calculate_sha256(artifact_paths["figure"]),
        },
        "integrity": evidence["integrity"],
        "diagnostics": evidence["diagnostics"],
    }
    write_json_atomic(manifest, artifact_paths["manifest"])
    return {
        "action": "BUILD",
        "frames": split_frames,
        "manifest": manifest,
        "evidence": evidence,
        "paths": artifact_paths,
    }
