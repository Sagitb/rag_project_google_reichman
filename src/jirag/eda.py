"""Focused exploratory analysis and reusable figure artifacts for jiRAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .artifacts import calculate_sha256, write_json_atomic
from .data_pipeline import calculate_canonical_fingerprint


EDA_SCHEMA_VERSION = "eda_summary_v2"
EDA_VERSION = "2.0"

OPERATIONAL_ORDERS = {
    "Priority": ["Highest", "High", "Medium", "Low"],
    "Status": ["To Do", "In Progress", "Done"],
    "Work Type": ["Bug", "Task", "Story"],
    "solution_type": [
        "solution-verified",
        "solution-partial",
        "solution-workaround",
        "solution-unresolved",
    ],
}


def _count_dict(series: pd.Series, order: list[str]) -> dict[str, int]:
    """Return deterministic category counts, including explicit zero values."""
    counts = series.value_counts()
    return {label: int(counts.get(label, 0)) for label in order}


def _validate_input(tickets_df: pd.DataFrame, canonical_sha256: str) -> None:
    """Confirm that EDA receives the unchanged output approved in Section 1."""
    required_columns = {
        "ticket_id",
        "Summary",
        "Work Type",
        "Priority",
        "Status",
        "Component",
        "family",
        "solution_type",
    }
    missing_columns = required_columns.difference(tickets_df.columns)
    if missing_columns:
        raise RuntimeError(f"EDA input is missing columns: {sorted(missing_columns)}")
    if tickets_df.empty:
        raise RuntimeError("EDA input is empty")
    if calculate_canonical_fingerprint(tickets_df) != canonical_sha256:
        raise RuntimeError("EDA input differs from the dataset approved in Section 1")


def _save_figure_atomic(fig: plt.Figure, output_path: Path) -> None:
    """Save a complete PNG before promoting it to its stable artifact path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary_path, dpi=150, bbox_inches="tight", format="png")
    temporary_path.replace(output_path)


def _build_operational_figure(
    distributions: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    """Visualize the four ticket fields used in operational questions."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    titles = {
        "Priority": "Tickets by Priority",
        "Status": "Tickets by Status",
        "Work Type": "Tickets by Work Type",
        "solution_type": "Tickets by Solution Type",
    }

    for ax, column in zip(axes.flat, OPERATIONAL_ORDERS):
        counts = distributions[column]
        labels = [label.removeprefix("solution-") for label in counts]
        values = list(counts.values())
        sns.barplot(x=values, y=labels, hue=labels, palette="viridis", legend=False, ax=ax)
        ax.set_title(titles[column])
        ax.set_xlabel("Ticket count")
        ax.set_ylabel("")
        for patch, value in zip(ax.patches, values):
            ax.text(value + 5, patch.get_y() + patch.get_height() / 2, str(value), va="center")
        ax.set_xlim(0, max(values) * 1.15)

    fig.suptitle("Operational Dataset Profile", fontsize=16)
    fig.tight_layout()
    _save_figure_atomic(fig, output_path)
    plt.close(fig)


def _build_family_solution_figure(
    family_solution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize solution coverage within each technical family."""
    display_table = family_solution.copy()
    display_table.index = display_table.index.str.removeprefix("family-")
    display_table.columns = display_table.columns.str.removeprefix("solution-")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(display_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Solution Coverage within Each Ticket Family")
    ax.set_xlabel("Solution type")
    ax.set_ylabel("Ticket family")
    fig.tight_layout()
    _save_figure_atomic(fig, output_path)
    plt.close(fig)


def _build_status_solution_figure(
    status_solution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize why workflow status and solution quality are not equivalent."""
    display_table = status_solution.copy()
    display_table.index = display_table.index.str.removeprefix("solution-")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(display_table, annot=True, fmt="d", cmap="OrRd", ax=ax)
    ax.set_title("Workflow Status by Solution Type")
    ax.set_xlabel("Jira status")
    ax.set_ylabel("Solution type")
    fig.tight_layout()
    _save_figure_atomic(fig, output_path)
    plt.close(fig)


def _artifacts_are_compatible(
    artifact_paths: dict[str, Path],
    canonical_sha256: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Return a saved summary only when every EDA artifact is compatible."""
    if not all(Path(path).is_file() for path in artifact_paths.values()):
        return False, None

    try:
        with Path(artifact_paths["summary"]).open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False, None

    expected = {
        "schema_version": EDA_SCHEMA_VERSION,
        "eda_version": EDA_VERSION,
        "canonical_sha256": canonical_sha256,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return False, None

    expected_hashes = summary.get("figure_sha256", {})
    for name in ("operational", "family_solution", "status_solution"):
        if expected_hashes.get(name) != calculate_sha256(artifact_paths[name]):
            return False, None
    return True, summary


def _validate_result(
    tickets_df: pd.DataFrame,
    summary: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    """Reject incomplete summaries, coverage gaps or empty persisted files."""
    errors = []
    ticket_count = len(tickets_df)
    if summary.get("overview", {}).get("tickets") != ticket_count:
        errors.append("EDA overview does not match the input row count")
    if sum(summary.get("distributions", {}).get("Status", {}).values()) != ticket_count:
        errors.append("Status distribution does not cover every ticket")

    status_solution = pd.DataFrame.from_dict(
        summary.get("status_solution_counts", {}),
        orient="index",
    )
    family_solution = pd.DataFrame.from_dict(
        summary.get("family_solution_counts", {}),
        orient="index",
    )
    if status_solution.empty or int(status_solution.to_numpy().sum()) != ticket_count:
        errors.append("Status × solution table does not cover every ticket")
    if family_solution.empty or int(family_solution.to_numpy().sum()) != ticket_count:
        errors.append("Family × solution table does not cover every ticket")
    for name, artifact_path in artifact_paths.items():
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            errors.append(f"Missing or empty EDA artifact: {name}")
    if errors:
        raise RuntimeError("EDA validation failed:\n- " + "\n- ".join(errors))


def ensure_eda_artifacts(
    tickets_df: pd.DataFrame,
    canonical_sha256: str,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """BUILD focused EDA figures once or LOAD compatible saved artifacts."""
    artifact_paths = {name: Path(path) for name, path in artifact_paths.items()}
    _validate_input(tickets_df, canonical_sha256)
    compatible, summary = _artifacts_are_compatible(artifact_paths, canonical_sha256)
    if compatible:
        _validate_result(tickets_df, summary, artifact_paths)
        return {"action": "LOAD", "summary": summary, "paths": artifact_paths}

    distributions = {
        column: _count_dict(tickets_df[column], order)
        for column, order in OPERATIONAL_ORDERS.items()
    }
    family_solution = pd.crosstab(
        tickets_df["family"],
        tickets_df["solution_type"],
    ).reindex(columns=OPERATIONAL_ORDERS["solution_type"], fill_value=0)
    status_solution = pd.crosstab(
        tickets_df["solution_type"],
        tickets_df["Status"],
    ).reindex(
        index=OPERATIONAL_ORDERS["solution_type"],
        columns=OPERATIONAL_ORDERS["Status"],
        fill_value=0,
    )

    _build_operational_figure(distributions, artifact_paths["operational"])
    _build_family_solution_figure(family_solution, artifact_paths["family_solution"])
    _build_status_solution_figure(status_solution, artifact_paths["status_solution"])

    summary = {
        "schema_version": EDA_SCHEMA_VERSION,
        "eda_version": EDA_VERSION,
        "canonical_sha256": canonical_sha256,
        "overview": {
            "tickets": int(len(tickets_df)),
            "families": int(tickets_df["family"].nunique()),
            "solution_types": int(tickets_df["solution_type"].nunique()),
            "unique_components": int(tickets_df["Component"].nunique()),
            "open_tickets": int(tickets_df["Status"].ne("Done").sum()),
            "done_tickets": int(tickets_df["Status"].eq("Done").sum()),
        },
        "distributions": distributions,
        "family_solution_counts": family_solution.astype(int).to_dict(orient="index"),
        "status_solution_counts": status_solution.astype(int).to_dict(orient="index"),
        "figure_sha256": {
            name: calculate_sha256(artifact_paths[name])
            for name in ("operational", "family_solution", "status_solution")
        },
    }
    write_json_atomic(summary, artifact_paths["summary"])
    _validate_result(tickets_df, summary, artifact_paths)
    return {"action": "BUILD", "summary": summary, "paths": artifact_paths}
