"""Loading, validation and persistent BUILD/LOAD logic for the Jira dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import calculate_sha256, write_json_atomic


DATASET_MANIFEST_SCHEMA = "dataset_manifest_v2"

EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
CREDENTIAL_PATTERN = re.compile(
    r"(?i)(password|token|secret|api_key)\s*[=:]\s*[^\s]{4,}"
)
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def load_dataset_contract(contract_path: Path) -> dict[str, Any]:
    """Load the versioned rules that define the approved dataset."""
    with Path(contract_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect_raw_csv(
    raw_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate Jira-compatible headers and row widths before using pandas."""
    errors = []
    row_count = 0

    with Path(raw_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"header": [], "row_count": 0, "errors": ["Raw CSV is empty"]}

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != len(header):
                errors.append(
                    f"Row {row_number} has {len(row)} fields; expected {len(header)}"
                )
            if not any(field.strip() for field in row):
                errors.append(f"Row {row_number} is completely empty")

    expected_header = contract["expected_raw_header"]
    if header != expected_header:
        errors.append("Raw header does not match the approved Jira CSV contract")

    expected_rows = contract["expected_row_count"]
    if row_count != expected_rows:
        errors.append(f"Raw CSV has {row_count} rows; expected {expected_rows}")

    return {"header": header, "row_count": row_count, "errors": errors}


def load_and_normalize_tickets(
    raw_path: Path,
    contract: dict[str, Any],
) -> pd.DataFrame:
    """Load the raw CSV and normalize only the three repeated Labels columns."""
    raw_frame = pd.read_csv(raw_path, encoding="utf-8-sig")
    return raw_frame.rename(columns=contract["label_rename_map"]).copy()


def calculate_canonical_fingerprint(frame: pd.DataFrame) -> str:
    """Hash normalized tabular content independently of CSV line endings."""
    rows = [list(frame.columns), *frame.values.tolist()]
    canonical_rows = [
        [str(value).replace("\r\n", "\n").replace("\r", "\n") for value in row]
        for row in rows
    ]
    serialized = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_schema(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> list[str]:
    """Validate table shape, column order and non-empty values."""
    errors = []
    expected_shape = (
        contract["expected_row_count"],
        len(contract["normalized_columns"]),
    )
    if frame.shape != expected_shape:
        errors.append(f"Normalized shape is {frame.shape}; expected {expected_shape}")
    if list(frame.columns) != contract["normalized_columns"]:
        errors.append("Normalized column order does not match the contract")
    if frame.isna().any().any():
        errors.append("Dataset contains missing values")

    for column in frame.columns:
        if frame[column].astype(str).str.strip().eq("").any():
            errors.append(f"Column '{column}' contains blank values")
    return errors


def _validate_categories(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> list[str]:
    """Validate approved categories and designed dataset balance."""
    errors = []
    category_contracts = {
        "Work Type": set(contract["allowed_work_types"]),
        "Priority": set(contract["allowed_priorities"]),
        "Status": set(contract["allowed_statuses"]),
        "family": set(contract["expected_families"]),
        "solution_type": set(contract["expected_solution_distribution"]),
    }
    for column, expected_values in category_contracts.items():
        actual_values = set(frame[column].unique())
        if actual_values != expected_values:
            errors.append(
                f"{column} values differ from the contract: {sorted(actual_values)}"
            )

    family_counts = frame["family"].value_counts().to_dict()
    expected_family_count = contract["expected_tickets_per_family"]
    for family in contract["expected_families"]:
        if family_counts.get(family) != expected_family_count:
            errors.append(
                f"{family} has {family_counts.get(family, 0)} tickets; "
                f"expected {expected_family_count}"
            )

    solution_counts = frame["solution_type"].value_counts().to_dict()
    if solution_counts != contract["expected_solution_distribution"]:
        errors.append(
            f"Solution distribution mismatch: {solution_counts}"
        )
    return errors


def _validate_identifiers(frame: pd.DataFrame) -> list[str]:
    """Validate unique content and the deterministic ticket-ID sequence."""
    errors = []
    for column in ("ticket_id", "Summary", "Description"):
        if not frame[column].is_unique:
            errors.append(f"Column '{column}' contains duplicate values")

    expected_ids = [f"tckt-{number:04d}" for number in range(1, len(frame) + 1)]
    if frame["ticket_id"].tolist() != expected_ids:
        errors.append("Ticket IDs are not sequential from tckt-0001")
    return errors


def _extract_description_sections(
    description: str,
    headings: list[str],
) -> tuple[list[str], list[str]]:
    """Return ordered headings and their intervening text blocks."""
    matches = []
    for heading in headings:
        heading_matches = list(
            re.finditer(rf"^{re.escape(heading)}\s*$", description, re.MULTILINE)
        )
        if len(heading_matches) != 1:
            return [], []
        matches.append((heading_matches[0].start(), heading_matches[0].end(), heading))

    matches.sort(key=lambda item: item[0])
    ordered_headings = [item[2] for item in matches]
    sections = []
    for index, (_, end, _) in enumerate(matches):
        next_start = matches[index + 1][0] if index + 1 < len(matches) else len(description)
        sections.append(description[end:next_start].strip())
    return ordered_headings, sections


def _validate_ticket_rows(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Validate workflow meaning, description structure and privacy signals."""
    errors = []
    warnings = []
    headings = contract["description_headings"]
    word_range = contract["description_word_count"]
    status_rules = contract["allowed_statuses_by_solution"]

    for row in frame.itertuples(index=False, name="Ticket"):
        ticket_id = row.ticket_id
        description = str(row.Description)
        summary = str(row.Summary)

        allowed_statuses = status_rules[row.solution_type]
        if row.Status not in allowed_statuses:
            errors.append(
                f"{ticket_id}: status '{row.Status}' is invalid for {row.solution_type}"
            )

        ordered_headings, sections = _extract_description_sections(
            description,
            headings,
        )
        if ordered_headings != headings:
            errors.append(f"{ticket_id}: description headings are missing or out of order")
        elif any(not section for section in sections):
            errors.append(f"{ticket_id}: at least one description section is empty")

        word_count = len(description.split())
        if not word_range["minimum"] <= word_count <= word_range["maximum"]:
            errors.append(
                f"{ticket_id}: description length {word_count} is outside the contract"
            )

        combined_text = f"{summary} {description}"
        if EMAIL_PATTERN.search(combined_text):
            errors.append(f"{ticket_id}: email address detected")
        if CREDENTIAL_PATTERN.search(combined_text):
            errors.append(f"{ticket_id}: credential assignment detected")
        if IPV4_PATTERN.search(combined_text):
            warnings.append(f"{ticket_id}: IPv4-like pattern requires review")

    return errors, warnings


def validate_ticket_dataset(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Run all approved dataset validation gates and return errors and warnings."""
    errors = _validate_schema(frame, contract)
    if errors:
        return errors, []

    errors.extend(_validate_categories(frame, contract))
    errors.extend(_validate_identifiers(frame))
    row_errors, warnings = _validate_ticket_rows(frame, contract)
    errors.extend(row_errors)
    return errors, warnings


def _write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> Path:
    """Persist a DataFrame without exposing a partially written CSV artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8", lineterminator="\n")
    temporary_path.replace(output_path)
    return output_path


def _artifacts_are_compatible(
    artifact_paths: dict[str, Path],
    expected: dict[str, str],
) -> bool:
    """Check whether persisted dataset artifacts match current inputs and rules."""
    if not all(Path(path).is_file() for path in artifact_paths.values()):
        return False

    try:
        with Path(artifact_paths["manifest"]).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    checks = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "dataset_version": expected["dataset_version"],
        "raw_sha256": expected["raw_sha256"],
        "canonical_sha256": expected["canonical_sha256"],
        "contract_sha256": expected["contract_sha256"],
    }
    if any(manifest.get(key) != value for key, value in checks.items()):
        return False

    return (
        manifest.get("processed_sha256")
        == calculate_sha256(artifact_paths["processed"])
        and manifest.get("qa_report_sha256")
        == calculate_sha256(artifact_paths["qa_report"])
    )


def ensure_validated_dataset(
    raw_path: Path,
    contract_path: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """Validate the approved raw data and BUILD or LOAD compatible artifacts."""
    raw_path = Path(raw_path)
    contract_path = Path(contract_path)
    artifact_paths = {name: Path(path) for name, path in artifact_paths.items()}
    raw_sha256_before = calculate_sha256(raw_path)
    contract = load_dataset_contract(contract_path)

    raw_inspection = inspect_raw_csv(raw_path, contract)
    tickets_frame = load_and_normalize_tickets(raw_path, contract)
    canonical_sha256 = calculate_canonical_fingerprint(tickets_frame)
    validation_errors, validation_warnings = validate_ticket_dataset(
        tickets_frame,
        contract,
    )
    validation_errors.extend(raw_inspection["errors"])

    expected_canonical = contract.get("expected_canonical_sha256")
    if expected_canonical and canonical_sha256 != expected_canonical:
        validation_errors.append(
            "Canonical fingerprint differs from the approved dataset contract"
        )

    if validation_errors:
        preview = "\n- ".join(validation_errors[:20])
        raise RuntimeError(f"Dataset validation failed:\n- {preview}")

    expected = {
        "dataset_version": contract["dataset_version"],
        "raw_sha256": raw_sha256_before,
        "canonical_sha256": canonical_sha256,
        "contract_sha256": calculate_sha256(contract_path),
    }
    compatible = _artifacts_are_compatible(artifact_paths, expected)

    if compatible:
        processed_frame = pd.read_csv(
            artifact_paths["processed"],
            encoding="utf-8",
        )
        processed_errors, _ = validate_ticket_dataset(processed_frame, contract)
        if processed_errors:
            raise RuntimeError(
                "Persisted processed dataset failed validation:\n- "
                + "\n- ".join(processed_errors[:20])
            )
        if calculate_canonical_fingerprint(processed_frame) != canonical_sha256:
            raise RuntimeError("Persisted processed dataset differs from raw content")

        with artifact_paths["qa_report"].open("r", encoding="utf-8") as handle:
            qa_report = json.load(handle)
        with artifact_paths["manifest"].open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        action = "LOAD"
        tickets_frame = processed_frame
    else:
        _write_csv_atomic(tickets_frame, artifact_paths["processed"])
        processed_sha256 = calculate_sha256(artifact_paths["processed"])
        timestamp = datetime.now(timezone.utc).isoformat()

        qa_report = {
            "schema_version": "dataset_qa_v2",
            "dataset_version": contract["dataset_version"],
            "validation_rule_version": contract["contract_version"],
            "timestamp_utc": timestamp,
            "result": "passed",
            "rows": len(tickets_frame),
            "columns": len(tickets_frame.columns),
            "errors": [],
            "warnings": validation_warnings,
            "raw_sha256": raw_sha256_before,
            "canonical_sha256": canonical_sha256,
        }
        write_json_atomic(qa_report, artifact_paths["qa_report"])
        manifest = {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "dataset_version": contract["dataset_version"],
            "validation_rule_version": contract["contract_version"],
            "created_at_utc": timestamp,
            "source_repo_path": f"data/raw/{raw_path.name}",
            "processed_artifact": artifact_paths["processed"].name,
            "row_count": len(tickets_frame),
            "column_count": len(tickets_frame.columns),
            "raw_sha256": raw_sha256_before,
            "canonical_sha256": canonical_sha256,
            "processed_sha256": processed_sha256,
            "qa_report_sha256": calculate_sha256(artifact_paths["qa_report"]),
            "contract_sha256": expected["contract_sha256"],
        }
        write_json_atomic(manifest, artifact_paths["manifest"])
        action = "BUILD"

    if calculate_sha256(raw_path) != raw_sha256_before:
        raise RuntimeError("Raw dataset changed during Section 1 execution")

    return {
        "action": action,
        "tickets_df": tickets_frame,
        "raw_header": raw_inspection["header"],
        "raw_row_count": raw_inspection["row_count"],
        "errors": [],
        "warnings": validation_warnings,
        "raw_sha256": raw_sha256_before,
        "canonical_sha256": canonical_sha256,
        "qa_report": qa_report,
        "manifest": manifest,
        "artifact_paths": artifact_paths,
    }
