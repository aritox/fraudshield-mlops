"""Validate the PaySim CSV schema and data quality."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fraudshield.data.config import DataConfig, load_data_config, raw_data_directory
from fraudshield.data.download import find_matching_csv

VALID_TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
NUMERIC_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
BINARY_COLUMNS = ["isFraud", "isFlaggedFraud"]


@dataclass
class ValidationResult:
    """Structured output from CSV validation."""

    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    manifest_path: Path
    quality_report_path: Path
    passed: bool
    errors: list[str] = field(default_factory=list)


def calculate_sha256(csv_path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum without loading the file into memory."""

    digest = hashlib.sha256()
    with csv_path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def _to_builtin_mapping(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp using a Z suffix."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_invalid_values_error(errors: list[str], column: str, invalid_values: set[Any]) -> None:
    values = ", ".join(repr(value) for value in sorted(invalid_values, key=str))
    errors.append(f"Column {column} contains invalid values: {values}")


def _validate_chunk(
    chunk: pd.DataFrame,
    errors: list[str],
    missing_counts: Counter[str],
    transaction_counts: Counter[str],
    fraud_by_type: Counter[str],
) -> dict[str, int | float | None]:
    missing_counts.update({column: int(value) for column, value in chunk.isna().sum().items()})
    missing_total = int(chunk.isna().sum().sum())
    if missing_total:
        errors.append(f"Detected {missing_total} missing values in a chunk")

    for column in NUMERIC_COLUMNS:
        numeric_values = pd.to_numeric(chunk[column], errors="coerce")
        invalid_numeric = numeric_values.isna() & chunk[column].notna()
        if bool(invalid_numeric.any()):
            errors.append(f"Column {column} contains non-numeric values")
        chunk[column] = numeric_values

    negative_amount_count = int((chunk["amount"] < 0).sum())
    if negative_amount_count:
        errors.append(f"Detected {negative_amount_count} negative amount values")

    for column in BINARY_COLUMNS:
        invalid_values = set(chunk.loc[~chunk[column].isin([0, 1]), column].dropna().unique())
        if invalid_values:
            _append_invalid_values_error(errors, column, invalid_values)
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    invalid_type_mask = ~chunk["type"].isin(VALID_TRANSACTION_TYPES)
    invalid_types = set(chunk.loc[invalid_type_mask, "type"].dropna().unique())
    if invalid_types:
        _append_invalid_values_error(errors, "type", invalid_types)

    transaction_counts.update(str(value) for value in chunk["type"].dropna())
    fraud_rows = chunk[chunk["isFraud"] == 1]
    fraud_by_type.update(str(value) for value in fraud_rows["type"].dropna())

    amount_min = chunk["amount"].min(skipna=True)
    amount_max = chunk["amount"].max(skipna=True)

    return {
        "rows": int(len(chunk)),
        "fraud": int((chunk["isFraud"] == 1).sum()),
        "flagged": int((chunk["isFlaggedFraud"] == 1).sum()),
        "amount_min": None if pd.isna(amount_min) else float(amount_min),
        "amount_max": None if pd.isna(amount_max) else float(amount_max),
    }


def validate_csv(
    csv_path: Path,
    config: DataConfig,
    root: Path,
    artifacts_directory: Path | None = None,
) -> ValidationResult:
    """Validate a CSV and write manifest and data quality reports."""

    errors: list[str] = []
    actual_columns = _read_header(csv_path)
    if actual_columns != config.expected_columns:
        missing = sorted(set(config.expected_columns) - set(actual_columns))
        unexpected = sorted(set(actual_columns) - set(config.expected_columns))
        raise ValueError(
            "CSV columns do not match expected schema. "
            f"Missing: {missing or 'none'}; unexpected: {unexpected or 'none'}"
        )

    total_rows = 0
    fraud_count = 0
    flagged_fraud_count = 0
    amount_min: float | None = None
    amount_max: float | None = None
    missing_counts: Counter[str] = Counter({column: 0 for column in config.expected_columns})
    transaction_counts: Counter[str] = Counter()
    fraud_by_type: Counter[str] = Counter()

    for chunk in pd.read_csv(csv_path, chunksize=config.chunk_size):
        chunk_stats = _validate_chunk(
            chunk=chunk,
            errors=errors,
            missing_counts=missing_counts,
            transaction_counts=transaction_counts,
            fraud_by_type=fraud_by_type,
        )
        total_rows += int(chunk_stats["rows"])
        fraud_count += int(chunk_stats["fraud"])
        flagged_fraud_count += int(chunk_stats["flagged"])

        chunk_amount_min = chunk_stats["amount_min"]
        chunk_amount_max = chunk_stats["amount_max"]
        if chunk_amount_min is not None:
            amount_min = (
                float(chunk_amount_min)
                if amount_min is None
                else min(amount_min, float(chunk_amount_min))
            )
        if chunk_amount_max is not None:
            amount_max = (
                float(chunk_amount_max)
                if amount_max is None
                else max(amount_max, float(chunk_amount_max))
            )

    non_fraud_count = total_rows - fraud_count
    fraud_percentage = (fraud_count / total_rows * 100) if total_rows else 0.0
    validation_status = "failed" if errors else "passed"
    checksum = calculate_sha256(csv_path)
    timestamp = utc_timestamp()

    artifact_root = artifacts_directory or root / "artifacts" / "data"
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "dataset_manifest.json"
    quality_report_path = artifact_root / "data_quality_report.json"

    manifest = {
        "dataset_name": config.dataset_name,
        "kaggle_handle": config.kaggle_handle,
        "original_filename": csv_path.name,
        "relative_local_path": csv_path.relative_to(root).as_posix(),
        "file_size_bytes": int(csv_path.stat().st_size),
        "sha256_checksum": checksum,
        "validation_timestamp_utc": timestamp,
        "total_rows": int(total_rows),
        "total_columns": int(len(config.expected_columns)),
    }

    quality_report = {
        "dataset_name": config.dataset_name,
        "validation_status": validation_status,
        "validation_errors": errors,
        "total_rows": int(total_rows),
        "total_columns": int(len(config.expected_columns)),
        "fraud_transaction_count": int(fraud_count),
        "non_fraud_transaction_count": int(non_fraud_count),
        "fraud_percentage": float(fraud_percentage),
        "flagged_fraud_count": int(flagged_fraud_count),
        "missing_value_count_by_column": _to_builtin_mapping(missing_counts),
        "minimum_transaction_amount": amount_min,
        "maximum_transaction_amount": amount_max,
        "transaction_count_by_type": _to_builtin_mapping(transaction_counts),
        "fraud_count_by_transaction_type": _to_builtin_mapping(fraud_by_type),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    quality_report_path.write_text(json.dumps(quality_report, indent=2) + "\n", encoding="utf-8")

    return ValidationResult(
        manifest=manifest,
        quality_report=quality_report,
        manifest_path=manifest_path,
        quality_report_path=quality_report_path,
        passed=validation_status == "passed",
        errors=errors,
    )


def _print_summary(result: ValidationResult) -> None:
    report = result.quality_report
    print("Data validation summary")
    print(f"Status: {report['validation_status']}")
    print(f"Rows: {report['total_rows']}")
    print(f"Columns: {report['total_columns']}")
    print(f"Fraud count: {report['fraud_transaction_count']}")
    print(f"Fraud percentage: {report['fraud_percentage']:.6f}%")
    print(f"Flagged fraud count: {report['flagged_fraud_count']}")
    print("Transaction counts by type:")
    for transaction_type, count in report["transaction_count_by_type"].items():
        print(f"  {transaction_type}: {count}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Quality report: {result.quality_report_path}")
    if result.errors:
        print("Validation errors:")
        for error in result.errors:
            print(f"  - {error}")


def main() -> int:
    root = Path(__file__).resolve().parents[3]

    try:
        config = load_data_config(root=root)
        raw_dir = raw_data_directory(config, root)
        csv_path = find_matching_csv(raw_dir, config.expected_columns)
        if csv_path is None:
            raise FileNotFoundError(
                f"No PaySim CSV with the expected schema was found under {raw_dir}."
            )
        result = validate_csv(csv_path=csv_path, config=config, root=root)
        _print_summary(result)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
