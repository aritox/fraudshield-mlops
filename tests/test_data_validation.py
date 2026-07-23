from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fraudshield.data.config import DataConfig
from fraudshield.data.validate import validate_csv

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def make_config(chunk_size: int = 2) -> DataConfig:
    return DataConfig(
        dataset_name="paysim",
        kaggle_handle="ealaxi/paysim1",
        target_column="isFraud",
        chunk_size=chunk_size,
        raw_data_directory=Path("data/raw"),
        expected_columns=EXPECTED_COLUMNS,
    )


def write_csv(root: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> Path:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "synthetic_paysim.csv"
    selected_columns = columns or EXPECTED_COLUMNS

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        file.write(",".join(selected_columns) + "\n")
        for row in rows:
            file.write(",".join(str(row.get(column, "")) for column in selected_columns) + "\n")

    return csv_path


def valid_rows() -> list[dict[str, object]]:
    return [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 100.5,
            "nameOrig": "C1",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 899.5,
            "nameDest": "M1",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 250.0,
            "nameOrig": "C2",
            "oldbalanceOrg": 250.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C3",
            "oldbalanceDest": 10.0,
            "newbalanceDest": 260.0,
            "isFraud": 1,
            "isFlaggedFraud": 1,
        },
        {
            "step": 2,
            "type": "CASH_OUT",
            "amount": 10.0,
            "nameOrig": "C4",
            "oldbalanceOrg": 10.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C5",
            "oldbalanceDest": 5.0,
            "newbalanceDest": 15.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
    ]


def test_valid_schema_acceptance(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, valid_rows())

    result = validate_csv(csv_path=csv_path, config=make_config(), root=tmp_path)

    assert result.passed is True
    assert result.quality_report["validation_status"] == "passed"
    assert result.quality_report["total_rows"] == 3
    assert result.quality_report["fraud_transaction_count"] == 1
    assert result.quality_report["transaction_count_by_type"] == {
        "CASH_OUT": 1,
        "PAYMENT": 1,
        "TRANSFER": 1,
    }


def test_missing_column_rejection(tmp_path: Path) -> None:
    rows = valid_rows()
    columns = [column for column in EXPECTED_COLUMNS if column != "isFlaggedFraud"]
    csv_path = write_csv(tmp_path, rows, columns=columns)

    with pytest.raises(ValueError, match="CSV columns do not match expected schema"):
        validate_csv(csv_path=csv_path, config=make_config(), root=tmp_path)


def test_invalid_binary_target_rejection(tmp_path: Path) -> None:
    rows = valid_rows()
    rows[0]["isFraud"] = 2
    csv_path = write_csv(tmp_path, rows)

    result = validate_csv(csv_path=csv_path, config=make_config(), root=tmp_path)

    assert result.passed is False
    assert result.quality_report["validation_status"] == "failed"
    assert any("isFraud" in error for error in result.errors)


def test_negative_amount_rejection(tmp_path: Path) -> None:
    rows = valid_rows()
    rows[0]["amount"] = -1.0
    csv_path = write_csv(tmp_path, rows)

    result = validate_csv(csv_path=csv_path, config=make_config(), root=tmp_path)

    assert result.passed is False
    assert result.quality_report["validation_status"] == "failed"
    assert any("negative amount" in error for error in result.errors)


def test_report_creation(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, valid_rows())

    result = validate_csv(csv_path=csv_path, config=make_config(), root=tmp_path)

    assert result.manifest_path.exists()
    assert result.quality_report_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    quality_report = json.loads(result.quality_report_path.read_text(encoding="utf-8"))

    assert manifest["dataset_name"] == "paysim"
    assert manifest["relative_local_path"] == "data/raw/synthetic_paysim.csv"
    assert "validation_timestamp_utc" in manifest
    validation_timestamp = manifest["validation_timestamp_utc"]
    assert validation_timestamp.endswith("Z")
    parsed_timestamp = datetime.fromisoformat(validation_timestamp.replace("Z", "+00:00"))
    assert parsed_timestamp.tzinfo == UTC
    assert manifest["total_rows"] == 3
    assert quality_report["validation_status"] == "passed"
