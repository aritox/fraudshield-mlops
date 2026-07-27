"""Frozen aggregate training-reference profile tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from fraudshield.monitoring import reference as reference_module
from fraudshield.monitoring.config import load_monitoring_config
from fraudshield.monitoring.reference import (
    SELECTED_COLUMNS,
    build_reference_profile,
    export_reference,
    install_reference_access_guard,
    validate_reference_columns,
    validate_reference_source,
)
from fraudshield.tracking.mlflow_setup import sha256_file


def _root(tmp_path: Path) -> tuple[Path, object]:
    config_source = yaml.safe_load(Path("configs/monitoring.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "configs" / "monitoring.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config_source), encoding="utf-8")
    train_path = tmp_path / "data" / "processed" / "train.parquet"
    train_path.parent.mkdir(parents=True)
    table = pa.table(
        {
            "step": [1, 1, 2, 2],
            "type": ["TRANSFER", "PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [0.0, 10.0, 100.0, 1000.0],
            "oldbalanceOrg": [0.0, 20.0, 200.0, 2000.0],
            "oldbalanceDest": [0.0, 30.0, 300.0, 3000.0],
            "isFraud": [0, 0, 1, 1],
            "newbalanceOrig": [0.0, 0.0, 0.0, 0.0],
            "nameOrig": ["A", "B", "C", "D"],
        }
    )
    pq.write_table(table, train_path)
    return train_path, load_monitoring_config(config_path, root=tmp_path)


def test_reference_profile_is_deterministic_aggregate_and_selected_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _train, config = _root(tmp_path)
    original_parquet_file = reference_module.pq.ParquetFile
    observed_columns: list[str] = []

    class ParquetProxy:
        def __init__(self, path: Path) -> None:
            self.inner = original_parquet_file(path)
            self.metadata = self.inner.metadata

        def iter_batches(self, *args, **kwargs):
            columns = list(kwargs["columns"])
            observed_columns.extend(columns)
            assert set(columns).issubset(SELECTED_COLUMNS)
            return self.inner.iter_batches(*args, **kwargs)

    monkeypatch.setattr(reference_module.pq, "ParquetFile", ParquetProxy)
    first, metadata = build_reference_profile(config)
    second, second_metadata = build_reference_profile(config)

    assert first == second
    assert metadata == second_metadata
    assert metadata["source_row_count"] == 4
    assert metadata["source_step_minimum"] == 1
    assert metadata["source_step_maximum"] == 2
    assert set(observed_columns) == set(SELECTED_COLUMNS)
    assert first["source_split"] == "train"
    assert set(first["numeric_features"]) == {
        "step",
        "log1p(amount)",
        "log1p(oldbalanceOrg)",
        "log1p(oldbalanceDest)",
    }
    serialized = json.dumps(first)
    for forbidden in ("isFraud", "newbalanceOrig", "nameOrig", "raw_rows", "samples"):
        assert forbidden not in serialized
    for profile in first["numeric_features"].values():
        assert sum(profile["reference_counts"]) == profile["non_missing_count"]
        assert sum(profile["reference_proportions"]) == pytest.approx(1.0)
        assert profile["minimum"] <= profile["maximum"]
    categorical = first["categorical_features"]["type"]
    assert sum(categorical["reference_counts"]) == categorical["non_missing_count"]
    assert categorical["unknown_count"] == 0


def test_reference_manifest_checksum_provenance_and_access_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _train, config = _root(tmp_path)
    monkeypatch.setattr(
        "fraudshield.monitoring.reference.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    manifest = export_reference(config)
    stored_profile = json.loads(config.reference_profile_path.read_text(encoding="utf-8"))

    assert manifest["profile_sha256"] == sha256_file(config.reference_profile_path)
    assert manifest["profile_version"] == "phase2d_train_v1"
    assert manifest["source_split"] == "train"
    assert manifest["selected_columns"] == list(SELECTED_COLUMNS)
    assert len(manifest["source_selected_columns_sha256"]) == 64
    assert manifest["source_provenance_kind"] == "ordered_selected_columns_sha256"
    assert manifest["label_accessed"] is False
    assert manifest["validation_accessed"] is False
    assert manifest["test_accessed"] is False
    assert manifest["raw_data_accessed"] is False
    assert manifest["raw_rows_stored"] is False
    assert stored_profile["source_row_count"] == 4
    assert str(tmp_path) not in json.dumps(manifest)


def test_reference_guards_reject_raw_validation_test_and_labels(tmp_path: Path) -> None:
    train = tmp_path / "data" / "processed" / "train.parquet"
    validation = tmp_path / "data" / "processed" / "validation.parquet"
    test = tmp_path / "data" / "processed" / "test.parquet"
    raw = tmp_path / "data" / "raw" / "data.csv"
    train.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    for path in (train, validation, test, raw):
        path.write_bytes(b"fixture")

    assert validate_reference_source(tmp_path, train) == train.resolve()
    for prohibited in (validation, test, raw):
        with pytest.raises(ValueError):
            validate_reference_source(tmp_path, prohibited)
    for prohibited_column in ("isFraud", "isFlaggedFraud", "newbalanceOrig", "nameDest"):
        with pytest.raises(ValueError):
            validate_reference_columns(("step", prohibited_column))

    install_reference_access_guard(tmp_path)
    for prohibited in (validation, test, raw):
        with pytest.raises(PermissionError):
            prohibited.open("rb")
