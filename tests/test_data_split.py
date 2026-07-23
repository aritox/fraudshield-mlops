from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from fraudshield.data.split import SplitConfig, create_splits, load_split_config

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


def make_split_config() -> SplitConfig:
    return SplitConfig(
        time_column="step",
        target_column="isFraud",
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        chunk_size=3,
        output_directory=Path("data/processed"),
        compression="snappy",
        random_seed=42,
    )


def write_data_config(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "data.yaml").write_text(
        "\n".join(
            [
                "dataset_name: paysim",
                "kaggle_handle: ealaxi/paysim1",
                "target_column: isFraud",
                "chunk_size: 3",
                "raw_data_directory: data/raw",
                "expected_columns:",
                *[f"  - {column}" for column in EXPECTED_COLUMNS],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_split_config(root: Path, train: float = 0.70, validation: float = 0.15) -> Path:
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    config_path = configs / "split.yaml"
    test = 1.0 - train - validation
    config_path.write_text(
        "\n".join(
            [
                "time_column: step",
                "target_column: isFraud",
                f"train_fraction: {train}",
                f"validation_fraction: {validation}",
                f"test_fraction: {test}",
                "chunk_size: 3",
                "output_directory: data/processed",
                "compression: snappy",
                "random_seed: 42",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def write_paysim_csv(root: Path) -> Path:
    raw_dir = root / "data" / "raw" / "paysim_download"
    raw_dir.mkdir(parents=True)
    rows = []
    for step in range(1, 7):
        rows.extend(
            [
                {
                    "step": step,
                    "type": "PAYMENT",
                    "amount": float(step * 10),
                    "nameOrig": f"C{step}A",
                    "oldbalanceOrg": 100.0,
                    "newbalanceOrig": 90.0,
                    "nameDest": f"M{step}",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "isFraud": 0,
                    "isFlaggedFraud": 0,
                },
                {
                    "step": step,
                    "type": "TRANSFER",
                    "amount": float(step * 100),
                    "nameOrig": f"C{step}B",
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 0.0,
                    "nameDest": f"C{step}C",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 1000.0,
                    "isFraud": 1,
                    "isFlaggedFraud": 1,
                },
            ]
        )
    csv_path = raw_dir / "paysim.csv"
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    return csv_path


def prepare_root(root: Path) -> Path:
    write_data_config(root)
    write_split_config(root)
    return write_paysim_csv(root)


def test_fractions_must_sum_to_one(tmp_path: Path) -> None:
    write_split_config(tmp_path, train=0.70, validation=0.20)
    config_path = tmp_path / "configs" / "split.yaml"
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("test_fraction: 0.10000000000000003", "test_fraction: 0.2"))

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_split_config(root=tmp_path)


def test_temporal_split_outputs_and_manifest(tmp_path: Path) -> None:
    prepare_root(tmp_path)

    result = create_splits(root=tmp_path, config=make_split_config())

    assert result.passed is True
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    parsed = datetime.fromisoformat(manifest["split_timestamp_utc"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None

    train = pd.read_parquet(tmp_path / "data" / "processed" / "train.parquet")
    validation = pd.read_parquet(tmp_path / "data" / "processed" / "validation.parquet")
    test = pd.read_parquet(tmp_path / "data" / "processed" / "test.parquet")

    train_steps = set(train["step"])
    validation_steps = set(validation["step"])
    test_steps = set(test["step"])
    assert train_steps.isdisjoint(validation_steps)
    assert train_steps.isdisjoint(test_steps)
    assert validation_steps.isdisjoint(test_steps)
    assert max(train_steps) < min(validation_steps)
    assert max(validation_steps) < min(test_steps)
    assert len(train) + len(validation) + len(test) == 12
    assert int(train["isFraud"].sum() + validation["isFraud"].sum() + test["isFraud"].sum()) == 6
    assert all((tmp_path / "data" / "processed" / filename).exists() for filename in (
        "train.parquet",
        "validation.parquet",
        "test.parquet",
    ))
    assert manifest["checks"]["total_row_conservation"] is True
    assert manifest["checks"]["total_fraud_conservation"] is True
    assert manifest["checks"]["no_step_overlap"] is True
    assert manifest["checks"]["chronological_ordering_check"] is True


def test_existing_valid_output_can_be_reused(tmp_path: Path) -> None:
    prepare_root(tmp_path)

    first = create_splits(root=tmp_path, config=make_split_config())
    second = create_splits(root=tmp_path, config=make_split_config())

    assert first.reused is False
    assert second.reused is True
    assert second.manifest["split_timestamp_utc"] == first.manifest["split_timestamp_utc"]


def test_force_replaces_only_generated_outputs_and_preserves_raw(tmp_path: Path) -> None:
    csv_path = prepare_root(tmp_path)
    raw_before = csv_path.read_text(encoding="utf-8")
    keep_path = tmp_path / "data" / "processed" / "keep.txt"

    create_splits(root=tmp_path, config=make_split_config())
    keep_path.write_text("preserve", encoding="utf-8")
    train_path = tmp_path / "data" / "processed" / "train.parquet"
    train_path.write_text("stale generated output", encoding="utf-8")

    result = create_splits(root=tmp_path, config=make_split_config(), force=True)

    assert result.passed is True
    assert result.reused is False
    assert csv_path.read_text(encoding="utf-8") == raw_before
    assert keep_path.read_text(encoding="utf-8") == "preserve"
    assert pd.read_parquet(train_path).shape[0] > 0
