from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fraudshield.analysis.eda import (
    PLOT_FILENAMES,
    create_training_eda,
    deterministic_training_sample,
)


def write_training_split(root: Path) -> Path:
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True)
    rows = [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 10.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 90.0,
            "nameDest": "M1",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 1000.0,
            "nameOrig": "C2",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C3",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "isFraud": 1,
            "isFlaggedFraud": 1,
        },
        {
            "step": 2,
            "type": "CASH_OUT",
            "amount": 50.0,
            "nameOrig": "C4",
            "oldbalanceOrg": 50.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C5",
            "oldbalanceDest": 10.0,
            "newbalanceDest": 60.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
        {
            "step": 2,
            "type": "TRANSFER",
            "amount": 500.0,
            "nameOrig": "C6",
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C7",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 500.0,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        },
    ]
    train_path = processed_dir / "train.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    return train_path


def test_eda_reads_only_training_split_and_creates_artifacts(tmp_path: Path) -> None:
    write_training_split(tmp_path)

    result = create_training_eda(root=tmp_path, max_non_fraud_sample=2)

    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["total_training_rows"] == 4
    assert summary["fraud_count"] == 2
    assert summary["non_fraud_count"] == 2
    for field in (
        "transaction_count_by_type",
        "fraud_count_by_type",
        "fraud_rate_by_type",
        "transaction_count_by_step",
        "fraud_count_by_step",
        "fraud_rate_by_step",
        "amount_statistics_by_class",
        "balance_statistics_by_class",
        "flagged_fraud_crosstab",
        "zero_balance_frequencies",
        "missing_values_by_column",
    ):
        assert field in summary
    assert len(result.plot_paths) == len(PLOT_FILENAMES)
    assert all(path.exists() for path in result.plot_paths)


def test_deterministic_sampling_is_reproducible(tmp_path: Path) -> None:
    train_path = write_training_split(tmp_path)

    first = deterministic_training_sample(train_path, max_non_fraud=1, seed=42)
    second = deterministic_training_sample(train_path, max_non_fraud=1, seed=42)

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert int((first["isFraud"] == 1).sum()) == 2
    assert int((first["isFraud"] == 0).sum()) == 1
