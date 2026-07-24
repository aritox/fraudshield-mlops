from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudshield.features.baseline import feature_names, forbidden_raw_columns
from fraudshield.models import promote_sgd as promote
from fraudshield.models.train_baseline import reject_test_path


def _frame(rows: int = 12) -> pd.DataFrame:
    values = []
    for index in range(rows):
        fraud = int(index % 4 == 0)
        values.append(
            {
                "step": index + 1,
                "type": "TRANSFER" if fraud else "PAYMENT",
                "amount": 800.0 if fraud else 20.0 + index,
                "oldbalanceOrg": 700.0 if fraud else 200.0,
                "oldbalanceDest": 0.0 if fraud else 50.0,
                "isFraud": fraud,
            }
        )
    return pd.DataFrame(values)


def _write_fixture(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "artifacts" / "data").mkdir(parents=True)
    (root / "artifacts" / "tuning").mkdir(parents=True)
    (root / "artifacts" / "governance").mkdir(parents=True)
    (root / "artifacts" / "modeling").mkdir(parents=True)
    (root / "configs" / "feature_policy.yaml").write_text("target: [isFraud]\n", encoding="utf-8")
    config = Path("configs/production.yaml").read_text(encoding="utf-8")
    (root / "configs" / "production.yaml").write_text(config, encoding="utf-8")
    frame = _frame()
    frame.iloc[:8].to_parquet(root / "data" / "processed" / "train.parquet", index=False)
    frame.iloc[8:].to_parquet(root / "data" / "processed" / "validation.parquet", index=False)
    (root / "data" / "processed" / "test.parquet").write_text("sealed", encoding="utf-8")
    (root / "artifacts" / "data" / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "artifacts" / "tuning" / "frozen_candidate_configs.json").write_text(
        json.dumps(
            {
                "best_sgd_configuration": {
                    "alpha": 0.00001,
                    "epochs": 3,
                    "positive_class_weight": 5.0,
                }
            }
        ),
        encoding="utf-8",
    )


def test_production_config_and_feature_policy_are_frozen() -> None:
    config = promote.load_production_config()
    assert config["alpha"] == 0.00001
    assert config["epochs"] == 3
    assert config["positive_class_weight"] == 5.0
    assert config["operational_threshold"] == 0.98310834
    assert feature_names() == [
        "step",
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "log_amount",
        "log_oldbalance_origin",
        "log_oldbalance_destination",
        "log_amount_to_origin_balance",
        "log_amount_to_destination_balance",
        "origin_balance_zero_before",
        "destination_balance_zero_before",
        "amount_exceeds_origin_balance",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]
    assert set(feature_names()).isdisjoint(forbidden_raw_columns())


def test_promotion_rejects_test_path() -> None:
    with pytest.raises(ValueError, match="sealed test split"):
        reject_test_path(Path("data/processed/test.parquet"))


def test_tiny_promotion_creates_bundle_without_opening_test(tmp_path: Path, monkeypatch) -> None:
    _write_fixture(tmp_path)
    config = promote.load_production_config(root=tmp_path)
    transformer = promote.BaselineFeatureTransformer()
    scaler, _counts = promote._fit_scaler_and_count(
        tmp_path / config["train_path"], config["batch_size"], transformer
    )
    model = promote._train_sgd(
        tmp_path / config["train_path"], config["batch_size"], 42, scaler, transformer
    )
    y_true, amount, scores = promote._collect_validation(
        tmp_path / config["validation_path"], config["batch_size"], scaler, model
    )
    compact = promote._compact_metrics(y_true, amount, scores, 0.98310834, [0.1, 0.5, 1.0])
    reference = {
        "phase1c_unweighted_logistic": promote.REFERENCE_METRICS["phase1c_unweighted_logistic"],
        "phase1d_tuned_sgd_logistic": {
            "average_precision": compact["average_precision"],
            "roc_auc": compact["roc_auc"],
            "selected_threshold_metrics": compact["threshold"],
        },
        "phase1d_tuned_xgboost": promote.REFERENCE_METRICS["phase1d_tuned_xgboost"],
    }
    monkeypatch.setattr(promote, "_phase1d_reference", lambda _root: reference)
    original = promote.pq.ParquetFile

    def guarded(path):  # noqa: ANN001
        assert Path(path).name != "test.parquet"
        return original(path)

    monkeypatch.setattr(promote.pq, "ParquetFile", guarded)
    result = promote.promote_sgd(root=tmp_path, force=True)
    bundle = promote.joblib.load(result["artifact_path"])
    assert result["artifact_path"].exists()
    assert bundle["ordered_feature_names"] == feature_names()
    assert bundle["class_weights"] == {"0": 1.0, "1": 5.0}
    assert result["manifest"]["test_set_accessed"] is False
    assert result["manifest"]["validation_reproduction_status"] == "passed"


def test_training_uses_positive_weight_five_and_deterministic_shuffle(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    config = promote.load_production_config(root=tmp_path)
    transformer = promote.BaselineFeatureTransformer()
    scaler, _counts = promote._fit_scaler_and_count(
        tmp_path / config["train_path"], config["batch_size"], transformer
    )
    first = promote._train_sgd(
        tmp_path / config["train_path"], config["batch_size"], 42, scaler, transformer
    )
    second = promote._train_sgd(
        tmp_path / config["train_path"], config["batch_size"], 42, scaler, transformer
    )
    np.testing.assert_allclose(first.coef_, second.coef_)
    assert first.get_params()["alpha"] == 0.00001
