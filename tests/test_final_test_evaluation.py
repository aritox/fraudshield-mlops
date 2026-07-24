from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from fraudshield.data.validate import calculate_sha256
from fraudshield.models import evaluate_final_test as evaluator
from fraudshield.models import promote_sgd as promote
from fraudshield.models.train_baseline import reject_test_path


def _write_fixture(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "artifacts" / "data").mkdir(parents=True)
    (root / "artifacts" / "governance").mkdir(parents=True)
    (root / "artifacts" / "evaluation").mkdir(parents=True)
    (root / "configs" / "feature_policy.yaml").write_text("target: [isFraud]\n", encoding="utf-8")
    (root / "configs" / "production.yaml").write_text(
        Path("configs/production.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    frame = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["PAYMENT", "TRANSFER", "PAYMENT", "TRANSFER"],
            "amount": [10.0, 900.0, 20.0, 800.0],
            "oldbalanceOrg": [100.0, 1000.0, 200.0, 900.0],
            "oldbalanceDest": [50.0, 0.0, 40.0, 0.0],
            "isFraud": [0, 1, 0, 1],
        }
    )
    frame.to_parquet(root / "data" / "processed" / "test.parquet", index=False)
    (root / "artifacts" / "data" / "split_manifest.json").write_text("{}\n", encoding="utf-8")


def _write_bundle(root: Path) -> None:
    frame = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["PAYMENT", "TRANSFER", "PAYMENT", "TRANSFER"],
            "amount": [10.0, 900.0, 20.0, 800.0],
            "oldbalanceOrg": [100.0, 1000.0, 200.0, 900.0],
            "oldbalanceDest": [50.0, 0.0, 40.0, 0.0],
            "isFraud": [0, 1, 0, 1],
        }
    )
    transformer = promote.BaselineFeatureTransformer()
    features = transformer.transform(frame)
    scaler = StandardScaler().fit(features)
    model = SGDClassifier(loss="log_loss", alpha=0.00001, random_state=42).fit(
        scaler.transform(features), frame["isFraud"]
    )
    bundle = {
        "model": model,
        "model_family": "sgd_logistic",
        "scaler": scaler,
        "ordered_feature_names": promote.EXPECTED_FEATURES,
        "operational_threshold": 0.98310834,
        "frozen_hyperparameters": {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 0.00001,
            "epochs": 3,
            "positive_class_weight": 5.0,
        },
        "expected_raw_columns": promote.expected_raw_input_columns(),
        "forbidden_columns": promote.forbidden_raw_columns(),
        "feature_policy_metadata": {"test_set_accessed": False},
        "git_commit_hash": "test-commit",
    }
    artifact = root / promote.ARTIFACT_RELATIVE
    artifact.parent.mkdir(parents=True, exist_ok=True)
    promote.joblib.dump(bundle, artifact)
    config_hash = calculate_sha256(root / promote.CONFIG_RELATIVE)
    artifact_hash = calculate_sha256(artifact)
    manifest = {
        "production_config_sha256": config_hash,
        "artifact_sha256": artifact_hash,
        "git_commit_hash": "test-commit",
        "validation_reproduction_status": "passed",
        "test_set_accessed": False,
        "validation_verification_metrics": {
            "row_count": 4,
            "fraud_count": 2,
            "average_precision": 1.0,
            "roc_auc": 1.0,
            "threshold": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "f_beta": 1.0,
                "alert_rate": 0.5,
                "false_positive_rate": 0.0,
                "fraud_amount_recall": 1.0,
            },
            "top_k": [],
        },
    }
    (root / promote.MANIFEST_RELATIVE).write_text(json.dumps(manifest), encoding="utf-8")
    (root / promote.DECISION_RELATIVE).write_text(
        json.dumps({"decision_status": "approved_for_production", "test_set_accessed": False}),
        encoding="utf-8",
    )


def test_final_evaluation_uses_frozen_threshold_and_creates_marker(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_bundle(tmp_path)
    result = evaluator.evaluate_final_test(root=tmp_path)
    assert result["metrics"]["threshold"] == 0.98310834
    assert result["metrics"]["test_set_accessed"] is True
    assert (tmp_path / evaluator.MARKER_RELATIVE).exists()
    assert (tmp_path / evaluator.FINAL_MANIFEST_RELATIVE).exists()
    assert (tmp_path / evaluator.MODEL_CARD_RELATIVE).exists()


def test_completed_evaluation_is_reused_without_rescoring(tmp_path: Path, monkeypatch) -> None:
    _write_fixture(tmp_path)
    _write_bundle(tmp_path)
    first = evaluator.evaluate_final_test(root=tmp_path)

    def fail_score(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("completed final evaluation should be reused")

    monkeypatch.setattr(evaluator, "_score_test", fail_score)
    second = evaluator.evaluate_final_test(root=tmp_path)
    assert first["marker"] == second["marker"]
    assert second["reused"] is True


def test_non_final_training_module_rejects_test_path() -> None:
    try:
        reject_test_path(Path("data/processed/test.parquet"))
    except ValueError:
        return
    raise AssertionError("training modules must reject the sealed test path")
