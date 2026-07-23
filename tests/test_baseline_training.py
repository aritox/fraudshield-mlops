from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudshield.models import train_baseline as train_module
from fraudshield.models.train_baseline import (
    load_champion_bundle,
    predict_parquet_with_bundle,
    predict_with_bundle,
    reject_test_path,
    train_baseline,
)


def rows() -> list[dict[str, object]]:
    base = []
    for step in range(1, 13):
        fraud = int(step % 4 == 0)
        base.append(
            {
                "step": step,
                "type": "TRANSFER" if fraud else "PAYMENT",
                "amount": 900.0 if fraud else 20.0 + step,
                "oldbalanceOrg": 500.0 if fraud else 200.0,
                "oldbalanceDest": 0.0 if fraud else 50.0,
                "isFraud": fraud,
                "isFlaggedFraud": 0,
                "nameOrig": f"C{step}",
                "nameDest": f"M{step}",
                "newbalanceOrig": 0.0,
                "newbalanceDest": 0.0,
            }
        )
    return base


def write_config(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "feature_policy.yaml").write_text("forbidden: true\n", encoding="utf-8")
    config_path = config_dir / "modeling.yaml"
    config_path.write_text(
        "\n".join(
            [
                "batch_size: 4",
                "random_seed: 42",
                "training_epochs: 2",
                "train_path: data/processed/train.parquet",
                "validation_path: data/processed/validation.parquet",
                "selection_metric: average_precision",
                "threshold_metric: f_beta",
                "threshold_beta: 2.0",
                "top_k_percentages:",
                "  - 50.0",
                "models:",
                "  unweighted_logistic:",
                "    loss: log_loss",
                "    penalty: l2",
                "    alpha: 0.0001",
                "    learning_rate: optimal",
                "    average: true",
                "  weighted_logistic:",
                "    loss: log_loss",
                "    penalty: l2",
                "    alpha: 0.0001",
                "    learning_rate: optimal",
                "    average: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def write_fixtures(root: Path) -> Path:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    frame = pd.DataFrame(rows())
    frame.iloc[:8].to_parquet(processed / "train.parquet", index=False)
    frame.iloc[8:].to_parquet(processed / "validation.parquet", index=False)
    (processed / "test.parquet").write_text("sealed", encoding="utf-8")
    artifact_data = root / "artifacts" / "data"
    artifact_data.mkdir(parents=True)
    (artifact_data / "split_manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"row_count": 8, "fraud_count": 2},
                    "validation": {"row_count": 4, "fraud_count": 1},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return write_config(root)


def test_test_parquet_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="sealed test split"):
        reject_test_path(Path("data/processed/test.parquet"))


def test_tiny_end_to_end_training_creates_artifacts_and_bundle(tmp_path: Path) -> None:
    config_path = write_fixtures(tmp_path)

    result = train_baseline(root=tmp_path, config_path=config_path, force=True)
    bundle = load_champion_bundle(result.model_artifact_path)
    validation = pd.read_parquet(tmp_path / "data" / "processed" / "validation.parquet")
    probabilities, predictions = predict_with_bundle(bundle, validation)

    assert result.model_artifact_path.exists()
    assert result.metrics_path.exists()
    assert result.manifest_path.exists()
    assert all(path.exists() for path in result.plot_paths)
    assert len(bundle["feature_names"]) == 17
    assert probabilities.shape == predictions.shape == (4,)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()
    assert result.metrics["test_data_was_not_accessed"] is True
    assert result.manifest["test_set_accessed"] is False


def test_class_weights_use_training_counts_only(tmp_path: Path) -> None:
    config_path = write_fixtures(tmp_path)

    result = train_baseline(root=tmp_path, config_path=config_path, force=True)

    assert result.manifest["training_rows"] == 8
    assert result.manifest["training_fraud_count"] == 2
    if result.manifest["class_weights"] is not None:
        assert result.manifest["class_weights"]["0"] == pytest.approx(8 / (2 * 6))
        assert result.manifest["class_weights"]["1"] == pytest.approx(8 / (2 * 2))


def test_validation_is_not_used_in_scaler_partial_fit(tmp_path: Path, monkeypatch) -> None:
    config_path = write_fixtures(tmp_path)
    seen_fit_sizes = []
    original = train_module.StandardScaler

    class RecordingScaler(original):
        def partial_fit(self, x, y=None, sample_weight=None):  # noqa: ANN001
            seen_fit_sizes.append(len(x))
            return super().partial_fit(x, y=y, sample_weight=sample_weight)

    def write_placeholder_bundle(bundle, path):  # noqa: ANN001
        Path(path).write_bytes(b"placeholder bundle")

    monkeypatch.setattr(train_module, "StandardScaler", RecordingScaler)
    monkeypatch.setattr(train_module.joblib, "dump", write_placeholder_bundle)

    train_baseline(root=tmp_path, config_path=config_path, force=True)

    assert seen_fit_sizes == [4, 4]


def test_training_never_opens_test_fixture(tmp_path: Path, monkeypatch) -> None:
    config_path = write_fixtures(tmp_path)
    original = train_module.pq.ParquetFile

    def guarded_parquet_file(path):  # noqa: ANN001
        assert Path(path).name != "test.parquet"
        return original(path)

    monkeypatch.setattr(train_module.pq, "ParquetFile", guarded_parquet_file)

    train_baseline(root=tmp_path, config_path=config_path, force=True)


def test_loaded_bundle_rejects_test_parquet_as_evaluation_source(tmp_path: Path) -> None:
    config_path = write_fixtures(tmp_path)
    result = train_baseline(root=tmp_path, config_path=config_path, force=True)
    bundle = load_champion_bundle(result.model_artifact_path)

    with pytest.raises(ValueError, match="sealed test split"):
        predict_parquet_with_bundle(bundle, tmp_path / "data" / "processed" / "test.parquet")


def test_force_affects_only_phase_1c_outputs(tmp_path: Path) -> None:
    config_path = write_fixtures(tmp_path)
    keep = tmp_path / "artifacts" / "data" / "split_manifest.json"
    before = keep.read_text(encoding="utf-8")

    train_baseline(root=tmp_path, config_path=config_path, force=True)
    stale = (
        tmp_path
        / "artifacts"
        / "modeling"
        / "plots"
        / "01_validation_precision_recall_curve.png"
    )
    stale.write_text("stale", encoding="utf-8")
    train_baseline(root=tmp_path, config_path=config_path, force=True)

    assert keep.read_text(encoding="utf-8") == before
    assert stale.read_bytes() != b"stale"
