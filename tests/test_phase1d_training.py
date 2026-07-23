from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.models import tune_models as tune_module
from fraudshield.models.tune_models import (
    FROZEN_CONFIGS_RELATIVE,
    load_phase1d_bundle,
    predict_with_phase1d_bundle,
    tune_models,
)


def _rows(start: int, stop: int) -> list[dict[str, object]]:
    rows = []
    for step in range(start, stop + 1):
        fraud = int(step % 5 == 0)
        rows.append(
            {
                "step": step,
                "type": "TRANSFER" if fraud else "PAYMENT",
                "amount": 900.0 if fraud else 25.0 + step,
                "oldbalanceOrg": 800.0 if fraud else 300.0,
                "oldbalanceDest": 0.0 if fraud else 40.0,
                "isFraud": fraud,
                "isFlaggedFraud": 0,
                "nameOrig": f"C{step}",
                "nameDest": f"M{step}",
                "newbalanceOrig": 0.0,
                "newbalanceDest": 0.0,
            }
        )
    return rows


def _write_tuning_config(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "feature_policy.yaml").write_text("forbidden: true\n", encoding="utf-8")
    path = config_dir / "tuning.yaml"
    path.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "batch_size: 4",
                "train_path: data/processed/train.parquet",
                "validation_path: data/processed/validation.parquet",
                "selection_metric: average_precision",
                "threshold_metric: f_beta",
                "threshold_beta: 2.0",
                "minimum_pr_auc_improvement: 0.001",
                "inner_tuning:",
                "  method: chronological_whole_step",
                "  fit_fraction: 0.8",
                "  tuning_fraction: 0.2",
                "sgd_search:",
                "  positive_class_weights: [1.0, 2.0, 5.0]",
                "  alpha_values: [0.0001, 0.001]",
                "  epoch_values: [2, 3]",
                "  maximum_trials: 4",
                "xgboost_search:",
                "  enabled: true",
                "  maximum_trials: 1",
                "  final_nonfraud_sample_limit: 12",
                "  inner_nonfraud_sample_limit: 8",
                "  tree_method: hist",
                "  objective: binary:logistic",
                "  eval_metric: aucpr",
                "  n_jobs: 1",
                "  early_stopping_rounds: 2",
                "  parameter_space:",
                "    max_depth: [2]",
                "    learning_rate: [0.1]",
                "    n_estimators: [5]",
                "    min_child_weight: [1]",
                "    subsample: [1.0]",
                "    colsample_bytree: [1.0]",
                "    reg_lambda: [1.0]",
                "    reg_alpha: [0.0]",
                "top_k_percentages: [0.1, 0.5, 1.0]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_fixture(root: Path) -> Path:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(_rows(1, 20)).to_parquet(processed / "train.parquet", index=False)
    pd.DataFrame(_rows(21, 30)).to_parquet(processed / "validation.parquet", index=False)
    (processed / "test.parquet").write_text("sealed", encoding="utf-8")
    artifact_data = root / "artifacts" / "data"
    artifact_data.mkdir(parents=True)
    (artifact_data / "split_manifest.json").write_text(
        json.dumps(
            {
                "splitting_method": "chronological_whole_step",
                "splits": {
                    "train": {"row_count": 20, "fraud_count": 4},
                    "validation": {"row_count": 10, "fraud_count": 2},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return _write_tuning_config(root)


def test_tiny_end_to_end_phase1d_creates_artifacts_bundle_and_predictions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    original_validation = tune_module._validation_scores_for_candidates
    original_parquet_file = tune_module.pq.ParquetFile

    def guarded_validation(*args, **kwargs):  # noqa: ANN002, ANN003
        assert (tmp_path / FROZEN_CONFIGS_RELATIVE).exists()
        return original_validation(*args, **kwargs)

    def guarded_parquet_file(path):  # noqa: ANN001
        assert Path(path).name != "test.parquet"
        return original_parquet_file(path)

    monkeypatch.setattr(tune_module, "_validation_scores_for_candidates", guarded_validation)
    monkeypatch.setattr(tune_module.pq, "ParquetFile", guarded_parquet_file)

    result = tune_models(root=tmp_path, config_path=config_path, force=True)
    bundle = load_phase1d_bundle(result.model_artifact_path)
    validation = pd.read_parquet(tmp_path / "data" / "processed" / "validation.parquet")
    probabilities, predictions = predict_with_phase1d_bundle(bundle, validation)

    assert result.model_artifact_path.exists()
    assert result.validation_metrics["test_set_accessed"] is False
    assert result.validation_metrics["official_validation_accessed_during_search"] is False
    assert result.tuning_manifest["test_set_accessed"] is False
    assert result.model_manifest["test_set_accessed"] is False
    assert result.model_manifest["artifact_sha256"]
    assert all(path.exists() for path in result.plot_paths)
    assert probabilities.shape == predictions.shape == (10,)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()


def test_force_replaces_only_phase1d_outputs(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    keep = tmp_path / "artifacts" / "data" / "split_manifest.json"
    before = keep.read_text(encoding="utf-8")

    first = tune_models(root=tmp_path, config_path=config_path, force=True)
    stale_plot = tmp_path / "artifacts" / "tuning" / "plots" / "01_sgd_trial_pr_auc.png"
    stale_plot.write_text("stale", encoding="utf-8")
    second = tune_models(root=tmp_path, config_path=config_path, force=True)

    assert first.model_artifact_path.exists()
    assert second.model_artifact_path.exists()
    assert keep.read_text(encoding="utf-8") == before
    assert stale_plot.read_bytes() != b"stale"
