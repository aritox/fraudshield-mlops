from __future__ import annotations

from pathlib import Path

import pytest

from fraudshield.models.tune_models import (
    SgdSearchConfig,
    XgboostSearchConfig,
    deterministic_sgd_trials,
    deterministic_xgboost_trials,
    load_tuning_config,
)


def test_sgd_trial_generation_is_deterministic_capped_and_moderate() -> None:
    config = SgdSearchConfig(
        positive_class_weights=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0],
        alpha_values=[0.00001, 0.0001, 0.001],
        epoch_values=[2, 3, 5],
        maximum_trials=12,
    )

    first = deterministic_sgd_trials(config)
    second = deterministic_sgd_trials(config)

    assert first == second
    assert len(first) == 12
    assert any(trial["positive_class_weight"] == 1.0 for trial in first)
    assert any(trial["positive_class_weight"] in {2.0, 5.0, 10.0} for trial in first)
    assert len({trial["alpha"] for trial in first}) > 1
    assert len({trial["epochs"] for trial in first}) > 1
    assert all(trial["positive_class_weight"] != pytest.approx(612.0) for trial in first)


def test_xgboost_trial_generation_is_deterministic_and_capped() -> None:
    config = XgboostSearchConfig(
        enabled=True,
        maximum_trials=3,
        final_nonfraud_sample_limit=100,
        inner_nonfraud_sample_limit=50,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=1,
        early_stopping_rounds=2,
        parameter_space={
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [5, 8],
            "min_child_weight": [1],
            "subsample": [1.0],
            "colsample_bytree": [1.0],
            "reg_lambda": [1.0],
            "reg_alpha": [0.0],
        },
    )

    first = deterministic_xgboost_trials(config, seed=42)
    second = deterministic_xgboost_trials(config, seed=42)

    assert first == second
    assert len(first) == 3
    assert all("n_estimators" in trial["parameters"] for trial in first)


def test_tuning_config_rejects_test_path_and_large_balanced_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "batch_size: 4",
                "train_path: data/processed/test.parquet",
                "validation_path: data/processed/validation.parquet",
                "selection_metric: average_precision",
                "threshold_metric: f_beta",
                "threshold_beta: 2.0",
                "inner_tuning:",
                "  method: chronological_whole_step",
                "  fit_fraction: 0.8",
                "  tuning_fraction: 0.2",
                "sgd_search:",
                "  positive_class_weights: [1.0, 612.0]",
                "  alpha_values: [0.0001]",
                "  epoch_values: [2]",
                "  maximum_trials: 2",
                "xgboost_search:",
                "  enabled: false",
                "  maximum_trials: 1",
                "  final_nonfraud_sample_limit: 10",
                "  inner_nonfraud_sample_limit: 5",
                "  tree_method: hist",
                "  objective: binary:logistic",
                "  eval_metric: aucpr",
                "  n_jobs: 1",
                "  early_stopping_rounds: 2",
                "  parameter_space:",
                "    max_depth: [3]",
                "top_k_percentages: [0.5]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sealed test split"):
        load_tuning_config(config_path=config_path, root=tmp_path)
