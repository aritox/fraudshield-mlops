from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.features.baseline import feature_names, forbidden_raw_columns
from fraudshield.models.tune_models import (
    InnerTuningConfig,
    SgdSearchConfig,
    TuningConfig,
    XgboostSearchConfig,
    sample_xgboost_fitting_data,
)


def _config() -> TuningConfig:
    return TuningConfig(
        random_seed=42,
        batch_size=4,
        train_path=Path("data/processed/train.parquet"),
        validation_path=Path("data/processed/validation.parquet"),
        selection_metric="average_precision",
        threshold_metric="f_beta",
        threshold_beta=2.0,
        minimum_pr_auc_improvement=0.001,
        inner_tuning=InnerTuningConfig(
            method="chronological_whole_step",
            fit_fraction=0.8,
            tuning_fraction=0.2,
        ),
        sgd_search=SgdSearchConfig(
            positive_class_weights=[1.0, 2.0],
            alpha_values=[0.0001],
            epoch_values=[2],
            maximum_trials=2,
        ),
        xgboost_search=XgboostSearchConfig(
            enabled=True,
            maximum_trials=1,
            final_nonfraud_sample_limit=3,
            inner_nonfraud_sample_limit=3,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=1,
            early_stopping_rounds=2,
            parameter_space={"max_depth": [3], "n_estimators": [5]},
        ),
        top_k_percentages=[0.5],
    )


def _write_training_data(path: Path) -> None:
    rows = []
    for step in range(1, 11):
        fraud = int(step in {2, 5, 9})
        rows.append(
            {
                "step": step,
                "type": "TRANSFER" if fraud else "PAYMENT",
                "amount": 100.0 + step,
                "oldbalanceOrg": 500.0,
                "oldbalanceDest": 20.0,
                "isFraud": fraud,
                "isFlaggedFraud": 0,
                "nameOrig": f"C{step}",
                "nameDest": f"M{step}",
                "newbalanceOrig": 0.0,
                "newbalanceDest": 0.0,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_xgboost_sampling_is_deterministic_retains_frauds_and_weights(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    _write_training_data(train_path)
    config = _config()

    first = sample_xgboost_fitting_data(train_path, config, None, nonfraud_limit=3)
    second = sample_xgboost_fitting_data(train_path, config, None, nonfraud_limit=3)

    x_first, y_first, weights_first, metadata = first
    x_second, y_second, weights_second, metadata_second = second

    np.testing.assert_allclose(x_first, x_second)
    np.testing.assert_array_equal(y_first, y_second)
    np.testing.assert_allclose(weights_first, weights_second)
    assert metadata == metadata_second
    assert int((y_first == 1).sum()) == 3
    assert int((y_first == 0).sum()) == 3
    assert metadata["original_nonfraud_rows"] == 7
    assert metadata["sample_weights"]["nonfraud"] == 7 / 3
    assert metadata["sample_weights"]["fraud"] == 1.0
    assert x_first.shape[1] == len(feature_names())


def test_raw_identifiers_are_not_model_features() -> None:
    assert set(feature_names()).isdisjoint({"nameOrig", "nameDest"})
    assert {"nameOrig", "nameDest"}.issubset(set(forbidden_raw_columns()))
