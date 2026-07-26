"""Tests for the inference-only production MLflow wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.features.baseline import feature_names
from fraudshield.tracking.model_wrapper import (
    OUTPUT_COLUMNS,
    ProductionSGDPyFuncModel,
    synthetic_input_example,
)


class IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class DeterministicScoreModel:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        scores = np.array([0.1, 0.7, 0.99], dtype=np.float64)[: len(values)]
        return np.column_stack([1 - scores, scores])

    def fit(self, *_args, **_kwargs):
        raise AssertionError("fit must never be called")

    def partial_fit(self, *_args, **_kwargs):
        raise AssertionError("partial_fit must never be called")


def _wrapper() -> ProductionSGDPyFuncModel:
    wrapper = ProductionSGDPyFuncModel(0.5, 0.98310834)
    wrapper._bundle = {
        "feature_names": feature_names(),
        "scaler": IdentityScaler(),
        "model": DeterministicScoreModel(),
        "operational_threshold": 0.98310834,
    }
    return wrapper


def test_wrapper_preserves_order_input_and_frozen_threshold() -> None:
    frame = synthetic_input_example()
    frame.index = [9, 3, 7]
    original = frame.copy(deep=True)

    output = _wrapper().predict(None, frame)

    assert list(output.columns) == list(OUTPUT_COLUMNS)
    assert output.index.tolist() == [9, 3, 7]
    assert output["fraud_score"].tolist() == [0.1, 0.7, 0.99]
    assert output["prediction"].tolist() == [0, 0, 1]
    assert output["threshold"].tolist() == [0.98310834] * 3
    assert output["risk_level"].tolist() == ["low", "medium", "high"]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"isFraud": 0}, "Forbidden"),
        ({"unexpected": 1}, "Unexpected"),
        ({"amount": -1.0}, "non-negative"),
        ({"oldbalanceOrg": np.inf}, "finite"),
        ({"type": "UNKNOWN"}, "Unknown"),
    ],
)
def test_wrapper_rejects_invalid_inputs(change: dict[str, object], match: str) -> None:
    frame = synthetic_input_example()
    for column, value in change.items():
        frame[column] = value
    with pytest.raises(ValueError, match=match):
        _wrapper().predict(None, frame)


def test_wrapper_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing"):
        _wrapper().predict(None, synthetic_input_example().drop(columns="amount"))
