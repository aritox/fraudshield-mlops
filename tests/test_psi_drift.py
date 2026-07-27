"""Population Stability Index behavior and safety tests."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from fraudshield.monitoring.config import DriftConfig
from fraudshield.monitoring.drift import (
    categorical_drift,
    numeric_drift,
    population_stability_index,
)

THRESHOLDS = DriftConfig(0.10, 0.25, 0.25)
PROFILE = {
    "transformation": "identity",
    "bin_edges": [0.0, 1.0, 2.0],
    "reference_counts": [5, 5],
    "reference_proportions": [0.5, 0.5],
}


def test_identical_moderate_and_significant_numeric_shift() -> None:
    identical = numeric_drift(
        PROFILE,
        [0.5] * 5 + [1.5] * 5,
        minimum_events=10,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    moderate = numeric_drift(
        PROFILE,
        [0.5] * 7 + [1.5] * 3,
        minimum_events=10,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    significant = numeric_drift(
        PROFILE,
        [0.5] * 9 + [1.5],
        minimum_events=10,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    assert identical.metric_value == pytest.approx(0.0)
    assert identical.severity == "stable"
    assert moderate.severity == "moderate"
    assert significant.severity == "significant"


def test_zero_bins_out_of_range_and_log_transform_remain_finite() -> None:
    value = population_stability_index([1.0, 0.0], [0.5, 0.5], 1e-6)
    result = numeric_drift(
        PROFILE,
        [-100.0] * 5 + [100.0] * 5,
        minimum_events=10,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    log_profile = {
        **PROFILE,
        "transformation": "log1p",
        "bin_edges": [0.0, float(np.log1p(10.0)), float(np.log1p(100.0))],
    }
    transformed = numeric_drift(
        log_profile,
        [0.0] * 5 + [1000.0] * 5,
        minimum_events=10,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    assert math.isfinite(value)
    assert result.current_counts == [5, 5]
    assert transformed.current_counts == [5, 5]
    with pytest.raises(ValueError):
        numeric_drift(
            log_profile,
            [-1.0] * 10,
            minimum_events=10,
            epsilon=1e-6,
            thresholds=THRESHOLDS,
        )


def test_unknown_categories_insufficient_sample_and_baseline_immutability() -> None:
    profile = {
        "allowed_categories": ["A", "B"],
        "reference_counts": [5, 5, 0],
        "reference_proportions": [0.5, 0.5, 0.0],
    }
    original = copy.deepcopy(profile)
    observed = categorical_drift(
        profile,
        ["A"] * 5 + ["B"] * 4 + ["C"],
        minimum_events=10,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    insufficient = categorical_drift(
        profile,
        ["A", "B", "C"],
        minimum_events=5,
        epsilon=1e-6,
        thresholds=THRESHOLDS,
    )
    assert observed.current_counts == [5, 4, 1]
    assert observed.metric_value is not None and math.isfinite(observed.metric_value)
    assert insufficient.severity == "insufficient_data"
    assert insufficient.metric_value is None
    assert insufficient.current_counts == [1, 1, 1]
    assert profile == original
