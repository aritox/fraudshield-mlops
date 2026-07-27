"""Transparent Population Stability Index calculations over frozen bins."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fraudshield.monitoring.config import DriftConfig


@dataclass(frozen=True)
class DriftResult:
    metric_value: float | None
    severity: str
    sample_size: int
    current_counts: list[int]
    current_proportions: list[float]


def classify_psi(value: float, thresholds: DriftConfig) -> str:
    """Classify PSI independently from the production model threshold."""

    if value < thresholds.stable_below:
        return "stable"
    if value < thresholds.moderate_below:
        return "moderate"
    return "significant"


def population_stability_index(
    reference_proportions: Iterable[float],
    current_proportions: Iterable[float],
    epsilon: float,
) -> float:
    """Calculate sum((current-reference) * ln(current/reference))."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("PSI epsilon must be finite and positive")
    reference = np.asarray(list(reference_proportions), dtype=np.float64)
    current = np.asarray(list(current_proportions), dtype=np.float64)
    if reference.shape != current.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError("PSI distributions must have equal non-empty shapes")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("PSI distributions must be finite")
    if (reference < 0).any() or (current < 0).any():
        raise ValueError("PSI distributions must be non-negative")
    if reference.sum() <= 0 or current.sum() <= 0:
        raise ValueError("PSI distributions must contain observations")

    reference = reference / reference.sum()
    current = current / current.sum()
    safe_reference = np.maximum(reference, epsilon)
    safe_current = np.maximum(current, epsilon)
    value = float(
        np.sum((safe_current - safe_reference) * np.log(safe_current / safe_reference))
    )
    if not math.isfinite(value):
        raise ValueError("PSI result is not finite")
    return max(0.0, value)


def _numeric_current(profile: dict[str, Any], values: Iterable[float]) -> np.ndarray:
    current = np.asarray(list(values), dtype=np.float64)
    current = current[np.isfinite(current)]
    transformation = str(profile.get("transformation", "identity"))
    if transformation == "log1p":
        if (current < 0).any():
            raise ValueError("log1p drift inputs must be non-negative")
        current = np.log1p(current)
    elif transformation != "identity":
        raise ValueError("Numeric reference transformation is unsupported")
    return current


def numeric_drift(
    profile: dict[str, Any],
    values: Iterable[float],
    *,
    minimum_events: int,
    epsilon: float,
    thresholds: DriftConfig,
) -> DriftResult:
    """Calculate numeric PSI with current values assigned to frozen reference bins."""

    if minimum_events <= 0:
        raise ValueError("Minimum event count must be positive")
    current = _numeric_current(profile, values)
    sample_size = int(current.size)
    reference_counts = list(profile["reference_counts"])
    bin_count = len(reference_counts)
    edges = np.asarray(profile["bin_edges"], dtype=np.float64)
    if bin_count == 0 or edges.size != bin_count + 1 or not np.isfinite(edges).all():
        raise ValueError("Numeric reference bins are invalid")
    assignments = np.searchsorted(edges[1:-1], current, side="right")
    assignments = np.clip(assignments, 0, bin_count - 1)
    counts = np.bincount(assignments, minlength=bin_count).astype(np.int64)
    if sample_size < minimum_events:
        return DriftResult(None, "insufficient_data", sample_size, counts.tolist(), [])
    proportions = counts.astype(np.float64) / sample_size
    value = population_stability_index(profile["reference_proportions"], proportions, epsilon)
    return DriftResult(
        value,
        classify_psi(value, thresholds),
        sample_size,
        counts.tolist(),
        proportions.tolist(),
    )


def categorical_drift(
    profile: dict[str, Any],
    values: Iterable[str | None],
    *,
    minimum_events: int,
    epsilon: float,
    thresholds: DriftConfig,
) -> DriftResult:
    """Calculate categorical PSI with an explicit unknown-category bucket."""

    if minimum_events <= 0:
        raise ValueError("Minimum event count must be positive")
    categories = list(profile["allowed_categories"])
    positions = {value: index for index, value in enumerate(categories)}
    counts = np.zeros(len(categories) + 1, dtype=np.int64)
    sample_size = 0
    for raw_value in values:
        position = positions.get(str(raw_value), len(categories))
        counts[position] += 1
        sample_size += 1
    if len(profile["reference_proportions"]) != len(counts):
        raise ValueError("Categorical reference buckets are invalid")
    if sample_size < minimum_events:
        return DriftResult(None, "insufficient_data", sample_size, counts.tolist(), [])
    proportions = counts.astype(np.float64) / sample_size
    value = population_stability_index(profile["reference_proportions"], proportions, epsilon)
    return DriftResult(
        value,
        classify_psi(value, thresholds),
        sample_size,
        counts.tolist(),
        proportions.tolist(),
    )
