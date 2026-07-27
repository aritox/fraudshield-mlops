"""Delayed-outcome performance calculations over persisted predictions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class LabeledPrediction(Protocol):
    prediction: int
    actual_fraud: int | None
    amount: float


@dataclass(frozen=True)
class PerformanceResult:
    status: str
    labeled_count: int
    metrics: dict[str, float]

    @property
    def available(self) -> bool:
        return self.status == "available"


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    value = float(numerator / denominator)
    if not math.isfinite(value):
        raise ValueError("Performance metric is not finite")
    return value


def calculate_performance(
    events: Iterable[LabeledPrediction],
    *,
    minimum_labeled_events: int,
) -> PerformanceResult:
    """Calculate performance without rescoring or invoking a model."""

    if minimum_labeled_events <= 0:
        raise ValueError("Minimum labeled event count must be positive")
    labeled = [event for event in events if event.actual_fraud is not None]
    if len(labeled) < minimum_labeled_events:
        return PerformanceResult("insufficient_labeled_data", len(labeled), {})

    for event in labeled:
        if event.prediction not in (0, 1) or event.actual_fraud not in (0, 1):
            raise ValueError("Persisted prediction labels must be zero or one")
        if not math.isfinite(event.amount) or event.amount < 0:
            raise ValueError("Persisted transaction amount must be finite and non-negative")

    true_positive = sum(
        event.prediction == 1 and event.actual_fraud == 1 for event in labeled
    )
    false_positive = sum(
        event.prediction == 1 and event.actual_fraud == 0 for event in labeled
    )
    false_negative = sum(
        event.prediction == 0 and event.actual_fraud == 1 for event in labeled
    )
    true_negative = sum(
        event.prediction == 0 and event.actual_fraud == 0 for event in labeled
    )
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    f2 = _ratio(5 * precision * recall, 4 * precision + recall)
    false_positive_rate = _ratio(false_positive, false_positive + true_negative)
    fraud_amount = sum(event.amount for event in labeled if event.actual_fraud == 1)
    captured_fraud_amount = sum(
        event.amount
        for event in labeled
        if event.actual_fraud == 1 and event.prediction == 1
    )
    fraud_amount_recall = _ratio(captured_fraud_amount, fraud_amount)
    return PerformanceResult(
        status="available",
        labeled_count=len(labeled),
        metrics={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "false_positive_rate": false_positive_rate,
            "fraud_amount_recall": fraud_amount_recall,
        },
    )
