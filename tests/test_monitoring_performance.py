"""Delayed-outcome monitoring performance tests."""

from types import SimpleNamespace

import pytest

from fraudshield.monitoring.performance import calculate_performance


def test_performance_formulas_use_persisted_predictions_and_amounts() -> None:
    events = [
        SimpleNamespace(prediction=1, actual_fraud=1, amount=40.0),
        SimpleNamespace(prediction=1, actual_fraud=1, amount=60.0),
        SimpleNamespace(prediction=0, actual_fraud=1, amount=50.0),
        SimpleNamespace(prediction=1, actual_fraud=0, amount=20.0),
        SimpleNamespace(prediction=1, actual_fraud=0, amount=10.0),
        SimpleNamespace(prediction=0, actual_fraud=0, amount=30.0),
        SimpleNamespace(prediction=0, actual_fraud=0, amount=40.0),
    ]
    result = calculate_performance(events, minimum_labeled_events=7)

    assert result.available is True
    assert result.labeled_count == 7
    assert result.metrics["precision"] == pytest.approx(0.5)
    assert result.metrics["recall"] == pytest.approx(2 / 3)
    assert result.metrics["f1"] == pytest.approx(4 / 7)
    assert result.metrics["f2"] == pytest.approx(0.625)
    assert result.metrics["false_positive_rate"] == pytest.approx(0.5)
    assert result.metrics["fraud_amount_recall"] == pytest.approx(2 / 3)


def test_insufficient_labels_do_not_fabricate_performance_values() -> None:
    events = [SimpleNamespace(prediction=1, actual_fraud=1, amount=100.0)]
    result = calculate_performance(events, minimum_labeled_events=2)

    assert result.status == "insufficient_labeled_data"
    assert result.available is False
    assert result.labeled_count == 1
    assert result.metrics == {}
