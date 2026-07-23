from __future__ import annotations

import numpy as np
import pytest

from fraudshield.models.metrics import evaluate_scores, top_k_metrics


def test_ranking_threshold_confusion_and_fraud_amount_metrics() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    y_score = np.array([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    amount = np.array([10.0, 20.0, 100.0, 300.0], dtype=np.float64)

    metrics = evaluate_scores(y_true, y_score, amount, [50.0], threshold_beta=2.0)
    threshold = metrics["threshold_0_5"]

    assert metrics["average_precision"] == pytest.approx(0.8333333333)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert threshold["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert threshold["precision"] == pytest.approx(0.5)
    assert threshold["recall"] == pytest.approx(0.5)
    assert threshold["f1"] == pytest.approx(0.5)
    assert threshold["f_beta"] == pytest.approx(0.5)
    assert threshold["fraud_amount_recall"] == pytest.approx(0.25)
    assert metrics["top_k"][0]["reviewed_transactions"] == 2
    assert metrics["top_k"][0]["frauds_captured"] == 1


def test_top_k_calculations_are_deterministic_for_identical_scores() -> None:
    y_true = np.array([1, 0, 1, 0], dtype=np.int8)
    y_score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    amount = np.array([100.0, 10.0, 200.0, 20.0], dtype=np.float64)

    first = top_k_metrics(y_true, y_score, amount, [50.0])
    second = top_k_metrics(y_true, y_score, amount, [50.0])

    assert first == second
    assert first[0]["reviewed_transactions"] == 2
    assert first[0]["frauds_captured"] == 1


def test_no_predicted_positives_and_no_frauds_are_safe() -> None:
    y_true = np.array([0, 0, 0], dtype=np.int8)
    y_score = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    amount = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    with pytest.warns(UserWarning, match="No positive class found in y_true"):
        metrics = evaluate_scores(y_true, y_score, amount, [10.0], threshold_beta=2.0)

    assert metrics["average_precision"] == 0.0
    assert metrics["roc_auc"] == 0.0
    assert metrics["threshold_0_5"]["predicted_positive_count"] == 0
    assert metrics["threshold_0_5"]["precision"] == 0.0
    assert metrics["threshold_0_5"]["recall"] == 0.0
    assert metrics["threshold_0_5"]["fraud_amount_recall"] == 0.0
    assert metrics["top_k"][0]["fraud_amount_recall"] == 0.0
