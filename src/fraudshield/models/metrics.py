"""Validation metrics for FraudShield baseline models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    beta: float,
) -> dict[str, Any]:
    threshold = float(np.clip(threshold, 0.0, 1.0))
    predicted = (y_score >= threshold).astype(np.int8)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=labels).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        beta=1.0,
        average="binary",
        zero_division=0,
    )
    _, _, f_beta, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        beta=beta,
        average="binary",
        zero_division=0,
    )
    negative_count = tn + fp
    positive_predictions = tp + fp
    total_rows = len(y_true)
    fraud_amount_total = float(amount[y_true == 1].sum())
    fraud_amount_captured = float(amount[(y_true == 1) & (predicted == 1)].sum())
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f_beta": float(f_beta),
        "specificity": float(tn / negative_count) if negative_count else 0.0,
        "false_positive_rate": float(fp / negative_count) if negative_count else 0.0,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "predicted_positive_count": int(positive_predictions),
        "alert_rate": float(positive_predictions / total_rows) if total_rows else 0.0,
        "fraud_amount_captured": fraud_amount_captured,
        "fraud_amount_recall": (
            float(fraud_amount_captured / fraud_amount_total) if fraud_amount_total else 0.0
        ),
    }


def _best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    beta: float,
) -> tuple[float, dict[str, Any]]:
    if len(y_true) == 0:
        metrics = _binary_metrics(y_true, y_score, amount, 0.5, beta)
        return 0.5, metrics

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        threshold = 0.5
        return threshold, _binary_metrics(y_true, y_score, amount, threshold, beta)

    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    scores = np.divide(
        (1 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    best_index = int(np.argmax(scores))
    threshold = float(thresholds[best_index])
    return threshold, _binary_metrics(y_true, y_score, amount, threshold, beta)


def _ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    fraud_count = int(y_true.sum())
    non_fraud_count = int(len(y_true) - fraud_count)
    average_precision = (
        float(average_precision_score(y_true, y_score)) if fraud_count else 0.0
    )
    roc_auc = (
        float(roc_auc_score(y_true, y_score)) if fraud_count and non_fraud_count else 0.0
    )
    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }


def top_k_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    percentages: list[float],
) -> list[dict[str, Any]]:
    """Calculate top-k validation review metrics."""

    total_rows = len(y_true)
    fraud_total = int(y_true.sum())
    fraud_amount_total = float(amount[y_true == 1].sum())
    order = np.lexsort((np.arange(total_rows), -y_score)) if total_rows else np.array([], dtype=int)
    results = []
    for percentage in percentages:
        reviewed = int(np.ceil(total_rows * float(percentage) / 100.0)) if total_rows else 0
        reviewed = max(1, reviewed) if total_rows and percentage > 0 else reviewed
        reviewed = min(reviewed, total_rows)
        selected = order[:reviewed]
        frauds_captured = int(y_true[selected].sum()) if reviewed else 0
        fraud_amount_captured = (
            float(amount[selected][y_true[selected] == 1].sum()) if reviewed else 0.0
        )
        results.append(
            {
                "top_k_percentage": float(percentage),
                "reviewed_transactions": int(reviewed),
                "frauds_captured": frauds_captured,
                "recall": float(frauds_captured / fraud_total) if fraud_total else 0.0,
                "precision": float(frauds_captured / reviewed) if reviewed else 0.0,
                "fraud_amount_captured": fraud_amount_captured,
                "fraud_amount_recall": (
                    float(fraud_amount_captured / fraud_amount_total)
                    if fraud_amount_total
                    else 0.0
                ),
            }
        )
    return results


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    top_k_percentages: list[float],
    threshold_beta: float = 2.0,
    include_curve: bool = True,
) -> dict[str, Any]:
    """Evaluate ranking, threshold, and top-k metrics from validation scores."""

    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    amount = np.asarray(amount, dtype=np.float64)
    if not (len(y_true) == len(y_score) == len(amount)):
        raise ValueError("y_true, y_score, and amount must have equal lengths")
    if len(y_true) and not np.isfinite(y_score).all():
        raise ValueError("Scores contain NaN or infinite values")

    best_f1_threshold, best_f1_metrics = _best_threshold(y_true, y_score, amount, beta=1.0)
    best_f2_threshold, best_f2_metrics = _best_threshold(
        y_true,
        y_score,
        amount,
        beta=threshold_beta,
    )
    result = {
        **_ranking_metrics(y_true, y_score),
        "threshold_0_5": _binary_metrics(y_true, y_score, amount, 0.5, threshold_beta),
        "best_f1_threshold": float(best_f1_threshold),
        "best_f1": best_f1_metrics,
        "best_f2_threshold": float(best_f2_threshold),
        "best_f2": best_f2_metrics,
        "selected_threshold": float(best_f2_threshold),
        "selected_threshold_metrics": best_f2_metrics,
        "top_k": top_k_metrics(y_true, y_score, amount, top_k_percentages),
        "score_summary": score_summary(y_score),
    }
    if include_curve:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        result["precision_recall_curve"] = {
            "precision": _as_float_list(precision),
            "recall": _as_float_list(recall),
            "thresholds": _as_float_list(thresholds),
        }
    return result


def score_summary(y_score: np.ndarray) -> dict[str, Any]:
    """Summarize probability range, useful quantiles, and exact saturation counts."""

    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_score) == 0:
        quantiles = {str(value): 0.0 for value in (0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1)}
        return {
            "minimum": 0.0,
            "maximum": 0.0,
            "quantiles": quantiles,
            "exact_zero_count": 0,
            "exact_one_count": 0,
        }

    probabilities = np.clip(y_score, 0.0, 1.0)
    quantile_points = np.array([0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1])
    return {
        "minimum": float(np.min(probabilities)),
        "maximum": float(np.max(probabilities)),
        "quantiles": {
            str(float(point)): float(value)
            for point, value in zip(
                quantile_points,
                np.quantile(probabilities, quantile_points),
                strict=True,
            )
        },
        "exact_zero_count": int(np.sum(probabilities == 0.0)),
        "exact_one_count": int(np.sum(probabilities == 1.0)),
    }
