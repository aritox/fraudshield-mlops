"""Run the one-time final holdout evaluation for the promoted SGD model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, roc_curve

from fraudshield.data.config import repository_root
from fraudshield.data.validate import calculate_sha256, utc_timestamp
from fraudshield.features.baseline import BaselineFeatureTransformer, expected_raw_input_columns
from fraudshield.models.metrics import evaluate_scores
from fraudshield.models.promote_sgd import (
    ARTIFACT_RELATIVE,
    CONFIG_RELATIVE,
    DECISION_RELATIVE,
    EXPECTED_FEATURES,
    MANIFEST_RELATIVE,
    _positive_probability,
    load_production_config,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_COLUMN = "isFraud"
EVALUATION_RELATIVE = "artifacts/evaluation/final_test_metrics.json"
FINAL_MANIFEST_RELATIVE = "artifacts/evaluation/final_test_manifest.json"
MARKER_RELATIVE = "artifacts/evaluation/final_test_evaluation_complete.json"
PLOTS_RELATIVE = "artifacts/evaluation/plots"
MODEL_CARD_RELATIVE = "artifacts/model_card/production_sgd_model_card.md"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _validate_gates(
    root: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    decision = _read_json(root / DECISION_RELATIVE)
    manifest = _read_json(root / MANIFEST_RELATIVE)
    artifact = root / ARTIFACT_RELATIVE
    if decision is None or manifest is None or not artifact.exists():
        raise RuntimeError("production decision, promotion manifest, and bundle are required")
    if manifest.get("validation_reproduction_status") != "passed":
        raise RuntimeError("validation reproduction did not pass")
    if manifest.get("test_set_accessed") is not False:
        raise RuntimeError("production manifest does not prove test protection")
    if decision.get("decision_status") != "approved_for_production":
        raise RuntimeError("production model decision is not approved")
    if decision.get("test_set_accessed") is not False:
        raise RuntimeError("production decision does not prove test protection")
    if manifest.get("artifact_sha256") != calculate_sha256(artifact):
        raise RuntimeError("production artifact checksum mismatch")
    config_path = root / CONFIG_RELATIVE
    if manifest.get("production_config_sha256") != calculate_sha256(config_path):
        raise RuntimeError("production configuration checksum mismatch")
    threshold = float(config["operational_threshold"])
    bundle = joblib.load(artifact)
    required = {
        "model",
        "scaler",
        "model_family",
        "ordered_feature_names",
        "operational_threshold",
        "frozen_hyperparameters",
        "expected_raw_columns",
        "forbidden_columns",
        "feature_policy_metadata",
        "git_commit_hash",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise RuntimeError(f"production bundle is missing keys: {', '.join(missing)}")
    if bundle["git_commit_hash"] != manifest.get("git_commit_hash"):
        raise RuntimeError("production bundle and manifest Git commit metadata differ")
    if list(bundle["ordered_feature_names"]) != EXPECTED_FEATURES:
        raise RuntimeError("production bundle feature order is not frozen")
    if float(bundle["operational_threshold"]) != threshold:
        raise RuntimeError("production bundle threshold differs from production config")
    if bundle["feature_policy_metadata"].get("test_set_accessed") is not False:
        raise RuntimeError("production bundle does not prove test protection")
    frozen = bundle["frozen_hyperparameters"]
    for key, expected in {"alpha": 0.00001, "epochs": 3, "positive_class_weight": 5.0}.items():
        if float(frozen[key]) != expected:
            raise RuntimeError(f"production hyperparameter {key} is not frozen")
    return artifact, bundle, manifest


def _iter_test_frames(path: Path, batch_size: int):
    parquet_file = pq.ParquetFile(path)
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def _score_test(
    path: Path,
    batch_size: int,
    bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformer = BaselineFeatureTransformer()
    y_parts: list[np.ndarray] = []
    amount_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    for frame in _iter_test_frames(path, batch_size):
        features = transformer.transform(frame).astype(np.float32, copy=False)
        features = bundle["scaler"].transform(features).astype(np.float32, copy=False)
        y_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
        score_parts.append(_positive_probability(bundle["model"], features))
        step_parts.append(frame["step"].to_numpy(dtype=np.int64))
    if not y_parts:
        raise ValueError("final test split is empty")
    return (
        np.concatenate(y_parts),
        np.concatenate(amount_parts),
        np.concatenate(score_parts),
        np.concatenate(step_parts),
    )


def _score_summary(scores: np.ndarray) -> dict[str, Any]:
    points = {"p90": 0.90, "p95": 0.95, "p99": 0.99, "p99_9": 0.999}
    return {
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        **{name: float(np.quantile(scores, point)) for name, point in points.items()},
        "exact_zero_count": int(np.sum(scores == 0.0)),
        "exact_one_count": int(np.sum(scores == 1.0)),
    }


def _comparison(validation: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "fraud_prevalence",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "alert_rate",
        "false_positive_rate",
        "fraud_amount_recall",
    )
    result = {}
    for key in keys:
        before = float(validation[key])
        after = float(final[key])
        absolute = after - before
        result[key] = {
            "validation": before,
            "final_test": after,
            "absolute_change": absolute,
            "relative_change": absolute / abs(before) if before else None,
            "improved": absolute > 0,
        }
    validation_by_k = {
        item["top_k_percentage"]: item for item in validation.get("top_k", [])
    }
    result["top_k"] = [
        {
            "top_k_percentage": final_item["top_k_percentage"],
            "validation_recall": validation_item["recall"],
            "final_test_recall": final_item["recall"],
            "recall_absolute_change": final_item["recall"] - validation_item["recall"],
            "validation_fraud_amount_recall": validation_item["fraud_amount_recall"],
            "final_test_fraud_amount_recall": final_item["fraud_amount_recall"],
            "fraud_amount_recall_absolute_change": (
                final_item["fraud_amount_recall"] - validation_item["fraud_amount_recall"]
            ),
        }
        for final_item in final.get("top_k", [])
        if (validation_item := validation_by_k.get(final_item["top_k_percentage"])) is not None
    ]
    return result


def _plot_outputs(
    root: Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    metrics: dict[str, Any],
    validation: dict[str, Any],
) -> list[Path]:
    directory = root / PLOTS_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{index:02d}_{name}.png" for index, name in enumerate(
        [
            "final_test_precision_recall_curve",
            "final_test_roc_curve",
            "validation_vs_test_metrics",
            "final_test_confusion_matrix",
            "final_test_top_k_recall",
            "final_test_fraud_amount_recall",
            "validation_vs_test_alert_rate",
            "final_test_score_distribution",
        ],
        start=1,
    )]
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(recall, precision)
    ax.set_title("Final test precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(paths[0], dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--", color="#64748b")
    ax.set_title("Final test ROC curve")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Recall")
    fig.tight_layout()
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)

    labels = ["PR-AUC", "ROC-AUC", "F1", "F2"]
    validation_values = [
        validation["average_precision"],
        validation["roc_auc"],
        validation["f1"],
        validation["f2"],
    ]
    final_values = [
        metrics["average_precision"],
        metrics["roc_auc"],
        metrics["f1"],
        metrics["f2"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - 0.18, validation_values, 0.36, label="Validation")
    ax.bar(x + 0.18, final_values, 0.36, label="Final test")
    ax.set_xticks(x, labels)
    ax.set_title("Validation vs final test metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[2], dpi=150)
    plt.close(fig)

    matrix = metrics["confusion_matrix"]
    values = np.array(
        [
            [matrix["true_negative"], matrix["false_positive"]],
            [matrix["false_negative"], matrix["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title("Final test confusion matrix")
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(paths[3], dpi=150)
    plt.close(fig)

    top_k = metrics["top_k"]
    labels = [f"{item['top_k_percentage']}%" for item in top_k]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, [item["recall"] for item in top_k])
    ax.set_title("Final test top-k fraud recall")
    ax.set_ylabel("Recall")
    fig.tight_layout()
    fig.savefig(paths[4], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, [item["fraud_amount_recall"] for item in top_k], color="#0f766e")
    ax.set_title("Final test top-k fraud amount recall")
    ax.set_ylabel("Fraud amount recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(paths[5], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Validation", "Final test"], [validation["alert_rate"], metrics["alert_rate"]])
    ax.set_title("Validation vs final test alert rate")
    ax.set_ylabel("Alert rate")
    fig.tight_layout()
    fig.savefig(paths[6], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(scores, bins=80, color="#2563eb")
    ax.set_title("Final test score distribution")
    ax.set_xlabel("Fraud score")
    ax.set_ylabel("Transactions")
    fig.tight_layout()
    fig.savefig(paths[7], dpi=150)
    plt.close(fig)
    return paths


def _model_card(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    training_rows = manifest.get("training_rows", "recorded in promotion manifest")
    training_frauds = manifest.get("training_fraud_count", "recorded in promotion manifest")
    validation_rows = manifest.get("validation_rows", "recorded in promotion manifest")
    validation_frauds = manifest.get(
        "validation_fraud_count",
        "recorded in promotion manifest",
    )
    top_k_lines = "\n".join(
        f"- {item['top_k_percentage']}%: reviewed {item['reviewed_transactions']}, "
        f"recall {item['recall']:.6f}, fraud amount recall {item['fraud_amount_recall']:.6f}"
        for item in metrics["top_k"]
    )
    return f"""# Production SGD Model Card

## Model Overview

Production model: `production_sgd_logistic`, an incremental `SGDClassifier` logistic model.
PaySim is synthetic and does not represent real customers or real-world fraud probabilities.

## Intended Use

Rank transactions for fraud review using pre-transaction fields and the frozen
operational threshold.

## Out-of-Scope Use

Do not use this model as a calibrated probability, for identity decisions, demographic decisions,
or outside a monitored fraud-review workflow.

## Dataset And Splits

PaySim is synthetic. The chronological design keeps complete steps together:

- Training: steps 1-323, {training_rows} rows, {training_frauds} frauds.
- Validation: official development holdout, {validation_rows} rows, {validation_frauds} frauds.
- Final test: steps 378-743, evaluated only after the SGD pipeline was frozen.

## Features

{', '.join(EXPECTED_FEATURES)}

Forbidden features: `isFraud`, `isFlaggedFraud`, `nameOrig`, `nameDest`,
`newbalanceOrig`, `newbalanceDest`.

## Hyperparameters And Threshold

- Loss: `log_loss`; penalty: `l2`; alpha: `0.00001`.
- Epochs: `3`; positive sample weight: `5.0`; random seed: `42`.
- Operational threshold: `0.98310834`, selected by validation F2.

## Performance

Validation PR-AUC: `{validation['average_precision']:.6f}`; ROC-AUC: `{validation['roc_auc']:.6f}`.

Final-test PR-AUC: `{metrics['average_precision']:.6f}`; ROC-AUC: `{metrics['roc_auc']:.6f}`.

Final-test precision: `{metrics['precision']:.6f}`; recall: `{metrics['recall']:.6f}`;
F1: `{metrics['f1']:.6f}`; F2: `{metrics['f2']:.6f}`;
fraud amount recall: `{metrics['fraud_amount_recall']:.6f}`.

Top-k final-test metrics:

{top_k_lines}

## Model Choice And Limitations

SGD was selected instead of XGBoost for incremental training, low latency, small size,
interpretability, deployment simplicity, monitoring simplicity, and lower dependence on PaySim
simulator shortcuts. XGBoost remains a benchmark. Its near-perfect PaySim results rely heavily
on deterministic synthetic balance rules. SGD scores are ranking scores and are not guaranteed to
be calibrated real-world fraud probabilities.

## Fairness, Monitoring, And Retraining

PaySim does not provide the demographic information needed for a meaningful fairness assessment.
Fairness and subgroup performance must be assessed before real-world use. Monitor score drift,
alert volume, precision, recall, fraud amount capture, missing inputs, and latency. Retraining
requires a new time-aware development cycle, frozen decision record, validation threshold review,
and a separately governed holdout.

## Reproducibility

The production artifact was trained only on the official training split. The test set was evaluated
only after the model, features, hyperparameters, and threshold were frozen. Artifact SHA-256:
`{manifest['production_model_sha256']}`.
"""


def evaluate_final_test(
    root: Path | None = None,
    acknowledge_final_holdout_rerun: bool = False,
) -> dict[str, Any]:
    """Evaluate the sealed test split exactly once unless rerun is acknowledged."""

    repo_root = root or repository_root()
    config = load_production_config(root=repo_root)
    artifact_path, bundle, promotion_manifest = _validate_gates(repo_root, config)
    test_path = repo_root / config["test_path"]
    artifact_sha = calculate_sha256(artifact_path)
    test_sha = calculate_sha256(test_path)
    marker_path = repo_root / MARKER_RELATIVE
    existing_marker = _read_json(marker_path)
    if existing_marker is not None and not acknowledge_final_holdout_rerun:
        metrics_path = repo_root / EVALUATION_RELATIVE
        if (
            existing_marker.get("production_model_checksum") == artifact_sha
            and existing_marker.get("test_data_checksum") == test_sha
            and existing_marker.get("metrics_checksum") == calculate_sha256(metrics_path)
        ):
            return {
                "reused": True,
                "metrics": _read_json(metrics_path) or {},
                "marker": existing_marker,
            }
        raise RuntimeError("completed final evaluation marker does not match current checksums")

    if existing_marker is not None and acknowledge_final_holdout_rerun:
        print("Acknowledged final holdout rerun: test data will be scored again.")
    y_true, amount, scores, steps = _score_test(
        test_path,
        int(config["batch_size"]),
        bundle,
    )
    ranking = evaluate_scores(
        y_true,
        scores,
        amount,
        list(config["top_k_percentages"]),
        threshold_beta=2.0,
        include_curve=False,
    )
    threshold = float(config["operational_threshold"])
    predicted = scores >= threshold
    tn = int(((y_true == 0) & ~predicted).sum())
    fp = int(((y_true == 0) & predicted).sum())
    fn = int(((y_true == 1) & ~predicted).sum())
    tp = int(((y_true == 1) & predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    fraud_total = float(amount[y_true == 1].sum())
    captured = float(amount[(y_true == 1) & predicted].sum())
    negative_total = tn + fp
    threshold_metrics = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "specificity": tn / negative_total if negative_total else 0.0,
        "false_positive_rate": fp / negative_total if negative_total else 0.0,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "predicted_positive_count": int(predicted.sum()),
        "alert_rate": float(predicted.mean()),
        "fraud_amount_captured": captured,
        "fraud_amount_recall": captured / fraud_total if fraud_total else 0.0,
    }
    validation_metrics = promotion_manifest["validation_verification_metrics"]
    validation_summary = {
        "fraud_prevalence": validation_metrics["fraud_count"] / validation_metrics["row_count"],
        "average_precision": validation_metrics["average_precision"],
        "roc_auc": validation_metrics["roc_auc"],
        "precision": validation_metrics["threshold"]["precision"],
        "recall": validation_metrics["threshold"]["recall"],
        "f1": validation_metrics["threshold"]["f1"],
        "f2": validation_metrics["threshold"]["f_beta"],
        "alert_rate": validation_metrics["threshold"]["alert_rate"],
        "false_positive_rate": validation_metrics["threshold"]["false_positive_rate"],
        "fraud_amount_recall": validation_metrics["threshold"]["fraud_amount_recall"],
        "top_k": validation_metrics["top_k"],
    }
    final_summary = {
        "fraud_prevalence": float(y_true.mean()),
        "average_precision": ranking["average_precision"],
        "roc_auc": ranking["roc_auc"],
        "top_k": ranking["top_k"],
        **threshold_metrics,
    }
    step_minimum = int(steps.min())
    step_maximum = int(steps.max())
    metrics = {
        "evaluation_timestamp_utc": utc_timestamp(),
        "test_set_accessed": True,
        "test_rows": int(len(y_true)),
        "test_fraud_count": int(y_true.sum()),
        "test_step_minimum": step_minimum,
        "test_step_maximum": step_maximum,
        "threshold": threshold,
        "threshold_0_5": ranking["threshold_0_5"],
        "best_f1_threshold": ranking["best_f1_threshold"],
        "best_f1": ranking["best_f1"],
        "best_f2_threshold": ranking["best_f2_threshold"],
        "best_f2": ranking["best_f2"],
        "threshold_metrics": threshold_metrics,
        "average_precision": ranking["average_precision"],
        "roc_auc": ranking["roc_auc"],
        "top_k": ranking["top_k"],
        "score_summary": _score_summary(scores),
        "validation_summary": validation_summary,
        "validation_vs_final_test": _comparison(validation_summary, final_summary),
        "final_evaluation_status": "completed_rerun" if existing_marker else "completed",
    }
    evaluation_path = repo_root / EVALUATION_RELATIVE
    _write_json(evaluation_path, metrics)
    plot_paths = _plot_outputs(repo_root, y_true, scores, final_summary, validation_summary)
    final_manifest = {
        "evaluation_timestamp_utc": metrics["evaluation_timestamp_utc"],
        "git_commit_hash": _git_commit(repo_root),
        "production_model_artifact_path": ARTIFACT_RELATIVE,
        "production_model_sha256": artifact_sha,
        "production_configuration_sha256": calculate_sha256(repo_root / CONFIG_RELATIVE),
        "test_file_relative_path": config["test_path"],
        "test_file_size": int(test_path.stat().st_size),
        "test_file_sha256": test_sha,
        "test_set_accessed": True,
        "test_rows": int(len(y_true)),
        "test_fraud_count": int(y_true.sum()),
        "test_step_minimum": step_minimum,
        "test_step_maximum": step_maximum,
        "frozen_threshold": threshold,
        "threshold_changed_after_test": False,
        "hyperparameters_changed_after_test": False,
        "features_changed_after_test": False,
        "model_changed_after_test": False,
        "test_used_for_model_selection": False,
        "plot_paths": [path.relative_to(repo_root).as_posix() for path in plot_paths],
        "final_evaluation_status": metrics["final_evaluation_status"],
    }
    _write_json(repo_root / FINAL_MANIFEST_RELATIVE, final_manifest)
    metrics_sha = calculate_sha256(evaluation_path)
    marker = {
        "evaluation_completed": True,
        "timestamp": metrics["evaluation_timestamp_utc"],
        "production_model_checksum": artifact_sha,
        "test_data_checksum": test_sha,
        "metrics_checksum": metrics_sha,
        "statement": "This was the final holdout evaluation after production freezing.",
    }
    _write_json(marker_path, marker)
    (repo_root / MODEL_CARD_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    final_manifest["production_model_sha256"] = artifact_sha
    (repo_root / MODEL_CARD_RELATIVE).write_text(
        _model_card(final_summary, validation_summary, {**promotion_manifest, **final_manifest}),
        encoding="utf-8",
    )
    return {"reused": False, "metrics": metrics, "manifest": final_manifest, "marker": marker}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-final-holdout-rerun",
        action="store_true",
        help="explicitly authorize rescoring an already completed final holdout",
    )
    args = parser.parse_args()
    result = evaluate_final_test(
        acknowledge_final_holdout_rerun=args.acknowledge_final_holdout_rerun,
    )
    metrics = result["metrics"]
    threshold = metrics.get("threshold_metrics", {})
    print(
        f"Final test {'reused' if result['reused'] else 'completed'}: "
        f"PR-AUC={metrics.get('average_precision', 0.0):.6f} "
        f"ROC-AUC={metrics.get('roc_auc', 0.0):.6f} "
        f"precision={threshold.get('precision', 0.0):.6f} "
        f"recall={threshold.get('recall', 0.0):.6f} "
        f"F2={threshold.get('f2', 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
