"""Promote the frozen Phase 1D SGD candidate for production use."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
import sklearn
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from fraudshield.data.config import repository_root
from fraudshield.data.validate import calculate_sha256, utc_timestamp
from fraudshield.features.baseline import (
    BaselineFeatureTransformer,
    expected_raw_input_columns,
    feature_names,
    forbidden_raw_columns,
)
from fraudshield.models.metrics import evaluate_scores
from fraudshield.models.train_baseline import reject_test_path

TARGET_COLUMN = "isFraud"
CLASSES = np.array([0, 1], dtype=np.int8)
CONFIG_RELATIVE = "configs/production.yaml"
TRAINING_CONFIG_RELATIVE = "configs/tuning.yaml"
FEATURE_POLICY_RELATIVE = "configs/feature_policy.yaml"
SPLIT_MANIFEST_RELATIVE = "artifacts/data/split_manifest.json"
PHASE1D_METRICS_RELATIVE = "artifacts/modeling/phase1d_validation_metrics.json"
FROZEN_CONFIGS_RELATIVE = "artifacts/tuning/frozen_candidate_configs.json"
DECISION_RELATIVE = "artifacts/governance/production_model_decision.json"
MANIFEST_RELATIVE = "artifacts/governance/production_sgd_manifest.json"
ARTIFACT_RELATIVE = "artifacts/models/production_sgd.joblib"

EXPECTED_FEATURES = feature_names()
EXPECTED_TRAINING = {
    "alpha": 0.00001,
    "epochs": 3,
    "positive_class_weight": 5.0,
}
REFERENCE_METRICS = {
    "phase1c_unweighted_logistic": {
        "average_precision": 0.375911239819689,
        "roc_auc": 0.9839841632643104,
    },
    "phase1d_tuned_sgd_logistic": {
        "average_precision": 0.5406332160823262,
        "roc_auc": 0.9959320872851961,
        "selected_threshold": 0.9831083416938782,
        "selected_threshold_metrics": {
            "precision": 0.394919168591224,
            "recall": 0.6107142857142858,
            "f1": 0.4796633941093969,
            "f_beta": 0.5505473277527366,
            "alert_rate": 0.0009180643471937021,
            "false_positive_rate": 0.0005558331185314125,
            "fraud_amount_recall": 0.7505388032744525,
            "confusion_matrix": {
                "true_negative": 942205,
                "false_positive": 524,
                "false_negative": 218,
                "true_positive": 342,
            },
        },
    },
    "phase1d_tuned_xgboost": {
        "average_precision": 0.9989050653446082,
        "roc_auc": 0.9999994184815724,
        "selected_threshold": 0.317007452249527,
        "selected_threshold_metrics": {
            "precision": 0.9876325088339223,
            "recall": 0.9982142857142857,
            "f1": 0.9928952042628775,
            "f_beta": 0.9960798289379901,
            "alert_rate": 0.0006000281992051216,
            "false_positive_rate": 7.4252515834349e-06,
            "fraud_amount_recall": 0.9999950465337962,
        },
    },
}


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


def load_production_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Load and strictly validate the immutable production configuration."""

    repo_root = root or repository_root()
    path = config_path or repo_root / CONFIG_RELATIVE
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {
        "model_name",
        "model_family",
        "loss",
        "penalty",
        "alpha",
        "epochs",
        "positive_class_weight",
        "random_seed",
        "batch_size",
        "train_path",
        "validation_path",
        "test_path",
        "operational_threshold",
        "threshold_source",
        "top_k_percentages",
        "final_test_policy",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing production config keys: {', '.join(missing)}")
    if raw["model_name"] != "production_sgd_logistic" or raw["model_family"] != "SGDClassifier":
        raise ValueError("production model identity does not match the frozen SGD model")
    if raw["loss"] != "log_loss" or raw["penalty"] != "l2":
        raise ValueError("production SGD loss and penalty are frozen to log_loss and l2")
    for key, expected in EXPECTED_TRAINING.items():
        if float(raw[key]) != float(expected):
            raise ValueError(f"production config {key} does not match Phase 1D frozen settings")
    if int(raw["random_seed"]) != 42 or int(raw["batch_size"]) != 250000:
        raise ValueError("production seed and batch size are frozen")
    if float(raw["operational_threshold"]) != 0.98310834:
        raise ValueError("production operational threshold is not frozen Phase 1D F2 threshold")
    if raw["threshold_source"] != "phase1d_validation_f2":
        raise ValueError("unsupported threshold source")
    paths = {key: Path(str(raw[key])) for key in ("train_path", "validation_path", "test_path")}
    if any(path.is_absolute() for path in paths.values()):
        raise ValueError("production paths must be relative to the repository root")
    reject_test_path(paths["train_path"])
    reject_test_path(paths["validation_path"])
    if paths["train_path"].name != "train.parquet":
        raise ValueError("train_path must point to train.parquet")
    if paths["validation_path"].name != "validation.parquet":
        raise ValueError("validation_path must point to validation.parquet")
    if paths["test_path"].name != "test.parquet":
        raise ValueError("test_path must point to test.parquet")
    top_k = [float(value) for value in raw["top_k_percentages"]]
    if top_k != [0.1, 0.5, 1.0]:
        raise ValueError("top_k_percentages must be exactly [0.1, 0.5, 1.0]")
    policy = raw["final_test_policy"]
    if any(policy.get(key) is not False for key in policy):
        raise ValueError("all final_test_policy controls must be false")
    return {
        **raw,
        **{key: value.as_posix() for key, value in paths.items()},
        "top_k_percentages": top_k,
    }


def _validate_frozen_config(root: Path) -> dict[str, Any]:
    frozen = _read_json(root / FROZEN_CONFIGS_RELATIVE)
    if frozen is None:
        raise FileNotFoundError("Phase 1D frozen candidate configuration is missing")
    candidate = frozen.get("best_sgd_configuration", {})
    for key, expected in EXPECTED_TRAINING.items():
        if float(candidate.get(key, -1)) != float(expected):
            raise ValueError(f"Phase 1D frozen SGD {key} does not match production config")
    return frozen


def _phase1d_reference(root: Path) -> dict[str, Any]:
    payload = _read_json(root / PHASE1D_METRICS_RELATIVE)
    if payload is None:
        return REFERENCE_METRICS
    candidates = payload.get("candidates", {})
    reference = dict(REFERENCE_METRICS)
    for name in reference:
        if name in candidates:
            candidate = candidates[name]
            reference[name] = {
                key: candidate[key]
                for key in reference[name]
                if key in candidate
            }
            if "selected_threshold_metrics" in candidate:
                reference[name]["selected_threshold_metrics"] = candidate[
                    "selected_threshold_metrics"
                ]
    return reference


def _iter_frames(path: Path, batch_size: int, columns: list[str]):
    reject_test_path(path)
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def _fit_scaler_and_count(
    path: Path,
    batch_size: int,
    transformer: BaselineFeatureTransformer,
) -> tuple[StandardScaler, dict[int, int]]:
    scaler = StandardScaler()
    counts = {0: 0, 1: 0}
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_frames(path, batch_size, columns):
        scaler.partial_fit(transformer.transform(frame))
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        counts[0] += int((labels == 0).sum())
        counts[1] += int((labels == 1).sum())
    if not hasattr(scaler, "mean_") or not all(counts.values()):
        raise ValueError("training split must contain both classes")
    return scaler, counts


def _train_sgd(
    path: Path,
    batch_size: int,
    seed: int,
    scaler: StandardScaler,
    transformer: BaselineFeatureTransformer,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.00001,
        learning_rate="optimal",
        average=True,
        random_state=seed,
    )
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for epoch in range(3):
        for batch_index, frame in enumerate(_iter_frames(path, batch_size, columns)):
            features = scaler.transform(transformer.transform(frame)).astype(np.float32, copy=False)
            labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
            weights = np.where(labels == 1, 5.0, 1.0).astype(np.float32)
            rng = np.random.default_rng(seed + epoch * 1_000_003 + batch_index)
            order = rng.permutation(len(labels))
            model.partial_fit(
                features[order],
                labels[order],
                classes=CLASSES,
                sample_weight=weights[order],
            )
    return model


def _positive_probability(model: SGDClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities[:, int(np.where(model.classes_ == 1)[0][0])]


def _collect_validation(
    path: Path,
    batch_size: int,
    scaler: StandardScaler,
    model: SGDClassifier,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformer = BaselineFeatureTransformer()
    y_parts: list[np.ndarray] = []
    amount_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_frames(path, batch_size, columns):
        features = scaler.transform(transformer.transform(frame)).astype(np.float32, copy=False)
        y_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
        score_parts.append(_positive_probability(model, features))
    return np.concatenate(y_parts), np.concatenate(amount_parts), np.concatenate(score_parts)


def _metric_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amount: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
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
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f_beta": float(f2),
        "specificity": float(tn / negative_total) if negative_total else 0.0,
        "false_positive_rate": float(fp / negative_total) if negative_total else 0.0,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "predicted_positive_count": int(predicted.sum()),
        "alert_rate": float(predicted.mean()) if len(predicted) else 0.0,
        "fraud_amount_captured": captured,
        "fraud_amount_recall": float(captured / fraud_total) if fraud_total else 0.0,
    }


def _compact_metrics(
    y_true: np.ndarray,
    amount: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    top_k: list[float],
) -> dict[str, Any]:
    ranking = evaluate_scores(
        y_true,
        scores,
        amount,
        top_k,
        threshold_beta=2.0,
        include_curve=False,
    )
    return {
        "average_precision": ranking["average_precision"],
        "roc_auc": ranking["roc_auc"],
        "threshold": _metric_at_threshold(y_true, scores, amount, threshold),
        "best_f1_threshold": ranking["best_f1_threshold"],
        "best_f2_threshold": ranking["best_f2_threshold"],
        "top_k": ranking["top_k"],
        "score_summary": ranking["score_summary"],
        "row_count": int(len(y_true)),
        "fraud_count": int(y_true.sum()),
    }


def _metrics_close(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    scalar_keys = ("average_precision", "roc_auc")
    if any(not np.isclose(actual[key], expected[key], rtol=0.0, atol=1e-6) for key in scalar_keys):
        return False
    actual_threshold = actual["threshold"]
    expected_threshold = expected["selected_threshold_metrics"]
    keys = (
        "precision",
        "recall",
        "f1",
        "f_beta",
        "alert_rate",
        "false_positive_rate",
        "fraud_amount_recall",
    )
    for key in keys:
        if not np.isclose(actual_threshold[key], expected_threshold[key], rtol=0.0, atol=1e-5):
            return False
    return actual_threshold["confusion_matrix"] == expected_threshold["confusion_matrix"]


def _decision_payload(
    root: Path,
    config: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "production_model": "phase1d_tuned_sgd",
        "benchmark_model": "phase1d_tuned_xgboost",
        "baseline_model": "phase1c_unweighted_logistic",
        "decision_timestamp_utc": utc_timestamp(),
        "production_sgd_hyperparameters": {
            "loss": config["loss"],
            "penalty": config["penalty"],
            "alpha": float(config["alpha"]),
            "epochs": int(config["epochs"]),
            "positive_class_weight": float(config["positive_class_weight"]),
            "random_seed": int(config["random_seed"]),
            "batch_size": int(config["batch_size"]),
        },
        "frozen_feature_names": EXPECTED_FEATURES,
        "frozen_operational_threshold": float(config["operational_threshold"]),
        "validation_metrics": reference["phase1d_tuned_sgd_logistic"],
        "benchmark_xgboost_validation_metrics": reference["phase1d_tuned_xgboost"],
        "baseline_validation_metrics": reference["phase1c_unweighted_logistic"],
        "model_selection_rationale": (
            "SGD was chosen for production because it provides credible validation performance, "
            "incremental partial_fit training, low inference latency, small model size, "
            "interpretability, simple deployment, easy monitoring, and lower dependence on "
            "PaySim synthetic simulator shortcuts. XGBoost remains a benchmark; its near-perfect "
            "PaySim performance is driven substantially by deterministic synthetic balance rules. "
            "SGD does not have better raw validation metrics than XGBoost."
        ),
        "decision_status": "approved_for_production",
        "official_validation_used_for_tuning": False,
        "test_set_accessed": False,
        "git_commit_hash": _git_commit(root),
    }


def _artifact_reusable(root: Path, config_path: Path) -> bool:
    manifest = _read_json(root / MANIFEST_RELATIVE)
    artifact = root / ARTIFACT_RELATIVE
    if manifest is None or not artifact.exists():
        return False
    return all(
        manifest.get(key) == value
        for key, value in {
            "production_config_sha256": calculate_sha256(config_path),
            "source_split_manifest_sha256": calculate_sha256(root / SPLIT_MANIFEST_RELATIVE),
            "artifact_sha256": calculate_sha256(artifact),
        }.items()
    ) and manifest.get("test_set_accessed") is False


def promote_sgd(root: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Train or reuse the frozen production SGD model without opening test data."""

    repo_root = root or repository_root()
    config_path = repo_root / CONFIG_RELATIVE
    config = load_production_config(config_path, repo_root)
    _validate_frozen_config(repo_root)
    reference = _phase1d_reference(repo_root)
    decision_path = repo_root / DECISION_RELATIVE
    if not decision_path.exists():
        _write_json(decision_path, _decision_payload(repo_root, config, reference))
    if not force and _artifact_reusable(repo_root, config_path):
        return {
            "reused": True,
            "artifact_path": repo_root / ARTIFACT_RELATIVE,
            "manifest": _read_json(repo_root / MANIFEST_RELATIVE) or {},
        }

    start = time.perf_counter()
    train_path = repo_root / config["train_path"]
    validation_path = repo_root / config["validation_path"]
    transformer = BaselineFeatureTransformer()
    scaler, counts = _fit_scaler_and_count(train_path, int(config["batch_size"]), transformer)
    model = _train_sgd(
        train_path,
        int(config["batch_size"]),
        int(config["random_seed"]),
        scaler,
        transformer,
    )
    y_true, amount, scores = _collect_validation(
        validation_path,
        int(config["batch_size"]),
        scaler,
        model,
    )
    metrics = _compact_metrics(
        y_true,
        amount,
        scores,
        float(config["operational_threshold"]),
        list(config["top_k_percentages"]),
    )
    if not _metrics_close(metrics, reference["phase1d_tuned_sgd_logistic"]):
        raise RuntimeError("production SGD validation metrics differ materially from Phase 1D")

    bundle = {
        "model": model,
        "model_family": "sgd_logistic",
        "model_name": config["model_name"],
        "scaler": scaler,
        "ordered_feature_names": EXPECTED_FEATURES,
        "feature_names": EXPECTED_FEATURES,
        "operational_threshold": float(config["operational_threshold"]),
        "selected_threshold": float(config["operational_threshold"]),
        "frozen_hyperparameters": {
            "loss": config["loss"],
            "penalty": config["penalty"],
            "alpha": float(config["alpha"]),
            "epochs": int(config["epochs"]),
            "positive_class_weight": float(config["positive_class_weight"]),
            "random_seed": int(config["random_seed"]),
            "batch_size": int(config["batch_size"]),
        },
        "class_weights": {"0": 1.0, "1": 5.0},
        "class_or_sample_weights": {"sgd_positive_class_weight": 5.0},
        "expected_raw_columns": expected_raw_input_columns(),
        "forbidden_columns": forbidden_raw_columns(),
        "feature_policy_metadata": {
            "policy": "phase1d_pre_transaction_leakage_safe",
            "test_set_accessed": False,
        },
        "training_rows": int(sum(counts.values())),
        "training_fraud_count": int(counts[1]),
        "validation_verification_metrics": metrics,
        "git_commit_hash": _git_commit(repo_root),
        "created_at_utc": utc_timestamp(),
    }
    artifact_path = repo_root / ARTIFACT_RELATIVE
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    artifact_sha = calculate_sha256(artifact_path)
    manifest = {
        "created_at_utc": utc_timestamp(),
        "git_commit_hash": _git_commit(repo_root),
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "configuration_sha256": calculate_sha256(config_path),
        "production_config_sha256": calculate_sha256(config_path),
        "source_split_manifest_sha256": calculate_sha256(repo_root / SPLIT_MANIFEST_RELATIVE),
        "feature_policy_sha256": calculate_sha256(repo_root / FEATURE_POLICY_RELATIVE),
        "training_path": config["train_path"],
        "validation_path": config["validation_path"],
        "training_rows": int(sum(counts.values())),
        "training_fraud_count": int(counts[1]),
        "validation_rows": int(metrics["row_count"]),
        "validation_fraud_count": int(metrics["fraud_count"]),
        "feature_names": EXPECTED_FEATURES,
        "hyperparameters": bundle["frozen_hyperparameters"],
        "class_weights": bundle["class_weights"],
        "threshold": float(config["operational_threshold"]),
        "threshold_source": config["threshold_source"],
        "artifact_path": ARTIFACT_RELATIVE,
        "artifact_size_bytes": int(artifact_path.stat().st_size),
        "artifact_sha256": artifact_sha,
        "validation_reproduction_status": "passed",
        "validation_verification_metrics": metrics,
        "training_duration_seconds": float(time.perf_counter() - start),
        "test_set_accessed": False,
        "reproducibility_status": "reproducible",
    }
    _write_json(repo_root / MANIFEST_RELATIVE, manifest)
    return {"reused": False, "artifact_path": artifact_path, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace Phase 1E production outputs")
    args = parser.parse_args()
    result = promote_sgd(force=args.force)
    metrics = result["manifest"].get("validation_verification_metrics", {})
    threshold = metrics.get("threshold", {})
    print(
        f"Production SGD {'reused' if result['reused'] else 'trained'}: "
        f"{result['artifact_path']}"
    )
    print(
        f"Validation PR-AUC={metrics.get('average_precision', 0.0):.6f} "
        f"ROC-AUC={metrics.get('roc_auc', 0.0):.6f} "
        f"F2={threshold.get('f_beta', 0.0):.6f} "
        f"test_set_accessed={result['manifest'].get('test_set_accessed')}"
    )


if __name__ == "__main__":
    main()
