"""Train leakage-safe baseline fraud models on the chronological train split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_COLUMN = "isFraud"
TEST_FILENAME = "test.parquet"
MODEL_ARTIFACT_RELATIVE = "artifacts/models/baseline_champion.joblib"
METRICS_RELATIVE = "artifacts/modeling/baseline_metrics.json"
MANIFEST_RELATIVE = "artifacts/modeling/model_manifest.json"
PLOTS_RELATIVE = "artifacts/modeling/plots"
FEATURE_POLICY_RELATIVE = "configs/feature_policy.yaml"
SPLIT_MANIFEST_RELATIVE = "artifacts/data/split_manifest.json"
CLASSES = np.array([0, 1], dtype=np.int8)
PHASE_1C_OUTPUTS = (
    MODEL_ARTIFACT_RELATIVE,
    METRICS_RELATIVE,
    MANIFEST_RELATIVE,
)
PLOT_FILENAMES = (
    "01_validation_precision_recall_curve.png",
    "02_validation_roc_curve.png",
    "03_model_comparison_pr_auc.png",
    "04_champion_confusion_matrix.png",
    "05_threshold_precision_recall.png",
    "06_top_k_fraud_capture.png",
    "07_champion_feature_coefficients.png",
)


@dataclass(frozen=True)
class ModelSpec:
    """SGDClassifier hyperparameters from modeling config."""

    loss: str
    penalty: str
    alpha: float
    learning_rate: str
    average: bool

    def to_kwargs(self, seed: int) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "penalty": self.penalty,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "average": self.average,
            "random_state": seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "penalty": self.penalty,
            "alpha": float(self.alpha),
            "learning_rate": self.learning_rate,
            "average": bool(self.average),
        }


@dataclass(frozen=True)
class ModelingConfig:
    """Configuration for Phase 1C baseline training."""

    batch_size: int
    random_seed: int
    training_epochs: int
    train_path: Path
    validation_path: Path
    selection_metric: str
    threshold_metric: str
    threshold_beta: float
    top_k_percentages: list[float]
    models: dict[str, ModelSpec]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": int(self.batch_size),
            "random_seed": int(self.random_seed),
            "training_epochs": int(self.training_epochs),
            "train_path": self.train_path.as_posix(),
            "validation_path": self.validation_path.as_posix(),
            "selection_metric": self.selection_metric,
            "threshold_metric": self.threshold_metric,
            "threshold_beta": float(self.threshold_beta),
            "top_k_percentages": [float(value) for value in self.top_k_percentages],
            "models": {name: spec.to_dict() for name, spec in self.models.items()},
        }


@dataclass
class TrainingResult:
    """Structured output from baseline training."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    model_artifact_path: Path
    metrics_path: Path
    manifest_path: Path
    plot_paths: list[Path]
    reused: bool
    training_duration_seconds: float


def reject_test_path(path: Path | str) -> None:
    """Reject sealed holdout paths at runtime."""

    if Path(path).name == TEST_FILENAME:
        raise ValueError("The sealed test split must not be accessed during Phase 1C")


def load_modeling_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> ModelingConfig:
    """Load and validate Phase 1C modeling configuration."""

    repo_root = root or repository_root()
    resolved_path = config_path or repo_root / "configs" / "modeling.yaml"
    raw_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if "test_path" in raw_config:
        raise ValueError("modeling config must not contain test_path")

    required = {
        "batch_size",
        "random_seed",
        "training_epochs",
        "train_path",
        "validation_path",
        "selection_metric",
        "threshold_metric",
        "threshold_beta",
        "top_k_percentages",
        "models",
    }
    missing = sorted(required.difference(raw_config))
    if missing:
        raise ValueError(f"Missing required modeling config keys: {', '.join(missing)}")

    batch_size = int(raw_config["batch_size"])
    training_epochs = int(raw_config["training_epochs"])
    threshold_beta = float(raw_config["threshold_beta"])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if training_epochs <= 0:
        raise ValueError("training_epochs must be positive")
    if threshold_beta <= 0:
        raise ValueError("threshold_beta must be positive")
    if raw_config["selection_metric"] != "average_precision":
        raise ValueError("selection_metric must be average_precision")
    if raw_config["threshold_metric"] != "f_beta":
        raise ValueError("threshold_metric must be f_beta")

    train_path = Path(str(raw_config["train_path"]))
    validation_path = Path(str(raw_config["validation_path"]))
    if train_path.is_absolute() or validation_path.is_absolute():
        raise ValueError("train_path and validation_path must be relative to the repository root")
    reject_test_path(train_path)
    reject_test_path(validation_path)
    if train_path.name != "train.parquet":
        raise ValueError("train_path must point to train.parquet")
    if validation_path.name != "validation.parquet":
        raise ValueError("validation_path must point to validation.parquet")

    top_k_percentages = [float(value) for value in raw_config["top_k_percentages"]]
    if not top_k_percentages or any(value <= 0 for value in top_k_percentages):
        raise ValueError("top_k_percentages must contain positive values")

    raw_models = raw_config["models"]
    expected_models = {"unweighted_logistic", "weighted_logistic"}
    if set(raw_models) != expected_models:
        raise ValueError("models must define unweighted_logistic and weighted_logistic")
    models = {}
    for name, raw_spec in raw_models.items():
        spec = ModelSpec(
            loss=str(raw_spec["loss"]),
            penalty=str(raw_spec["penalty"]),
            alpha=float(raw_spec["alpha"]),
            learning_rate=str(raw_spec["learning_rate"]),
            average=bool(raw_spec["average"]),
        )
        if spec.loss != "log_loss":
            raise ValueError(f"{name}.loss must be log_loss")
        if spec.alpha <= 0:
            raise ValueError(f"{name}.alpha must be positive")
        models[name] = spec

    return ModelingConfig(
        batch_size=batch_size,
        random_seed=int(raw_config["random_seed"]),
        training_epochs=training_epochs,
        train_path=train_path,
        validation_path=validation_path,
        selection_metric=str(raw_config["selection_metric"]),
        threshold_metric=str(raw_config["threshold_metric"]),
        threshold_beta=threshold_beta,
        top_k_percentages=top_k_percentages,
        models=models,
    )


def _iter_batches(path: Path, batch_size: int, columns: list[str] | None = None):
    reject_test_path(path)
    parquet_file = pq.ParquetFile(path)
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=columns)


def _labels_from_batch(batch: Any) -> np.ndarray:
    column_index = batch.schema.get_field_index(TARGET_COLUMN)
    if column_index < 0:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return batch.column(column_index).to_numpy(zero_copy_only=False).astype(np.int8, copy=False)


def _frame_from_batch(batch: Any):
    return batch.to_pandas()


def _fit_scaler_and_count_training(
    train_path: Path,
    batch_size: int,
    transformer: BaselineFeatureTransformer,
) -> tuple[StandardScaler, dict[int, int]]:
    scaler = StandardScaler()
    class_counts = {0: 0, 1: 0}
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for batch in _iter_batches(train_path, batch_size, columns=columns):
        frame = _frame_from_batch(batch)
        features = transformer.transform(frame)
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        scaler.partial_fit(features)
        class_counts[0] += int((labels == 0).sum())
        class_counts[1] += int((labels == 1).sum())
    if not hasattr(scaler, "mean_"):
        raise ValueError("Training split is empty")
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError("Training split must contain both classes")
    return scaler, class_counts


def _class_weights(class_counts: dict[int, int]) -> dict[int, float]:
    total = class_counts[0] + class_counts[1]
    return {
        0: float(total / (2 * class_counts[0])),
        1: float(total / (2 * class_counts[1])),
    }


def _new_classifier(model_name: str, config: ModelingConfig) -> SGDClassifier:
    spec = config.models[model_name]
    return SGDClassifier(**spec.to_kwargs(config.random_seed))


def _shuffle_batch(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epoch: int,
    batch_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + epoch * 1_000_003 + batch_index)
    order = rng.permutation(len(labels))
    return features[order], labels[order], order


def _train_models(
    train_path: Path,
    config: ModelingConfig,
    scaler: StandardScaler,
    transformer: BaselineFeatureTransformer,
    class_weights: dict[int, float],
) -> dict[str, SGDClassifier]:
    models = {
        "unweighted_logistic": _new_classifier("unweighted_logistic", config),
        "weighted_logistic": _new_classifier("weighted_logistic", config),
    }
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for epoch in range(config.training_epochs):
        for batch_index, batch in enumerate(_iter_batches(train_path, config.batch_size, columns)):
            frame = _frame_from_batch(batch)
            features = scaler.transform(transformer.transform(frame)).astype(np.float32, copy=False)
            labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
            shuffled_features, shuffled_labels, order = _shuffle_batch(
                features,
                labels,
                config.random_seed,
                epoch,
                batch_index,
            )
            models["unweighted_logistic"].partial_fit(
                shuffled_features,
                shuffled_labels,
                classes=CLASSES,
            )
            weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float32)
            models["weighted_logistic"].partial_fit(
                shuffled_features,
                shuffled_labels,
                classes=CLASSES,
                sample_weight=weights[order],
            )
    return models


def _positive_probability(model: SGDClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    class_index = int(np.where(model.classes_ == 1)[0][0])
    return probabilities[:, class_index]


def _collect_validation_scores(
    validation_path: Path,
    config: ModelingConfig,
    scaler: StandardScaler,
    transformer: BaselineFeatureTransformer,
    models: dict[str, SGDClassifier],
    prior_probability: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    y_true_parts = []
    amount_parts = []
    type_values: list[str] = []
    score_parts: dict[str, list[np.ndarray]] = {
        "prior_probability": [],
        "unweighted_logistic": [],
        "weighted_logistic": [],
    }
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for batch in _iter_batches(validation_path, config.batch_size, columns):
        frame = _frame_from_batch(batch)
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        features = scaler.transform(transformer.transform(frame)).astype(np.float32, copy=False)
        y_true_parts.append(labels)
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
        type_values.extend(frame["type"].astype(str).tolist())
        score_parts["prior_probability"].append(
            np.full(len(labels), prior_probability, dtype=np.float64)
        )
        for model_name, model in models.items():
            score_parts[model_name].append(_positive_probability(model, features))

    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=np.int8)
    amount = np.concatenate(amount_parts) if amount_parts else np.array([], dtype=np.float64)
    scores = {
        model_name: np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        for model_name, parts in score_parts.items()
    }
    return y_true, amount, scores, type_values


def _evaluate_candidates(
    y_true: np.ndarray,
    amount: np.ndarray,
    scores: dict[str, np.ndarray],
    config: ModelingConfig,
) -> dict[str, Any]:
    return {
        model_name: evaluate_scores(
            y_true=y_true,
            y_score=score,
            amount=amount,
            top_k_percentages=config.top_k_percentages,
            threshold_beta=config.threshold_beta,
        )
        for model_name, score in scores.items()
    }


def _select_champion(candidate_metrics: dict[str, Any]) -> tuple[str, str]:
    learned = ["unweighted_logistic", "weighted_logistic"]
    prior_ap = candidate_metrics["prior_probability"]["average_precision"]
    learned_best_ap = max(candidate_metrics[name]["average_precision"] for name in learned)
    eligible = list(candidate_metrics)
    if learned_best_ap > prior_ap:
        eligible.remove("prior_probability")

    def key(model_name: str) -> tuple[float, float, float, int]:
        metrics = candidate_metrics[model_name]
        simplicity = 1 if model_name == "unweighted_logistic" else 0
        return (
            float(metrics["average_precision"]),
            float(metrics["selected_threshold_metrics"]["f_beta"]),
            -float(metrics["selected_threshold_metrics"]["alert_rate"]),
            simplicity,
        )

    champion = max(eligible, key=key)
    rationale = (
        "Selected by validation average precision, then selected-threshold F2, "
        "then lower alert rate, then unweighted logistic simplicity. "
        "The prior baseline was excluded from champion selection because a learned model "
        "outperformed it on average precision."
        if "prior_probability" not in eligible
        else "Selected by the configured tie-breakers; the prior baseline remained eligible "
        "because neither learned model outperformed it on validation average precision."
    )
    return champion, rationale


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


def _plot_precision_recall(candidate_metrics: dict[str, Any], champion: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, metrics in candidate_metrics.items():
        curve = metrics["precision_recall_curve"]
        ax.plot(curve["recall"], curve["precision"], label=model_name)
    ax.set_title("Validation Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.annotate(f"Champion: {champion}", xy=(0.02, 0.04), xycoords="axes fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, score in scores.items():
        if int(y_true.sum()) == 0 or int(len(y_true) - y_true.sum()) == 0:
            ax.plot([0, 1], [0, 1], label=f"{model_name} unavailable")
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title("Validation ROC Curve")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_model_comparison(candidate_metrics: dict[str, Any], path: Path) -> None:
    labels = list(candidate_metrics)
    values = [candidate_metrics[name]["average_precision"] for name in labels]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, values, color=["#64748b", "#2563eb", "#0f766e"])
    ax.set_title("Validation Model Comparison by PR-AUC")
    ax.set_ylabel("Average precision")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(metrics: dict[str, Any], champion: str, path: Path) -> None:
    matrix = metrics["selected_threshold_metrics"]["confusion_matrix"]
    values = np.array(
        [
            [matrix["true_negative"], matrix["false_positive"]],
            [matrix["false_negative"], matrix["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(f"Validation Confusion Matrix: {champion}")
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_threshold_precision_recall(metrics: dict[str, Any], champion: str, path: Path) -> None:
    curve = metrics["precision_recall_curve"]
    thresholds = curve["thresholds"]
    fig, ax = plt.subplots(figsize=(9, 6))
    if thresholds:
        ax.plot(thresholds, curve["precision"][:-1], label="Precision")
        ax.plot(thresholds, curve["recall"][:-1], label="Recall")
    ax.axvline(metrics["selected_threshold"], color="#dc2626", linestyle="--", label="Selected")
    ax.set_title(f"Validation Threshold Precision and Recall: {champion}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_top_k(metrics: dict[str, Any], champion: str, path: Path) -> None:
    top_k = metrics["top_k"]
    labels = [f"{item['top_k_percentage']}%" for item in top_k]
    values = [item["fraud_amount_recall"] for item in top_k]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, values, color="#0f766e")
    ax.set_title(f"Validation Top-K Fraud Amount Capture: {champion}")
    ax.set_xlabel("Reviewed highest-risk validation transactions")
    ax.set_ylabel("Fraud amount recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coefficients(model: SGDClassifier, champion: str, path: Path) -> None:
    coefficients = model.coef_[0]
    names = np.asarray(feature_names())
    order = np.argsort(np.abs(coefficients))[-min(14, len(coefficients)) :]
    ordered = order[np.argsort(coefficients[order])]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in coefficients[ordered]]
    ax.barh(names[ordered], coefficients[ordered], color=colors)
    ax.set_title(f"Strongest Standardized Coefficients: {champion}")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _create_plots(
    root: Path,
    candidate_metrics: dict[str, Any],
    champion: str,
    champion_model: SGDClassifier | None,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
) -> list[Path]:
    plots_dir = root / PLOTS_RELATIVE
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = [plots_dir / filename for filename in PLOT_FILENAMES]
    _plot_precision_recall(candidate_metrics, champion, paths[0])
    _plot_roc(y_true, scores, paths[1])
    _plot_model_comparison(candidate_metrics, paths[2])
    _plot_confusion(candidate_metrics[champion], champion, paths[3])
    _plot_threshold_precision_recall(candidate_metrics[champion], champion, paths[4])
    _plot_top_k(candidate_metrics[champion], champion, paths[5])
    if champion_model is None:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title("Validation Feature Coefficients Unavailable: prior_probability")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(paths[6], dpi=150)
        plt.close(fig)
    else:
        _plot_coefficients(champion_model, champion, paths[6])
    return paths


def _remove_phase_1c_outputs(root: Path) -> None:
    for relative in PHASE_1C_OUTPUTS:
        path = root / relative
        if path.exists():
            path.unlink()
    plots_dir = root / PLOTS_RELATIVE
    if plots_dir.exists():
        for filename in PLOT_FILENAMES:
            path = plots_dir / filename
            if path.exists():
                path.unlink()


def _artifact_inputs_match(root: Path, manifest: dict[str, Any] | None, config_path: Path) -> bool:
    if manifest is None:
        return False
    artifact_path = root / MODEL_ARTIFACT_RELATIVE
    if not artifact_path.exists():
        return False
    expected = {
        "split_manifest_sha256": calculate_sha256(root / SPLIT_MANIFEST_RELATIVE),
        "modeling_config_sha256": calculate_sha256(config_path),
        "feature_policy_sha256": calculate_sha256(root / FEATURE_POLICY_RELATIVE),
        "model_artifact_sha256": calculate_sha256(artifact_path),
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def _build_bundle(
    model: SGDClassifier | None,
    scaler: StandardScaler,
    champion: str,
    selected_threshold: float,
    class_weights: dict[int, float] | None,
    config: ModelingConfig,
) -> dict[str, Any]:
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names(),
        "selected_threshold": float(selected_threshold),
        "model_name": champion,
        "class_weights": class_weights,
        "feature_policy_metadata": {
            "policy": "pre_transaction_baseline",
            "test_set_accessed": False,
        },
        "training_config": config.to_dict(),
        "expected_raw_input_columns": expected_raw_input_columns(),
        "forbidden_columns": forbidden_raw_columns(),
        "created_at_utc": utc_timestamp(),
    }


def _required_bundle_keys() -> set[str]:
    return {
        "model",
        "scaler",
        "feature_names",
        "selected_threshold",
        "model_name",
        "class_weights",
        "feature_policy_metadata",
        "training_config",
        "expected_raw_input_columns",
        "forbidden_columns",
        "created_at_utc",
    }


def load_champion_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Load a saved champion bundle and validate required keys."""

    path = Path(bundle_path)
    reject_test_path(path)
    bundle = joblib.load(path)
    missing = sorted(_required_bundle_keys().difference(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {', '.join(missing)}")
    return bundle


def predict_with_bundle(bundle: dict[str, Any], frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return fraud probabilities and selected-threshold predictions for a batch."""

    transformer = BaselineFeatureTransformer()
    features = transformer.transform(frame)
    scaled = bundle["scaler"].transform(features).astype(np.float32, copy=False)
    model = bundle["model"]
    if model is None:
        raise ValueError("Prior-probability bundles do not support fitted-model prediction")
    probabilities = _positive_probability(model, scaled)
    predictions = (probabilities >= float(bundle["selected_threshold"])).astype(np.int8)
    return probabilities, predictions


def predict_parquet_with_bundle(
    bundle: dict[str, Any],
    parquet_path: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a Parquet source unless it is the sealed holdout filename."""

    path = Path(parquet_path)
    reject_test_path(path)
    probabilities = []
    predictions = []
    for batch in _iter_batches(path, int(bundle["training_config"]["batch_size"])):
        batch_probabilities, batch_predictions = predict_with_bundle(
            bundle,
            _frame_from_batch(batch),
        )
        probabilities.append(batch_probabilities)
        predictions.append(batch_predictions)
    return (
        np.concatenate(probabilities) if probabilities else np.array([], dtype=np.float64),
        np.concatenate(predictions) if predictions else np.array([], dtype=np.int8),
    )


def _build_metrics_payload(
    candidate_metrics: dict[str, Any],
    champion: str,
    rationale: str,
    y_true: np.ndarray,
) -> dict[str, Any]:
    return {
        "candidates": candidate_metrics,
        "champion_model": champion,
        "selection_metric": "average_precision",
        "selection_rationale": rationale,
        "validation_row_count": int(len(y_true)),
        "validation_fraud_count": int(y_true.sum()),
        "test_data_was_not_accessed": True,
    }


def _build_manifest(
    root: Path,
    config_path: Path,
    config: ModelingConfig,
    metrics: dict[str, Any],
    model_artifact_path: Path,
    train_counts: dict[int, int],
    y_true: np.ndarray,
    champion: str,
    selected_threshold: float,
    class_weights: dict[int, float] | None,
) -> dict[str, Any]:
    artifact_relative = model_artifact_path.relative_to(root).as_posix()
    return {
        "created_at_utc": utc_timestamp(),
        "git_commit_hash": _git_commit(root),
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "source_train_path": config.train_path.as_posix(),
        "source_validation_path": config.validation_path.as_posix(),
        "split_manifest_sha256": calculate_sha256(root / SPLIT_MANIFEST_RELATIVE),
        "modeling_config_sha256": calculate_sha256(config_path),
        "feature_policy_sha256": calculate_sha256(root / FEATURE_POLICY_RELATIVE),
        "training_rows": int(train_counts[0] + train_counts[1]),
        "training_fraud_count": int(train_counts[1]),
        "validation_rows": int(len(y_true)),
        "validation_fraud_count": int(y_true.sum()),
        "model_name": champion,
        "training_config": config.to_dict(),
        "model_hyperparameters": (
            config.models[champion].to_dict() if champion in config.models else {"type": champion}
        ),
        "class_weights": {str(label): float(weight) for label, weight in class_weights.items()}
        if class_weights is not None
        else None,
        "champion_class_weights": (
            {str(label): float(weight) for label, weight in class_weights.items()}
            if champion == "weighted_logistic" and class_weights is not None
            else None
        ),
        "feature_names": feature_names(),
        "selected_threshold": float(selected_threshold),
        "model_artifact_relative_path": artifact_relative,
        "model_artifact_file_size": int(model_artifact_path.stat().st_size),
        "model_artifact_sha256": calculate_sha256(model_artifact_path),
        "test_set_accessed": False,
        "reproducibility_status": "reproducible",
        "metrics_summary": {
            "champion_average_precision": metrics["candidates"][champion]["average_precision"],
            "champion_roc_auc": metrics["candidates"][champion]["roc_auc"],
        },
    }


def train_baseline(
    root: Path | None = None,
    config_path: Path | None = None,
    force: bool = False,
) -> TrainingResult:
    """Train or reuse the Phase 1C baseline champion artifacts."""

    start = time.perf_counter()
    repo_root = root or repository_root()
    resolved_config_path = config_path or repo_root / "configs" / "modeling.yaml"
    config = load_modeling_config(resolved_config_path, repo_root)
    model_artifact_path = repo_root / MODEL_ARTIFACT_RELATIVE
    metrics_path = repo_root / METRICS_RELATIVE
    manifest_path = repo_root / MANIFEST_RELATIVE

    if not force and _artifact_inputs_match(
        repo_root,
        _read_json(manifest_path),
        resolved_config_path,
    ):
        manifest = _read_json(manifest_path)
        metrics = _read_json(metrics_path)
        if manifest is not None and metrics is not None:
            plot_paths = [repo_root / PLOTS_RELATIVE / filename for filename in PLOT_FILENAMES]
            return TrainingResult(
                metrics=metrics,
                manifest=manifest,
                model_artifact_path=model_artifact_path,
                metrics_path=metrics_path,
                manifest_path=manifest_path,
                plot_paths=plot_paths,
                reused=True,
                training_duration_seconds=float(time.perf_counter() - start),
            )

    if force:
        _remove_phase_1c_outputs(repo_root)

    train_path = repo_root / config.train_path
    validation_path = repo_root / config.validation_path
    reject_test_path(train_path)
    reject_test_path(validation_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation split not found: {validation_path}")

    transformer = BaselineFeatureTransformer()
    scaler, train_counts = _fit_scaler_and_count_training(
        train_path,
        config.batch_size,
        transformer,
    )
    class_weights = _class_weights(train_counts)
    prior_probability = float(train_counts[1] / (train_counts[0] + train_counts[1]))
    learned_models = _train_models(
        train_path,
        config,
        scaler,
        transformer,
        class_weights,
    )
    y_true, amount, scores, _type_values = _collect_validation_scores(
        validation_path,
        config,
        scaler,
        transformer,
        learned_models,
        prior_probability,
    )
    candidate_metrics = _evaluate_candidates(y_true, amount, scores, config)
    champion, rationale = _select_champion(candidate_metrics)
    champion_model = learned_models.get(champion)
    champion_class_weights = class_weights if champion == "weighted_logistic" else None
    selected_threshold = float(candidate_metrics[champion]["selected_threshold"])

    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(
        champion_model,
        scaler,
        champion,
        selected_threshold,
        champion_class_weights,
        config,
    )
    joblib.dump(bundle, model_artifact_path)

    plot_paths = _create_plots(
        repo_root,
        candidate_metrics,
        champion,
        champion_model,
        y_true,
        scores,
    )
    metrics = _build_metrics_payload(candidate_metrics, champion, rationale, y_true)
    manifest = _build_manifest(
        root=repo_root,
        config_path=resolved_config_path,
        config=config,
        metrics=metrics,
        model_artifact_path=model_artifact_path,
        train_counts=train_counts,
        y_true=y_true,
        champion=champion,
        selected_threshold=selected_threshold,
        class_weights=class_weights,
    )
    metrics["generated_plots"] = [path.relative_to(repo_root).as_posix() for path in plot_paths]
    metrics["model_artifact_path"] = model_artifact_path.relative_to(repo_root).as_posix()
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    return TrainingResult(
        metrics=metrics,
        manifest=manifest,
        model_artifact_path=model_artifact_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        plot_paths=plot_paths,
        reused=False,
        training_duration_seconds=float(time.perf_counter() - start),
    )


def _print_summary(result: TrainingResult) -> None:
    metrics = result.metrics
    manifest = result.manifest
    champion = metrics["champion_model"]
    champion_metrics = metrics["candidates"][champion]
    selected = champion_metrics["selected_threshold_metrics"]
    print("Phase 1C baseline modeling")
    print(f"Mode: {'reused' if result.reused else 'trained'}")
    print(f"Training rows: {manifest['training_rows']}")
    print(f"Training fraud count: {manifest['training_fraud_count']}")
    print(f"Class weights: {manifest['class_weights']}")
    print(f"Feature count: {len(manifest['feature_names'])}")
    print(f"Training epochs: {manifest['training_config']['training_epochs']}")
    print(f"Validation rows: {manifest['validation_rows']}")
    print(f"Validation fraud count: {manifest['validation_fraud_count']}")
    print("Candidate validation metrics:")
    for model_name, model_metrics in metrics["candidates"].items():
        print(
            f"  {model_name}: PR-AUC={model_metrics['average_precision']:.6f}, "
            f"ROC-AUC={model_metrics['roc_auc']:.6f}"
        )
    print(f"Selected model: {champion}")
    print(f"Selected threshold: {champion_metrics['selected_threshold']:.8f}")
    print(
        "Selected-threshold metrics: "
        f"precision={selected['precision']:.6f}, recall={selected['recall']:.6f}, "
        f"F1={selected['f1']:.6f}, F2={selected['f_beta']:.6f}"
    )
    print(f"Alert rate: {selected['alert_rate']:.6f}")
    print(f"Fraud amount recall: {selected['fraud_amount_recall']:.6f}")
    print(f"Model artifact: {result.model_artifact_path}")
    print(f"Training duration seconds: {result.training_duration_seconds:.2f}")
    print("Test data access: false; sealed test.parquet was not accessed")
    print("Final status: passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1C baseline fraud models.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only Phase 1C generated model artifacts, metrics, manifest, and plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = train_baseline(force=args.force)
        _print_summary(result)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
