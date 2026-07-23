"""Phase 1D time-aware tuning and stronger model comparison."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
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
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import xgboost
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraudshield.data.config import repository_root
from fraudshield.data.validate import calculate_sha256, utc_timestamp
from fraudshield.features.baseline import (
    BaselineFeatureTransformer,
    expected_raw_input_columns,
    feature_names,
    forbidden_raw_columns,
)
from fraudshield.models.metrics import evaluate_scores
from fraudshield.models.temporal_tuning import InnerSplit, StepWindow, create_inner_split
from fraudshield.models.train_baseline import (
    load_champion_bundle,
    predict_with_bundle,
    reject_test_path,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_COLUMN = "isFraud"
TEST_FILENAME = "test.parquet"
CLASSES = np.array([0, 1], dtype=np.int8)
CONFIG_RELATIVE = "configs/tuning.yaml"
FEATURE_POLICY_RELATIVE = "configs/feature_policy.yaml"
SPLIT_MANIFEST_RELATIVE = "artifacts/data/split_manifest.json"
BASELINE_MODEL_RELATIVE = "artifacts/models/baseline_champion.joblib"
MODEL_ARTIFACT_RELATIVE = "artifacts/models/phase1d_champion.joblib"
TUNING_RESULTS_RELATIVE = "artifacts/tuning/tuning_results.json"
TUNING_MANIFEST_RELATIVE = "artifacts/tuning/tuning_manifest.json"
FROZEN_CONFIGS_RELATIVE = "artifacts/tuning/frozen_candidate_configs.json"
VALIDATION_METRICS_RELATIVE = "artifacts/modeling/phase1d_validation_metrics.json"
MODEL_MANIFEST_RELATIVE = "artifacts/modeling/phase1d_model_manifest.json"
TUNING_PLOTS_RELATIVE = "artifacts/tuning/plots"
MODELING_PLOTS_RELATIVE = "artifacts/modeling/phase1d_plots"
PHASE_1D_JSON_OUTPUTS = (
    TUNING_RESULTS_RELATIVE,
    TUNING_MANIFEST_RELATIVE,
    FROZEN_CONFIGS_RELATIVE,
    VALIDATION_METRICS_RELATIVE,
    MODEL_MANIFEST_RELATIVE,
    "artifacts/tuning/inner_split_manifest.json",
)
TUNING_PLOT_FILENAMES = (
    "01_sgd_trial_pr_auc.png",
    "02_sgd_weight_vs_pr_auc.png",
    "03_xgboost_trial_pr_auc.png",
    "04_inner_tuning_model_comparison.png",
)
MODELING_PLOT_FILENAMES = (
    "01_validation_precision_recall_comparison.png",
    "02_validation_roc_comparison.png",
    "03_validation_pr_auc_comparison.png",
    "04_selected_threshold_metrics.png",
    "05_top_k_recall_comparison.png",
    "06_top_k_fraud_amount_recall.png",
    "07_xgboost_feature_importance.png",
    "08_champion_confusion_matrix.png",
    "09_alert_rate_vs_recall.png",
)


@dataclass(frozen=True)
class InnerTuningConfig:
    method: str
    fit_fraction: float
    tuning_fraction: float


@dataclass(frozen=True)
class SgdSearchConfig:
    positive_class_weights: list[float]
    alpha_values: list[float]
    epoch_values: list[int]
    maximum_trials: int


@dataclass(frozen=True)
class XgboostSearchConfig:
    enabled: bool
    maximum_trials: int
    final_nonfraud_sample_limit: int
    inner_nonfraud_sample_limit: int
    tree_method: str
    objective: str
    eval_metric: str
    n_jobs: int
    early_stopping_rounds: int
    parameter_space: dict[str, list[Any]]


@dataclass(frozen=True)
class TuningConfig:
    random_seed: int
    batch_size: int
    train_path: Path
    validation_path: Path
    selection_metric: str
    threshold_metric: str
    threshold_beta: float
    minimum_pr_auc_improvement: float
    inner_tuning: InnerTuningConfig
    sgd_search: SgdSearchConfig
    xgboost_search: XgboostSearchConfig
    top_k_percentages: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_seed": int(self.random_seed),
            "batch_size": int(self.batch_size),
            "train_path": self.train_path.as_posix(),
            "validation_path": self.validation_path.as_posix(),
            "selection_metric": self.selection_metric,
            "threshold_metric": self.threshold_metric,
            "threshold_beta": float(self.threshold_beta),
            "minimum_pr_auc_improvement": float(self.minimum_pr_auc_improvement),
            "inner_tuning": {
                "method": self.inner_tuning.method,
                "fit_fraction": float(self.inner_tuning.fit_fraction),
                "tuning_fraction": float(self.inner_tuning.tuning_fraction),
            },
            "sgd_search": {
                "positive_class_weights": [
                    float(value) for value in self.sgd_search.positive_class_weights
                ],
                "alpha_values": [float(value) for value in self.sgd_search.alpha_values],
                "epoch_values": [int(value) for value in self.sgd_search.epoch_values],
                "maximum_trials": int(self.sgd_search.maximum_trials),
            },
            "xgboost_search": {
                "enabled": bool(self.xgboost_search.enabled),
                "maximum_trials": int(self.xgboost_search.maximum_trials),
                "final_nonfraud_sample_limit": int(
                    self.xgboost_search.final_nonfraud_sample_limit
                ),
                "inner_nonfraud_sample_limit": int(
                    self.xgboost_search.inner_nonfraud_sample_limit
                ),
                "tree_method": self.xgboost_search.tree_method,
                "objective": self.xgboost_search.objective,
                "eval_metric": self.xgboost_search.eval_metric,
                "n_jobs": int(self.xgboost_search.n_jobs),
                "early_stopping_rounds": int(self.xgboost_search.early_stopping_rounds),
                "parameter_space": self.xgboost_search.parameter_space,
            },
            "top_k_percentages": [float(value) for value in self.top_k_percentages],
        }


@dataclass
class TuningRunResult:
    tuning_results: dict[str, Any]
    validation_metrics: dict[str, Any]
    tuning_manifest: dict[str, Any]
    model_manifest: dict[str, Any]
    model_artifact_path: Path
    plot_paths: list[Path]
    reused: bool
    duration_seconds: float


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_relative_parquet(path: Path, expected_name: str, label: str) -> None:
    if path.is_absolute():
        raise ValueError(f"{label} must be relative to the repository root")
    reject_test_path(path)
    if path.name != expected_name:
        raise ValueError(f"{label} must point to {expected_name}")


def load_tuning_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> TuningConfig:
    """Load and validate Phase 1D tuning configuration."""

    repo_root = root or repository_root()
    resolved_path = config_path or repo_root / CONFIG_RELATIVE
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if "test_path" in raw:
        raise ValueError("tuning config must not contain test_path")
    required = {
        "random_seed",
        "batch_size",
        "train_path",
        "validation_path",
        "selection_metric",
        "threshold_metric",
        "threshold_beta",
        "inner_tuning",
        "sgd_search",
        "xgboost_search",
        "top_k_percentages",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing required tuning config keys: {', '.join(missing)}")

    train_path = Path(str(raw["train_path"]))
    validation_path = Path(str(raw["validation_path"]))
    _validate_relative_parquet(train_path, "train.parquet", "train_path")
    _validate_relative_parquet(validation_path, "validation.parquet", "validation_path")

    batch_size = int(raw["batch_size"])
    _require_positive("batch_size", batch_size)
    threshold_beta = float(raw["threshold_beta"])
    _require_positive("threshold_beta", threshold_beta)
    if raw["selection_metric"] != "average_precision":
        raise ValueError("selection_metric must be average_precision")
    if raw["threshold_metric"] != "f_beta":
        raise ValueError("threshold_metric must be f_beta")

    raw_inner = raw["inner_tuning"]
    fit_fraction = float(raw_inner["fit_fraction"])
    tuning_fraction = float(raw_inner["tuning_fraction"])
    if raw_inner["method"] != "chronological_whole_step":
        raise ValueError("inner_tuning.method must be chronological_whole_step")
    if fit_fraction <= 0 or tuning_fraction <= 0:
        raise ValueError("inner tuning fractions must be positive")
    if abs((fit_fraction + tuning_fraction) - 1.0) > 1e-9:
        raise ValueError("inner tuning fractions must sum to 1.0")

    raw_sgd = raw["sgd_search"]
    positive_weights = [float(value) for value in raw_sgd["positive_class_weights"]]
    alpha_values = [float(value) for value in raw_sgd["alpha_values"]]
    epoch_values = [int(value) for value in raw_sgd["epoch_values"]]
    if 1.0 not in positive_weights:
        raise ValueError("sgd_search must include positive weight 1.0")
    if len(set(positive_weights).intersection({2.0, 5.0, 10.0, 25.0, 50.0})) < 1:
        raise ValueError("sgd_search must include moderate positive weights")
    if any(value <= 0 for value in positive_weights + alpha_values):
        raise ValueError("SGD weights and alpha values must be positive")
    if any(value <= 0 for value in epoch_values):
        raise ValueError("SGD epoch values must be positive")
    if any(value >= 100 for value in positive_weights):
        raise ValueError("SGD positive weights must stay moderate and below 100")
    maximum_sgd_trials = int(raw_sgd["maximum_trials"])
    _require_positive("sgd_search.maximum_trials", maximum_sgd_trials)

    raw_xgb = raw["xgboost_search"]
    parameter_space = {
        str(key): list(value) for key, value in dict(raw_xgb["parameter_space"]).items()
    }
    if any(not values for values in parameter_space.values()):
        raise ValueError("xgboost parameter_space values must be non-empty")
    xgb_config = XgboostSearchConfig(
        enabled=bool(raw_xgb["enabled"]),
        maximum_trials=int(raw_xgb["maximum_trials"]),
        final_nonfraud_sample_limit=int(raw_xgb["final_nonfraud_sample_limit"]),
        inner_nonfraud_sample_limit=int(raw_xgb["inner_nonfraud_sample_limit"]),
        tree_method=str(raw_xgb["tree_method"]),
        objective=str(raw_xgb["objective"]),
        eval_metric=str(raw_xgb["eval_metric"]),
        n_jobs=int(raw_xgb["n_jobs"]),
        early_stopping_rounds=int(raw_xgb["early_stopping_rounds"]),
        parameter_space=parameter_space,
    )
    _require_positive("xgboost_search.maximum_trials", xgb_config.maximum_trials)
    _require_positive(
        "xgboost_search.final_nonfraud_sample_limit",
        xgb_config.final_nonfraud_sample_limit,
    )
    _require_positive(
        "xgboost_search.inner_nonfraud_sample_limit",
        xgb_config.inner_nonfraud_sample_limit,
    )
    _require_positive("xgboost_search.n_jobs", xgb_config.n_jobs)
    _require_positive("xgboost_search.early_stopping_rounds", xgb_config.early_stopping_rounds)
    if xgb_config.objective != "binary:logistic":
        raise ValueError("xgboost objective must be binary:logistic")
    if xgb_config.eval_metric != "aucpr":
        raise ValueError("xgboost eval_metric must be aucpr")

    top_k_percentages = [float(value) for value in raw["top_k_percentages"]]
    if not top_k_percentages or any(value <= 0 for value in top_k_percentages):
        raise ValueError("top_k_percentages must contain positive values")

    return TuningConfig(
        random_seed=int(raw["random_seed"]),
        batch_size=batch_size,
        train_path=train_path,
        validation_path=validation_path,
        selection_metric=str(raw["selection_metric"]),
        threshold_metric=str(raw["threshold_metric"]),
        threshold_beta=threshold_beta,
        minimum_pr_auc_improvement=float(raw.get("minimum_pr_auc_improvement", 0.001)),
        inner_tuning=InnerTuningConfig(
            method=str(raw_inner["method"]),
            fit_fraction=fit_fraction,
            tuning_fraction=tuning_fraction,
        ),
        sgd_search=SgdSearchConfig(
            positive_class_weights=positive_weights,
            alpha_values=alpha_values,
            epoch_values=epoch_values,
            maximum_trials=maximum_sgd_trials,
        ),
        xgboost_search=xgb_config,
        top_k_percentages=top_k_percentages,
    )


def _iter_batches(
    path: Path,
    batch_size: int,
    columns: list[str],
    window: StepWindow | None = None,
):
    reject_test_path(path)
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if window is not None:
            frame = frame.loc[
                (frame["step"] >= window.minimum) & (frame["step"] <= window.maximum)
            ]
        if not frame.empty:
            yield frame


def _features_and_labels(
    frame: Any,
    transformer: BaselineFeatureTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    features = transformer.transform(frame)
    labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
    return features, labels


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    class_index = int(np.where(model.classes_ == 1)[0][0])
    return probabilities[:, class_index]


def _named_feature_frame(features: np.ndarray) -> pd.DataFrame:
    """Return a named feature matrix for estimators that preserve feature names."""

    return pd.DataFrame(features, columns=feature_names())


def _shuffle(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    epoch: int,
    batch_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed + epoch * 1_000_003 + batch_index)
    order = rng.permutation(len(labels))
    ordered_weight = sample_weight[order] if sample_weight is not None else None
    return features[order], labels[order], ordered_weight


def _fit_scaler_and_count(
    path: Path,
    config: TuningConfig,
    transformer: BaselineFeatureTransformer,
    window: StepWindow | None,
) -> tuple[StandardScaler, dict[int, int]]:
    scaler = StandardScaler()
    counts = {0: 0, 1: 0}
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_batches(path, config.batch_size, columns, window):
        features, labels = _features_and_labels(frame, transformer)
        scaler.partial_fit(features)
        counts[0] += int((labels == 0).sum())
        counts[1] += int((labels == 1).sum())
    if not hasattr(scaler, "mean_"):
        raise ValueError("Fitting period is empty")
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("Fitting period must contain both classes")
    return scaler, counts


def deterministic_sgd_trials(config: SgdSearchConfig) -> list[dict[str, Any]]:
    """Generate deterministic controlled SGD trials capped by maximum_trials."""

    alphas = sorted(set(config.alpha_values))
    epochs = sorted(set(config.epoch_values))
    weights = sorted(set(config.positive_class_weights))
    anchor_alpha = 0.0001 if 0.0001 in alphas else alphas[len(alphas) // 2]
    anchor_epoch = 3 if 3 in epochs else epochs[len(epochs) // 2]
    staged: list[tuple[float, int, float]] = []
    for weight in weights:
        staged.append((anchor_alpha, anchor_epoch, weight))
    for alpha in alphas:
        staged.append((alpha, anchor_epoch, 1.0))
        if len(weights) > 1:
            staged.append((alpha, anchor_epoch, weights[min(2, len(weights) - 1)]))
    for epoch in epochs:
        staged.append((anchor_alpha, epoch, 1.0))
        if len(weights) > 1:
            staged.append((anchor_alpha, epoch, weights[-2]))

    full = list(itertools.product(alphas, epochs, weights))
    staged.extend(full)
    seen: set[tuple[float, int, float]] = set()
    trials = []
    for alpha, epoch, weight in staged:
        key = (float(alpha), int(epoch), float(weight))
        if key in seen:
            continue
        seen.add(key)
        trials.append(
            {
                "trial_id": f"sgd_{len(trials) + 1:02d}",
                "alpha": float(alpha),
                "epochs": int(epoch),
                "positive_class_weight": float(weight),
            }
        )
        if len(trials) >= config.maximum_trials:
            break
    return trials


def deterministic_xgboost_trials(
    config: XgboostSearchConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate deterministic XGBoost parameter candidates capped by maximum_trials."""

    keys = sorted(config.parameter_space)
    combinations = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(config.parameter_space[key] for key in keys))
    ]
    anchor = {
        key: config.parameter_space[key][len(config.parameter_space[key]) // 2] for key in keys
    }
    ordered = [anchor] if anchor in combinations else []
    remaining = [combo for combo in combinations if combo != anchor]
    remaining.sort(
        key=lambda combo: hashlib.sha256(
            (json.dumps(combo, sort_keys=True) + f"|{seed}").encode("utf-8")
        ).hexdigest()
    )
    ordered.extend(remaining)
    trials = []
    for params in ordered[: config.maximum_trials]:
        trials.append({"trial_id": f"xgb_{len(trials) + 1:02d}", "parameters": params})
    return trials


def _train_sgd(
    path: Path,
    config: TuningConfig,
    trial: dict[str, Any],
    scaler: StandardScaler,
    transformer: BaselineFeatureTransformer,
    window: StepWindow | None,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(trial["alpha"]),
        learning_rate="optimal",
        average=True,
        random_state=config.random_seed,
    )
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    positive_weight = float(trial["positive_class_weight"])
    for epoch in range(int(trial["epochs"])):
        for batch_index, frame in enumerate(
            _iter_batches(path, config.batch_size, columns, window)
        ):
            features, labels = _features_and_labels(frame, transformer)
            scaled = scaler.transform(features).astype(np.float32, copy=False)
            weights = np.where(labels == 1, positive_weight, 1.0).astype(np.float32)
            scaled, labels, weights = _shuffle(
                scaled,
                labels,
                weights,
                config.random_seed,
                epoch,
                batch_index,
            )
            model.partial_fit(scaled, labels, classes=CLASSES, sample_weight=weights)
    return model


def _collect_scores(
    path: Path,
    config: TuningConfig,
    transformer: BaselineFeatureTransformer,
    score_function: Any,
    window: StepWindow | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_parts = []
    amount_parts = []
    score_parts = []
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_batches(path, config.batch_size, columns, window):
        y_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
        score_parts.append(score_function(frame, transformer))
    return (
        np.concatenate(y_parts) if y_parts else np.array([], dtype=np.int8),
        np.concatenate(amount_parts) if amount_parts else np.array([], dtype=np.float64),
        np.concatenate(score_parts) if score_parts else np.array([], dtype=np.float64),
    )


def _evaluate_trial_scores(
    y_true: np.ndarray,
    amount: np.ndarray,
    scores: np.ndarray,
    config: TuningConfig,
) -> dict[str, Any]:
    return evaluate_scores(
        y_true=y_true,
        y_score=scores,
        amount=amount,
        top_k_percentages=config.top_k_percentages,
        threshold_beta=config.threshold_beta,
        include_curve=False,
    )


def run_sgd_search(
    root: Path,
    config: TuningConfig,
    inner_split: InnerSplit,
) -> tuple[dict[str, Any], SGDClassifier, StandardScaler]:
    """Tune SGD logistic models only on the internal train/tuning periods."""

    train_path = root / config.train_path
    transformer = BaselineFeatureTransformer()
    trials = deterministic_sgd_trials(config.sgd_search)
    trial_results = []
    best_model: SGDClassifier | None = None
    best_scaler: StandardScaler | None = None
    best_record: dict[str, Any] | None = None
    for trial in trials:
        start = time.perf_counter()
        scaler, counts = _fit_scaler_and_count(
            train_path,
            config,
            transformer,
            inner_split.fit_window,
        )
        model = _train_sgd(train_path, config, trial, scaler, transformer, inner_split.fit_window)

        def score_function(
            frame: Any,
            local_transformer: BaselineFeatureTransformer,
            local_scaler: StandardScaler = scaler,
            local_model: SGDClassifier = model,
        ) -> np.ndarray:
            features = local_scaler.transform(local_transformer.transform(frame)).astype(
                np.float32,
                copy=False,
            )
            return _positive_probability(local_model, features)

        y_true, amount, scores = _collect_scores(
            train_path,
            config,
            transformer,
            score_function,
            inner_split.tuning_window,
        )
        metrics = _evaluate_trial_scores(y_true, amount, scores, config)
        record = {
            **trial,
            "training_duration_seconds": float(time.perf_counter() - start),
            "training_counts": {"0": counts[0], "1": counts[1]},
            "inner_tuning_metrics": metrics,
            "probability_minimum": metrics["score_summary"]["minimum"],
            "probability_maximum": metrics["score_summary"]["maximum"],
            "probability_exact_zero_count": metrics["score_summary"]["exact_zero_count"],
            "probability_exact_one_count": metrics["score_summary"]["exact_one_count"],
        }
        trial_results.append(record)
        if best_record is None or _sgd_key(record) > _sgd_key(best_record):
            best_record = record
            best_model = model
            best_scaler = scaler
    assert best_record is not None and best_model is not None and best_scaler is not None
    return {
        "trials": trial_results,
        "best_trial": best_record,
        "selection_metric": "inner_tuning_average_precision",
        "tie_breakers": [
            "higher F2",
            "lower alert rate",
            "lower positive-class weight",
            "fewer epochs",
        ],
    }, best_model, best_scaler


def _sgd_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = record["inner_tuning_metrics"]
    selected = metrics["selected_threshold_metrics"]
    return (
        float(metrics["average_precision"]),
        float(selected["f_beta"]),
        -float(selected["alert_rate"]),
        -float(record["positive_class_weight"]),
        -float(record["epochs"]),
    )


def _stable_row_hash(seed: int, ordinal: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{ordinal}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def sample_xgboost_fitting_data(
    path: Path,
    config: TuningConfig,
    window: StepWindow | None,
    nonfraud_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample non-fraud deterministically while retaining all fraud rows."""

    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    transformer = BaselineFeatureTransformer()
    total_rows = 0
    total_fraud = 0
    total_nonfraud = 0
    selected_heap: list[tuple[int, int]] = []
    for frame in _iter_batches(path, config.batch_size, columns, window):
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        for _local_index, label in enumerate(labels):
            ordinal = total_rows
            total_rows += 1
            if int(label) == 1:
                total_fraud += 1
                continue
            total_nonfraud += 1
            row_hash = _stable_row_hash(config.random_seed, ordinal)
            heap_entry = (-row_hash, ordinal)
            if len(selected_heap) < nonfraud_limit:
                heapq.heappush(selected_heap, heap_entry)
            elif heap_entry > selected_heap[0]:
                heapq.heapreplace(selected_heap, heap_entry)

    selected_ordinals = {ordinal for _negative_hash, ordinal in selected_heap}
    selected_nonfraud = len(selected_ordinals)
    features_parts = []
    label_parts = []
    weights_parts = []
    ordinal = 0
    nonfraud_weight = float(total_nonfraud / selected_nonfraud) if selected_nonfraud else 0.0
    for frame in _iter_batches(path, config.batch_size, columns, window):
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        keep_mask = np.zeros(len(labels), dtype=bool)
        weights = np.zeros(len(labels), dtype=np.float32)
        for local_index, label in enumerate(labels):
            keep = int(label) == 1 or ordinal in selected_ordinals
            if keep:
                keep_mask[local_index] = True
                weights[local_index] = 1.0 if int(label) == 1 else nonfraud_weight
            ordinal += 1
        if not bool(keep_mask.any()):
            continue
        kept_frame = frame.loc[keep_mask]
        features_parts.append(transformer.transform(kept_frame))
        label_parts.append(kept_frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        weights_parts.append(weights[keep_mask])

    features = (
        np.vstack(features_parts).astype(np.float32, copy=False)
        if features_parts
        else np.empty((0, len(feature_names())), dtype=np.float32)
    )
    labels = np.concatenate(label_parts) if label_parts else np.array([], dtype=np.int8)
    weights = np.concatenate(weights_parts) if weights_parts else np.array([], dtype=np.float32)
    sampled_fraud = int((labels == 1).sum())
    sampled_nonfraud = int((labels == 0).sum())
    metadata = {
        "original_fitting_rows": int(total_rows),
        "original_fraud_rows": int(total_fraud),
        "original_nonfraud_rows": int(total_nonfraud),
        "sampled_fraud_rows": int(sampled_fraud),
        "sampled_nonfraud_rows": int(sampled_nonfraud),
        "fraud_sampling_fraction": 1.0 if total_fraud else 0.0,
        "nonfraud_sampling_fraction": (
            float(sampled_nonfraud / total_nonfraud) if total_nonfraud else 0.0
        ),
        "sample_weights": {"fraud": 1.0, "nonfraud": float(nonfraud_weight)},
        "deterministic_seed": int(config.random_seed),
        "raw_identifiers_used_as_model_features": False,
    }
    if sampled_fraud != total_fraud:
        raise RuntimeError("XGBoost sampling failed to retain all fraud rows")
    if sampled_nonfraud > nonfraud_limit:
        raise RuntimeError("XGBoost sampling exceeded the configured non-fraud limit")
    return features, labels, weights, metadata


def _fit_xgboost_model(
    params: dict[str, Any],
    config: TuningConfig,
    n_estimators: int,
    early_stopping_rounds: int | None,
) -> XGBClassifier:
    return XGBClassifier(
        **params,
        n_estimators=int(n_estimators),
        objective=config.xgboost_search.objective,
        tree_method=config.xgboost_search.tree_method,
        eval_metric=config.xgboost_search.eval_metric,
        n_jobs=config.xgboost_search.n_jobs,
        random_state=config.random_seed,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0,
    )


def _collect_feature_matrix(
    path: Path,
    config: TuningConfig,
    window: StepWindow | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformer = BaselineFeatureTransformer()
    feature_parts = []
    label_parts = []
    amount_parts = []
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_batches(path, config.batch_size, columns, window):
        feature_parts.append(transformer.transform(frame))
        label_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
    return (
        np.vstack(feature_parts).astype(np.float32, copy=False)
        if feature_parts
        else np.empty((0, len(feature_names())), dtype=np.float32),
        np.concatenate(label_parts) if label_parts else np.array([], dtype=np.int8),
        np.concatenate(amount_parts) if amount_parts else np.array([], dtype=np.float64),
    )


def run_xgboost_search(
    root: Path,
    config: TuningConfig,
    inner_split: InnerSplit,
) -> tuple[dict[str, Any], XGBClassifier | None, dict[str, Any] | None]:
    """Tune XGBoost using sampled inner-fit rows and complete inner tuning evaluation."""

    if not config.xgboost_search.enabled:
        return {"enabled": False, "trials": [], "best_trial": None}, None, None
    train_path = root / config.train_path
    trials = deterministic_xgboost_trials(config.xgboost_search, config.random_seed)
    x_fit, y_fit, w_fit, sample_metadata = sample_xgboost_fitting_data(
        train_path,
        config,
        inner_split.fit_window,
        config.xgboost_search.inner_nonfraud_sample_limit,
    )
    x_eval, y_eval, amount_eval = _collect_feature_matrix(
        train_path,
        config,
        inner_split.tuning_window,
    )
    trial_results = []
    best_model: XGBClassifier | None = None
    best_record: dict[str, Any] | None = None
    for trial in trials:
        params = dict(trial["parameters"])
        configured_estimators = int(params.pop("n_estimators"))
        start = time.perf_counter()
        model = _fit_xgboost_model(
            params,
            config,
            configured_estimators,
            config.xgboost_search.early_stopping_rounds,
        )
        model.fit(
            x_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(x_eval, y_eval)],
            verbose=False,
        )
        model.get_booster().feature_names = feature_names()
        scores = model.predict_proba(_named_feature_frame(x_eval))[:, 1]
        metrics = _evaluate_trial_scores(y_eval, amount_eval, scores, config)
        best_iteration = getattr(model, "best_iteration", None)
        boosting_rounds = (
            int(best_iteration + 1) if best_iteration is not None else configured_estimators
        )
        importance = model.get_booster().get_score(importance_type="gain")
        named_importance = _named_feature_importance(importance)
        record = {
            **trial,
            "parameters": {**trial["parameters"], "n_estimators": configured_estimators},
            "boosting_rounds_used": boosting_rounds,
            "early_stopping_best_iteration": (
                int(best_iteration) if best_iteration is not None else None
            ),
            "training_duration_seconds": float(time.perf_counter() - start),
            "inner_tuning_metrics": metrics,
            "probability_minimum": metrics["score_summary"]["minimum"],
            "probability_maximum": metrics["score_summary"]["maximum"],
            "feature_importance_by_gain": named_importance,
            "sampling_metadata": sample_metadata,
        }
        trial_results.append(record)
        if best_record is None or _xgb_key(record) > _xgb_key(best_record):
            best_record = record
            best_model = model
    assert best_record is not None
    return {
        "enabled": True,
        "trials": trial_results,
        "best_trial": best_record,
        "selection_metric": "inner_tuning_average_precision",
        "tie_breakers": [
            "higher F2",
            "lower alert rate",
            "fewer trees",
            "shallower depth",
        ],
    }, best_model, sample_metadata


def _xgb_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = record["inner_tuning_metrics"]
    selected = metrics["selected_threshold_metrics"]
    return (
        float(metrics["average_precision"]),
        float(selected["f_beta"]),
        -float(selected["alert_rate"]),
        -float(record["boosting_rounds_used"]),
        -float(record["parameters"]["max_depth"]),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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


def _remove_phase1d_outputs(root: Path) -> None:
    for relative in PHASE_1D_JSON_OUTPUTS + (MODEL_ARTIFACT_RELATIVE,):
        path = root / relative
        if path.exists():
            path.unlink()
    for relative, filenames in (
        (TUNING_PLOTS_RELATIVE, TUNING_PLOT_FILENAMES),
        (MODELING_PLOTS_RELATIVE, MODELING_PLOT_FILENAMES),
    ):
        directory = root / relative
        if directory.exists():
            for filename in filenames:
                path = directory / filename
                if path.exists():
                    path.unlink()


def _phase1d_outputs_valid(root: Path, config_path: Path) -> bool:
    manifest = _read_json(root / TUNING_MANIFEST_RELATIVE)
    model_manifest = _read_json(root / MODEL_MANIFEST_RELATIVE)
    if manifest is None or model_manifest is None:
        return False
    artifact = root / MODEL_ARTIFACT_RELATIVE
    if not artifact.exists():
        return False
    expected = {
        "tuning_config_sha256": calculate_sha256(config_path),
        "source_split_manifest_sha256": calculate_sha256(root / SPLIT_MANIFEST_RELATIVE),
        "feature_policy_sha256": calculate_sha256(root / FEATURE_POLICY_RELATIVE),
    }
    if not all(manifest.get(key) == value for key, value in expected.items()):
        return False
    if model_manifest.get("model_artifact_sha256") != calculate_sha256(artifact):
        return False
    return all((root / relative).exists() for relative in PHASE_1D_JSON_OUTPUTS)


def _freeze_configs(
    root: Path,
    sgd_results: dict[str, Any],
    xgb_results: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "created_at_utc": utc_timestamp(),
        "best_sgd_configuration": {
            key: sgd_results["best_trial"][key]
            for key in ("trial_id", "alpha", "epochs", "positive_class_weight")
        },
        "best_sgd_inner_tuning_metrics": sgd_results["best_trial"]["inner_tuning_metrics"],
        "best_xgboost_configuration": (
            {
                "trial_id": xgb_results["best_trial"]["trial_id"],
                "parameters": xgb_results["best_trial"]["parameters"],
                "boosting_rounds_used": xgb_results["best_trial"]["boosting_rounds_used"],
            }
            if xgb_results.get("best_trial") is not None
            else None
        ),
        "best_xgboost_inner_tuning_metrics": (
            xgb_results["best_trial"]["inner_tuning_metrics"]
            if xgb_results.get("best_trial") is not None
            else None
        ),
        "selection_rationale": {
            "sgd": sgd_results["tie_breakers"],
            "xgboost": xgb_results.get("tie_breakers", []),
        },
        "official_validation_accessed_during_search": False,
        "test_set_accessed": False,
    }
    _write_json(root / FROZEN_CONFIGS_RELATIVE, payload)
    return payload


def _retrain_sgd(
    root: Path,
    config: TuningConfig,
    frozen: dict[str, Any],
) -> tuple[SGDClassifier, StandardScaler, dict[int, int], float]:
    start = time.perf_counter()
    transformer = BaselineFeatureTransformer()
    train_path = root / config.train_path
    scaler, counts = _fit_scaler_and_count(train_path, config, transformer, None)
    model = _train_sgd(train_path, config, frozen, scaler, transformer, None)
    return model, scaler, counts, float(time.perf_counter() - start)


def _retrain_xgboost(
    root: Path,
    config: TuningConfig,
    frozen: dict[str, Any] | None,
) -> tuple[XGBClassifier | None, dict[str, Any] | None, float]:
    if frozen is None:
        return None, None, 0.0
    start = time.perf_counter()
    params = dict(frozen["parameters"])
    params.pop("n_estimators", None)
    model = _fit_xgboost_model(
        params,
        config,
        int(frozen["boosting_rounds_used"]),
        early_stopping_rounds=None,
    )
    x_fit, y_fit, w_fit, sampling = sample_xgboost_fitting_data(
        root / config.train_path,
        config,
        None,
        config.xgboost_search.final_nonfraud_sample_limit,
    )
    model.fit(x_fit, y_fit, sample_weight=w_fit, verbose=False)
    model.get_booster().feature_names = feature_names()
    return model, sampling, float(time.perf_counter() - start)


def _train_phase1c_baseline_fallback(
    root: Path,
    config: TuningConfig,
) -> tuple[SGDClassifier, StandardScaler]:
    trial = {"alpha": 0.0001, "epochs": 3, "positive_class_weight": 1.0}
    transformer = BaselineFeatureTransformer()
    train_path = root / config.train_path
    scaler, _counts = _fit_scaler_and_count(train_path, config, transformer, None)
    model = _train_sgd(train_path, config, trial, scaler, transformer, None)
    return model, scaler


def _validation_scores_for_candidates(
    root: Path,
    config: TuningConfig,
    sgd_model: SGDClassifier,
    sgd_scaler: StandardScaler,
    xgb_model: XGBClassifier | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], SGDClassifier, StandardScaler]:
    validation_path = root / config.validation_path
    transformer = BaselineFeatureTransformer()
    scores: dict[str, list[np.ndarray]] = {
        "phase1c_unweighted_logistic": [],
        "phase1d_tuned_sgd_logistic": [],
    }
    if xgb_model is not None:
        scores["phase1d_tuned_xgboost"] = []
    y_parts = []
    amount_parts = []
    baseline_bundle = None
    if (root / BASELINE_MODEL_RELATIVE).exists():
        baseline_bundle = load_champion_bundle(root / BASELINE_MODEL_RELATIVE)
    fallback_model: SGDClassifier | None = None
    fallback_scaler: StandardScaler | None = None
    if baseline_bundle is None:
        fallback_model, fallback_scaler = _train_phase1c_baseline_fallback(root, config)
    else:
        fallback_model = baseline_bundle["model"]
        fallback_scaler = baseline_bundle["scaler"]

    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_batches(validation_path, config.batch_size, columns):
        y_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
        if baseline_bundle is not None:
            baseline_scores, _predictions = predict_with_bundle(baseline_bundle, frame)
        else:
            assert fallback_model is not None and fallback_scaler is not None
            features = fallback_scaler.transform(transformer.transform(frame)).astype(
                np.float32,
                copy=False,
            )
            baseline_scores = _positive_probability(fallback_model, features)
        scores["phase1c_unweighted_logistic"].append(baseline_scores)

        sgd_features = sgd_scaler.transform(transformer.transform(frame)).astype(
            np.float32,
            copy=False,
        )
        scores["phase1d_tuned_sgd_logistic"].append(_positive_probability(sgd_model, sgd_features))
        if xgb_model is not None:
            xgb_features = transformer.transform(frame).astype(np.float32, copy=False)
            scores["phase1d_tuned_xgboost"].append(
                xgb_model.predict_proba(_named_feature_frame(xgb_features))[:, 1]
            )

    return (
        np.concatenate(y_parts) if y_parts else np.array([], dtype=np.int8),
        np.concatenate(amount_parts) if amount_parts else np.array([], dtype=np.float64),
        {name: np.concatenate(parts) for name, parts in scores.items()},
        fallback_model,
        fallback_scaler,
    )


def _evaluate_validation_candidates(
    y_true: np.ndarray,
    amount: np.ndarray,
    scores: dict[str, np.ndarray],
    config: TuningConfig,
) -> dict[str, Any]:
    return {
        name: evaluate_scores(
            y_true,
            score,
            amount,
            config.top_k_percentages,
            config.threshold_beta,
            include_curve=True,
        )
        for name, score in scores.items()
    }


def _select_phase1d_champion(metrics: dict[str, Any], tolerance: float) -> tuple[str, str]:
    baseline = "phase1c_unweighted_logistic"
    learned = [name for name in metrics if name != baseline]

    def key(name: str) -> tuple[float, float, float, float, int]:
        selected = metrics[name]["selected_threshold_metrics"]
        top05 = next(
            item for item in metrics[name]["top_k"] if item["top_k_percentage"] == 0.5
        )
        simplicity = 1 if name == "phase1d_tuned_sgd_logistic" else 0
        return (
            float(metrics[name]["average_precision"]),
            float(selected["f_beta"]),
            float(top05["fraud_amount_recall"]),
            -float(selected["alert_rate"]),
            simplicity,
        )

    best_learned = max(learned, key=key)
    improvement = (
        float(metrics[best_learned]["average_precision"])
        - float(metrics[baseline]["average_precision"])
    )
    if improvement >= tolerance:
        return (
            best_learned,
            "Selected by official validation PR-AUC with configured tie-breakers; "
            f"improvement over Phase 1C baseline was {improvement:.6f}.",
        )
    return (
        baseline,
        "Retained Phase 1C baseline because the best Phase 1D candidate did not improve "
        f"validation PR-AUC by at least {tolerance:.6f}; improvement was {improvement:.6f}.",
    )


def _build_bundle(
    root: Path,
    champion: str,
    baseline_model: SGDClassifier,
    baseline_scaler: StandardScaler,
    sgd_model: SGDClassifier,
    sgd_scaler: StandardScaler,
    xgb_model: XGBClassifier | None,
    selected_threshold: float,
    frozen: dict[str, Any],
    full_xgb_sampling: dict[str, Any] | None,
    config: TuningConfig,
) -> dict[str, Any]:
    if champion == "phase1d_tuned_xgboost":
        model = xgb_model
        scaler = None
        family = "xgboost"
        hyperparameters = frozen["best_xgboost_configuration"]
        sampling = full_xgb_sampling
    elif champion == "phase1d_tuned_sgd_logistic":
        model = sgd_model
        scaler = sgd_scaler
        family = "sgd_logistic"
        hyperparameters = frozen["best_sgd_configuration"]
        sampling = None
    else:
        model = baseline_model
        scaler = baseline_scaler
        family = "phase1c_unweighted_logistic"
        hyperparameters = {
            "source": BASELINE_MODEL_RELATIVE,
            "fallback_used": not (root / BASELINE_MODEL_RELATIVE).exists(),
        }
        sampling = None
    return {
        "model": model,
        "model_family": family,
        "scaler": scaler,
        "ordered_feature_names": feature_names(),
        "feature_names": feature_names(),
        "operational_threshold": float(selected_threshold),
        "selected_threshold": float(selected_threshold),
        "frozen_hyperparameters": hyperparameters,
        "sampling_metadata": sampling,
        "class_or_sample_weights": {
            "sgd_positive_class_weight": frozen["best_sgd_configuration"]["positive_class_weight"],
            "xgboost_sample_weights": full_xgb_sampling["sample_weights"]
            if full_xgb_sampling is not None
            else None,
        },
        "expected_raw_columns": expected_raw_input_columns(),
        "expected_raw_input_columns": expected_raw_input_columns(),
        "forbidden_columns": forbidden_raw_columns(),
        "created_at_utc": utc_timestamp(),
        "feature_policy_metadata": {
            "policy": "phase1d_pre_transaction_leakage_safe",
            "test_set_accessed": False,
        },
        "training_config": config.to_dict(),
    }


def load_phase1d_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Load a Phase 1D champion bundle."""

    path = Path(bundle_path)
    reject_test_path(path)
    bundle = joblib.load(path)
    required = {
        "model",
        "model_family",
        "ordered_feature_names",
        "operational_threshold",
        "frozen_hyperparameters",
        "expected_raw_columns",
        "forbidden_columns",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"Phase 1D bundle is missing required keys: {', '.join(missing)}")
    return bundle


def predict_with_phase1d_bundle(
    bundle: dict[str, Any],
    frame: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate probabilities and selected-threshold predictions from a Phase 1D bundle."""

    transformer = BaselineFeatureTransformer()
    features = transformer.transform(frame).astype(np.float32, copy=False)
    if bundle["scaler"] is not None:
        features = bundle["scaler"].transform(features).astype(np.float32, copy=False)
    model = bundle["model"]
    if bundle["model_family"] == "xgboost":
        probabilities = _positive_probability(model, _named_feature_frame(features))
    else:
        probabilities = _positive_probability(model, features)
    predictions = (probabilities >= float(bundle["operational_threshold"])).astype(np.int8)
    return probabilities, predictions


def _plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, values, color=["#2563eb", "#0f766e", "#f59e0b", "#64748b"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _named_feature_importance(raw_importance: dict[str, float]) -> dict[str, float]:
    names = feature_names()
    mapped = {}
    for key, value in raw_importance.items():
        if key.startswith("f") and key[1:].isdigit():
            mapped[names[int(key[1:])]] = float(value)
        else:
            mapped[key] = float(value)
    return mapped


def _create_tuning_plots(
    root: Path,
    sgd_results: dict[str, Any],
    xgb_results: dict[str, Any],
) -> list[Path]:
    directory = root / TUNING_PLOTS_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / filename for filename in TUNING_PLOT_FILENAMES]
    sgd_trials = sgd_results["trials"]
    _plot_bar(
        [trial["trial_id"] for trial in sgd_trials],
        [trial["inner_tuning_metrics"]["average_precision"] for trial in sgd_trials],
        "Inner Tuning SGD Trial PR-AUC",
        "Average precision",
        paths[0],
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        [trial["positive_class_weight"] for trial in sgd_trials],
        [trial["inner_tuning_metrics"]["average_precision"] for trial in sgd_trials],
    )
    ax.set_xscale("log")
    ax.set_title("Inner Tuning SGD Weight vs PR-AUC")
    ax.set_xlabel("Positive-class weight")
    ax.set_ylabel("Average precision")
    fig.tight_layout()
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)

    xgb_trials = xgb_results.get("trials", [])
    _plot_bar(
        [trial["trial_id"] for trial in xgb_trials] or ["disabled"],
        [trial["inner_tuning_metrics"]["average_precision"] for trial in xgb_trials] or [0.0],
        "Inner Tuning XGBoost Trial PR-AUC",
        "Average precision",
        paths[2],
    )
    labels = ["best_sgd"]
    values = [sgd_results["best_trial"]["inner_tuning_metrics"]["average_precision"]]
    if xgb_results.get("best_trial") is not None:
        labels.append("best_xgboost")
        values.append(xgb_results["best_trial"]["inner_tuning_metrics"]["average_precision"])
    _plot_bar(labels, values, "Inner Tuning Model Comparison", "Average precision", paths[3])
    return paths


def _create_modeling_plots(
    root: Path,
    validation_metrics: dict[str, Any],
    validation_scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    champion: str,
    xgb_model: XGBClassifier | None,
) -> list[Path]:
    from sklearn.metrics import roc_curve

    directory = root / MODELING_PLOTS_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / filename for filename in MODELING_PLOT_FILENAMES]

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, metrics in validation_metrics.items():
        curve = metrics["precision_recall_curve"]
        ax.plot(curve["recall"], curve["precision"], label=name)
    ax.set_title("Official Validation Precision-Recall Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[0], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, score in validation_scores.items():
        fpr, tpr, _thresholds = roc_curve(y_true, score)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("Official Validation ROC Comparison")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)

    labels = list(validation_metrics)
    _plot_bar(
        labels,
        [validation_metrics[name]["average_precision"] for name in labels],
        "Official Validation PR-AUC Comparison",
        "Average precision",
        paths[2],
    )
    _plot_bar(
        labels,
        [validation_metrics[name]["selected_threshold_metrics"]["f_beta"] for name in labels],
        "Official Validation Selected Threshold F2",
        "F2",
        paths[3],
    )
    _plot_bar(
        labels,
        [validation_metrics[name]["top_k"][1]["recall"] for name in labels],
        "Official Validation Top-K Recall Comparison",
        "Recall at top 0.5%",
        paths[4],
    )
    _plot_bar(
        labels,
        [validation_metrics[name]["top_k"][1]["fraud_amount_recall"] for name in labels],
        "Official Validation Top-K Fraud Amount Recall",
        "Fraud amount recall at top 0.5%",
        paths[5],
    )
    importance = {}
    if xgb_model is not None:
        raw = xgb_model.get_booster().get_score(importance_type="gain")
        importance = _named_feature_importance(raw)
    ordered = sorted(importance.items(), key=lambda item: item[1])[-12:]
    _plot_bar(
        [name for name, _value in ordered] or ["unavailable"],
        [value for _name, value in ordered] or [0.0],
        "XGBoost Feature Importance By Gain",
        "Gain",
        paths[6],
    )
    matrix = validation_metrics[champion]["selected_threshold_metrics"]["confusion_matrix"]
    values = np.array(
        [
            [matrix["true_negative"], matrix["false_positive"]],
            [matrix["false_negative"], matrix["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(f"Official Validation Champion Confusion Matrix: {champion}")
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(paths[7], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in labels:
        selected = validation_metrics[name]["selected_threshold_metrics"]
        ax.scatter(selected["alert_rate"], selected["recall"], label=name)
    ax.set_title("Official Validation Alert Rate vs Recall")
    ax.set_xlabel("Alert rate")
    ax.set_ylabel("Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[8], dpi=150)
    plt.close(fig)
    return paths


def _build_manifests(
    root: Path,
    config_path: Path,
    config: TuningConfig,
    inner_split: InnerSplit,
    tuning_results: dict[str, Any],
    validation_payload: dict[str, Any],
    artifact_path: Path,
    full_train_counts: dict[int, int],
    full_xgb_sampling: dict[str, Any] | None,
    selected_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_relative = artifact_path.relative_to(root).as_posix()
    shared = {
        "created_at_utc": utc_timestamp(),
        "git_commit_hash": _git_commit(root),
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "tuning_config_sha256": calculate_sha256(config_path),
        "source_split_manifest_sha256": calculate_sha256(root / SPLIT_MANIFEST_RELATIVE),
        "feature_policy_sha256": calculate_sha256(root / FEATURE_POLICY_RELATIVE),
        "train_path": config.train_path.as_posix(),
        "validation_path": config.validation_path.as_posix(),
        "inner_split_information": inner_split.manifest,
        "training_counts": {
            "rows": int(full_train_counts[0] + full_train_counts[1]),
            "frauds": int(full_train_counts[1]),
            "non_frauds": int(full_train_counts[0]),
        },
        "sampling_counts": full_xgb_sampling,
        "selected_threshold": float(selected_threshold),
        "artifact_path": artifact_relative,
        "artifact_size_bytes": int(artifact_path.stat().st_size),
        "artifact_sha256": calculate_sha256(artifact_path),
        "model_artifact_sha256": calculate_sha256(artifact_path),
        "test_set_accessed": False,
        "reproducibility_status": "reproducible",
    }
    tuning_manifest = {
        **shared,
        "phase": "1D_tuning",
        "official_validation_accessed_during_search": False,
        "sgd_trial_count": len(tuning_results["sgd_search"]["trials"]),
        "xgboost_trial_count": len(tuning_results["xgboost_search"].get("trials", [])),
    }
    model_manifest = {
        **shared,
        "phase": "1D_modeling",
        "model_hyperparameters": validation_payload["frozen_candidate_configs"],
        "champion_model": validation_payload["champion_model"],
        "selection_rationale": validation_payload["selection_rationale"],
    }
    return tuning_manifest, model_manifest


def tune_models(
    root: Path | None = None,
    config_path: Path | None = None,
    force: bool = False,
) -> TuningRunResult:
    """Run or reuse Phase 1D tuning, retraining, validation comparison, and artifacts."""

    start = time.perf_counter()
    repo_root = root or repository_root()
    resolved_config_path = config_path or repo_root / CONFIG_RELATIVE
    config = load_tuning_config(resolved_config_path, repo_root)
    if not force and _phase1d_outputs_valid(repo_root, resolved_config_path):
        return TuningRunResult(
            tuning_results=_read_json(repo_root / TUNING_RESULTS_RELATIVE) or {},
            validation_metrics=_read_json(repo_root / VALIDATION_METRICS_RELATIVE) or {},
            tuning_manifest=_read_json(repo_root / TUNING_MANIFEST_RELATIVE) or {},
            model_manifest=_read_json(repo_root / MODEL_MANIFEST_RELATIVE) or {},
            model_artifact_path=repo_root / MODEL_ARTIFACT_RELATIVE,
            plot_paths=[
                repo_root / TUNING_PLOTS_RELATIVE / filename for filename in TUNING_PLOT_FILENAMES
            ]
            + [
                repo_root / MODELING_PLOTS_RELATIVE / filename
                for filename in MODELING_PLOT_FILENAMES
            ],
            reused=True,
            duration_seconds=float(time.perf_counter() - start),
        )
    if force:
        _remove_phase1d_outputs(repo_root)

    inner_split = create_inner_split(
        root=repo_root,
        train_path=config.train_path,
        batch_size=config.batch_size,
        fit_fraction=config.inner_tuning.fit_fraction,
        method=config.inner_tuning.method,
    )
    sgd_results, _inner_sgd_model, _inner_sgd_scaler = run_sgd_search(
        repo_root,
        config,
        inner_split,
    )
    xgb_results, _inner_xgb_model, _inner_xgb_sampling = run_xgboost_search(
        repo_root,
        config,
        inner_split,
    )
    frozen = _freeze_configs(repo_root, sgd_results, xgb_results)

    sgd_model, sgd_scaler, full_train_counts, sgd_duration = _retrain_sgd(
        repo_root,
        config,
        frozen["best_sgd_configuration"],
    )
    xgb_model, full_xgb_sampling, xgb_duration = _retrain_xgboost(
        repo_root,
        config,
        frozen["best_xgboost_configuration"],
    )
    y_val, amount_val, validation_scores, baseline_model, baseline_scaler = (
        _validation_scores_for_candidates(
            repo_root,
            config,
            sgd_model,
            sgd_scaler,
            xgb_model,
        )
    )
    validation_candidate_metrics = _evaluate_validation_candidates(
        y_val,
        amount_val,
        validation_scores,
        config,
    )
    champion, rationale = _select_phase1d_champion(
        validation_candidate_metrics,
        config.minimum_pr_auc_improvement,
    )
    selected_threshold = float(validation_candidate_metrics[champion]["selected_threshold"])
    bundle = _build_bundle(
        repo_root,
        champion,
        baseline_model,
        baseline_scaler,
        sgd_model,
        sgd_scaler,
        xgb_model,
        selected_threshold,
        frozen,
        full_xgb_sampling,
        config,
    )
    artifact_path = repo_root / MODEL_ARTIFACT_RELATIVE
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)

    tuning_plot_paths = _create_tuning_plots(repo_root, sgd_results, xgb_results)
    modeling_plot_paths = _create_modeling_plots(
        repo_root,
        validation_candidate_metrics,
        validation_scores,
        y_val,
        champion,
        xgb_model,
    )
    tuning_results = {
        "created_at_utc": utc_timestamp(),
        "inner_split": inner_split.manifest,
        "sgd_search": sgd_results,
        "xgboost_search": xgb_results,
        "frozen_candidate_configs_path": FROZEN_CONFIGS_RELATIVE,
        "official_validation_accessed_during_search": False,
        "test_set_accessed": False,
        "retraining_durations_seconds": {
            "phase1d_tuned_sgd_logistic": sgd_duration,
            "phase1d_tuned_xgboost": xgb_duration,
        },
        "generated_plots": [
            path.relative_to(repo_root).as_posix() for path in tuning_plot_paths
        ],
    }
    validation_payload = {
        "created_at_utc": utc_timestamp(),
        "candidates": validation_candidate_metrics,
        "champion_model": champion,
        "selection_rationale": rationale,
        "selected_threshold": selected_threshold,
        "validation_row_count": int(len(y_val)),
        "validation_fraud_count": int(y_val.sum()),
        "frozen_candidate_configs": frozen,
        "generated_plots": [
            path.relative_to(repo_root).as_posix() for path in modeling_plot_paths
        ],
        "model_artifact_path": MODEL_ARTIFACT_RELATIVE,
        "official_validation_accessed_during_search": False,
        "test_set_accessed": False,
    }
    tuning_manifest, model_manifest = _build_manifests(
        root=repo_root,
        config_path=resolved_config_path,
        config=config,
        inner_split=inner_split,
        tuning_results=tuning_results,
        validation_payload=validation_payload,
        artifact_path=artifact_path,
        full_train_counts=full_train_counts,
        full_xgb_sampling=full_xgb_sampling,
        selected_threshold=selected_threshold,
    )
    _write_json(repo_root / TUNING_RESULTS_RELATIVE, tuning_results)
    _write_json(repo_root / VALIDATION_METRICS_RELATIVE, validation_payload)
    _write_json(repo_root / TUNING_MANIFEST_RELATIVE, tuning_manifest)
    _write_json(repo_root / MODEL_MANIFEST_RELATIVE, model_manifest)
    return TuningRunResult(
        tuning_results=tuning_results,
        validation_metrics=validation_payload,
        tuning_manifest=tuning_manifest,
        model_manifest=model_manifest,
        model_artifact_path=artifact_path,
        plot_paths=tuning_plot_paths + modeling_plot_paths,
        reused=False,
        duration_seconds=float(time.perf_counter() - start),
    )


def _print_summary(result: TuningRunResult) -> None:
    tuning = result.tuning_results
    validation = result.validation_metrics
    inner = tuning["inner_split"]
    champion = validation["champion_model"]
    champion_metrics = validation["candidates"][champion]
    selected = champion_metrics["selected_threshold_metrics"]
    print("Phase 1D time-aware tuning and validation comparison")
    print(f"Mode: {'reused' if result.reused else 'trained'}")
    print(f"Inner boundary step: {inner['selected_step_boundary']}")
    print(
        "Inner fit rows/frauds: "
        f"{inner['inner_fit_row_count']}/{inner['inner_fit_fraud_count']}; "
        "inner tuning rows/frauds: "
        f"{inner['inner_tuning_row_count']}/{inner['inner_tuning_fraud_count']}"
    )
    print(f"SGD trials: {len(tuning['sgd_search']['trials'])}")
    print(f"XGBoost trials: {len(tuning['xgboost_search'].get('trials', []))}")
    print(f"Best SGD: {validation['frozen_candidate_configs']['best_sgd_configuration']}")
    print(f"Best XGBoost: {validation['frozen_candidate_configs']['best_xgboost_configuration']}")
    sampling = result.model_manifest.get("sampling_counts")
    print(f"Full-train XGBoost sampling: {sampling}")
    print("Official validation candidate metrics:")
    for name, metrics in validation["candidates"].items():
        print(
            f"  {name}: PR-AUC={metrics['average_precision']:.6f}, "
            f"ROC-AUC={metrics['roc_auc']:.6f}, "
            f"threshold={metrics['selected_threshold']:.8f}"
        )
    print(f"Selected champion: {champion}")
    print(f"Selected threshold: {champion_metrics['selected_threshold']:.8f}")
    print(
        "Selected-threshold metrics: "
        f"precision={selected['precision']:.6f}, recall={selected['recall']:.6f}, "
        f"F1={selected['f1']:.6f}, F2={selected['f_beta']:.6f}"
    )
    print(f"Alert rate: {selected['alert_rate']:.6f}")
    print(f"Fraud amount recall: {selected['fraud_amount_recall']:.6f}")
    print("Top-k results:")
    for item in champion_metrics["top_k"]:
        print(
            f"  top {item['top_k_percentage']}%: recall={item['recall']:.6f}, "
            f"fraud_amount_recall={item['fraud_amount_recall']:.6f}"
        )
    print(f"Model artifact: {result.model_artifact_path}")
    print("Official validation used during search: false")
    print("Test data access: false; sealed test.parquet was not accessed")
    print(f"Duration seconds: {result.duration_seconds:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1D time-aware model tuning.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only Phase 1D tuning/model artifacts and plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = tune_models(force=args.force)
        _print_summary(result)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
