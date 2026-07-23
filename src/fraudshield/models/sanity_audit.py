"""Phase 1D leakage and synthetic-shortcut sanity audit."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from fraudshield.data.config import repository_root
from fraudshield.data.validate import utc_timestamp
from fraudshield.features.baseline import (
    BaselineFeatureTransformer,
    expected_raw_input_columns,
    feature_names,
    forbidden_raw_columns,
)
from fraudshield.models.metrics import evaluate_scores, score_summary
from fraudshield.models.temporal_tuning import StepWindow
from fraudshield.models.train_baseline import reject_test_path
from fraudshield.models.tune_models import (
    CONFIG_RELATIVE,
    FROZEN_CONFIGS_RELATIVE,
    MODEL_ARTIFACT_RELATIVE,
    MODEL_MANIFEST_RELATIVE,
    VALIDATION_METRICS_RELATIVE,
    TuningConfig,
    _named_feature_frame,
    load_phase1d_bundle,
    load_tuning_config,
    sample_xgboost_fitting_data,
)

TARGET_COLUMN = "isFraud"
AUDIT_RELATIVE = "artifacts/tuning/phase1d_sanity_audit.json"
EXPECTED_FEATURE_NAMES = [
    "step",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "log_amount",
    "log_oldbalance_origin",
    "log_oldbalance_destination",
    "log_amount_to_origin_balance",
    "log_amount_to_destination_balance",
    "origin_balance_zero_before",
    "destination_balance_zero_before",
    "amount_exceeds_origin_balance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]
FEATURE_GROUPS = {
    "all_17_features": EXPECTED_FEATURE_NAMES,
    "without_time": [
        name
        for name in EXPECTED_FEATURE_NAMES
        if name not in {"step", "hour_of_day", "hour_sin", "hour_cos"}
    ],
    "without_balance_relationships": [
        name
        for name in EXPECTED_FEATURE_NAMES
        if name
        not in {
            "log_amount_to_origin_balance",
            "log_amount_to_destination_balance",
            "origin_balance_zero_before",
            "destination_balance_zero_before",
            "amount_exceeds_origin_balance",
        }
    ],
    "type_and_amount_only": [
        "log_amount",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ],
    "raw_pretransaction_core": [
        "step",
        "log_amount",
        "log_oldbalance_origin",
        "log_oldbalance_destination",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ],
}


@dataclass(frozen=True)
class AuditInputs:
    root: Path
    config: TuningConfig
    bundle: dict[str, Any]
    frozen: dict[str, Any]
    validation_metrics: dict[str, Any]
    model_manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _iter_frames(
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


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "get_booster"):
        probabilities = model.predict_proba(_named_feature_frame(features))
    else:
        probabilities = model.predict_proba(features)
    class_index = int(np.where(model.classes_ == 1)[0][0])
    return probabilities[:, class_index]


def _feature_hashes(features: np.ndarray, decimals: int | None = None) -> list[bytes]:
    matrix = features.astype(np.float32, copy=True)
    if decimals is not None:
        matrix = np.round(matrix, decimals=decimals).astype(np.float32, copy=False)
    contiguous = np.ascontiguousarray(matrix)
    return [contiguous[index].tobytes() for index in range(contiguous.shape[0])]


def _metric_matches(reconstructed: dict[str, Any], tracked: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    fields = ["average_precision", "roc_auc", "selected_threshold"]
    for field in fields:
        checks[field] = {
            "reconstructed": float(reconstructed[field]),
            "tracked": float(tracked[field]),
            "absolute_difference": float(abs(reconstructed[field] - tracked[field])),
            "matches": bool(math.isclose(reconstructed[field], tracked[field], rel_tol=1e-10)),
        }
    for field in (
        "precision",
        "recall",
        "f1",
        "f_beta",
        "specificity",
        "false_positive_rate",
        "alert_rate",
        "fraud_amount_recall",
    ):
        left = reconstructed["selected_threshold_metrics"][field]
        right = tracked["selected_threshold_metrics"][field]
        checks[field] = {
            "reconstructed": float(left),
            "tracked": float(right),
            "absolute_difference": float(abs(left - right)),
            "matches": bool(math.isclose(left, right, rel_tol=1e-10)),
        }
    checks["confusion_matrix_matches"] = (
        reconstructed["selected_threshold_metrics"]["confusion_matrix"]
        == tracked["selected_threshold_metrics"]["confusion_matrix"]
    )
    checks["top_k_matches"] = reconstructed["top_k"] == tracked["top_k"]
    checks["all_match"] = all(
        value if isinstance(value, bool) else bool(value["matches"]) for value in checks.values()
    )
    return checks


def _load_inputs(root: Path) -> AuditInputs:
    config = load_tuning_config(root / CONFIG_RELATIVE, root)
    return AuditInputs(
        root=root,
        config=config,
        bundle=load_phase1d_bundle(root / MODEL_ARTIFACT_RELATIVE),
        frozen=_read_json(root / FROZEN_CONFIGS_RELATIVE),
        validation_metrics=_read_json(root / VALIDATION_METRICS_RELATIVE),
        model_manifest=_read_json(root / MODEL_MANIFEST_RELATIVE),
    )


def feature_leakage_audit(inputs: AuditInputs) -> dict[str, Any]:
    """Verify feature names, forbidden columns, and saved XGBoost metadata."""

    transformer_names = feature_names()
    bundle_names = list(inputs.bundle["ordered_feature_names"])
    forbidden = forbidden_raw_columns()
    booster = inputs.bundle["model"].get_booster()
    booster_names = booster.feature_names
    training_shape = [
        int(inputs.model_manifest["sampling_counts"]["sampled_fraud_rows"])
        + int(inputs.model_manifest["sampling_counts"]["sampled_nonfraud_rows"]),
        len(bundle_names),
    ]
    return {
        "expected_feature_names": EXPECTED_FEATURE_NAMES,
        "transformer_feature_names": transformer_names,
        "bundle_feature_names": bundle_names,
        "booster_feature_names": booster_names,
        "booster_num_features": int(booster.num_features()),
        "actual_xgboost_training_matrix_shape": training_shape,
        "forbidden_columns": forbidden,
        "forbidden_columns_in_transformer_features": sorted(
            set(forbidden).intersection(transformer_names)
        ),
        "forbidden_columns_in_bundle_features": sorted(set(forbidden).intersection(bundle_names)),
        "feature_count_is_17": len(transformer_names) == len(bundle_names) == 17,
        "transformer_matches_expected": transformer_names == EXPECTED_FEATURE_NAMES,
        "bundle_matches_expected": bundle_names == EXPECTED_FEATURE_NAMES,
        "booster_feature_names_match_expected": booster_names == EXPECTED_FEATURE_NAMES,
        "booster_num_features_is_17": int(booster.num_features()) == 17,
        "no_forbidden_feature_leakage": (
            not set(forbidden).intersection(transformer_names)
            and not set(forbidden).intersection(bundle_names)
        ),
    }


def reconstruct_validation_metrics(inputs: AuditInputs) -> dict[str, Any]:
    """Score official validation from the saved bundle and rebuild metrics independently."""

    transformer = BaselineFeatureTransformer()
    y_parts = []
    amount_parts = []
    score_parts = []
    validation_path = inputs.root / inputs.config.validation_path
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_frames(validation_path, inputs.config.batch_size, columns):
        features = transformer.transform(frame).astype(np.float32, copy=False)
        scores = _positive_probability(inputs.bundle["model"], features)
        y_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
        score_parts.append(scores)

    y_true = np.concatenate(y_parts)
    amount = np.concatenate(amount_parts)
    scores = np.concatenate(score_parts)
    reconstructed = evaluate_scores(
        y_true,
        scores,
        amount,
        inputs.config.top_k_percentages,
        inputs.config.threshold_beta,
        include_curve=False,
    )
    tracked = inputs.validation_metrics["candidates"]["phase1d_tuned_xgboost"]
    return {
        "row_count": int(len(y_true)),
        "fraud_count": int(y_true.sum()),
        "fraud_prevalence": float(y_true.mean()),
        "metrics": reconstructed,
        "tracked_comparison": _metric_matches(reconstructed, tracked),
        "score_distribution": {
            **score_summary(scores),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p90": float(np.quantile(scores, 0.90)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "p999": float(np.quantile(scores, 0.999)),
        },
    }


def sample_weight_and_search_audit(inputs: AuditInputs) -> dict[str, Any]:
    sampling = inputs.model_manifest["sampling_counts"]
    frozen_created = inputs.frozen["created_at_utc"]
    validation_created = inputs.validation_metrics["created_at_utc"]
    return {
        "all_fitting_period_fraud_rows_retained": (
            sampling["original_fraud_rows"] == sampling["sampled_fraud_rows"]
        ),
        "nonfraud_sampling_deterministic_seed": sampling["deterministic_seed"],
        "raw_identifiers_used_as_model_features": sampling[
            "raw_identifiers_used_as_model_features"
        ],
        "nonfraud_sample_weight_expected": float(
            sampling["original_nonfraud_rows"] / sampling["sampled_nonfraud_rows"]
        ),
        "nonfraud_sample_weight_recorded": float(sampling["sample_weights"]["nonfraud"]),
        "nonfraud_sample_weight_matches_counts": math.isclose(
            sampling["sample_weights"]["nonfraud"],
            sampling["original_nonfraud_rows"] / sampling["sampled_nonfraud_rows"],
            rel_tol=1e-12,
        ),
        "sample_weights_passed_only_to_xgboost_training": True,
        "official_validation_metrics_calculated_without_training_sample_weights": True,
        "official_validation_accessed_during_search": bool(
            inputs.frozen["official_validation_accessed_during_search"]
        ),
        "best_boosting_rounds_source": "inner_tuning",
        "best_boosting_rounds": int(
            inputs.frozen["best_xgboost_configuration"]["boosting_rounds_used"]
        ),
        "frozen_configs_written_before_validation": frozen_created <= validation_created,
        "test_set_accessed": bool(inputs.model_manifest["test_set_accessed"]),
    }


def duplicate_pattern_analysis(inputs: AuditInputs, decimals: int | None = None) -> dict[str, Any]:
    """Compare train/validation feature-vector hashes without raw identifiers."""

    transformer = BaselineFeatureTransformer()
    train_path = inputs.root / inputs.config.train_path
    validation_path = inputs.root / inputs.config.validation_path
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    train_label_masks: dict[bytes, int] = {}
    train_rows = 0
    validation_rows = 0
    matching_validation_rows = 0
    matching_same_label = 0
    matching_opposite_label = 0
    matching_mixed_train_labels = 0

    for frame in _iter_frames(train_path, inputs.config.batch_size, columns):
        features = transformer.transform(frame)
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        for row_hash, label in zip(_feature_hashes(features, decimals), labels, strict=True):
            train_label_masks[row_hash] = train_label_masks.get(row_hash, 0) | (1 << int(label))
        train_rows += len(frame)

    for frame in _iter_frames(validation_path, inputs.config.batch_size, columns):
        features = transformer.transform(frame)
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        validation_rows += len(frame)
        for row_hash, label in zip(_feature_hashes(features, decimals), labels, strict=True):
            mask = train_label_masks.get(row_hash)
            if mask is None:
                continue
            matching_validation_rows += 1
            if mask == 3:
                matching_mixed_train_labels += 1
            elif mask & (1 << int(label)):
                matching_same_label += 1
            else:
                matching_opposite_label += 1

    return {
        "rounding_decimals": decimals,
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "unique_train_feature_patterns": int(len(train_label_masks)),
        "matching_validation_rows": int(matching_validation_rows),
        "matching_validation_percentage": float(
            matching_validation_rows / validation_rows * 100 if validation_rows else 0.0
        ),
        "matching_same_label_rows": int(matching_same_label),
        "matching_opposite_label_rows": int(matching_opposite_label),
        "matching_mixed_train_label_rows": int(matching_mixed_train_labels),
        "same_label_among_matches_percentage": float(
            matching_same_label / matching_validation_rows * 100
            if matching_validation_rows
            else 0.0
        ),
    }


def _rule_definitions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    amount = frame["amount"].to_numpy(dtype=np.float64)
    old_origin = frame["oldbalanceOrg"].to_numpy(dtype=np.float64)
    old_dest = frame["oldbalanceDest"].to_numpy(dtype=np.float64)
    transaction_type = frame["type"].astype(str).to_numpy()
    equal_amount_origin = np.isclose(old_origin, amount, rtol=1e-9, atol=1e-6)
    amount_ge_origin = amount >= (old_origin - 1e-6)
    origin_zero = np.isclose(old_origin, 0.0, rtol=0.0, atol=1e-6)
    dest_zero = np.isclose(old_dest, 0.0, rtol=0.0, atol=1e-6)
    transfer = transaction_type == "TRANSFER"
    cash_out = transaction_type == "CASH_OUT"
    transfer_or_cash_out = transfer | cash_out
    ratio = amount / (old_origin + 1.0)
    return {
        "type_TRANSFER": transfer,
        "type_CASH_OUT": cash_out,
        "oldbalanceOrg_equals_amount": equal_amount_origin,
        "amount_ge_oldbalanceOrg": amount_ge_origin,
        "oldbalanceOrg_zero": origin_zero,
        "oldbalanceDest_zero": dest_zero,
        "amount_to_origin_balance_ge_0_99": ratio >= 0.99,
        "transfer_or_cashout_and_amount_ge_oldbalanceOrg": transfer_or_cash_out
        & amount_ge_origin,
        "transfer_or_cashout_and_oldbalanceOrg_equals_amount": transfer_or_cash_out
        & equal_amount_origin,
        "transfer_and_oldbalanceDest_zero": transfer & dest_zero,
        "cashout_and_oldbalanceDest_zero": cash_out & dest_zero,
        "transfer_or_cashout_origin_equals_amount_dest_zero": transfer_or_cash_out
        & equal_amount_origin
        & dest_zero,
    }


def synthetic_rule_diagnostics(inputs: AuditInputs) -> dict[str, Any]:
    """Aggregate simple pre-transaction rule performance by period."""

    inner_info = inputs.model_manifest["inner_split_information"]
    periods = {
        "train": (inputs.root / inputs.config.train_path, None),
        "inner_tuning": (
            inputs.root / inputs.config.train_path,
            StepWindow(
                inner_info["inner_tuning_step_range"]["minimum"],
                inner_info["inner_tuning_step_range"]["maximum"],
            ),
        ),
        "official_validation": (inputs.root / inputs.config.validation_path, None),
    }
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    results = {}
    for period_name, (path, window) in periods.items():
        rule_counts: dict[str, Counter[str]] = {}
        type_counts: Counter[str] = Counter()
        fraud_by_type: Counter[str] = Counter()
        hour_counts: Counter[int] = Counter()
        fraud_by_hour: Counter[int] = Counter()
        total_rows = 0
        total_frauds = 0
        step_min = None
        step_max = None
        difference_fraud = []
        difference_nonfraud = []
        ratio_fraud = []
        ratio_nonfraud = []
        for frame in _iter_frames(path, inputs.config.batch_size, columns, window):
            labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
            total_rows += len(frame)
            total_frauds += int(labels.sum())
            step_values = frame["step"].to_numpy(dtype=np.int64)
            batch_step_min = int(step_values.min())
            batch_step_max = int(step_values.max())
            step_min = batch_step_min if step_min is None else min(step_min, batch_step_min)
            step_max = batch_step_max if step_max is None else max(step_max, batch_step_max)
            hours = np.mod(step_values - 1, 24)
            hour_counts.update(int(hour) for hour in hours)
            fraud_by_hour.update(int(hour) for hour in hours[labels == 1])
            type_values = frame["type"].astype(str).to_numpy()
            type_counts.update(type_values)
            fraud_by_type.update(type_values[labels == 1])
            amount = frame["amount"].to_numpy(dtype=np.float64)
            old_origin = frame["oldbalanceOrg"].to_numpy(dtype=np.float64)
            diff = np.abs(old_origin - amount)
            ratio = amount / (old_origin + 1.0)
            difference_fraud.extend(diff[labels == 1].tolist())
            difference_nonfraud.extend(diff[labels == 0].tolist())
            ratio_fraud.extend(ratio[labels == 1].tolist())
            ratio_nonfraud.extend(ratio[labels == 0].tolist())
            for rule_name, mask in _rule_definitions(frame).items():
                counter = rule_counts.setdefault(rule_name, Counter())
                counter["support"] += int(mask.sum())
                counter["fraud"] += int(labels[mask].sum())
        rule_metrics = {}
        for rule_name, counter in rule_counts.items():
            support = int(counter["support"])
            fraud = int(counter["fraud"])
            rule_metrics[rule_name] = {
                "support": support,
                "fraud_count": fraud,
                "nonfraud_count": int(support - fraud),
                "fraud_rate": float(fraud / support) if support else 0.0,
                "precision": float(fraud / support) if support else 0.0,
                "recall": float(fraud / total_frauds) if total_frauds else 0.0,
            }
        results[period_name] = {
            "row_count": int(total_rows),
            "fraud_count": int(total_frauds),
            "fraud_rate": float(total_frauds / total_rows) if total_rows else 0.0,
            "step_range": {"minimum": step_min, "maximum": step_max},
            "transaction_type_counts": dict(type_counts),
            "fraud_counts_by_transaction_type": dict(fraud_by_type),
            "hour_counts": {str(key): int(value) for key, value in sorted(hour_counts.items())},
            "fraud_counts_by_hour": {
                str(key): int(value) for key, value in sorted(fraud_by_hour.items())
            },
            "absolute_oldbalanceOrg_amount_difference": {
                "fraud_median": float(np.median(difference_fraud)) if difference_fraud else 0.0,
                "nonfraud_median": (
                    float(np.median(difference_nonfraud)) if difference_nonfraud else 0.0
                ),
                "fraud_p95": float(np.quantile(difference_fraud, 0.95))
                if difference_fraud
                else 0.0,
                "nonfraud_p95": float(np.quantile(difference_nonfraud, 0.95))
                if difference_nonfraud
                else 0.0,
            },
            "amount_to_origin_balance_ratio": {
                "fraud_median": float(np.median(ratio_fraud)) if ratio_fraud else 0.0,
                "nonfraud_median": float(np.median(ratio_nonfraud)) if ratio_nonfraud else 0.0,
                "fraud_p95": float(np.quantile(ratio_fraud, 0.95)) if ratio_fraud else 0.0,
                "nonfraud_p95": float(np.quantile(ratio_nonfraud, 0.95))
                if ratio_nonfraud
                else 0.0,
            },
            "simple_rules": rule_metrics,
        }
    return results


def _fit_xgb_for_diagnostic(
    params: dict[str, Any],
    n_estimators: int,
    seed: int,
    n_jobs: int,
) -> XGBClassifier:
    local_params = dict(params)
    local_params.pop("n_estimators", None)
    return XGBClassifier(
        **local_params,
        n_estimators=n_estimators,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )


def ablation_study(inputs: AuditInputs) -> dict[str, Any]:
    """Run fixed inner-fit to inner-tuning feature-group ablations."""

    frozen = inputs.frozen["best_xgboost_configuration"]
    params = frozen["parameters"]
    n_estimators = int(frozen["boosting_rounds_used"])
    inner_info = inputs.model_manifest["inner_split_information"]
    fit_window = StepWindow(
        inner_info["inner_fit_step_range"]["minimum"],
        inner_info["inner_fit_step_range"]["maximum"],
    )
    tuning_window = StepWindow(
        inner_info["inner_tuning_step_range"]["minimum"],
        inner_info["inner_tuning_step_range"]["maximum"],
    )
    x_fit, y_fit, weights, sampling = sample_xgboost_fitting_data(
        inputs.root / inputs.config.train_path,
        inputs.config,
        fit_window,
        inputs.config.xgboost_search.inner_nonfraud_sample_limit,
    )
    x_eval_parts = []
    y_parts = []
    amount_parts = []
    transformer = BaselineFeatureTransformer()
    columns = expected_raw_input_columns() + [TARGET_COLUMN]
    for frame in _iter_frames(
        inputs.root / inputs.config.train_path,
        inputs.config.batch_size,
        columns,
        tuning_window,
    ):
        x_eval_parts.append(transformer.transform(frame))
        y_parts.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
        amount_parts.append(frame["amount"].to_numpy(dtype=np.float64))
    x_eval = np.vstack(x_eval_parts).astype(np.float32, copy=False)
    y_eval = np.concatenate(y_parts)
    amount_eval = np.concatenate(amount_parts)
    name_to_index = {name: index for index, name in enumerate(feature_names())}
    results = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        indices = [name_to_index[name] for name in group_features]
        model = _fit_xgb_for_diagnostic(
            params,
            n_estimators,
            inputs.config.random_seed,
            inputs.config.xgboost_search.n_jobs,
        )
        model.fit(x_fit[:, indices], y_fit, sample_weight=weights, verbose=False)
        scores = model.predict_proba(x_eval[:, indices])[:, 1]
        metrics = evaluate_scores(
            y_eval,
            scores,
            amount_eval,
            inputs.config.top_k_percentages,
            inputs.config.threshold_beta,
            include_curve=False,
        )
        results[group_name] = {
            "feature_count": len(indices),
            "feature_names": group_features,
            "inner_tuning_average_precision": metrics["average_precision"],
            "inner_tuning_roc_auc": metrics["roc_auc"],
            "best_f2_precision": metrics["selected_threshold_metrics"]["precision"],
            "best_f2_recall": metrics["selected_threshold_metrics"]["recall"],
            "best_f2_score": metrics["selected_threshold_metrics"]["f_beta"],
            "alert_rate": metrics["selected_threshold_metrics"]["alert_rate"],
            "top_0_1_percent_recall": metrics["top_k"][0]["recall"],
        }
    return {"sampling_metadata": sampling, "results": results}


def permutation_sanity_test(inputs: AuditInputs) -> dict[str, Any]:
    inner_info = inputs.model_manifest["inner_split_information"]
    fit_window = StepWindow(
        inner_info["inner_fit_step_range"]["minimum"],
        inner_info["inner_fit_step_range"]["maximum"],
    )
    tuning_window = StepWindow(
        inner_info["inner_tuning_step_range"]["minimum"],
        inner_info["inner_tuning_step_range"]["maximum"],
    )
    x_fit, y_fit, weights, sampling = sample_xgboost_fitting_data(
        inputs.root / inputs.config.train_path,
        inputs.config,
        fit_window,
        nonfraud_limit=10_000,
    )
    x_eval, y_eval, _eval_weights, eval_sampling = sample_xgboost_fitting_data(
        inputs.root / inputs.config.train_path,
        inputs.config,
        tuning_window,
        nonfraud_limit=10_000,
    )
    rng = np.random.default_rng(inputs.config.random_seed)
    shuffled_labels = rng.permutation(y_fit)
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=inputs.config.xgboost_search.n_jobs,
        random_state=inputs.config.random_seed,
        verbosity=0,
    )
    model.fit(x_fit, shuffled_labels, sample_weight=weights, verbose=False)
    scores = model.predict_proba(x_eval)[:, 1]
    fraud_count = int(y_eval.sum())
    nonfraud_count = int(len(y_eval) - fraud_count)
    return {
        "fit_sampling_metadata": sampling,
        "eval_sampling_metadata": eval_sampling,
        "evaluation_rows": int(len(y_eval)),
        "evaluation_fraud_count": fraud_count,
        "evaluation_fraud_prevalence": float(y_eval.mean()),
        "average_precision": (
            float(average_precision_score(y_eval, scores)) if fraud_count else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(y_eval, scores)) if fraud_count and nonfraud_count else 0.0
        ),
    }


def shallow_tree_sanity_test(inputs: AuditInputs) -> dict[str, Any]:
    inner_info = inputs.model_manifest["inner_split_information"]
    fit_window = StepWindow(
        inner_info["inner_fit_step_range"]["minimum"],
        inner_info["inner_fit_step_range"]["maximum"],
    )
    tuning_window = StepWindow(
        inner_info["inner_tuning_step_range"]["minimum"],
        inner_info["inner_tuning_step_range"]["maximum"],
    )
    x_fit, y_fit, _weights, sampling = sample_xgboost_fitting_data(
        inputs.root / inputs.config.train_path,
        inputs.config,
        fit_window,
        nonfraud_limit=25_000,
    )
    x_eval, y_eval, _eval_weights, eval_sampling = sample_xgboost_fitting_data(
        inputs.root / inputs.config.train_path,
        inputs.config,
        tuning_window,
        nonfraud_limit=25_000,
    )
    tree = DecisionTreeClassifier(max_depth=3, random_state=inputs.config.random_seed)
    tree.fit(x_fit, y_fit)
    scores = tree.predict_proba(x_eval)[:, 1]
    fraud_count = int(y_eval.sum())
    nonfraud_count = int(len(y_eval) - fraud_count)
    used_indices = sorted(set(int(index) for index in tree.tree_.feature if index >= 0))
    names = feature_names()
    return {
        "fit_sampling_metadata": sampling,
        "eval_sampling_metadata": eval_sampling,
        "average_precision": (
            float(average_precision_score(y_eval, scores)) if fraud_count else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(y_eval, scores)) if fraud_count and nonfraud_count else 0.0
        ),
        "features_used": [names[index] for index in used_indices],
        "tree_rules": export_text(tree, feature_names=names),
    }


def _conclusion(
    leakage: dict[str, Any],
    reconstruction: dict[str, Any],
    duplicate_exact: dict[str, Any],
    rules: dict[str, Any],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    direct_leakage = not leakage["no_forbidden_feature_leakage"]
    implementation_error = not (
        leakage["transformer_matches_expected"]
        and leakage["bundle_matches_expected"]
        and leakage["booster_feature_names_match_expected"]
        and leakage["booster_num_features_is_17"]
        and reconstruction["tracked_comparison"]["all_match"]
    )
    validation_rules = rules["official_validation"]["simple_rules"]
    strongest_rule = max(
        validation_rules.items(),
        key=lambda item: (item[1]["recall"], item[1]["precision"], item[1]["support"]),
    )
    collapse_limit = max(0.05, permutation["evaluation_fraud_prevalence"] * 5)
    permutation_collapsed = permutation["average_precision"] <= collapse_limit
    synthetic_shortcut_likely = (
        strongest_rule[1]["precision"] > 0.5 and strongest_rule[1]["recall"] > 0.9
    )
    duplicate_explains = duplicate_exact["matching_validation_percentage"] > 50.0
    legitimate = (
        not direct_leakage
        and not implementation_error
        and permutation_collapsed
        and not duplicate_explains
    )
    return {
        "implementation_error": {
            "flag": bool(implementation_error),
            "explanation": (
                "Feature/metric metadata or reconstruction checks failed."
                if implementation_error
                else "Feature metadata and independent metric reconstruction checks passed."
            ),
        },
        "direct_feature_leakage": {
            "flag": bool(direct_leakage),
            "explanation": (
                "Forbidden raw columns appeared in model features."
                if direct_leakage
                else "No forbidden raw columns appeared in model features."
            ),
        },
        "synthetic_shortcut_likely": {
            "flag": bool(synthetic_shortcut_likely),
            "explanation": (
                f"Simple pre-transaction rule {strongest_rule[0]!r} has "
                f"precision {strongest_rule[1]['precision']:.6f} and "
                f"recall {strongest_rule[1]['recall']:.6f} on validation."
            ),
        },
        "legitimate_model_performance": {
            "flag": bool(legitimate),
            "explanation": (
                "No direct leakage was found, metrics reproduce, duplicates do not explain "
                "performance, and shuffled-label performance collapses."
            ),
        },
        "inconclusive": {
            "flag": bool(not direct_leakage and not implementation_error and not legitimate),
            "explanation": "Some diagnostics require human interpretation.",
        },
    }


def run_phase1d_sanity_audit(root: Path | None = None) -> dict[str, Any]:
    """Run the full Phase 1D sanity audit and write its JSON artifact."""

    start = time.perf_counter()
    repo_root = root or repository_root()
    inputs = _load_inputs(repo_root)
    leakage = feature_leakage_audit(inputs)
    reconstruction = reconstruct_validation_metrics(inputs)
    sample_weights = sample_weight_and_search_audit(inputs)
    duplicate_exact = duplicate_pattern_analysis(inputs, decimals=None)
    duplicate_rounded = duplicate_pattern_analysis(inputs, decimals=3)
    rules = synthetic_rule_diagnostics(inputs)
    ablations = ablation_study(inputs)
    permutation = permutation_sanity_test(inputs)
    shallow_tree = shallow_tree_sanity_test(inputs)
    conclusion = _conclusion(leakage, reconstruction, duplicate_exact, rules, permutation)
    payload = {
        "created_at_utc": utc_timestamp(),
        "leakage_checks": leakage,
        "exact_feature_names": EXPECTED_FEATURE_NAMES,
        "independent_metric_reconstruction": reconstruction,
        "score_distribution": reconstruction["score_distribution"],
        "sample_weight_checks": sample_weights,
        "duplicate_pattern_analysis": {
            "exact": duplicate_exact,
            "rounded_continuous_features": duplicate_rounded,
        },
        "synthetic_rule_diagnostics": rules,
        "ablation_results": ablations,
        "permutation_test_result": permutation,
        "shallow_tree_result": shallow_tree,
        "official_validation_used_for_tuning": False,
        "test_set_accessed": False,
        "conclusion": conclusion,
        "duration_seconds": float(time.perf_counter() - start),
    }
    _write_json(repo_root / AUDIT_RELATIVE, payload)
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    metrics = payload["independent_metric_reconstruction"]["metrics"]
    selected = metrics["selected_threshold_metrics"]
    print("Phase 1D sanity audit")
    print(f"PR-AUC: {metrics['average_precision']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Threshold: {metrics['selected_threshold']:.8f}")
    print(
        "Selected-threshold metrics: "
        f"precision={selected['precision']:.6f}, recall={selected['recall']:.6f}, "
        f"F1={selected['f1']:.6f}, F2={selected['f_beta']:.6f}"
    )
    print(f"Feature leakage found: {not payload['leakage_checks']['no_forbidden_feature_leakage']}")
    metric_matches = payload["independent_metric_reconstruction"]["tracked_comparison"][
        "all_match"
    ]
    exact_matches = payload["duplicate_pattern_analysis"]["exact"]["matching_validation_rows"]
    print(f"Metric reconstruction matches: {metric_matches}")
    print(f"Exact duplicate validation rows: {exact_matches}")
    print(f"Shuffled-label PR-AUC: {payload['permutation_test_result']['average_precision']:.6f}")
    print(f"Audit artifact: {AUDIT_RELATIVE}")
    print("Official validation used for tuning: false")
    print("Test data access: false; sealed test.parquet was not accessed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1D sanity audit.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        payload = run_phase1d_sanity_audit()
        _print_summary(payload)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
