"""Import existing FraudShield reports and frozen models into local MLflow."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from fraudshield.data.config import repository_root
from fraudshield.features.baseline import feature_names
from fraudshield.tracking.config import MlflowConfig, load_mlflow_config
from fraudshield.tracking.mlflow_setup import (
    RUN_KEY_TAG,
    VERSION_KEY_TAG,
    configure_local_mlflow,
    ensure_experiment,
    ensure_registered_model,
    install_prohibited_data_guard,
    model_versions_with_key,
    relative_existing_files,
    runs_with_key,
    set_model_version_tags,
    sha256_file,
    stable_key,
    utc_timestamp,
    write_json,
)
from fraudshield.tracking.model_wrapper import (
    WRAPPER_VERSION,
    BenchmarkXGBoostPyFuncModel,
    ProductionSGDPyFuncModel,
    benchmark_signature,
    production_signature,
    synthetic_input_example,
)

REQUIRED_ARTIFACTS = (
    "artifacts/models/baseline_champion.joblib",
    "artifacts/models/phase1d_champion.joblib",
    "artifacts/models/production_sgd.joblib",
    "artifacts/modeling/baseline_metrics.json",
    "artifacts/modeling/phase1d_validation_metrics.json",
    "artifacts/evaluation/final_test_metrics.json",
    "artifacts/evaluation/final_test_manifest.json",
    "artifacts/evaluation/final_test_evaluation_complete.json",
    "artifacts/governance/production_model_decision.json",
    "artifacts/governance/production_sgd_manifest.json",
    "artifacts/model_card/production_sgd_model_card.md",
)
CONFIG_RELATIVE = "configs/mlflow.yaml"
PRODUCTION_CONFIG_RELATIVE = "configs/production.yaml"
TUNING_CONFIG_RELATIVE = "configs/tuning.yaml"
FEATURE_POLICY_RELATIVE = "configs/feature_policy.yaml"
BASELINE_MANIFEST_RELATIVE = "artifacts/modeling/model_manifest.json"
PHASE1D_MANIFEST_RELATIVE = "artifacts/modeling/phase1d_model_manifest.json"
FROZEN_CONFIG_RELATIVE = "artifacts/tuning/frozen_candidate_configs.json"
SANITY_AUDIT_RELATIVE = "artifacts/tuning/phase1d_sanity_audit.json"
SNAPSHOT_RELATIVE = "artifacts/mlflow/registry_snapshot.json"
MLFLOW_MANIFEST_RELATIVE = "artifacts/mlflow/mlflow_manifest.json"
PRODUCTION_MODEL_RELATIVE = "artifacts/models/production_sgd.joblib"
BENCHMARK_MODEL_RELATIVE = "artifacts/models/phase1d_champion.joblib"


@dataclass(frozen=True)
class RunImport:
    name: str
    experiment: str
    phase: str
    role: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, Any]
    artifacts: tuple[str, ...]
    source_checksums: dict[str, str]

    @property
    def run_key(self) -> str:
        return stable_key(
            self.experiment,
            self.name,
            self.role,
            self.source_checksums,
        )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _checksums(root: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in relative_paths}


def _flatten_params(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            params.update(_flatten_params(name, value))
        elif isinstance(value, (list, tuple)):
            params[name] = json.dumps(value, separators=(",", ":"))
        elif value is not None:
            params[name] = value
    return params


def _metric_values(candidate: dict[str, Any], prefix: str) -> dict[str, float]:
    threshold = candidate.get("selected_threshold_metrics", candidate.get("threshold_metrics", {}))
    f2_value = threshold.get("f_beta", threshold.get("f2"))
    if f2_value is None:
        raise ValueError(f"stored {prefix} threshold metrics do not contain F2")
    metrics = {
        f"{prefix}_pr_auc": float(candidate["average_precision"]),
        f"{prefix}_roc_auc": float(candidate["roc_auc"]),
        f"{prefix}_precision": float(threshold["precision"]),
        f"{prefix}_recall": float(threshold["recall"]),
        f"{prefix}_f1": float(threshold["f1"]),
        f"{prefix}_f2": float(f2_value),
        f"{prefix}_alert_rate": float(threshold["alert_rate"]),
        f"{prefix}_false_positive_rate": float(threshold["false_positive_rate"]),
        f"{prefix}_fraud_amount_recall": float(threshold["fraud_amount_recall"]),
    }
    for item in candidate.get("top_k", []):
        label = str(float(item["top_k_percentage"])).replace(".", "_")
        metrics[f"{prefix}_top_{label}_percent_recall"] = float(item["recall"])
        metrics[f"{prefix}_top_{label}_percent_fraud_amount_recall"] = float(
            item["fraud_amount_recall"]
        )
    return metrics


def _artifact_paths(root: Path, explicit: list[str], patterns: list[str]) -> tuple[str, ...]:
    paths = {path for path in explicit if (root / path).is_file()}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                paths.add(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def build_import_plan(root: Path, config: MlflowConfig) -> list[RunImport]:
    """Build immutable run payloads solely from existing tracked reports."""

    baseline_metrics_path = root / "artifacts/modeling/baseline_metrics.json"
    phase1d_metrics_path = root / "artifacts/modeling/phase1d_validation_metrics.json"
    final_metrics_path = root / "artifacts/evaluation/final_test_metrics.json"
    baseline_metrics = _read_json(baseline_metrics_path)
    phase1d_metrics = _read_json(phase1d_metrics_path)
    final_metrics = _read_json(final_metrics_path)
    baseline_manifest = _read_json(root / BASELINE_MANIFEST_RELATIVE)
    phase1d_manifest = _read_json(root / PHASE1D_MANIFEST_RELATIVE)
    frozen = _read_json(root / FROZEN_CONFIG_RELATIVE)
    production_manifest = _read_json(root / "artifacts/governance/production_sgd_manifest.json")
    final_manifest = _read_json(root / "artifacts/evaluation/final_test_manifest.json")

    shared_tags = {
        "project": "FraudShield",
        "dataset": "PaySim",
        "synthetic_dataset": True,
        "leakage_safe_features": True,
        "run_origin": "imported_existing_artifacts",
    }
    baseline_sources = (
        "artifacts/modeling/baseline_metrics.json",
        BASELINE_MANIFEST_RELATIVE,
        "configs/modeling.yaml",
        FEATURE_POLICY_RELATIVE,
    )
    phase1d_sources = (
        "artifacts/modeling/phase1d_validation_metrics.json",
        PHASE1D_MANIFEST_RELATIVE,
        FROZEN_CONFIG_RELATIVE,
        TUNING_CONFIG_RELATIVE,
        SANITY_AUDIT_RELATIVE,
        FEATURE_POLICY_RELATIVE,
    )
    final_sources = (
        "artifacts/evaluation/final_test_metrics.json",
        "artifacts/evaluation/final_test_manifest.json",
        "artifacts/evaluation/final_test_evaluation_complete.json",
        "artifacts/governance/production_model_decision.json",
        "artifacts/governance/production_sgd_manifest.json",
        "artifacts/model_card/production_sgd_model_card.md",
        PRODUCTION_CONFIG_RELATIVE,
    )

    baseline_candidate = baseline_metrics["candidates"]["unweighted_logistic"]
    baseline_params = {
        "model_family": "SGDClassifier",
        "model_role": "baseline",
        "feature_count": len(baseline_manifest["feature_names"]),
        "ordered_feature_names": baseline_manifest["feature_names"],
        "selected_threshold": baseline_candidate["selected_threshold"],
        "threshold_selection_metric": "f_beta",
        "model_selection_metric": "average_precision",
        "random_seed": baseline_manifest["training_config"]["random_seed"],
        "training_rows": baseline_manifest["training_rows"],
        "training_fraud_count": baseline_manifest["training_fraud_count"],
        **_flatten_params("hyperparameter", baseline_manifest["model_hyperparameters"]),
    }
    sgd_candidate = phase1d_metrics["candidates"]["phase1d_tuned_sgd_logistic"]
    sgd_config = frozen["best_sgd_configuration"]
    sgd_params = {
        "model_family": "SGDClassifier",
        "model_role": "production_candidate",
        "feature_count": len(feature_names()),
        "ordered_feature_names": feature_names(),
        "selected_threshold": sgd_candidate["selected_threshold"],
        "threshold_selection_metric": "f_beta",
        "model_selection_metric": "average_precision",
        "random_seed": 42,
        "training_rows": phase1d_manifest["training_counts"]["rows"],
        "training_fraud_count": phase1d_manifest["training_counts"]["frauds"],
        "alpha": sgd_config["alpha"],
        "epochs": sgd_config["epochs"],
        "positive_class_weight": sgd_config["positive_class_weight"],
    }
    xgb_candidate = phase1d_metrics["candidates"]["phase1d_tuned_xgboost"]
    xgb_config = frozen["best_xgboost_configuration"]
    xgb_params = {
        "model_family": "XGBoost",
        "model_role": "benchmark",
        "feature_count": len(feature_names()),
        "ordered_feature_names": feature_names(),
        "selected_threshold": xgb_candidate["selected_threshold"],
        "threshold_selection_metric": "f_beta",
        "model_selection_metric": "average_precision",
        "random_seed": 42,
        "training_rows": phase1d_manifest["training_counts"]["rows"],
        "training_fraud_count": phase1d_manifest["training_counts"]["frauds"],
        "boosting_rounds": xgb_config["boosting_rounds_used"],
        **_flatten_params("xgboost", xgb_config["parameters"]),
    }
    final_params = {
        "model_family": "SGDClassifier",
        "model_role": "production",
        "feature_count": len(production_manifest["feature_names"]),
        "ordered_feature_names": production_manifest["feature_names"],
        "frozen_threshold": production_manifest["threshold"],
        "threshold_selection_metric": "validation_f2",
        "model_selection_metric": "operational_governance",
        "random_seed": production_manifest["hyperparameters"]["random_seed"],
        "training_rows": production_manifest["training_rows"],
        "training_fraud_count": production_manifest["training_fraud_count"],
        **_flatten_params("hyperparameter", production_manifest["hyperparameters"]),
    }

    return [
        RunImport(
            name="phase1c_unweighted_logistic",
            experiment=config.experiment_names.development,
            phase="Phase 1C",
            role="baseline",
            parameters=baseline_params,
            metrics=_metric_values(baseline_candidate, "validation"),
            tags={
                **shared_tags,
                "phase": "Phase 1C",
                "model_role": "baseline",
                "test_set_accessed_for_development": False,
                "source_git_commit": baseline_manifest["git_commit_hash"],
            },
            artifacts=_artifact_paths(
                root,
                list(baseline_sources),
                ["artifacts/modeling/plots/*.png"],
            ),
            source_checksums=_checksums(root, baseline_sources),
        ),
        RunImport(
            name="phase1d_tuned_sgd_logistic",
            experiment=config.experiment_names.development,
            phase="Phase 1D",
            role="production_candidate",
            parameters=sgd_params,
            metrics=_metric_values(sgd_candidate, "validation"),
            tags={
                **shared_tags,
                "phase": "Phase 1D",
                "model_role": "production_candidate",
                "test_set_accessed_for_development": False,
                "source_git_commit": phase1d_manifest["git_commit_hash"],
            },
            artifacts=_artifact_paths(
                root,
                list(phase1d_sources),
                ["artifacts/tuning/plots/*.png", "artifacts/modeling/phase1d_plots/*.png"],
            ),
            source_checksums=_checksums(root, phase1d_sources),
        ),
        RunImport(
            name="phase1d_tuned_xgboost",
            experiment=config.experiment_names.development,
            phase="Phase 1D",
            role="benchmark",
            parameters=xgb_params,
            metrics=_metric_values(xgb_candidate, "validation"),
            tags={
                **shared_tags,
                "phase": "Phase 1D",
                "model_role": "benchmark",
                "test_set_accessed_for_development": False,
                "source_git_commit": phase1d_manifest["git_commit_hash"],
                "synthetic_shortcut_likely": True,
            },
            artifacts=_artifact_paths(
                root,
                list(phase1d_sources),
                ["artifacts/tuning/plots/*.png", "artifacts/modeling/phase1d_plots/*.png"],
            ),
            source_checksums=_checksums(root, phase1d_sources),
        ),
        RunImport(
            name="production_sgd_final_holdout",
            experiment=config.experiment_names.final_evaluation,
            phase="Phase 1E",
            role="production",
            parameters=final_params,
            metrics=_metric_values(final_metrics, "final_test"),
            tags={
                **shared_tags,
                "phase": "Phase 1E",
                "model_role": "production",
                "final_holdout": True,
                "final_holdout_evaluated_once": True,
                "threshold_frozen_before_test": True,
                "hyperparameters_frozen_before_test": True,
                "features_frozen_before_test": True,
                "test_used_for_model_selection": False,
                "test_rescored_by_mlflow_phase": False,
                "production_model": "production_sgd_logistic",
                "source_git_commit": final_manifest["git_commit_hash"],
            },
            artifacts=_artifact_paths(
                root,
                list(final_sources),
                ["artifacts/evaluation/plots/*.png", "artifacts/governance/*.json"],
            ),
            source_checksums=_checksums(root, final_sources),
        ),
    ]


def _import_run(
    client: MlflowClient,
    experiment_id: str,
    spec: RunImport,
    root: Path,
) -> tuple[str, bool]:
    tags = {
        **{
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in spec.tags.items()
        },
        RUN_KEY_TAG: spec.run_key,
        "source_artifact_checksums": json.dumps(spec.source_checksums, sort_keys=True),
        "mlflow.runName": spec.name,
    }
    existing = runs_with_key(client, experiment_id, spec.run_key)
    if existing:
        run_id = existing[0].info.run_id
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        return run_id, False
    with mlflow.start_run(experiment_id=experiment_id, run_name=spec.name, tags=tags) as run:
        mlflow.log_params(spec.parameters)
        mlflow.log_metrics(spec.metrics)
        for relative in spec.artifacts:
            mlflow.log_artifact(str(root / relative), artifact_path="imported_source_artifacts")
        return run.info.run_id, True


def _pip_requirements(include_xgboost: bool) -> list[str]:
    packages = ["mlflow", "pandas", "numpy", "joblib", "scikit-learn"]
    if include_xgboost:
        packages.append("xgboost")
    return [f"{name}=={importlib.metadata.version(name)}" for name in packages]


def _register_model(
    *,
    client: MlflowClient,
    root: Path,
    run_id: str,
    model_name: str,
    alias: str,
    model_path: str,
    configuration_path: str,
    wrapper_checksum: str,
    python_model: mlflow.pyfunc.PythonModel,
    signature: Any,
    tags: dict[str, Any],
    include_xgboost: bool,
) -> tuple[str, bool, str]:
    source_model_sha = sha256_file(root / model_path)
    configuration_sha = sha256_file(root / configuration_path)
    version_key = stable_key(source_model_sha, configuration_sha, wrapper_checksum)
    ensure_registered_model(client, model_name)
    existing = model_versions_with_key(client, model_name, version_key)
    created = False
    if existing:
        version = str(existing[0].version)
    else:
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name="inference_model",
                python_model=python_model,
                artifacts={"model_bundle": str(root / model_path)},
                code_paths=[str(root / "src")],
                registered_model_name=model_name,
                signature=signature,
                input_example=synthetic_input_example(),
                pip_requirements=_pip_requirements(include_xgboost),
                metadata={
                    "wrapper_version": WRAPPER_VERSION,
                    "wrapper_checksum": wrapper_checksum,
                    "source_model_sha256": source_model_sha,
                    "source_configuration_sha256": configuration_sha,
                    "ordered_raw_input_fields": feature_names()[:0]
                    + ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"],
                    "ordered_engineered_features": feature_names(),
                },
            )
        if info.registered_model_version is None:
            raise RuntimeError(f"MLflow did not register a version for {model_name}")
        version = str(info.registered_model_version)
        created = True
    version_tags = {
        **tags,
        VERSION_KEY_TAG: version_key,
        "source_model_sha256": source_model_sha,
        "source_configuration_sha256": configuration_sha,
        "wrapper_version": WRAPPER_VERSION,
        "wrapper_checksum": wrapper_checksum,
    }
    set_model_version_tags(client, model_name, version, version_tags)
    client.set_registered_model_alias(model_name, alias, version)
    return version, created, version_key


def _registry_snapshot(
    config: MlflowConfig,
    client: MlflowClient,
    experiment_ids: dict[str, str],
    run_ids: dict[str, str],
    model_versions: dict[str, str],
    source_checksums: dict[str, str],
) -> dict[str, Any]:
    aliases = {
        "production": {
            "alias": config.registry_aliases.production,
            "target_version": client.get_model_version_by_alias(
                config.registered_models.production,
                config.registry_aliases.production,
            ).version,
        },
        "benchmark": {
            "alias": config.registry_aliases.benchmark,
            "target_version": client.get_model_version_by_alias(
                config.registered_models.benchmark,
                config.registry_aliases.benchmark,
            ).version,
        },
    }
    return {
        "tracking_configuration": config.tracked_settings(),
        "experiments": {
            name: {"experiment_id": experiment_ids[name]} for name in sorted(experiment_ids)
        },
        "imported_runs": {
            name: {"run_id": run_ids[name]} for name in sorted(run_ids)
        },
        "registered_models": {
            "production": {
                "name": config.registered_models.production,
                "version": model_versions["production"],
            },
            "benchmark": {
                "name": config.registered_models.benchmark,
                "version": model_versions["benchmark"],
            },
        },
        "aliases": aliases,
        "production_version": model_versions["production"],
        "benchmark_version": model_versions["benchmark"],
        "source_checksums": source_checksums,
        "export_timestamp_utc": utc_timestamp(),
    }


def import_existing_runs(
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Import immutable source reports and package frozen model bundles."""

    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    relative_existing_files(repo_root, REQUIRED_ARTIFACTS)
    config = load_mlflow_config(config_path=config_path, root=repo_root)
    client, _tracking_uri, artifact_uri = configure_local_mlflow(config)
    experiment_ids = {
        config.experiment_names.development: ensure_experiment(
            client, config.experiment_names.development, artifact_uri
        ),
        config.experiment_names.final_evaluation: ensure_experiment(
            client, config.experiment_names.final_evaluation, artifact_uri
        ),
    }
    plan = build_import_plan(repo_root, config)
    run_ids: dict[str, str] = {}
    created_runs = 0
    for spec in plan:
        run_id, created = _import_run(
            client,
            experiment_ids[spec.experiment],
            spec,
            repo_root,
        )
        run_ids[spec.name] = run_id
        created_runs += int(created)

    wrapper_checksum = sha256_file(Path(__file__).with_name("model_wrapper.py"))
    production_manifest = _read_json(
        repo_root / "artifacts/governance/production_sgd_manifest.json"
    )
    phase1d_manifest = _read_json(repo_root / PHASE1D_MANIFEST_RELATIVE)
    production_version, production_created, _ = _register_model(
        client=client,
        root=repo_root,
        run_id=run_ids["production_sgd_final_holdout"],
        model_name=config.registered_models.production,
        alias=config.registry_aliases.production,
        model_path=PRODUCTION_MODEL_RELATIVE,
        configuration_path=PRODUCTION_CONFIG_RELATIVE,
        wrapper_checksum=wrapper_checksum,
        python_model=ProductionSGDPyFuncModel(
            config.risk_levels.medium_threshold,
            config.risk_levels.high_threshold,
        ),
        signature=production_signature(),
        include_xgboost=False,
        tags={
            "role": "production",
            "model_family": "SGDClassifier",
            "threshold_source": "phase1d_validation_f2",
            "operational_threshold": config.risk_levels.high_threshold,
            "synthetic_dataset": True,
            "test_evaluated": True,
            "test_used_for_selection": False,
            "production_decision": "operational_governance",
            "source_git_commit": production_manifest["git_commit_hash"],
        },
    )
    benchmark_version, benchmark_created, _ = _register_model(
        client=client,
        root=repo_root,
        run_id=run_ids["phase1d_tuned_xgboost"],
        model_name=config.registered_models.benchmark,
        alias=config.registry_aliases.benchmark,
        model_path=BENCHMARK_MODEL_RELATIVE,
        configuration_path=TUNING_CONFIG_RELATIVE,
        wrapper_checksum=wrapper_checksum,
        python_model=BenchmarkXGBoostPyFuncModel(),
        signature=benchmark_signature(),
        include_xgboost=True,
        tags={
            "role": "benchmark",
            "model_family": "XGBoost",
            "deployable_production_model": False,
            "synthetic_shortcut_likely": True,
            "final_test_evaluated": False,
            "test_used_for_selection": False,
            "source_git_commit": phase1d_manifest["git_commit_hash"],
        },
    )
    model_versions = {
        "production": production_version,
        "benchmark": benchmark_version,
    }
    all_source_checksums = {
        relative: sha256_file(repo_root / relative)
        for relative in sorted({path for spec in plan for path in spec.source_checksums})
    }
    snapshot = _registry_snapshot(
        config,
        client,
        experiment_ids,
        run_ids,
        model_versions,
        all_source_checksums,
    )
    write_json(repo_root / SNAPSHOT_RELATIVE, snapshot)
    write_json(
        repo_root / MLFLOW_MANIFEST_RELATIVE,
        {
            "mlflow_version": mlflow.__version__,
            "python_version": importlib.metadata.version("fraudshield-mlops")
            and __import__("platform").python_version(),
            "source_git_commit": _git_commit(repo_root),
            "backend_type": "SQLite",
            "artifact_store_type": "local filesystem",
            "database_relative_path": config.storage.backend_database.as_posix(),
            "artifact_root_relative_path": config.storage.artifact_root.as_posix(),
            "imported_source_artifacts": all_source_checksums,
            "models_retrained": False,
            "metrics_recomputed": False,
            "raw_data_accessed": False,
            "train_parquet_accessed": False,
            "validation_parquet_accessed": False,
            "test_parquet_accessed": False,
            "registry_verification_status": "pending verify_registry",
            "created_runs": created_runs,
            "created_model_versions": int(production_created) + int(benchmark_created),
            "export_timestamp_utc": utc_timestamp(),
        },
    )
    summary = {
        "experiment_ids": experiment_ids,
        "run_ids": run_ids,
        "model_versions": model_versions,
        "created_runs": created_runs,
        "created_model_versions": int(production_created) + int(benchmark_created),
        "test_parquet_accessed": False,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = import_existing_runs()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
