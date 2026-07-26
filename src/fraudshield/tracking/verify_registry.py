"""Verify the local FraudShield MLflow experiments and model registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from fraudshield.data.config import repository_root
from fraudshield.tracking.config import load_mlflow_config
from fraudshield.tracking.log_existing_runs import (
    MLFLOW_MANIFEST_RELATIVE,
    SNAPSHOT_RELATIVE,
    build_import_plan,
)
from fraudshield.tracking.mlflow_setup import (
    VERSION_KEY_TAG,
    configure_local_mlflow,
    install_prohibited_data_guard,
    model_versions_with_key,
    runs_with_key,
    utc_timestamp,
    write_json,
)
from fraudshield.tracking.model_wrapper import OUTPUT_COLUMNS, synthetic_input_example


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def verify_registry(
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Verify experiments, imports, aliases, prediction behavior, and stored metrics."""

    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    config = load_mlflow_config(config_path=config_path, root=repo_root)
    client, _tracking_uri, _artifact_uri = configure_local_mlflow(config)
    plan = build_import_plan(repo_root, config)

    experiment_ids: dict[str, str] = {}
    run_ids: dict[str, str] = {}
    for experiment_name in (
        config.experiment_names.development,
        config.experiment_names.final_evaluation,
    ):
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"missing MLflow experiment: {experiment_name}")
        experiment_ids[experiment_name] = experiment.experiment_id
    for spec in plan:
        equivalent = runs_with_key(client, experiment_ids[spec.experiment], spec.run_key)
        if len(equivalent) != 1:
            raise RuntimeError(f"expected exactly one imported run for {spec.name}")
        run = equivalent[0]
        run_ids[spec.name] = run.info.run_id
        for key, expected in spec.metrics.items():
            actual = run.data.metrics.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"stored MLflow metric mismatch for {spec.name}/{key}: "
                    f"{actual!r} != {expected!r}"
                )

    production = client.get_registered_model(config.registered_models.production)
    benchmark = client.get_registered_model(config.registered_models.benchmark)
    if production.name != config.registered_models.production:
        raise RuntimeError("production registered model identity mismatch")
    if benchmark.name != config.registered_models.benchmark:
        raise RuntimeError("benchmark registered model identity mismatch")

    production_alias = client.get_model_version_by_alias(
        config.registered_models.production,
        config.registry_aliases.production,
    )
    benchmark_alias = client.get_model_version_by_alias(
        config.registered_models.benchmark,
        config.registry_aliases.benchmark,
    )
    for name, version in (
        (config.registered_models.production, production_alias),
        (config.registered_models.benchmark, benchmark_alias),
    ):
        version_key = version.tags.get(VERSION_KEY_TAG)
        if not version_key:
            raise RuntimeError(f"model version lacks duplicate-prevention key: {name}")
        if len(model_versions_with_key(client, name, version_key)) != 1:
            raise RuntimeError(f"expected exactly one equivalent model version: {name}")

    model_uri = (
        f"models:/{config.registered_models.production}@{config.registry_aliases.production}"
    )
    loaded = mlflow.pyfunc.load_model(model_uri)
    example = synthetic_input_example()
    prediction = loaded.predict(example)
    if list(prediction.columns) != list(OUTPUT_COLUMNS):
        raise RuntimeError("production PyFunc output schema is incorrect")
    if not prediction.index.equals(example.index):
        raise RuntimeError("production PyFunc did not preserve row order")
    if not np.isfinite(prediction["fraud_score"].to_numpy()).all():
        raise RuntimeError("production PyFunc returned non-finite scores")
    if not np.equal(
        prediction["threshold"].to_numpy(),
        config.risk_levels.high_threshold,
    ).all():
        raise RuntimeError("production PyFunc threshold is not frozen")

    final_spec = next(spec for spec in plan if spec.name == "production_sgd_final_holdout")
    final_run = client.get_run(run_ids[final_spec.name])
    no_rescore_tag = final_run.data.tags.get("test_rescored_by_mlflow_phase", "")
    if no_rescore_tag.lower() not in {"false", "0"}:
        raise RuntimeError("final holdout run does not record the no-rescore guarantee")
    for key, expected in final_spec.metrics.items():
        if final_run.data.metrics.get(key) != expected:
            raise RuntimeError(f"final holdout metric changed in MLflow: {key}")

    tracking_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repo_root / "src/fraudshield/tracking").glob("*.py")
    )
    prohibited_training_calls = ("." + "fit(", "." + "partial_fit(")
    if any(call in tracking_sources for call in prohibited_training_calls):
        raise RuntimeError("Phase 2A tracking package contains a training call")

    manifest_path = repo_root / MLFLOW_MANIFEST_RELATIVE
    manifest = _read_json(manifest_path)
    guarantees = {
        "models_retrained": False,
        "metrics_recomputed": False,
        "raw_data_accessed": False,
        "train_parquet_accessed": False,
        "validation_parquet_accessed": False,
        "test_parquet_accessed": False,
    }
    for key, expected in guarantees.items():
        if manifest.get(key) is not expected:
            raise RuntimeError(f"MLflow manifest guarantee failed: {key}")
    manifest["registry_verification_status"] = "verified"
    manifest["verification_timestamp_utc"] = utc_timestamp()
    write_json(manifest_path, manifest)

    snapshot = _read_json(repo_root / SNAPSHOT_RELATIVE)
    result = {
        "status": "verified",
        "experiment_ids": experiment_ids,
        "run_ids": run_ids,
        "production_model": config.registered_models.production,
        "production_version": str(production_alias.version),
        "champion_alias_target": str(production_alias.version),
        "benchmark_model": config.registered_models.benchmark,
        "benchmark_version": str(benchmark_alias.version),
        "challenger_alias_target": str(benchmark_alias.version),
        "prediction_rows": len(prediction),
        "prediction_columns": list(prediction.columns),
        "prediction_thresholds": sorted(set(prediction["threshold"].tolist())),
        "duplicate_equivalent_runs": 0,
        "duplicate_equivalent_model_versions": 0,
        "snapshot_export_timestamp_utc": snapshot["export_timestamp_utc"],
        "models_retrained": False,
        "metrics_recomputed": False,
        "prohibited_data_accessed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(verify_registry(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
