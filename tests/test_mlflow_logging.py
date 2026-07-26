"""Tests for immutable metric import and run idempotency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraudshield.tracking.config import load_mlflow_config
from fraudshield.tracking.log_existing_runs import RunImport, _import_run, _metric_values
from fraudshield.tracking.mlflow_setup import (
    configure_local_mlflow,
    ensure_experiment,
    install_prohibited_data_guard,
    runs_with_key,
)
from test_mlflow_config import _write_config


def test_metrics_are_imported_from_stored_json_without_recomputation(tmp_path: Path) -> None:
    stored = {
        "average_precision": 0.25,
        "roc_auc": 0.75,
        "threshold_metrics": {
            "precision": 0.2,
            "recall": 0.4,
            "f1": 0.266,
            "f2": 0.333,
            "alert_rate": 0.01,
            "false_positive_rate": 0.009,
            "fraud_amount_recall": 0.5,
        },
        "top_k": [
            {"top_k_percentage": 0.1, "recall": 0.3, "fraud_amount_recall": 0.45}
        ],
    }
    path = tmp_path / "stored_metrics.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    metrics = _metric_values(loaded, "final_test")

    assert metrics["final_test_pr_auc"] == 0.25
    assert metrics["final_test_f2"] == 0.333
    assert metrics["final_test_top_0_1_percent_recall"] == 0.3


def test_run_import_is_idempotent_and_keeps_required_tags(tmp_path: Path) -> None:
    config = load_mlflow_config(_write_config(tmp_path), root=tmp_path)
    client, _, artifact_uri = configure_local_mlflow(config)
    experiment_id = ensure_experiment(
        client,
        config.experiment_names.development,
        artifact_uri,
    )
    spec = RunImport(
        name="synthetic_import",
        experiment=config.experiment_names.development,
        phase="test",
        role="benchmark",
        parameters={"model_family": "synthetic"},
        metrics={"validation_pr_auc": 0.25},
        tags={
            "project": "FraudShield",
            "run_origin": "imported_existing_artifacts",
            "test_set_accessed_for_development": False,
        },
        artifacts=(),
        source_checksums={"synthetic_metrics.json": "abc"},
    )

    first_id, first_created = _import_run(client, experiment_id, spec, tmp_path)
    second_id, second_created = _import_run(client, experiment_id, spec, tmp_path)

    assert first_created is True
    assert second_created is False
    assert first_id == second_id
    assert len(runs_with_key(client, experiment_id, spec.run_key)) == 1
    run = client.get_run(first_id)
    assert run.data.tags["project"] == "FraudShield"
    assert run.data.tags["run_origin"] == "imported_existing_artifacts"


def test_filesystem_guard_blocks_prohibited_data(tmp_path: Path) -> None:
    protected = tmp_path / "data" / "processed" / "test.parquet"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"synthetic fixture only")
    install_prohibited_data_guard(tmp_path)

    with pytest.raises(PermissionError, match="prohibited data access"):
        protected.read_bytes()
