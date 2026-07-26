"""Tests for repository-relative local MLflow configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraudshield.tracking.config import FROZEN_OPERATIONAL_THRESHOLD, load_mlflow_config
from fraudshield.tracking.mlflow_setup import artifact_root_uri, sqlite_tracking_uri


def _write_config(root: Path, medium: float = 0.5, high: float = 0.98310834) -> Path:
    path = root / "configs" / "mlflow.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""experiment_names:
  development: FraudShield-Development
  final_evaluation: FraudShield-Final-Evaluation
registered_models:
  production: fraudshield-production-sgd
  benchmark: fraudshield-xgboost-benchmark
registry_aliases:
  production: champion
  benchmark: challenger
risk_levels:
  medium_threshold: {medium}
  high_threshold: {high}
server:
  host: 127.0.0.1
  port: 5000
storage:
  backend_database: artifacts/mlflow/mlflow.db
  artifact_root: artifacts/mlflow/artifacts
""",
        encoding="utf-8",
    )
    return path


def test_relative_configuration_and_local_uris(tmp_path: Path) -> None:
    config = load_mlflow_config(_write_config(tmp_path), root=tmp_path)

    assert config.risk_levels.high_threshold == FROZEN_OPERATIONAL_THRESHOLD
    assert config.storage.backend_database == Path("artifacts/mlflow/mlflow.db")
    assert sqlite_tracking_uri(config).startswith("sqlite:///" )
    assert sqlite_tracking_uri(config).endswith("/artifacts/mlflow/mlflow.db")
    assert artifact_root_uri(config).startswith("file:///")
    assert config.tracked_settings()["repository_root"] == "."
    assert str(tmp_path) not in str(config.tracked_settings())


@pytest.mark.parametrize(
    ("medium", "high"),
    [(0.99, 0.98310834), (0.5, 0.9)],
)
def test_invalid_or_unfrozen_thresholds_are_rejected(
    tmp_path: Path,
    medium: float,
    high: float,
) -> None:
    with pytest.raises(ValueError):
        load_mlflow_config(_write_config(tmp_path, medium, high), root=tmp_path)
