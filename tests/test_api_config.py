"""Tests for strict local API configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraudshield.api.config import PRODUCTION_MODEL_URI, load_api_config


def _write_configs(
    root: Path,
    *,
    port: int = 8000,
    maximum_batch_size: int = 1000,
    threshold: float = 0.98310834,
    alias: str = "champion",
    model_uri: str = PRODUCTION_MODEL_URI,
) -> Path:
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "mlflow.yaml").write_text(
        """experiment_names:
  development: FraudShield-Development
  final_evaluation: FraudShield-Final-Evaluation
registered_models:
  production: fraudshield-production-sgd
  benchmark: fraudshield-xgboost-benchmark
registry_aliases:
  production: champion
  benchmark: challenger
risk_levels:
  medium_threshold: 0.50
  high_threshold: 0.98310834
server:
  host: 127.0.0.1
  port: 5000
storage:
  backend_database: artifacts/mlflow/mlflow.db
  artifact_root: artifacts/mlflow/artifacts
""",
        encoding="utf-8",
    )
    api_path = configs / "api.yaml"
    api_path.write_text(
        f"""application:
  title: FraudShield Inference API
  description: Real-time fraud scoring with the registered production SGD model
  version: 0.1.0
server:
  host: 127.0.0.1
  port: {port}
model:
  uri: {model_uri}
  registered_name: fraudshield-production-sgd
  alias: {alias}
  expected_family: SGDClassifier
  expected_threshold: {threshold}
  load_on_startup: true
inference:
  maximum_batch_size: {maximum_batch_size}
  medium_risk_threshold: 0.50
  high_risk_threshold: {threshold}
logging:
  level: INFO
  log_raw_inputs: false
  include_request_id: true
  include_latency: true
api:
  docs_url: /docs
  redoc_url: /redoc
  openapi_url: /openapi.json
""",
        encoding="utf-8",
    )
    return api_path


def test_valid_api_configuration(tmp_path: Path) -> None:
    config = load_api_config(_write_configs(tmp_path), root=tmp_path)

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8000
    assert config.model.uri == PRODUCTION_MODEL_URI
    assert config.model.alias == "champion"
    assert config.model.expected_threshold == 0.98310834
    assert config.inference.maximum_batch_size == 1000


@pytest.mark.parametrize(
    "overrides",
    [
        {"port": 0},
        {"maximum_batch_size": 0},
        {"threshold": 0.9},
        {"alias": "challenger"},
        {"model_uri": "models:/fraudshield-xgboost-benchmark@challenger"},
    ],
)
def test_invalid_api_configuration_is_rejected(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        load_api_config(_write_configs(tmp_path, **overrides), root=tmp_path)
