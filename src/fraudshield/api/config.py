"""Validated configuration for the local FraudShield inference API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fraudshield.data.config import repository_root
from fraudshield.tracking.config import FROZEN_OPERATIONAL_THRESHOLD, load_mlflow_config

CONFIG_RELATIVE_PATH = Path("configs/api.yaml")
PRODUCTION_MODEL_NAME = "fraudshield-production-sgd"
PRODUCTION_ALIAS = "champion"
PRODUCTION_MODEL_URI = f"models:/{PRODUCTION_MODEL_NAME}@{PRODUCTION_ALIAS}"


@dataclass(frozen=True)
class ApplicationConfig:
    title: str
    description: str
    version: str


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class ModelConfig:
    uri: str
    registered_name: str
    alias: str
    expected_family: str
    expected_threshold: float
    load_on_startup: bool


@dataclass(frozen=True)
class InferenceConfig:
    maximum_batch_size: int
    medium_risk_threshold: float
    high_risk_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_raw_inputs: bool
    include_request_id: bool
    include_latency: bool


@dataclass(frozen=True)
class RouteConfig:
    docs_url: str
    redoc_url: str
    openapi_url: str


@dataclass(frozen=True)
class ApiConfig:
    application: ApplicationConfig
    server: ServerConfig
    model: ModelConfig
    inference: InferenceConfig
    logging: LoggingConfig
    api: RouteConfig
    repository_root: Path
    config_path: Path


def _section(raw: dict[str, Any], name: str, expected: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"API configuration section {name!r} does not match its schema")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def load_api_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> ApiConfig:
    """Load and strictly validate repository-local API settings."""

    repo_root = (root or repository_root()).resolve()
    path = (config_path or repo_root / CONFIG_RELATIVE_PATH).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("API configuration must be inside the repository") from error
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(raw) != {"application", "server", "model", "inference", "logging", "api"}:
        raise ValueError("API configuration sections do not match the required schema")

    application = _section(raw, "application", {"title", "description", "version"})
    server = _section(raw, "server", {"host", "port"})
    model = _section(
        raw,
        "model",
        {
            "uri",
            "registered_name",
            "alias",
            "expected_family",
            "expected_threshold",
            "load_on_startup",
        },
    )
    inference = _section(
        raw,
        "inference",
        {"maximum_batch_size", "medium_risk_threshold", "high_risk_threshold"},
    )
    logging = _section(
        raw,
        "logging",
        {"level", "log_raw_inputs", "include_request_id", "include_latency"},
    )
    routes = _section(raw, "api", {"docs_url", "redoc_url", "openapi_url"})

    if any(not str(application[key]).strip() for key in application):
        raise ValueError("application title, description, and version must be non-empty")
    host = str(server["host"])
    port = int(server["port"])
    if host != "127.0.0.1":
        raise ValueError("Phase 2B API must bind only to 127.0.0.1")
    if not 1 <= port <= 65535:
        raise ValueError("API server port must be between 1 and 65535")

    model_uri = str(model["uri"])
    if model_uri != PRODUCTION_MODEL_URI or "xgboost" in model_uri.lower():
        raise ValueError("API model URI must reference only the production SGD champion")
    if model["registered_name"] != PRODUCTION_MODEL_NAME:
        raise ValueError("API registered model must be fraudshield-production-sgd")
    if model["alias"] != PRODUCTION_ALIAS:
        raise ValueError("API model alias must be champion")
    if model["expected_family"] != "SGDClassifier":
        raise ValueError("API model family must be SGDClassifier")
    expected_threshold = float(model["expected_threshold"])
    if expected_threshold != FROZEN_OPERATIONAL_THRESHOLD:
        raise ValueError("API expected threshold does not match the frozen threshold")

    maximum_batch_size = int(inference["maximum_batch_size"])
    medium_threshold = float(inference["medium_risk_threshold"])
    high_threshold = float(inference["high_risk_threshold"])
    if not 1 <= maximum_batch_size <= 10_000:
        raise ValueError("maximum_batch_size must be between 1 and 10000")
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("risk thresholds must satisfy 0 <= medium < high <= 1")
    if high_threshold != expected_threshold:
        raise ValueError("high-risk and frozen model thresholds must match")
    mlflow_config = load_mlflow_config(root=repo_root)
    if medium_threshold != mlflow_config.risk_levels.medium_threshold:
        raise ValueError("API medium-risk threshold differs from the MLflow model policy")
    if high_threshold != mlflow_config.risk_levels.high_threshold:
        raise ValueError("API high-risk threshold differs from the MLflow model policy")

    level = str(logging["level"]).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("unsupported API logging level")
    if _strict_bool(logging["log_raw_inputs"], "log_raw_inputs"):
        raise ValueError("raw input logging must remain disabled")
    route_values = {key: str(value) for key, value in routes.items()}
    if any(not value.startswith("/") for value in route_values.values()):
        raise ValueError("API documentation paths must begin with /")
    if len(set(route_values.values())) != len(route_values):
        raise ValueError("API documentation paths must be distinct")

    return ApiConfig(
        application=ApplicationConfig(**{key: str(value) for key, value in application.items()}),
        server=ServerConfig(host=host, port=port),
        model=ModelConfig(
            uri=model_uri,
            registered_name=str(model["registered_name"]),
            alias=str(model["alias"]),
            expected_family=str(model["expected_family"]),
            expected_threshold=expected_threshold,
            load_on_startup=_strict_bool(model["load_on_startup"], "load_on_startup"),
        ),
        inference=InferenceConfig(
            maximum_batch_size=maximum_batch_size,
            medium_risk_threshold=medium_threshold,
            high_risk_threshold=high_threshold,
        ),
        logging=LoggingConfig(
            level=level,
            log_raw_inputs=False,
            include_request_id=_strict_bool(
                logging["include_request_id"], "include_request_id"
            ),
            include_latency=_strict_bool(logging["include_latency"], "include_latency"),
        ),
        api=RouteConfig(**route_values),
        repository_root=repo_root,
        config_path=path,
    )
