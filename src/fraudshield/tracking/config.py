"""Validated repository-relative MLflow configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from fraudshield.data.config import repository_root

CONFIG_RELATIVE_PATH = Path("configs/mlflow.yaml")
FROZEN_OPERATIONAL_THRESHOLD = 0.98310834


@dataclass(frozen=True)
class ExperimentNames:
    development: str
    final_evaluation: str


@dataclass(frozen=True)
class RegisteredModels:
    production: str
    benchmark: str


@dataclass(frozen=True)
class RegistryAliases:
    production: str
    benchmark: str


@dataclass(frozen=True)
class RiskLevels:
    medium_threshold: float
    high_threshold: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class StorageConfig:
    backend_database: Path
    artifact_root: Path


@dataclass(frozen=True)
class MlflowConfig:
    experiment_names: ExperimentNames
    registered_models: RegisteredModels
    registry_aliases: RegistryAliases
    risk_levels: RiskLevels
    server: ServerConfig
    storage: StorageConfig
    repository_root: Path
    config_path: Path

    def tracked_settings(self) -> dict[str, Any]:
        """Return settings suitable for tracked JSON without machine-specific paths."""

        payload = asdict(self)
        payload["storage"] = {
            "backend_database": self.storage.backend_database.as_posix(),
            "artifact_root": self.storage.artifact_root.as_posix(),
        }
        payload["repository_root"] = "."
        payload["config_path"] = self.config_path.relative_to(
            self.repository_root
        ).as_posix()
        return payload


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"mlflow config section {key!r} must be a mapping")
    return value


def _relative_path(value: Any, name: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a repository-relative path")
    return path


def load_mlflow_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> MlflowConfig:
    """Load and strictly validate the local MLflow configuration."""

    repo_root = (root or repository_root()).resolve()
    path = (config_path or repo_root / CONFIG_RELATIVE_PATH).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("MLflow configuration must be inside the repository") from error
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    expected_sections = {
        "experiment_names",
        "registered_models",
        "registry_aliases",
        "risk_levels",
        "server",
        "storage",
    }
    if set(raw) != expected_sections:
        raise ValueError("MLflow configuration sections do not match the required schema")

    experiments = _require_mapping(raw, "experiment_names")
    models = _require_mapping(raw, "registered_models")
    aliases = _require_mapping(raw, "registry_aliases")
    risks = _require_mapping(raw, "risk_levels")
    server = _require_mapping(raw, "server")
    storage = _require_mapping(raw, "storage")

    expected_values = {
        "development": "FraudShield-Development",
        "final_evaluation": "FraudShield-Final-Evaluation",
        "production_model": "fraudshield-production-sgd",
        "benchmark_model": "fraudshield-xgboost-benchmark",
        "production_alias": "champion",
        "benchmark_alias": "challenger",
    }
    if experiments.get("development") != expected_values["development"]:
        raise ValueError("unsupported development experiment name")
    if experiments.get("final_evaluation") != expected_values["final_evaluation"]:
        raise ValueError("unsupported final-evaluation experiment name")
    if models.get("production") != expected_values["production_model"]:
        raise ValueError("unsupported production registered-model name")
    if models.get("benchmark") != expected_values["benchmark_model"]:
        raise ValueError("unsupported benchmark registered-model name")
    if aliases.get("production") != expected_values["production_alias"]:
        raise ValueError("production alias must be champion")
    if aliases.get("benchmark") != expected_values["benchmark_alias"]:
        raise ValueError("benchmark alias must be challenger")

    medium = float(risks.get("medium_threshold", -1))
    high = float(risks.get("high_threshold", -1))
    if not 0 <= medium < high <= 1:
        raise ValueError("risk thresholds must satisfy 0 <= medium < high <= 1")
    if high != FROZEN_OPERATIONAL_THRESHOLD:
        raise ValueError("high-risk threshold must match the frozen SGD threshold")
    host = str(server.get("host", ""))
    port = int(server.get("port", 0))
    if host != "127.0.0.1":
        raise ValueError("MLflow server must bind only to 127.0.0.1")
    if not 1 <= port <= 65535:
        raise ValueError("MLflow server port must be between 1 and 65535")

    backend_database = _relative_path(storage.get("backend_database"), "backend_database")
    artifact_root = _relative_path(storage.get("artifact_root"), "artifact_root")
    if backend_database.suffix != ".db":
        raise ValueError("backend_database must be a SQLite .db path")
    if backend_database != Path("artifacts/mlflow/mlflow.db"):
        raise ValueError("unsupported MLflow backend database path")
    if artifact_root != Path("artifacts/mlflow/artifacts"):
        raise ValueError("unsupported MLflow artifact root")

    return MlflowConfig(
        experiment_names=ExperimentNames(**experiments),
        registered_models=RegisteredModels(**models),
        registry_aliases=RegistryAliases(**aliases),
        risk_levels=RiskLevels(medium_threshold=medium, high_threshold=high),
        server=ServerConfig(host=host, port=port),
        storage=StorageConfig(
            backend_database=backend_database,
            artifact_root=artifact_root,
        ),
        repository_root=repo_root,
        config_path=path,
    )
