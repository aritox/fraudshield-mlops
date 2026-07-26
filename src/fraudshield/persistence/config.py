"""Validated, secret-safe database configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import URL, make_url

from fraudshield.data.config import repository_root

CONFIG_RELATIVE_PATH = Path("configs/database.yaml")


@dataclass(frozen=True)
class PoolConfig:
    connect_timeout_seconds: int
    pool_size: int
    max_overflow: int
    pool_timeout_seconds: int
    pool_recycle_seconds: int
    pool_pre_ping: bool


@dataclass(frozen=True)
class EnvironmentConfig:
    database_url_variable: str
    postgres_user_variable: str
    postgres_password_variable: str
    postgres_database_variable: str
    postgres_port_variable: str


@dataclass(frozen=True)
class PersistencePolicy:
    required_for_predictions: bool
    store_model_inputs: bool
    maximum_outcome_batch_size: int
    request_id_header: str
    replay_header: str
    payload_hash_algorithm: str


@dataclass(frozen=True)
class RetentionPolicy:
    local_demo_only: bool
    automatic_deletion_enabled: bool


@dataclass(frozen=True)
class DatabaseConfig:
    driver: str
    host: str
    port: int
    name: str
    pool: PoolConfig
    environment: EnvironmentConfig
    persistence: PersistencePolicy
    retention: RetentionPolicy
    repository_root: Path
    config_path: Path

    def url(self, environ: dict[str, str] | None = None) -> URL:
        values = os.environ if environ is None else environ
        override = values.get(self.environment.database_url_variable, "").strip()
        if override:
            try:
                parsed = make_url(override)
            except Exception as error:
                raise ValueError("Database URL override is invalid") from error
            if parsed.drivername != self.driver:
                raise ValueError("Database URL override must use postgresql+psycopg")
            return parsed
        user = values.get(self.environment.postgres_user_variable, "").strip()
        password = values.get(self.environment.postgres_password_variable, "")
        database = values.get(self.environment.postgres_database_variable, self.name).strip()
        port_text = values.get(self.environment.postgres_port_variable, str(self.port)).strip()
        if not user or not password or not database:
            raise ValueError("PostgreSQL credentials are not configured")
        try:
            port = int(port_text)
        except ValueError as error:
            raise ValueError("PostgreSQL port is invalid") from error
        if not 1 <= port <= 65535:
            raise ValueError("PostgreSQL port is invalid")
        return URL.create(
            self.driver,
            username=user,
            password=password,
            host=self.host,
            port=port,
            database=database,
        )

    def safe_settings(self) -> dict[str, Any]:
        return {
            "driver": self.driver,
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "pool_size": self.pool.pool_size,
            "max_overflow": self.pool.max_overflow,
            "persistence_required": self.persistence.required_for_predictions,
        }


def _section(raw: dict[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Database configuration section {name!r} is invalid")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def load_database_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> DatabaseConfig:
    repo_root = (root or repository_root()).resolve()
    path = (config_path or repo_root / CONFIG_RELATIVE_PATH).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("Database configuration must be inside the repository") from error
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(raw) != {"database", "environment", "persistence", "retention"}:
        raise ValueError("Database configuration sections are invalid")
    database = _section(
        raw,
        "database",
        {
            "driver",
            "host",
            "port",
            "name",
            "connect_timeout_seconds",
            "pool_size",
            "max_overflow",
            "pool_timeout_seconds",
            "pool_recycle_seconds",
            "pool_pre_ping",
        },
    )
    environment = _section(
        raw,
        "environment",
        {
            "database_url_variable",
            "postgres_user_variable",
            "postgres_password_variable",
            "postgres_database_variable",
            "postgres_port_variable",
        },
    )
    persistence = _section(
        raw,
        "persistence",
        {
            "required_for_predictions",
            "store_model_inputs",
            "maximum_outcome_batch_size",
            "request_id_header",
            "replay_header",
            "payload_hash_algorithm",
        },
    )
    retention = _section(
        raw,
        "retention",
        {"local_demo_only", "automatic_deletion_enabled"},
    )
    if database["driver"] != "postgresql+psycopg":
        raise ValueError("Only postgresql+psycopg is supported")
    if not str(database["host"]).strip() or not str(database["name"]).strip():
        raise ValueError("Database host and name must be non-empty")
    port = int(database["port"])
    if not 1 <= port <= 65535:
        raise ValueError("Database port is invalid")
    numeric = {
        key: int(database[key])
        for key in (
            "connect_timeout_seconds",
            "pool_size",
            "max_overflow",
            "pool_timeout_seconds",
            "pool_recycle_seconds",
        )
    }
    if numeric["connect_timeout_seconds"] <= 0 or numeric["pool_timeout_seconds"] <= 0:
        raise ValueError("Database connection and pool timeouts must be positive")
    if any(numeric[key] < 0 for key in ("pool_size", "max_overflow", "pool_recycle_seconds")):
        raise ValueError("Database pool values must be non-negative")
    if not _strict_bool(persistence["required_for_predictions"], "required_for_predictions"):
        raise ValueError("Prediction persistence must be required")
    if not _strict_bool(persistence["store_model_inputs"], "store_model_inputs"):
        raise ValueError("Model inputs must be stored for the local audit demo")
    maximum_outcomes = int(persistence["maximum_outcome_batch_size"])
    if maximum_outcomes <= 0:
        raise ValueError("Maximum outcome batch size must be positive")
    if str(persistence["payload_hash_algorithm"]).lower() != "sha256":
        raise ValueError("Payload hashing must use SHA-256")
    if not _strict_bool(retention["local_demo_only"], "local_demo_only"):
        raise ValueError("Retention policy must explicitly be local-demo-only")
    if _strict_bool(retention["automatic_deletion_enabled"], "automatic_deletion_enabled"):
        raise ValueError("Automatic deletion is not supported in Phase 2C")
    pool = PoolConfig(
        **numeric,
        pool_pre_ping=_strict_bool(database["pool_pre_ping"], "pool_pre_ping"),
    )
    return DatabaseConfig(
        driver=str(database["driver"]),
        host=str(database["host"]),
        port=port,
        name=str(database["name"]),
        pool=pool,
        environment=EnvironmentConfig(**{key: str(value) for key, value in environment.items()}),
        persistence=PersistencePolicy(
            required_for_predictions=True,
            store_model_inputs=True,
            maximum_outcome_batch_size=maximum_outcomes,
            request_id_header=str(persistence["request_id_header"]),
            replay_header=str(persistence["replay_header"]),
            payload_hash_algorithm="sha256",
        ),
        retention=RetentionPolicy(local_demo_only=True, automatic_deletion_enabled=False),
        repository_root=repo_root,
        config_path=path,
    )
