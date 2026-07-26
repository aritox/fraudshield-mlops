"""Shared local MLflow setup, idempotency, and safety helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from fraudshield.tracking.config import MlflowConfig

RUN_KEY_TAG = "fraudshield.run_key"
VERSION_KEY_TAG = "fraudshield.version_key"
_GUARDED_ROOTS: set[str] = set()


def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 Z form."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum without interpreting file contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: Any) -> str:
    """Build a deterministic identifier from JSON-serializable values."""

    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic tracked JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sqlite_tracking_uri(config: MlflowConfig) -> str:
    """Return the absolute runtime SQLite URI for the configured local database."""

    database = (config.repository_root / config.storage.backend_database).resolve()
    return f"sqlite:///{database.as_posix()}"


def artifact_root_uri(config: MlflowConfig) -> str:
    """Return the absolute runtime file URI for the local artifact store."""

    return (config.repository_root / config.storage.artifact_root).resolve().as_uri()


def configure_local_mlflow(config: MlflowConfig) -> tuple[MlflowClient, str, str]:
    """Create local storage directories and configure tracking and registry access."""

    database = config.repository_root / config.storage.backend_database
    artifact_root = config.repository_root / config.storage.artifact_root
    database.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = sqlite_tracking_uri(config)
    artifact_uri = artifact_root_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    return client, tracking_uri, artifact_uri


def ensure_experiment(client: MlflowClient, name: str, artifact_uri: str) -> str:
    """Create an experiment once and return its stable ID."""

    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(name=name, artifact_location=f"{artifact_uri}/{name}")


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create a registered model if it does not already exist."""

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def runs_with_key(client: MlflowClient, experiment_id: str, run_key: str) -> list[Any]:
    """Return equivalent active runs, failing if historical duplication exists."""

    escaped = run_key.replace("'", "\\'")
    runs = list(
        client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.`{RUN_KEY_TAG}` = '{escaped}'",
            max_results=100,
        )
    )
    if len(runs) > 1:
        raise RuntimeError(f"duplicate equivalent MLflow runs found for key {run_key}")
    return runs


def model_versions_with_key(
    client: MlflowClient,
    model_name: str,
    version_key: str,
) -> list[Any]:
    """Return equivalent versions, failing if duplicate versions exist."""

    versions = [
        version
        for version in client.search_model_versions(f"name = '{model_name}'")
        if version.tags.get(VERSION_KEY_TAG) == version_key
    ]
    if len(versions) > 1:
        raise RuntimeError(
            f"duplicate equivalent model versions found for {model_name}: {version_key}"
        )
    return versions


def set_model_version_tags(
    client: MlflowClient,
    name: str,
    version: str,
    tags: dict[str, Any],
) -> None:
    """Apply string-valued tags to a model version."""

    for key, value in tags.items():
        tag_value = str(value).lower() if isinstance(value, bool) else str(value)
        client.set_model_version_tag(name, version, key, tag_value)


def install_prohibited_data_guard(root: Path) -> None:
    """Fail closed if this process attempts to open raw or protected Parquet data."""

    resolved_root = os.path.normcase(os.path.abspath(root))
    if resolved_root in _GUARDED_ROOTS:
        return
    raw_prefix = os.path.normcase(os.path.abspath(root / "data" / "raw")) + os.sep
    protected = {
        os.path.normcase(os.path.abspath(root / "data" / "processed" / name))
        for name in ("train.parquet", "validation.parquet", "test.parquet")
    }

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(candidate)))
        if path in protected or path.startswith(raw_prefix):
            raise PermissionError(f"Phase 2A prohibited data access blocked: {path}")

    sys.addaudithook(audit_hook)
    _GUARDED_ROOTS.add(resolved_root)


def relative_existing_files(root: Path, paths: Iterable[str]) -> list[Path]:
    """Validate and resolve a set of required repository-relative non-data artifacts."""

    resolved: list[Path] = []
    for relative in paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"artifact path escapes repository: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {relative}")
        resolved.append(path)
    return resolved
