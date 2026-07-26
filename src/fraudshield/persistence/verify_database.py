"""Verify PostgreSQL connectivity, Alembic state, tables, constraints, and indexes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from fraudshield.data.config import repository_root
from fraudshield.persistence.config import load_database_config
from fraudshield.persistence.database import create_database_runtime
from fraudshield.persistence.models import Base
from fraudshield.tracking.mlflow_setup import (
    install_prohibited_data_guard,
    utc_timestamp,
    write_json,
)

DATABASE_MANIFEST = Path("artifacts/database/database_manifest.json")


def _head(root: Path) -> str:
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise RuntimeError("Alembic head revision is unavailable")
    return head


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
    ).stdout.strip()


def verify_database(root: Path | None = None, *, write_manifest: bool = True) -> dict[str, Any]:
    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    config = load_database_config(root=repo_root)
    runtime = create_database_runtime(config)
    expected_tables = set(Base.metadata.tables)
    head = _head(repo_root)
    try:
        with runtime.engine.connect() as connection:
            if connection.dialect.name != "postgresql":
                raise RuntimeError("Database is not PostgreSQL")
            current = connection.scalar(text("SELECT version_num FROM alembic_version"))
            if current != head:
                raise RuntimeError("Database migration is not current")
            inspector = inspect(connection)
            actual_tables = set(inspector.get_table_names())
            missing = sorted(expected_tables - actual_tables)
            if missing:
                raise RuntimeError("Required prediction audit tables are missing")
            objects = {}
            for table in sorted(expected_tables):
                objects[table] = {
                    "constraints": sorted(
                        item["name"]
                        for item in inspector.get_check_constraints(table)
                        if item["name"]
                    ),
                    "indexes": sorted(item["name"] for item in inspector.get_indexes(table)),
                    "foreign_keys": sorted(
                        item["name"] for item in inspector.get_foreign_keys(table) if item["name"]
                    ),
                }
    finally:
        runtime.engine.dispose()
    result = {
        "status": "verified",
        "database_type": "PostgreSQL",
        "migration_status": "current",
        "alembic_revision": head,
        "tables": sorted(expected_tables),
        "objects": objects,
    }
    if write_manifest:
        manifest = {
            "generation_timestamp_utc": utc_timestamp(),
            "source_git_commit": _git_commit(repo_root),
            "postgresql_image_tag": "postgres:16-alpine",
            "sqlalchemy_version": importlib.metadata.version("sqlalchemy"),
            "psycopg_version": importlib.metadata.version("psycopg"),
            "alembic_version": importlib.metadata.version("alembic"),
            "database_type": "PostgreSQL",
            "persistence_policy": "required; atomic request and event batches",
            "idempotency_policy": "UUID request ID plus canonical SHA-256 payload hash",
            "outcome_policy": "atomic delayed outcomes; immutable actual-fraud conflicts",
            "migration_status": "current",
            "alembic_revision": head,
            "raw_data_accessed": False,
            "parquet_data_accessed": False,
        }
        write_json(repo_root / DATABASE_MANIFEST, manifest)
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result = verify_database()
    except Exception as error:
        raise SystemExit(f"Database verification failed safely: {type(error).__name__}") from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
