"""Export the persistence schema manifest without reading production records."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql

from fraudshield.data.config import repository_root
from fraudshield.persistence.models import Base
from fraudshield.tracking.mlflow_setup import (
    install_prohibited_data_guard,
    utc_timestamp,
    write_json,
)

SCHEMA_MANIFEST = Path("artifacts/database/schema_manifest.json")


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
    ).stdout.strip()


def _head(root: Path) -> str:
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise RuntimeError("Alembic head revision is unavailable")
    return head


def export_schema(root: Path | None = None) -> dict[str, Any]:
    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    dialect = postgresql.dialect()
    tables = []
    for table in Base.metadata.sorted_tables:
        tables.append(
            {
                "name": table.name,
                "columns": [
                    {
                        "name": column.name,
                        "type": column.type.compile(dialect=dialect),
                        "nullable": column.nullable,
                        "primary_key": column.primary_key,
                    }
                    for column in table.columns
                ],
                "primary_keys": [column.name for column in table.primary_key.columns],
                "foreign_keys": sorted(
                    [
                        {
                            "columns": [element.parent.name for element in constraint.elements],
                            "referred_table": constraint.referred_table.name,
                            "referred_columns": [
                                element.column.name for element in constraint.elements
                            ],
                            "ondelete": next(iter(constraint.elements)).ondelete,
                        }
                        for constraint in table.foreign_key_constraints
                    ],
                    key=lambda item: item["columns"],
                ),
                "unique_constraints": sorted(
                    [column.name for column in constraint.columns]
                    for constraint in table.constraints
                    if constraint.__class__.__name__ == "UniqueConstraint"
                ),
                "check_constraints": sorted(
                    [
                        {"name": constraint.name, "sql": str(constraint.sqltext)}
                        for constraint in table.constraints
                        if constraint.__class__.__name__ == "CheckConstraint"
                    ],
                    key=lambda item: str(item["name"]),
                ),
                "indexes": sorted(
                    [
                        {
                            "name": index.name,
                            "columns": [column.name for column in index.columns],
                            "unique": index.unique,
                        }
                        for index in table.indexes
                    ],
                    key=lambda item: str(item["name"]),
                ),
            }
        )
    manifest = {
        "schema_generation_timestamp_utc": utc_timestamp(),
        "source_git_commit": _git_commit(repo_root),
        "alembic_head_revision": _head(repo_root),
        "tables": tables,
        "raw_data_accessed": False,
        "parquet_data_accessed": False,
    }
    serialized = json.dumps(manifest, sort_keys=True)
    if str(repo_root) in serialized or "password" in serialized.lower():
        raise ValueError("Schema manifest contains prohibited host or credential data")
    write_json(repo_root / SCHEMA_MANIFEST, manifest)
    return manifest


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(export_schema(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
