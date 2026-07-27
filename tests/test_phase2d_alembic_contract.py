"""Phase 2D monitoring migration and Compose contracts."""

from pathlib import Path

import yaml
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from fraudshield.persistence.models import Base


def test_phase2d_migration_and_metadata_define_monitoring_objects() -> None:
    migration_path = Path("alembic/versions/phase2d_001_add_monitoring_tables.py")
    migration = migration_path.read_text(encoding="utf-8")
    assert 'revision = "phase2d_001"' in migration
    assert 'down_revision = "phase2c_001"' in migration
    assert migration.index('op.create_table(\n        "monitoring_runs"') < migration.index(
        'op.create_table(\n        "monitoring_metrics"'
    )
    assert migration.index('op.drop_table("monitoring_metrics")') < migration.index(
        'op.drop_table("monitoring_runs")'
    )

    runs = Base.metadata.tables["monitoring_runs"]
    metrics = Base.metadata.tables["monitoring_metrics"]
    assert runs.primary_key.columns.keys() == ["run_id"]
    assert metrics.primary_key.columns.keys() == ["metric_id"]
    assert {index.name for index in runs.indexes} == {
        "ix_monitoring_runs_window_end",
        "ix_monitoring_runs_created_at",
        "ix_monitoring_runs_overall_drift_status",
    }
    assert {index.name for index in metrics.indexes} == {
        "ix_monitoring_metrics_run_id",
        "ix_monitoring_metrics_metric_name",
        "ix_monitoring_metrics_feature_name",
        "ix_monitoring_metrics_severity",
    }
    assert any(isinstance(item, CheckConstraint) for item in runs.constraints)
    assert any(isinstance(item, UniqueConstraint) for item in runs.constraints)
    assert any(isinstance(item, ForeignKeyConstraint) for item in metrics.constraints)
    assert any(isinstance(item, UniqueConstraint) for item in metrics.constraints)


def test_monitor_compose_service_is_internal_and_hardened() -> None:
    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    monitor = compose["services"]["monitor"]
    assert monitor["image"] == compose["services"]["api"]["image"]
    assert monitor["command"] == ["python", "-m", "fraudshield.monitoring.worker"]
    assert monitor["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert (
        monitor["depends_on"]["migrate"]["condition"]
        == "service_completed_successfully"
    )
    assert monitor["expose"] == ["8001"]
    assert "ports" not in monitor
    assert "USER 10001:10001" in Path("Dockerfile").read_text(encoding="utf-8")
    assert monitor["read_only"] is True
    assert monitor["security_opt"] == ["no-new-privileges:true"]
    assert monitor["cap_drop"] == ["ALL"]
    serialized = yaml.safe_dump(monitor)
    assert "data/raw" not in serialized
    assert "data/processed" not in serialized
    assert "/var/run/docker.sock" not in serialized


def test_application_startup_never_creates_monitoring_schema() -> None:
    application_source = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (Path("src/fraudshield/api"), Path("src/fraudshield/monitoring"))
        for path in directory.glob("*.py")
    )
    assert "create_all" not in application_source
