"""Add persistent Phase 2D monitoring runs and metrics."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "phase2d_001"
down_revision = "phase2c_001"
branch_labels = None
depends_on = None

MAX_DOUBLE = "1.7976931348623157e308"


def upgrade() -> None:
    op.create_table(
        "monitoring_runs",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_version", sa.String(length=64), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("labeled_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("overall_drift_status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "window_end > window_start", name="ck_monitoring_runs_window_order_valid"
        ),
        sa.CheckConstraint(
            "event_count >= 0", name="ck_monitoring_runs_event_count_nonnegative"
        ),
        sa.CheckConstraint(
            "labeled_count >= 0", name="ck_monitoring_runs_labeled_count_nonnegative"
        ),
        sa.CheckConstraint(
            "labeled_count <= event_count",
            name="ck_monitoring_runs_labeled_count_within_events",
        ),
        sa.CheckConstraint(
            "length(trim(reference_version)) > 0",
            name="ck_monitoring_runs_reference_version_nonempty",
        ),
        sa.CheckConstraint(
            "status IN ('completed','insufficient_data','insufficient_labeled_data','failed')",
            name="ck_monitoring_runs_status_valid",
        ),
        sa.CheckConstraint(
            "overall_drift_status IN "
            "('stable','moderate','significant','insufficient_data','failed')",
            name="ck_monitoring_runs_overall_drift_status_valid",
        ),
        sa.PrimaryKeyConstraint("run_id", name="pk_monitoring_runs"),
        sa.UniqueConstraint(
            "window_start",
            "window_end",
            "reference_version",
            name="uq_monitoring_runs_window_reference",
        ),
    )
    op.create_index("ix_monitoring_runs_window_end", "monitoring_runs", ["window_end"])
    op.create_index("ix_monitoring_runs_created_at", "monitoring_runs", ["created_at"])
    op.create_index(
        "ix_monitoring_runs_overall_drift_status",
        "monitoring_runs",
        ["overall_drift_status"],
    )

    op.create_table(
        "monitoring_metrics",
        sa.Column("metric_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("feature_name", sa.String(length=128), nullable=True),
        sa.Column("metric_value", sa.Float(precision=53), nullable=True),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(trim(metric_name)) > 0",
            name="ck_monitoring_metrics_metric_name_nonempty",
        ),
        sa.CheckConstraint(
            "metric_value IS NULL OR "
            f"(metric_value > -{MAX_DOUBLE} AND metric_value < {MAX_DOUBLE})",
            name="ck_monitoring_metrics_metric_value_finite",
        ),
        sa.CheckConstraint(
            "severity IN "
            "('stable','moderate','significant','insufficient_data','informational','unavailable')",
            name="ck_monitoring_metrics_severity_valid",
        ),
        sa.CheckConstraint(
            "sample_size >= 0", name="ck_monitoring_metrics_sample_size_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["monitoring_runs.run_id"],
            name="fk_monitoring_metrics_run_id_monitoring_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("metric_id", name="pk_monitoring_metrics"),
        sa.UniqueConstraint(
            "run_id",
            "metric_name",
            "feature_name",
            name="uq_monitoring_metrics_run_name_feature",
        ),
    )
    op.create_index("ix_monitoring_metrics_run_id", "monitoring_metrics", ["run_id"])
    op.create_index(
        "ix_monitoring_metrics_metric_name", "monitoring_metrics", ["metric_name"]
    )
    op.create_index(
        "ix_monitoring_metrics_feature_name", "monitoring_metrics", ["feature_name"]
    )
    op.create_index("ix_monitoring_metrics_severity", "monitoring_metrics", ["severity"])


def downgrade() -> None:
    op.drop_table("monitoring_metrics")
    op.drop_table("monitoring_runs")
