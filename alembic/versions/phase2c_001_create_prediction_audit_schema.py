"""Create the Phase 2C prediction audit schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "phase2c_001"
down_revision = None
branch_labels = None
depends_on = None

MAX_DOUBLE = "1.7976931348623157e308"


def upgrade() -> None:
    op.create_table(
        "prediction_requests",
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_alias", sa.String(length=64), nullable=False),
        sa.Column("threshold", sa.Float(precision=53), nullable=False),
        sa.Column("processing_time_ms", sa.Float(precision=53), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "length(payload_hash) = 64", name="ck_prediction_requests_payload_hash_length"
        ),
        sa.CheckConstraint("batch_size > 0", name="ck_prediction_requests_batch_size_positive"),
        sa.CheckConstraint(
            "threshold >= 0 AND threshold <= 1", name="ck_prediction_requests_threshold_range"
        ),
        sa.CheckConstraint(
            "processing_time_ms IS NULL OR "
            f"(processing_time_ms >= 0 AND processing_time_ms < {MAX_DOUBLE})",
            name="ck_prediction_requests_processing_time_finite_nonnegative",
        ),
        sa.PrimaryKeyConstraint("request_id", name="pk_prediction_requests"),
    )
    op.create_index("ix_prediction_requests_created_at", "prediction_requests", ["created_at"])
    op.create_index(
        "ix_prediction_requests_model_name_version",
        "prediction_requests",
        ["model_name", "model_version"],
    )
    op.create_table(
        "prediction_events",
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("transaction_type", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.Float(precision=53), nullable=False),
        sa.Column("oldbalance_origin", sa.Float(precision=53), nullable=False),
        sa.Column("oldbalance_destination", sa.Float(precision=53), nullable=False),
        sa.Column("fraud_score", sa.Float(precision=53), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("item_index >= 0", name="ck_prediction_events_item_index_nonnegative"),
        sa.CheckConstraint("step >= 1", name="ck_prediction_events_step_positive"),
        sa.CheckConstraint(
            "transaction_type IN ('CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER')",
            name="ck_prediction_events_transaction_type_valid",
        ),
        sa.CheckConstraint(
            f"amount >= 0 AND amount < {MAX_DOUBLE}",
            name="ck_prediction_events_amount_finite_nonnegative",
        ),
        sa.CheckConstraint(
            f"oldbalance_origin >= 0 AND oldbalance_origin < {MAX_DOUBLE}",
            name="ck_prediction_events_oldbalance_origin_finite_nonnegative",
        ),
        sa.CheckConstraint(
            f"oldbalance_destination >= 0 AND oldbalance_destination < {MAX_DOUBLE}",
            name="ck_prediction_events_oldbalance_destination_finite_nonnegative",
        ),
        sa.CheckConstraint(
            "fraud_score >= 0 AND fraud_score <= 1", name="ck_prediction_events_fraud_score_range"
        ),
        sa.CheckConstraint("prediction IN (0, 1)", name="ck_prediction_events_prediction_valid"),
        sa.CheckConstraint(
            "risk_level IN ('low', 'medium', 'high')", name="ck_prediction_events_risk_level_valid"
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["prediction_requests.request_id"],
            name="fk_prediction_events_request_id_prediction_requests",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("prediction_id", name="pk_prediction_events"),
        sa.UniqueConstraint("request_id", "item_index", name="uq_prediction_events_request_item"),
    )
    for name, columns in (
        ("ix_prediction_events_request_id", ["request_id"]),
        ("ix_prediction_events_created_at", ["created_at"]),
        ("ix_prediction_events_prediction", ["prediction"]),
        ("ix_prediction_events_risk_level", ["risk_level"]),
        ("ix_prediction_events_transaction_type", ["transaction_type"]),
    ):
        op.create_index(name, "prediction_events", columns)
    op.create_table(
        "prediction_outcomes",
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("actual_fraud", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "actual_fraud IN (0, 1)", name="ck_prediction_outcomes_actual_fraud_valid"
        ),
        sa.CheckConstraint(
            "length(trim(source)) > 0", name="ck_prediction_outcomes_source_nonempty"
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["prediction_events.prediction_id"],
            name="fk_prediction_outcomes_prediction_id_prediction_events",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("prediction_id", name="pk_prediction_outcomes"),
    )
    op.create_index("ix_prediction_outcomes_actual_fraud", "prediction_outcomes", ["actual_fraud"])
    op.create_index("ix_prediction_outcomes_observed_at", "prediction_outcomes", ["observed_at"])


def downgrade() -> None:
    op.drop_table("prediction_outcomes")
    op.drop_table("prediction_events")
    op.drop_table("prediction_requests")
