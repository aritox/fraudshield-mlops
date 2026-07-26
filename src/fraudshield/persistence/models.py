"""SQLAlchemy declarative audit models and database invariants."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
MAX_FINITE_DOUBLE = "1.7976931348623157e308"


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class PredictionRequest(Base):
    __tablename__ = "prediction_requests"
    __table_args__ = (
        CheckConstraint("length(payload_hash) = 64", name="payload_hash_length"),
        CheckConstraint("batch_size > 0", name="batch_size_positive"),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="threshold_range"),
        CheckConstraint(
            f"processing_time_ms IS NULL OR (processing_time_ms >= 0 AND "
            f"processing_time_ms < {MAX_FINITE_DOUBLE})",
            name="processing_time_finite_nonnegative",
        ),
        Index("ix_prediction_requests_created_at", "created_at"),
        Index("ix_prediction_requests_model_name_version", "model_name", "model_version"),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold: Mapped[float] = mapped_column(Float(53), nullable=False)
    processing_time_ms: Mapped[float | None] = mapped_column(Float(53), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list[PredictionEvent]] = relationship(
        back_populates="request", cascade="all, delete-orphan", passive_deletes=True
    )


class PredictionEvent(Base):
    __tablename__ = "prediction_events"
    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "item_index",
            name="uq_prediction_events_request_item",
        ),
        CheckConstraint("item_index >= 0", name="item_index_nonnegative"),
        CheckConstraint("step >= 1", name="step_positive"),
        CheckConstraint(
            "transaction_type IN ('CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER')",
            name="transaction_type_valid",
        ),
        CheckConstraint(
            f"amount >= 0 AND amount < {MAX_FINITE_DOUBLE}", name="amount_finite_nonnegative"
        ),
        CheckConstraint(
            f"oldbalance_origin >= 0 AND oldbalance_origin < {MAX_FINITE_DOUBLE}",
            name="oldbalance_origin_finite_nonnegative",
        ),
        CheckConstraint(
            f"oldbalance_destination >= 0 AND oldbalance_destination < {MAX_FINITE_DOUBLE}",
            name="oldbalance_destination_finite_nonnegative",
        ),
        CheckConstraint("fraud_score >= 0 AND fraud_score <= 1", name="fraud_score_range"),
        CheckConstraint("prediction IN (0, 1)", name="prediction_valid"),
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="risk_level_valid"),
        Index("ix_prediction_events_request_id", "request_id"),
        Index("ix_prediction_events_created_at", "created_at"),
        Index("ix_prediction_events_prediction", "prediction"),
        Index("ix_prediction_events_risk_level", "risk_level"),
        Index("ix_prediction_events_transaction_type", "transaction_type"),
    )

    prediction_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("prediction_requests.request_id", ondelete="CASCADE"),
        nullable=False,
    )
    item_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[float] = mapped_column(Float(53), nullable=False)
    oldbalance_origin: Mapped[float] = mapped_column(Float(53), nullable=False)
    oldbalance_destination: Mapped[float] = mapped_column(Float(53), nullable=False)
    fraud_score: Mapped[float] = mapped_column(Float(53), nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    request: Mapped[PredictionRequest] = relationship(back_populates="events")
    outcome: Mapped[PredictionOutcome | None] = relationship(
        back_populates="event", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )


class PredictionOutcome(Base):
    __tablename__ = "prediction_outcomes"
    __table_args__ = (
        CheckConstraint("actual_fraud IN (0, 1)", name="actual_fraud_valid"),
        CheckConstraint("length(trim(source)) > 0", name="source_nonempty"),
        Index("ix_prediction_outcomes_actual_fraud", "actual_fraud"),
        Index("ix_prediction_outcomes_observed_at", "observed_at"),
    )

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("prediction_events.prediction_id", ondelete="CASCADE"),
        primary_key=True,
    )
    actual_fraud: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    event: Mapped[PredictionEvent] = relationship(back_populates="outcome")
