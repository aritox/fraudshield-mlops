"""Database-only reads and writes for production monitoring."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from fraudshield.persistence.models import (
    MonitoringMetric,
    MonitoringRun,
    PredictionEvent,
)


@dataclass(frozen=True)
class PersistedMonitoringEvent:
    step: int
    transaction_type: str
    amount: float
    oldbalance_origin: float
    oldbalance_destination: float
    fraud_score: float
    prediction: int
    risk_level: str
    created_at: datetime
    actual_fraud: int | None
    observed_at: datetime | None


class MonitoringRepository:
    def load_events(
        self,
        session: Session,
        window_start: datetime,
        window_end: datetime,
    ) -> list[PersistedMonitoringEvent]:
        statement = (
            select(PredictionEvent)
            .where(
                PredictionEvent.created_at >= window_start,
                PredictionEvent.created_at < window_end,
            )
            .options(joinedload(PredictionEvent.outcome))
            .order_by(PredictionEvent.created_at, PredictionEvent.prediction_id)
        )
        events = session.scalars(statement).unique()
        return [
            PersistedMonitoringEvent(
                step=event.step,
                transaction_type=event.transaction_type,
                amount=float(event.amount),
                oldbalance_origin=float(event.oldbalance_origin),
                oldbalance_destination=float(event.oldbalance_destination),
                fraud_score=float(event.fraud_score),
                prediction=event.prediction,
                risk_level=event.risk_level,
                created_at=event.created_at,
                actual_fraud=event.outcome.actual_fraud if event.outcome is not None else None,
                observed_at=event.outcome.observed_at if event.outcome is not None else None,
            )
            for event in events
        ]

    def add_run(
        self,
        session: Session,
        run: MonitoringRun,
        metrics: list[MonitoringMetric],
    ) -> None:
        session.add(run)
        session.add_all(metrics)

    def get_run(self, session: Session, run_id: uuid.UUID) -> MonitoringRun | None:
        return session.scalar(
            select(MonitoringRun)
            .where(MonitoringRun.run_id == run_id)
            .options(joinedload(MonitoringRun.metrics))
        )

    def row_counts(self, session: Session) -> dict[str, int]:
        return {
            "monitoring_runs": int(
                session.scalar(select(func.count(MonitoringRun.run_id))) or 0
            ),
            "monitoring_metrics": int(
                session.scalar(select(func.count(MonitoringMetric.metric_id))) or 0
            ),
        }
