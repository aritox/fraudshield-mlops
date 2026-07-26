"""Database reads and writes for prediction audit records."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload, selectinload

from fraudshield.persistence.models import PredictionEvent, PredictionOutcome, PredictionRequest


class PredictionRepository:
    def get_request(
        self,
        session: Session,
        request_id: uuid.UUID,
        *,
        lock: bool = False,
    ) -> PredictionRequest | None:
        statement = (
            select(PredictionRequest)
            .where(PredictionRequest.request_id == request_id)
            .options(selectinload(PredictionRequest.events))
        )
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    def add_request(self, session: Session, request: PredictionRequest) -> None:
        session.add(request)

    def add_events(self, session: Session, events: Iterable[PredictionEvent]) -> None:
        session.add_all(list(events))

    def get_audit(self, session: Session, prediction_id: uuid.UUID) -> PredictionEvent | None:
        return session.scalar(
            select(PredictionEvent)
            .where(PredictionEvent.prediction_id == prediction_id)
            .options(joinedload(PredictionEvent.request), joinedload(PredictionEvent.outcome))
        )

    def lock_events(
        self,
        session: Session,
        prediction_ids: Iterable[uuid.UUID],
    ) -> dict[uuid.UUID, PredictionEvent]:
        identifiers = list(dict.fromkeys(prediction_ids))
        events = session.scalars(
            select(PredictionEvent)
            .where(PredictionEvent.prediction_id.in_(identifiers))
            .options(selectinload(PredictionEvent.outcome))
            .with_for_update()
        ).unique()
        return {event.prediction_id: event for event in events}

    def add_outcomes(self, session: Session, outcomes: Iterable[PredictionOutcome]) -> None:
        session.add_all(list(outcomes))

    def row_counts(self, session: Session) -> dict[str, int]:
        return {
            "prediction_requests": int(
                session.scalar(select(func.count(PredictionRequest.request_id))) or 0
            ),
            "prediction_events": int(
                session.scalar(select(func.count(PredictionEvent.prediction_id))) or 0
            ),
            "prediction_outcomes": int(
                session.scalar(select(func.count(PredictionOutcome.prediction_id))) or 0
            ),
        }
