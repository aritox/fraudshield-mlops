"""Transactional idempotency, hashing, prediction persistence, and outcomes."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from fraudshield.persistence.models import PredictionEvent, PredictionOutcome, PredictionRequest
from fraudshield.persistence.repository import PredictionRepository
from fraudshield.persistence.schemas import (
    AuditRecord,
    ModelIdentity,
    OutcomeValue,
    PersistedPrediction,
    PredictionResult,
    ScoredValue,
    StoredOutcome,
    TransactionValues,
)


class PersistenceUnavailableError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


class PredictionNotFoundError(RuntimeError):
    pass


class OutcomeConflictError(RuntimeError):
    pass


def canonical_payload_hash(endpoint: str, transactions: Sequence[TransactionValues]) -> str:
    payload = {
        "endpoint": endpoint,
        "transactions": [
            {
                "step": item.step,
                "type": item.transaction_type,
                "amount": item.amount,
                "oldbalanceOrg": item.oldbalance_origin,
                "oldbalanceDest": item.oldbalance_destination,
            }
            for item in transactions
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PredictionPersistenceService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        repository: PredictionRepository | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.repository = repository or PredictionRepository()

    @staticmethod
    def _stored_result(request: PredictionRequest, *, replayed: bool) -> PredictionResult:
        if request.completed_at is None or request.processing_time_ms is None:
            raise PersistenceUnavailableError("Persisted prediction request is incomplete")
        events = sorted(request.events, key=lambda item: item.item_index)
        if len(events) != request.batch_size:
            raise PersistenceUnavailableError("Persisted prediction batch is incomplete")
        model = ModelIdentity(
            name=request.model_name,
            version=request.model_version,
            alias=request.model_alias,
            threshold=float(request.threshold),
        )
        return PredictionResult(
            request_id=request.request_id,
            predictions=[
                PersistedPrediction(
                    prediction_id=event.prediction_id,
                    item_index=event.item_index,
                    fraud_score=float(event.fraud_score),
                    prediction=event.prediction,
                    threshold=float(request.threshold),
                    risk_level=event.risk_level,
                )
                for event in events
            ],
            model=model,
            processing_time_ms=float(request.processing_time_ms),
            replayed=replayed,
        )

    def _existing_result(
        self,
        request_id: uuid.UUID,
        payload_hash: str,
    ) -> PredictionResult:
        try:
            with self.session_factory() as session, session.begin():
                existing = self.repository.get_request(session, request_id, lock=True)
                if existing is None:
                    raise PersistenceUnavailableError("Concurrent request could not be resolved")
                if existing.payload_hash != payload_hash:
                    raise IdempotencyConflictError()
                return self._stored_result(existing, replayed=True)
        except (IdempotencyConflictError, PersistenceUnavailableError):
            raise
        except SQLAlchemyError as error:
            raise PersistenceUnavailableError("Prediction persistence is unavailable") from error

    def persist_predictions(
        self,
        *,
        request_id: uuid.UUID,
        endpoint: str,
        transactions: Sequence[TransactionValues],
        model: ModelIdentity,
        scorer: Callable[[], Sequence[ScoredValue]],
    ) -> PredictionResult:
        payload_hash = canonical_payload_hash(endpoint, transactions)
        started = time.perf_counter()
        session = self.session_factory()
        request_reserved = False
        try:
            with session.begin():
                existing = self.repository.get_request(session, request_id, lock=True)
                if existing is not None:
                    if existing.payload_hash != payload_hash:
                        raise IdempotencyConflictError()
                    return self._stored_result(existing, replayed=True)
                request = PredictionRequest(
                    request_id=request_id,
                    endpoint=endpoint,
                    payload_hash=payload_hash,
                    batch_size=len(transactions),
                    model_name=model.name,
                    model_version=model.version,
                    model_alias=model.alias,
                    threshold=model.threshold,
                )
                self.repository.add_request(session, request)
                session.flush()
                request_reserved = True
                scores = list(scorer())
                if len(scores) != len(transactions):
                    raise ValueError("Model output row count does not match the request")
                events = []
                for index, (inputs, score) in enumerate(zip(transactions, scores, strict=True)):
                    if score.item_index != index:
                        raise ValueError("Model output order does not match the request")
                    events.append(
                        PredictionEvent(
                            prediction_id=uuid.uuid4(),
                            request_id=request_id,
                            item_index=index,
                            step=inputs.step,
                            transaction_type=inputs.transaction_type,
                            amount=inputs.amount,
                            oldbalance_origin=inputs.oldbalance_origin,
                            oldbalance_destination=inputs.oldbalance_destination,
                            fraud_score=score.fraud_score,
                            prediction=score.prediction,
                            risk_level=score.risk_level,
                        )
                    )
                self.repository.add_events(session, events)
                request.processing_time_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                request.completed_at = datetime.now(UTC)
                session.flush()
                request.events = events
                return self._stored_result(request, replayed=False)
        except IntegrityError as error:
            if not request_reserved:
                return self._existing_result(request_id, payload_hash)
            raise PersistenceUnavailableError("Prediction persistence is unavailable") from error
        except (IdempotencyConflictError, PersistenceUnavailableError, ValueError):
            raise
        except SQLAlchemyError as error:
            raise PersistenceUnavailableError("Prediction persistence is unavailable") from error
        finally:
            session.close()

    def get_audit(self, prediction_id: uuid.UUID) -> AuditRecord:
        try:
            with self.session_factory() as session, session.begin():
                event = self.repository.get_audit(session, prediction_id)
                if event is None:
                    raise PredictionNotFoundError()
                request = event.request
                outcome = None
                if event.outcome is not None:
                    stored = event.outcome
                    outcome = StoredOutcome(
                        prediction_id=stored.prediction_id,
                        actual_fraud=stored.actual_fraud,
                        observed_at=stored.observed_at,
                        source=stored.source,
                        created_at=stored.created_at,
                        updated_at=stored.updated_at,
                        replayed=False,
                    )
                return AuditRecord(
                    prediction_id=event.prediction_id,
                    request_id=event.request_id,
                    item_index=event.item_index,
                    endpoint=request.endpoint,
                    model_name=request.model_name,
                    model_version=request.model_version,
                    model_alias=request.model_alias,
                    threshold=float(request.threshold),
                    step=event.step,
                    transaction_type=event.transaction_type,
                    amount=float(event.amount),
                    oldbalance_origin=float(event.oldbalance_origin),
                    oldbalance_destination=float(event.oldbalance_destination),
                    fraud_score=float(event.fraud_score),
                    prediction=event.prediction,
                    risk_level=event.risk_level,
                    request_created_at=request.created_at,
                    request_completed_at=request.completed_at,
                    prediction_created_at=event.created_at,
                    outcome=outcome,
                )
        except PredictionNotFoundError:
            raise
        except SQLAlchemyError as error:
            raise PersistenceUnavailableError("Prediction persistence is unavailable") from error

    def submit_outcomes(self, values: Sequence[OutcomeValue]) -> list[StoredOutcome]:
        try:
            with self.session_factory() as session, session.begin():
                events = self.repository.lock_events(
                    session, (value.prediction_id for value in values)
                )
                missing = [
                    value.prediction_id for value in values if value.prediction_id not in events
                ]
                if missing:
                    raise PredictionNotFoundError()
                requested: dict[uuid.UUID, OutcomeValue] = {}
                for value in values:
                    previous = requested.get(value.prediction_id)
                    if previous is not None and previous.actual_fraud != value.actual_fraud:
                        raise OutcomeConflictError()
                    requested.setdefault(value.prediction_id, value)
                created: dict[uuid.UUID, PredictionOutcome] = {}
                replayed_ids: set[uuid.UUID] = set()
                for prediction_id, value in requested.items():
                    existing = events[prediction_id].outcome
                    if existing is not None:
                        if existing.actual_fraud != value.actual_fraud:
                            raise OutcomeConflictError()
                        created[prediction_id] = existing
                        replayed_ids.add(prediction_id)
                    else:
                        created[prediction_id] = PredictionOutcome(
                            prediction_id=prediction_id,
                            actual_fraud=value.actual_fraud,
                            observed_at=value.observed_at,
                            source=value.source,
                        )
                self.repository.add_outcomes(
                    session,
                    [item for key, item in created.items() if key not in replayed_ids],
                )
                session.flush()
                seen: set[uuid.UUID] = set()
                results = []
                for value in values:
                    stored = created[value.prediction_id]
                    is_replay = value.prediction_id in replayed_ids or value.prediction_id in seen
                    seen.add(value.prediction_id)
                    results.append(
                        StoredOutcome(
                            prediction_id=stored.prediction_id,
                            actual_fraud=stored.actual_fraud,
                            observed_at=stored.observed_at,
                            source=stored.source,
                            created_at=stored.created_at,
                            updated_at=stored.updated_at,
                            replayed=is_replay,
                        )
                    )
                return results
        except (PredictionNotFoundError, OutcomeConflictError):
            raise
        except SQLAlchemyError as error:
            raise PersistenceUnavailableError("Outcome persistence is unavailable") from error

    def row_counts(self) -> dict[str, int]:
        try:
            with self.session_factory() as session, session.begin():
                return self.repository.row_counts(session)
        except SQLAlchemyError as error:
            raise PersistenceUnavailableError("Prediction persistence is unavailable") from error
