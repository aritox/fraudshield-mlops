"""Transactional idempotency service tests."""

import uuid

import pytest

from fraudshield.persistence.schemas import ModelIdentity, ScoredValue, TransactionValues
from fraudshield.persistence.service import (
    IdempotencyConflictError,
    PredictionPersistenceService,
    canonical_payload_hash,
)


def _inputs(amount: float = 100.0) -> list[TransactionValues]:
    return [TransactionValues(1, "TRANSFER", amount, 100.0, 0.0)]


MODEL = ModelIdentity("fraudshield-production-sgd", "1", "champion", 0.98310834)


def test_hash_is_deterministic_and_order_sensitive() -> None:
    first = _inputs()
    assert canonical_payload_hash("/predict", first) == canonical_payload_hash("/predict", first)
    assert canonical_payload_hash("/predict", first) != canonical_payload_hash(
        "/predict/batch", first
    )
    assert canonical_payload_hash("/predict", first) != canonical_payload_hash(
        "/predict", first + _inputs(101.0)
    )


def test_new_replay_conflict_and_model_not_called_on_replay(audit_session_factory) -> None:
    service = PredictionPersistenceService(audit_session_factory)
    request_id = uuid.uuid4()
    calls = 0

    def score():
        nonlocal calls
        calls += 1
        return [ScoredValue(0, 0.99, 1, "high")]

    first = service.persist_predictions(
        request_id=request_id,
        endpoint="/predict",
        transactions=_inputs(),
        model=MODEL,
        scorer=score,
    )
    replay = service.persist_predictions(
        request_id=request_id,
        endpoint="/predict",
        transactions=_inputs(),
        model=MODEL,
        scorer=score,
    )
    assert calls == 1
    assert replay.replayed is True
    assert replay.predictions == first.predictions
    assert service.row_counts() == {
        "prediction_requests": 1,
        "prediction_events": 1,
        "prediction_outcomes": 0,
    }
    with pytest.raises(IdempotencyConflictError):
        service.persist_predictions(
            request_id=request_id,
            endpoint="/predict",
            transactions=_inputs(999.0),
            model=MODEL,
            scorer=score,
        )
    assert calls == 1


def test_batch_rolls_back_when_scorer_fails(audit_session_factory) -> None:
    service = PredictionPersistenceService(audit_session_factory)

    def fail():
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        service.persist_predictions(
            request_id=uuid.uuid4(),
            endpoint="/predict/batch",
            transactions=_inputs() * 3,
            model=MODEL,
            scorer=fail,
        )
    assert service.row_counts()["prediction_requests"] == 0
