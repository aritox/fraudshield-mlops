"""Repository-only audit persistence tests."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from fraudshield.persistence.models import PredictionRequest
from fraudshield.persistence.repository import PredictionRepository


def _request(identifier: uuid.UUID) -> PredictionRequest:
    return PredictionRequest(
        request_id=identifier,
        endpoint="/predict",
        payload_hash="a" * 64,
        batch_size=1,
        model_name="fraudshield-production-sgd",
        model_version="1",
        model_alias="champion",
        threshold=0.98310834,
    )


def test_repository_insert_read_and_unique_rollback(audit_session_factory) -> None:
    repository = PredictionRepository()
    identifier = uuid.uuid4()
    with audit_session_factory() as session, session.begin():
        repository.add_request(session, _request(identifier))
    with audit_session_factory() as session, session.begin():
        assert repository.get_request(session, identifier) is not None
    with pytest.raises(IntegrityError), audit_session_factory() as session, session.begin():
        repository.add_request(session, _request(identifier))
        session.flush()
    with audit_session_factory() as session, session.begin():
        assert repository.row_counts(session)["prediction_requests"] == 1
