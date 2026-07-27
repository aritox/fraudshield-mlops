"""Monitoring database repository tests."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from fraudshield.monitoring.repository import MonitoringRepository
from fraudshield.persistence.models import (
    MonitoringMetric,
    MonitoringRun,
    PredictionEvent,
    PredictionRequest,
)


def _event(session, created_at: datetime, item_index: int) -> None:
    request_id = uuid.uuid4()
    session.add(
        PredictionRequest(
            request_id=request_id,
            endpoint="/predict",
            payload_hash=uuid.uuid4().hex * 2,
            batch_size=1,
            model_name="fraudshield-production-sgd",
            model_version="1",
            model_alias="champion",
            threshold=0.98310834,
        )
    )
    session.add(
        PredictionEvent(
            prediction_id=uuid.uuid4(),
            request_id=request_id,
            item_index=item_index,
            step=1,
            transaction_type="TRANSFER",
            amount=10.0,
            oldbalance_origin=20.0,
            oldbalance_destination=30.0,
            fraud_score=0.9,
            prediction=0,
            risk_level="medium",
            created_at=created_at,
        )
    )


def test_repository_uses_inclusive_start_and_exclusive_end(audit_session_factory) -> None:
    repository = MonitoringRepository()
    start = datetime(2026, 1, 2, tzinfo=UTC)
    end = start + timedelta(hours=1)
    with audit_session_factory() as session, session.begin():
        _event(session, start, 0)
        _event(session, end - timedelta(microseconds=1), 1)
        _event(session, end, 2)

    with audit_session_factory() as session, session.begin():
        events = repository.load_events(session, start, end)
    assert len(events) == 2
    assert all(start <= event.created_at.replace(tzinfo=UTC) < end for event in events)


def test_monitoring_run_and_metrics_roll_back_atomically(audit_session_factory) -> None:
    repository = MonitoringRepository()
    run_id = uuid.uuid4()
    start = datetime(2026, 1, 2, tzinfo=UTC)
    run = MonitoringRun(
        run_id=run_id,
        window_start=start,
        window_end=start + timedelta(hours=1),
        reference_version="phase2d_train_v1",
        event_count=0,
        labeled_count=0,
        status="insufficient_data",
        overall_drift_status="insufficient_data",
    )
    duplicate_metrics = [
        MonitoringMetric(
            metric_id=uuid.uuid4(),
            run_id=run_id,
            metric_name="feature_psi",
            feature_name="step",
            metric_value=None,
            severity="insufficient_data",
            sample_size=0,
        )
        for _ in range(2)
    ]
    with pytest.raises(IntegrityError), audit_session_factory() as session, session.begin():
        repository.add_run(session, run, duplicate_metrics)
        session.flush()
    with audit_session_factory() as session, session.begin():
        assert repository.row_counts(session) == {
            "monitoring_runs": 0,
            "monitoring_metrics": 0,
        }
