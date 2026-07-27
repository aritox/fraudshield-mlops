"""Transactional monitoring-service tests over persisted audit rows."""

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from fraudshield.monitoring.config import load_monitoring_config
from fraudshield.monitoring.repository import MonitoringRepository
from fraudshield.monitoring.service import MonitoringService
from fraudshield.persistence.models import (
    MonitoringMetric,
    MonitoringRun,
    PredictionEvent,
    PredictionOutcome,
    PredictionRequest,
)


def _config(*, minimum_events: int = 2, minimum_labeled_events: int = 2):
    config = load_monitoring_config()
    return replace(
        config,
        monitoring=replace(
            config.monitoring,
            minimum_events=minimum_events,
            minimum_labeled_events=minimum_labeled_events,
        ),
    )


def _reference() -> dict:
    numeric = {}
    for feature, transformation in (
        ("step", "identity"),
        ("log1p(amount)", "log1p"),
        ("log1p(oldbalanceOrg)", "log1p"),
        ("log1p(oldbalanceDest)", "log1p"),
    ):
        numeric[feature] = {
            "transformation": transformation,
            "bin_edges": [0.0, 2.0, 20.0],
            "reference_counts": [1, 1],
            "reference_proportions": [0.5, 0.5],
        }
    return {
        "numeric_features": numeric,
        "categorical_features": {
            "type": {
                "allowed_categories": ["PAYMENT", "TRANSFER"],
                "reference_counts": [1, 1, 0],
                "reference_proportions": [0.5, 0.5, 0.0],
            }
        },
    }


def _add_prediction(
    session,
    *,
    created_at: datetime,
    prediction: int,
    actual_fraud: int | None,
    transaction_type: str = "TRANSFER",
) -> uuid.UUID:
    request_id = uuid.uuid4()
    prediction_id = uuid.uuid4()
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
            prediction_id=prediction_id,
            request_id=request_id,
            item_index=0,
            step=2,
            transaction_type=transaction_type,
            amount=100.0,
            oldbalance_origin=200.0,
            oldbalance_destination=300.0,
            fraud_score=0.99 if prediction else 0.1,
            prediction=prediction,
            risk_level="high" if prediction else "low",
            created_at=created_at,
        )
    )
    if actual_fraud is not None:
        session.add(
            PredictionOutcome(
                prediction_id=prediction_id,
                actual_fraud=actual_fraud,
                observed_at=created_at + timedelta(minutes=5),
                source="synthetic-test",
            )
        )
    return prediction_id


def test_service_persists_psi_output_previous_window_and_performance_without_model_call(
    audit_session_factory,
    monkeypatch,
) -> None:
    end = datetime(2026, 1, 3, tzinfo=UTC)
    start = end - timedelta(hours=24)
    with audit_session_factory() as session, session.begin():
        _add_prediction(
            session,
            created_at=start - timedelta(hours=1),
            prediction=1,
            actual_fraud=None,
        )
        _add_prediction(
            session,
            created_at=start,
            prediction=1,
            actual_fraud=1,
        )
        _add_prediction(
            session,
            created_at=end - timedelta(seconds=1),
            prediction=0,
            actual_fraud=0,
            transaction_type="PAYMENT",
        )

    def model_call_forbidden(*_args, **_kwargs):
        raise AssertionError("Monitoring invoked model inference")

    monkeypatch.setattr(
        "fraudshield.api.model_service.ProductionModelService.predict_one", model_call_forbidden
    )
    monkeypatch.setattr(
        "fraudshield.api.model_service.ProductionModelService.predict_batch", model_call_forbidden
    )
    with audit_session_factory() as session:
        before = list(
            session.execute(
                select(
                    PredictionEvent.prediction_id,
                    PredictionEvent.fraud_score,
                    PredictionEvent.prediction,
                ).order_by(PredictionEvent.prediction_id)
            )
        )

    service = MonitoringService(
        audit_session_factory,
        _config(),
        reference_profile=_reference(),
    )
    result = service.run_once(window_end=end)

    assert result.status == "completed"
    assert result.event_count == 2
    assert result.labeled_count == 2
    assert result.metric("alert_rate") == 0.5
    assert result.metric("previous_window_alert_rate") == 1.0
    assert result.metric("performance_available") == 1.0
    assert result.metric("precision") == 1.0
    assert result.overall_drift_status in {"stable", "moderate", "significant"}
    with audit_session_factory() as session:
        stored_run = session.scalar(select(MonitoringRun))
        stored_metrics = list(session.scalars(select(MonitoringMetric)))
        after = list(
            session.execute(
                select(
                    PredictionEvent.prediction_id,
                    PredictionEvent.fraud_score,
                    PredictionEvent.prediction,
                ).order_by(PredictionEvent.prediction_id)
            )
        )
    assert stored_run is not None and stored_run.run_id == result.run_id
    assert len([item for item in stored_metrics if item.metric_name == "feature_psi"]) == 5
    assert before == after


def test_service_marks_insufficient_labels_without_performance_values(
    audit_session_factory,
) -> None:
    end = datetime(2026, 1, 3, tzinfo=UTC)
    with audit_session_factory() as session, session.begin():
        for offset in range(2):
            _add_prediction(
                session,
                created_at=end - timedelta(minutes=offset + 1),
                prediction=offset % 2,
                actual_fraud=None,
            )
    result = MonitoringService(
        audit_session_factory,
        _config(minimum_labeled_events=2),
        reference_profile=_reference(),
    ).run_once(window_end=end)

    assert result.status == "insufficient_labeled_data"
    assert result.metric("performance_available") == 0.0
    assert result.metric("precision") is None
    assert not any(item.metric_name == "precision" for item in result.metrics)


def test_failed_calculation_persists_failed_run_without_partial_metrics(
    audit_session_factory,
    monkeypatch,
) -> None:
    service = MonitoringService(
        audit_session_factory,
        _config(),
        repository=MonitoringRepository(),
        reference_profile=_reference(),
    )

    def fail(*_args, **_kwargs):
        raise ValueError("synthetic calculation failure")

    monkeypatch.setattr(service, "_calculate", fail)
    result = service.run_once(window_end=datetime(2026, 1, 3, tzinfo=UTC))

    assert result.status == "failed"
    with audit_session_factory() as session:
        run = session.scalar(select(MonitoringRun))
        metrics = list(session.scalars(select(MonitoringMetric)))
    assert run is not None and run.status == "failed"
    assert metrics == []
