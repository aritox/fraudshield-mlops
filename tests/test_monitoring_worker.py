"""Monitoring worker loop and metric exposition tests."""

import threading
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from prometheus_client import generate_latest

from fraudshield.monitoring.service import MonitoringMetricValue, MonitoringRunResult
from fraudshield.monitoring.worker import run_worker
from fraudshield.monitoring.worker_metrics import WorkerMetrics


def _result(*, performance: bool) -> MonitoringRunResult:
    end = datetime(2026, 1, 3, tzinfo=UTC)
    metrics = [
        MonitoringMetricValue("feature_psi", "step", 0.12, "moderate", 20),
        MonitoringMetricValue("alert_rate", None, 0.25, "informational", 20),
        MonitoringMetricValue("mean_fraud_score", None, 0.4, "informational", 20),
        MonitoringMetricValue(
            "performance_available", None, 1.0 if performance else 0.0, "informational", 4
        ),
    ]
    if performance:
        metrics.append(MonitoringMetricValue("precision", None, 0.75, "informational", 4))
    return MonitoringRunResult(
        run_id=uuid.uuid4(),
        window_start=end - timedelta(hours=24),
        window_end=end,
        reference_version="phase2d_train_v1",
        event_count=20,
        labeled_count=4,
        status="completed" if performance else "insufficient_labeled_data",
        overall_drift_status="moderate",
        metrics=tuple(metrics),
    )


def test_worker_metrics_are_bounded_and_omit_unavailable_performance() -> None:
    metrics = WorkerMetrics()
    metrics.record(_result(performance=False))
    unavailable = generate_latest(metrics.registry).decode("utf-8")
    assert "fraudshield_monitoring_window_events 20.0" in unavailable
    assert 'fraudshield_feature_psi{feature="step"} 0.12' in unavailable
    assert "fraudshield_performance_available 0.0" in unavailable
    assert "fraudshield_production_precision" not in unavailable
    for forbidden in ("request_id", "prediction_id", "amount", "oldbalance"):
        assert forbidden not in unavailable

    metrics.record(_result(performance=True))
    available = generate_latest(metrics.registry).decode("utf-8")
    assert "fraudshield_performance_available 1.0" in available
    assert "fraudshield_production_precision 0.75" in available


def test_worker_honors_a_preexisting_stop_signal(monkeypatch) -> None:
    calls: list[str] = []
    server = SimpleNamespace(
        shutdown=lambda: calls.append("shutdown"),
        server_close=lambda: calls.append("close"),
    )
    thread = SimpleNamespace(join=lambda timeout: calls.append(f"join:{timeout}"))
    runtime = SimpleNamespace(engine=SimpleNamespace(dispose=lambda: calls.append("dispose")))
    service = SimpleNamespace(
        config=SimpleNamespace(
            monitoring=SimpleNamespace(
                metrics_port=8001,
                metrics_host="127.0.0.1",
                interval_seconds=1,
            )
        )
    )
    monkeypatch.setattr(
        "fraudshield.monitoring.worker.build_monitoring_service",
        lambda: (service, runtime),
    )
    monkeypatch.setattr(
        "fraudshield.monitoring.worker.start_http_server",
        lambda **_kwargs: (server, thread),
    )
    stop = threading.Event()
    stop.set()
    run_worker(stop)
    assert calls == ["shutdown", "close", "join:5", "dispose"]
