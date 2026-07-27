"""Focused FastAPI Prometheus operational-metrics tests."""

from __future__ import annotations

from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from fraudshield.monitoring.api_metrics import ApiMetrics
from test_api_endpoints import ReadyService, _client, _transaction


def _samples(exposition: str, name: str):
    return [
        sample
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == name
    ]


def _sample_value(exposition: str, name: str, labels: dict[str, str]) -> float:
    matches = [sample for sample in _samples(exposition, name) if sample.labels == labels]
    assert len(matches) == 1
    return float(matches[0].value)


def test_required_metrics_have_only_bounded_labels() -> None:
    metrics = ApiMetrics(CollectorRegistry())
    metrics.observe_http(
        method="GET",
        normalized_route="/health/ready",
        status_code=200,
        duration_seconds=0.01,
    )
    metrics.observe_prediction(
        prediction=1,
        risk_level="high",
        transaction_type="TRANSFER",
        model_version="1",
        fraud_score=0.99,
    )
    metrics.set_model_readiness(
        True,
        {
            "registered_model_name": "fraudshield-production-sgd",
            "resolved_model_version": "1",
            "alias": "champion",
            "model_family": "SGDClassifier",
        },
    )
    metrics.set_database_readiness(True)
    rendered = metrics.render().decode("utf-8")

    assert metrics.http_requests._labelnames == (
        "method",
        "normalized_route",
        "status_class",
    )
    assert metrics.http_duration._labelnames == ("method", "normalized_route")
    assert metrics.predictions._labelnames == (
        "prediction",
        "risk_level",
        "transaction_type",
        "model_version",
    )
    assert metrics.prediction_score._labelnames == ("model_version",)
    assert metrics.model_info._labelnames == (
        "model_name",
        "model_version",
        "model_alias",
        "model_family",
    )
    for required in (
        "fraudshield_http_requests_total",
        "fraudshield_http_request_duration_seconds_bucket",
        "fraudshield_predictions_total",
        "fraudshield_prediction_score_bucket",
        "fraudshield_idempotent_replays_total",
        "fraudshield_idempotency_conflicts_total",
        "fraudshield_persistence_failures_total",
        "fraudshield_model_ready",
        "fraudshield_database_ready",
        "fraudshield_model_info",
    ):
        assert required in rendered
    for forbidden_label in (
        "request_id=",
        "prediction_id=",
        "amount=",
        "oldbalanceOrg=",
        "oldbalanceDest=",
        "fraud_score=",
        "account=",
    ):
        assert forbidden_label not in rendered


def test_metrics_endpoint_replay_conflict_and_readiness(tmp_path) -> None:
    request_id = "88888888-8888-4888-8888-888888888888"
    transaction = _transaction(amount=123456.789) | {
        "oldbalanceOrg": 987654.321,
        "oldbalanceDest": 456789.123,
    }
    service = ReadyService()

    with _client(tmp_path, service) as client:
        liveness = client.get("/health/live")
        first = client.post(
            "/predict",
            json=transaction,
            headers={"X-Request-ID": request_id},
        )
        replay = client.post(
            "/predict",
            json=transaction,
            headers={"X-Request-ID": request_id},
        )
        conflict = client.post(
            "/predict",
            json=transaction | {"amount": 123457.789},
            headers={"X-Request-ID": request_id},
        )
        metrics_response = client.get("/metrics")

    assert liveness.status_code == 200
    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    assert replay.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "idempotency_conflict"
    assert service.batch_calls == 1
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain;")
    exposition = metrics_response.text

    prediction_labels = {
        "model_version": "1",
        "prediction": "1",
        "risk_level": "high",
        "transaction_type": "TRANSFER",
    }
    assert _sample_value(
        exposition, "fraudshield_predictions_total", prediction_labels
    ) == 1.0
    assert _sample_value(
        exposition,
        "fraudshield_prediction_score_count",
        {"model_version": "1"},
    ) == 1.0
    assert _sample_value(exposition, "fraudshield_idempotent_replays_total", {}) == 1.0
    assert _sample_value(exposition, "fraudshield_idempotency_conflicts_total", {}) == 1.0
    assert _sample_value(
        exposition,
        "fraudshield_http_requests_total",
        {"method": "POST", "normalized_route": "/predict", "status_class": "2xx"},
    ) == 2.0
    assert _sample_value(
        exposition,
        "fraudshield_http_requests_total",
        {"method": "POST", "normalized_route": "/predict", "status_class": "4xx"},
    ) == 1.0
    assert _sample_value(
        exposition,
        "fraudshield_http_request_duration_seconds_count",
        {"method": "POST", "normalized_route": "/predict"},
    ) == 3.0
    assert _sample_value(exposition, "fraudshield_model_ready", {}) == 1.0
    assert _sample_value(exposition, "fraudshield_database_ready", {}) == 1.0
    assert _sample_value(
        exposition,
        "fraudshield_model_info",
        {
            "model_alias": "champion",
            "model_family": "SGDClassifier",
            "model_name": "fraudshield-production-sgd",
            "model_version": "1",
        },
    ) == 1.0
    assert not any(
        sample.labels.get("normalized_route") == "/metrics"
        for sample in _samples(exposition, "fraudshield_http_requests_total")
    )
    for sensitive_value in (
        request_id,
        first.json()["prediction_id"],
        "123456.789",
        "987654.321",
        "456789.123",
    ):
        assert sensitive_value not in exposition
    assert "X-Request-ID" in first.headers
    assert "X-Process-Time-Ms" in first.headers


def test_repeated_app_creation_uses_isolated_registries(tmp_path) -> None:
    with _client(tmp_path / "one", ReadyService()) as first:
        assert first.get("/metrics").status_code == 200
    with _client(tmp_path / "two", ReadyService()) as second:
        assert second.get("/metrics").status_code == 200
