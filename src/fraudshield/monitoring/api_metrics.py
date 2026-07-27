"""Bounded-cardinality Prometheus instrumentation for the inference API."""

from __future__ import annotations

import time
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class HttpMetricsMiddleware:
    """Measure HTTP traffic using resolved route templates instead of raw paths."""

    def __init__(
        self,
        app: ASGIApp,
        metrics: ApiMetrics,
        excluded_paths: set[str] | None = None,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.excluded_paths = excluded_paths or set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = str(scope.get("path", "unmatched"))
            if path not in self.excluded_paths:
                route = scope.get("route")
                normalized_route = str(getattr(route, "path", "unmatched"))
                self.metrics.observe_http(
                    method=str(scope.get("method", "UNKNOWN")),
                    normalized_route=normalized_route,
                    status_code=status_code,
                    duration_seconds=time.perf_counter() - started,
                )


class ApiMetrics:
    """One isolated collector registry for one FastAPI application instance."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "fraudshield_http_requests_total",
            "HTTP requests handled by the FraudShield API.",
            ("method", "normalized_route", "status_class"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "fraudshield_http_request_duration_seconds",
            "FraudShield API request duration in seconds.",
            ("method", "normalized_route"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.predictions = Counter(
            "fraudshield_predictions_total",
            "New persisted production prediction results.",
            ("prediction", "risk_level", "transaction_type", "model_version"),
            registry=self.registry,
        )
        self.prediction_score = Histogram(
            "fraudshield_prediction_score",
            "Production fraud score distribution; scores are not calibrated probabilities.",
            ("model_version",),
            buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0),
            registry=self.registry,
        )
        self.idempotent_replays = Counter(
            "fraudshield_idempotent_replays_total",
            "Identical persisted prediction request replays.",
            registry=self.registry,
        )
        self.idempotency_conflicts = Counter(
            "fraudshield_idempotency_conflicts_total",
            "Prediction request ID payload conflicts.",
            registry=self.registry,
        )
        self.persistence_failures = Counter(
            "fraudshield_persistence_failures_total",
            "Prediction-audit persistence failures.",
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "fraudshield_model_ready",
            "Whether the immutable production model is ready.",
            registry=self.registry,
        )
        self.database_ready = Gauge(
            "fraudshield_database_ready",
            "Whether PostgreSQL is reachable with the current migration.",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "fraudshield_model_info",
            "Immutable production model identity.",
            ("model_name", "model_version", "model_alias", "model_family"),
            registry=self.registry,
        )

    def observe_http(
        self,
        *,
        method: str,
        normalized_route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP request with bounded labels."""

        status_class = f"{max(1, min(5, status_code // 100))}xx"
        self.http_requests.labels(method, normalized_route, status_class).inc()
        self.http_duration.labels(method, normalized_route).observe(
            max(0.0, duration_seconds)
        )

    def observe_prediction(
        self,
        *,
        prediction: int,
        risk_level: str,
        transaction_type: str,
        model_version: str,
        fraud_score: float,
    ) -> None:
        """Record one newly persisted model result, never an idempotent replay."""

        self.predictions.labels(
            str(prediction), risk_level, transaction_type, model_version
        ).inc()
        self.prediction_score.labels(model_version).observe(fraud_score)

    def set_model_readiness(self, ready: bool, info: dict[str, Any] | None = None) -> None:
        """Publish readiness and the bounded immutable model identity."""

        self.model_ready.set(1 if ready else 0)
        self.model_info.clear()
        if ready and info is not None:
            self.model_info.labels(
                str(info["registered_model_name"]),
                str(info["resolved_model_version"]),
                str(info["alias"]),
                str(info["model_family"]),
            ).set(1)

    def set_database_readiness(self, ready: bool) -> None:
        """Publish current database and migration readiness."""

        self.database_ready.set(1 if ready else 0)

    def render(self) -> bytes:
        """Render this application's isolated registry."""

        return generate_latest(self.registry)
