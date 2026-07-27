"""Bounded-cardinality Prometheus exposition for the monitoring worker."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable

from prometheus_client import CollectorRegistry
from prometheus_client.core import GaugeMetricFamily

from fraudshield.monitoring.service import MonitoringRunResult

RUN_STATUSES = ("completed", "insufficient_data", "insufficient_labeled_data", "failed")
DRIFT_STATUSES = ("stable", "moderate", "significant", "insufficient_data", "failed")
PERFORMANCE_METRICS = (
    "precision",
    "recall",
    "f1",
    "f2",
    "false_positive_rate",
    "fraud_amount_recall",
)


class WorkerMetrics:
    """A dynamic collector that omits unavailable performance observations."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self._lock = threading.Lock()
        self._last_success = 0.0
        self._status = "failed"
        self._event_count = 0
        self._labeled_count = 0
        self._feature_psi: dict[str, float] = {}
        self._feature_severity: dict[str, str] = {}
        self._alert_rate: float | None = None
        self._mean_score: float | None = None
        self._performance_available = False
        self._performance: dict[str, float] = {}
        self.registry.register(self)

    def record(self, result: MonitoringRunResult) -> None:
        with self._lock:
            self._status = result.status if result.status in RUN_STATUSES else "failed"
            if result.status != "failed":
                self._last_success = time.time()
            self._event_count = result.event_count
            self._labeled_count = result.labeled_count
            self._feature_psi = {
                item.feature_name: item.metric_value
                for item in result.metrics
                if item.metric_name == "feature_psi"
                and item.feature_name is not None
                and item.metric_value is not None
            }
            self._feature_severity = {
                item.feature_name: item.severity
                for item in result.metrics
                if item.metric_name == "feature_psi"
                and item.feature_name is not None
                and item.severity in DRIFT_STATUSES
            }
            self._alert_rate = result.metric("alert_rate")
            self._mean_score = result.metric("mean_fraud_score")
            self._performance_available = result.metric("performance_available") == 1.0
            self._performance = {
                name: value
                for name in PERFORMANCE_METRICS
                if (value := result.metric(name)) is not None
            }

    def record_failure(self) -> None:
        with self._lock:
            self._status = "failed"
            self._event_count = 0
            self._labeled_count = 0
            self._feature_psi = {}
            self._feature_severity = {}
            self._alert_rate = None
            self._mean_score = None
            self._performance_available = False
            self._performance = {}

    def collect(self) -> Iterable[GaugeMetricFamily]:
        with self._lock:
            last_success = self._last_success
            status = self._status
            event_count = self._event_count
            labeled_count = self._labeled_count
            feature_psi = dict(self._feature_psi)
            feature_severity = dict(self._feature_severity)
            alert_rate = self._alert_rate
            mean_score = self._mean_score
            performance_available = self._performance_available
            performance = dict(self._performance)

        last = GaugeMetricFamily(
            "fraudshield_monitoring_last_success_timestamp_seconds",
            "Unix timestamp of the most recent successful monitoring run.",
        )
        last.add_metric([], last_success)
        yield last

        run_status = GaugeMetricFamily(
            "fraudshield_monitoring_run_status",
            "Current monitoring run status.",
            labels=["status"],
        )
        run_status.add_metric([status], 1.0)
        yield run_status

        window_events = GaugeMetricFamily(
            "fraudshield_monitoring_window_events",
            "Persisted predictions in the current monitoring window.",
        )
        window_events.add_metric([], event_count)
        yield window_events

        labeled_events = GaugeMetricFamily(
            "fraudshield_monitoring_labeled_events",
            "Persisted predictions with delayed outcomes in the current window.",
        )
        labeled_events.add_metric([], labeled_count)
        yield labeled_events

        psi = GaugeMetricFamily(
            "fraudshield_feature_psi",
            "Feature population stability index against the frozen train reference.",
            labels=["feature"],
        )
        for feature, value in sorted(feature_psi.items()):
            psi.add_metric([feature], value)
        yield psi

        severity = GaugeMetricFamily(
            "fraudshield_feature_drift_severity",
            "Current bounded drift severity for each feature.",
            labels=["feature", "severity"],
        )
        for feature, value in sorted(feature_severity.items()):
            severity.add_metric([feature, value], 1.0)
        yield severity

        alert = GaugeMetricFamily(
            "fraudshield_alert_rate", "Fraction of current predictions classified as fraud."
        )
        if alert_rate is not None:
            alert.add_metric([], alert_rate)
        yield alert

        score = GaugeMetricFamily(
            "fraudshield_mean_fraud_score", "Mean uncalibrated fraud score in the current window."
        )
        if mean_score is not None:
            score.add_metric([], mean_score)
        yield score

        available = GaugeMetricFamily(
            "fraudshield_performance_available",
            "Whether delayed-outcome performance has sufficient labeled events.",
        )
        available.add_metric([], 1.0 if performance_available else 0.0)
        yield available

        for name in PERFORMANCE_METRICS:
            if name not in performance:
                continue
            family = GaugeMetricFamily(
                f"fraudshield_production_{name}",
                f"Production {name.replace('_', ' ')} from persisted delayed outcomes.",
            )
            family.add_metric([], performance[name])
            yield family
