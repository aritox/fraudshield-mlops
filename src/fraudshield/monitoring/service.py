"""Transactional production-monitoring orchestration over persisted audit data."""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from fraudshield.monitoring.config import MonitoringConfig
from fraudshield.monitoring.drift import categorical_drift, numeric_drift
from fraudshield.monitoring.performance import PerformanceResult, calculate_performance
from fraudshield.monitoring.reference import load_reference_profile
from fraudshield.monitoring.repository import MonitoringRepository, PersistedMonitoringEvent
from fraudshield.persistence.models import MonitoringMetric, MonitoringRun

_DRIFT_RANK = {"insufficient_data": 0, "stable": 1, "moderate": 2, "significant": 3}


@dataclass(frozen=True)
class MonitoringMetricValue:
    metric_name: str
    feature_name: str | None
    metric_value: float | None
    severity: str
    sample_size: int


@dataclass(frozen=True)
class MonitoringRunResult:
    run_id: uuid.UUID
    window_start: datetime
    window_end: datetime
    reference_version: str
    event_count: int
    labeled_count: int
    status: str
    overall_drift_status: str
    metrics: tuple[MonitoringMetricValue, ...]

    def metric(self, name: str, feature: str | None = None) -> float | None:
        for item in self.metrics:
            if item.metric_name == name and item.feature_name == feature:
                return item.metric_value
        return None


class MonitoringUnavailableError(RuntimeError):
    pass


def _window_end(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("Monitoring window end must be timezone-aware")
    return current.astimezone(UTC).replace(microsecond=0)


def _informational_metric(
    name: str,
    value: float | None,
    sample_size: int,
    feature: str | None = None,
) -> MonitoringMetricValue:
    return MonitoringMetricValue(
        metric_name=name,
        feature_name=feature,
        metric_value=value,
        severity="informational" if value is not None else "unavailable",
        sample_size=sample_size,
    )


def _output_metrics(
    events: list[PersistedMonitoringEvent],
    *,
    prefix: str = "",
) -> list[MonitoringMetricValue]:
    sample_size = len(events)
    alert_rate = (
        sum(event.prediction == 1 for event in events) / sample_size if sample_size else None
    )
    mean_score = (
        sum(event.fraud_score for event in events) / sample_size if sample_size else None
    )
    metrics = [
        _informational_metric(f"{prefix}alert_rate", alert_rate, sample_size),
        _informational_metric(f"{prefix}mean_fraud_score", mean_score, sample_size),
    ]
    prediction_counts = Counter(event.prediction for event in events)
    for value in (0, 1):
        proportion = prediction_counts[value] / sample_size if sample_size else None
        metrics.append(
            _informational_metric(
                f"{prefix}prediction_distribution", proportion, sample_size, str(value)
            )
        )
    risk_counts = Counter(event.risk_level for event in events)
    for value in ("low", "medium", "high"):
        proportion = risk_counts[value] / sample_size if sample_size else None
        metrics.append(
            _informational_metric(
                f"{prefix}risk_level_distribution", proportion, sample_size, value
            )
        )
    return metrics


def _feature_metrics(
    events: list[PersistedMonitoringEvent],
    profile: dict,
    config: MonitoringConfig,
) -> list[MonitoringMetricValue]:
    numeric_values = {
        "step": (event.step for event in events),
        "log1p(amount)": (event.amount for event in events),
        "log1p(oldbalanceOrg)": (event.oldbalance_origin for event in events),
        "log1p(oldbalanceDest)": (event.oldbalance_destination for event in events),
    }
    metrics = []
    for feature, values in numeric_values.items():
        result = numeric_drift(
            profile["numeric_features"][feature],
            values,
            minimum_events=config.monitoring.minimum_events,
            epsilon=config.reference.epsilon,
            thresholds=config.drift,
        )
        metrics.append(
            MonitoringMetricValue(
                "feature_psi", feature, result.metric_value, result.severity, result.sample_size
            )
        )
    categorical = categorical_drift(
        profile["categorical_features"]["type"],
        (event.transaction_type for event in events),
        minimum_events=config.monitoring.minimum_events,
        epsilon=config.reference.epsilon,
        thresholds=config.drift,
    )
    metrics.append(
        MonitoringMetricValue(
            "feature_psi",
            "type",
            categorical.metric_value,
            categorical.severity,
            categorical.sample_size,
        )
    )
    return metrics


def _overall_drift(metrics: list[MonitoringMetricValue]) -> str:
    severities = [item.severity for item in metrics if item.metric_name == "feature_psi"]
    if not severities or all(item == "insufficient_data" for item in severities):
        return "insufficient_data"
    return max(severities, key=lambda item: _DRIFT_RANK[item])


def _performance_metrics(
    result: PerformanceResult,
) -> list[MonitoringMetricValue]:
    metrics = [
        _informational_metric(
            "performance_available", 1.0 if result.available else 0.0, result.labeled_count
        )
    ]
    if result.available:
        metrics.extend(
            _informational_metric(name, value, result.labeled_count)
            for name, value in result.metrics.items()
        )
    return metrics


class MonitoringService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        config: MonitoringConfig,
        repository: MonitoringRepository | None = None,
        reference_profile: dict | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.repository = repository or MonitoringRepository()
        self.reference_profile = reference_profile or load_reference_profile(config)

    def _calculate(
        self,
        current: list[PersistedMonitoringEvent],
        previous: list[PersistedMonitoringEvent],
    ) -> tuple[str, str, PerformanceResult, list[MonitoringMetricValue]]:
        feature_metrics = _feature_metrics(current, self.reference_profile, self.config)
        overall_drift = _overall_drift(feature_metrics)
        performance = calculate_performance(
            current,
            minimum_labeled_events=self.config.monitoring.minimum_labeled_events,
        )
        if len(current) < self.config.monitoring.minimum_events:
            status = "insufficient_data"
        elif not performance.available:
            status = "insufficient_labeled_data"
        else:
            status = "completed"
        metrics = [
            *feature_metrics,
            *_output_metrics(current),
            *_output_metrics(previous, prefix="previous_window_"),
            *_performance_metrics(performance),
        ]
        if any(
            item.metric_value is not None and not math.isfinite(item.metric_value)
            for item in metrics
        ):
            raise ValueError("Monitoring metric is not finite")
        return status, overall_drift, performance, metrics

    def run_once(self, *, window_end: datetime | None = None) -> MonitoringRunResult:
        end = _window_end(window_end)
        window = timedelta(hours=self.config.monitoring.window_hours)
        start = end - window
        previous_start = start - window
        run_id = uuid.uuid4()
        current: list[PersistedMonitoringEvent] = []
        try:
            with self.session_factory() as session, session.begin():
                current = self.repository.load_events(session, start, end)
                previous = self.repository.load_events(session, previous_start, start)
                status, overall, performance, values = self._calculate(current, previous)
                completed_at = datetime.now(UTC)
                run = MonitoringRun(
                    run_id=run_id,
                    window_start=start,
                    window_end=end,
                    reference_version=self.config.reference.version,
                    event_count=len(current),
                    labeled_count=performance.labeled_count,
                    status=status,
                    overall_drift_status=overall,
                    completed_at=completed_at,
                )
                rows = [
                    MonitoringMetric(
                        metric_id=uuid.uuid4(),
                        run_id=run_id,
                        metric_name=value.metric_name,
                        feature_name=value.feature_name,
                        metric_value=value.metric_value,
                        severity=value.severity,
                        sample_size=value.sample_size,
                    )
                    for value in values
                ]
                self.repository.add_run(session, run, rows)
                session.flush()
            return MonitoringRunResult(
                run_id=run_id,
                window_start=start,
                window_end=end,
                reference_version=self.config.reference.version,
                event_count=len(current),
                labeled_count=performance.labeled_count,
                status=status,
                overall_drift_status=overall,
                metrics=tuple(values),
            )
        except SQLAlchemyError as error:
            raise MonitoringUnavailableError("Monitoring persistence is unavailable") from error
        except Exception:
            self._persist_failed_run(run_id, start, end, len(current))
            return MonitoringRunResult(
                run_id=run_id,
                window_start=start,
                window_end=end,
                reference_version=self.config.reference.version,
                event_count=len(current),
                labeled_count=0,
                status="failed",
                overall_drift_status="failed",
                metrics=(),
            )

    def _persist_failed_run(
        self,
        run_id: uuid.UUID,
        window_start: datetime,
        window_end: datetime,
        event_count: int,
    ) -> None:
        try:
            with self.session_factory() as session, session.begin():
                self.repository.add_run(
                    session,
                    MonitoringRun(
                        run_id=run_id,
                        window_start=window_start,
                        window_end=window_end,
                        reference_version=self.config.reference.version,
                        event_count=event_count,
                        labeled_count=0,
                        status="failed",
                        overall_drift_status="failed",
                        completed_at=datetime.now(UTC),
                    ),
                    [],
                )
                session.flush()
        except SQLAlchemyError as error:
            raise MonitoringUnavailableError("Monitoring failure could not be persisted") from error
