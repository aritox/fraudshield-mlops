"""Prometheus scrape, recording-rule, and alert-rule contracts."""

from pathlib import Path

import yaml

PROMETHEUS_CONFIG = Path("monitoring/prometheus/prometheus.yml")
PROMETHEUS_RULES = Path("monitoring/prometheus/rules/fraudshield.rules.yml")


def test_prometheus_scrapes_only_expected_local_services() -> None:
    raw = PROMETHEUS_CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(raw)
    assert config["global"] == {
        "scrape_interval": "15s",
        "evaluation_interval": "15s",
    }
    assert config["rule_files"] == ["/etc/prometheus/rules/fraudshield.rules.yml"]
    jobs = {item["job_name"]: item for item in config["scrape_configs"]}
    assert set(jobs) == {"fraudshield-api", "fraudshield-monitor", "prometheus"}
    assert jobs["fraudshield-api"]["static_configs"] == [{"targets": ["api:8000"]}]
    assert jobs["fraudshield-monitor"]["static_configs"] == [
        {"targets": ["monitor:8001"]}
    ]
    assert jobs["prometheus"]["static_configs"] == [{"targets": ["prometheus:9090"]}]
    assert all(item["metrics_path"] == "/metrics" for item in jobs.values())
    assert "remote_write" not in config
    assert "alerting" not in config
    assert "alertmanager" not in raw.lower()
    assert "password" not in raw.lower()


def test_recording_and_alert_rules_are_complete_and_bounded() -> None:
    raw = PROMETHEUS_RULES.read_text(encoding="utf-8")
    document = yaml.safe_load(raw)
    groups = {group["name"]: group for group in document["groups"]}
    recordings = {
        item["record"]: item for item in groups["fraudshield-recording-rules"]["rules"]
    }
    assert set(recordings) == {
        "fraudshield:http_request_rate:5m",
        "fraudshield:http_error_rate:5m",
        "fraudshield:http_request_duration_seconds:p50_5m",
        "fraudshield:http_request_duration_seconds:p95_5m",
        "fraudshield:http_request_duration_seconds:p99_5m",
        "fraudshield:high_risk_prediction_rate:5m",
    }
    alerts = {item["alert"]: item for item in groups["fraudshield-alert-rules"]["rules"]}
    assert set(alerts) == {
        "FraudShieldAPIDown",
        "FraudShieldDatabaseNotReady",
        "FraudShieldHighErrorRate",
        "FraudShieldHighP95Latency",
        "FraudShieldMonitoringStale",
        "FraudShieldSignificantFeatureDrift",
    }
    assert all(item.get("for") not in (None, "0s") for item in alerts.values())
    assert all(item["labels"]["severity"] in {"warning", "critical"} for item in alerts.values())
    assert "request_id" not in raw
    assert "prediction_id" not in raw
    assert "remote_write" not in raw
    assert "alertmanager" not in raw.lower()
