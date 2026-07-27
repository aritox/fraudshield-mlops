"""Grafana provisioning and dashboard contracts."""

import json
from pathlib import Path

import yaml

DATASOURCE_PATH = Path("monitoring/grafana/provisioning/datasources/prometheus.yml")
DASHBOARD_PROVIDER_PATH = Path(
    "monitoring/grafana/provisioning/dashboards/dashboards.yml"
)
DASHBOARD_PATH = Path("monitoring/grafana/dashboards/fraudshield-overview.json")

REQUIRED_PANELS = {
    "API health",
    "Database readiness",
    "Monitoring worker freshness",
    "Request rate",
    "Error rate",
    "p50 latency",
    "p95 latency",
    "p99 latency",
    "Predictions by risk level",
    "Predictions by transaction type",
    "High-risk alert rate",
    "Fraud-score histogram",
    "Monitoring-window event count",
    "Feature PSI",
    "Feature drift severity",
    "Labeled outcome count",
    "Precision",
    "Recall",
    "F2",
    "Monitoring-run status",
}


def test_prometheus_datasource_is_default_and_local() -> None:
    datasource = yaml.safe_load(DATASOURCE_PATH.read_text(encoding="utf-8"))
    assert datasource["apiVersion"] == 1
    assert len(datasource["datasources"]) == 1
    prometheus = datasource["datasources"][0]
    assert prometheus == {
        "name": "Prometheus",
        "uid": "prometheus",
        "type": "prometheus",
        "access": "proxy",
        "url": "http://prometheus:9090",
        "isDefault": True,
        "editable": False,
        "jsonData": {"timeInterval": "15s", "httpMethod": "POST"},
    }


def test_dashboard_provider_and_dashboard_are_valid() -> None:
    provider = yaml.safe_load(DASHBOARD_PROVIDER_PATH.read_text(encoding="utf-8"))
    assert provider["apiVersion"] == 1
    assert provider["providers"][0]["options"]["path"] == "/etc/grafana/dashboards"
    assert provider["providers"][0]["allowUiUpdates"] is False
    assert provider["providers"][0]["disableDeletion"] is True

    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    assert dashboard["title"] == "FraudShield Production Monitoring"
    assert dashboard["uid"] == "fraudshield-production-monitoring"
    assert dashboard["refresh"] == "15s"
    panels = dashboard["panels"]
    assert {panel["title"] for panel in panels} == REQUIRED_PANELS
    assert len({panel["id"] for panel in panels}) == len(REQUIRED_PANELS)
    for panel in panels:
        assert panel["datasource"] == {"type": "prometheus", "uid": "prometheus"}
        assert panel["fieldConfig"]["defaults"]["unit"]
        assert panel["targets"]
        assert all(target["legendFormat"] for target in panel["targets"])
        assert all(target["expr"] for target in panel["targets"])


def test_grafana_files_contain_no_cloud_credentials_or_machine_paths() -> None:
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DATASOURCE_PATH, DASHBOARD_PROVIDER_PATH, DASHBOARD_PATH)
    ).lower()
    for forbidden in (
        "grafana.net",
        "grafana.com/api",
        "grafana cloud",
        "api_key",
        "bearer",
        "password",
        "c:\\\\users\\",
        "c:/users/",
        "/users/",
        "/home/",
    ):
        assert forbidden not in content
