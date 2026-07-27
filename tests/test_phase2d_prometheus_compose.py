"""Phase 2D.4 Prometheus Compose and stack-script contracts."""

from pathlib import Path

import yaml


def test_prometheus_compose_service_is_pinned_local_and_persistent() -> None:
    raw = Path("compose.yaml").read_text(encoding="utf-8")
    compose = yaml.safe_load(raw)
    service = compose["services"]["prometheus"]
    assert service["image"] == "prom/prometheus:v3.13.1"
    assert service["ports"] == ["127.0.0.1:9090:9090"]
    assert service["restart"] == "unless-stopped"
    assert "--storage.tsdb.retention.time=7d" in service["command"]
    assert service["healthcheck"]["test"] == [
        "CMD",
        "/bin/promtool",
        "check",
        "healthy",
        "--url=http://127.0.0.1:9090",
    ]
    assert service["depends_on"]["api"]["condition"] == "service_healthy"
    assert service["depends_on"]["monitor"]["condition"] == "service_healthy"
    assert "prometheus_data:/prometheus" in service["volumes"]
    assert "prometheus_data" in compose["volumes"]
    assert all(
        item.endswith(":ro")
        for item in service["volumes"]
        if item.startswith("./monitoring/")
    )
    assert "/var/run/docker.sock" not in raw
    assert "remote_write" not in raw
    assert "grafana" not in compose["services"]


def test_stack_scripts_report_health_and_preserve_prometheus_volume() -> None:
    start = Path("scripts/start_stack.ps1").read_text(encoding="utf-8")
    status = Path("scripts/stack_status.ps1").read_text(encoding="utf-8")
    stop = Path("scripts/stop_stack.ps1").read_text(encoding="utf-8")
    assert 'Service -eq "prometheus"' in start
    assert "http://127.0.0.1:9090" in start
    assert "prometheusHealth" in start
    assert "fraudshield_prometheus_data" in status
    assert "docker compose down" in stop
    assert "down -v" not in stop
