"""Phase 2D.5 Grafana Compose, environment, and stack-script contracts."""

from pathlib import Path

import yaml


def test_grafana_compose_service_is_pinned_local_and_persistent() -> None:
    raw = Path("compose.yaml").read_text(encoding="utf-8")
    compose = yaml.safe_load(raw)
    service = compose["services"]["grafana"]
    assert service["image"] == "grafana/grafana:13.1.0"
    assert service["ports"] == [
        "127.0.0.1:${FRAUDSHIELD_GRAFANA_PORT:-3000}:3000"
    ]
    assert service["depends_on"] == {
        "prometheus": {"condition": "service_healthy"}
    }
    assert service["restart"] == "unless-stopped"
    assert service["healthcheck"]["test"] == [
        "CMD",
        "wget",
        "-q",
        "-O",
        "/dev/null",
        "http://127.0.0.1:3000/api/health",
    ]
    assert "grafana_data:/var/lib/grafana" in service["volumes"]
    assert "grafana_data" in compose["volumes"]
    assert all(
        item.endswith(":ro")
        for item in service["volumes"]
        if item.startswith("./monitoring/grafana/")
    )
    environment = service["environment"]
    assert environment["GF_AUTH_ANONYMOUS_ENABLED"] == "true"
    assert environment["GF_AUTH_ANONYMOUS_ORG_ROLE"] == "Viewer"
    assert environment["GF_USERS_ALLOW_SIGN_UP"] == "false"
    assert environment["GF_ANALYTICS_REPORTING_ENABLED"] == "false"
    assert environment["GF_ANALYTICS_CHECK_FOR_UPDATES"] == "false"
    assert environment["GF_PLUGINS_PREINSTALL_DISABLED"] == "true"
    assert "${GRAFANA_ADMIN_PASSWORD:" in environment["GF_SECURITY_ADMIN_PASSWORD"]
    assert "/var/run/docker.sock" not in raw
    assert "GF_INSTALL_PLUGINS" not in raw


def test_local_environment_and_stack_scripts_cover_grafana_safely() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")
    init = Path("scripts/init_local_env.ps1").read_text(encoding="utf-8")
    start = Path("scripts/start_stack.ps1").read_text(encoding="utf-8")
    status = Path("scripts/stack_status.ps1").read_text(encoding="utf-8")
    stop = Path("scripts/stop_stack.ps1").read_text(encoding="utf-8")

    assert "GRAFANA_ADMIN_USER=admin" in example
    assert "GRAFANA_ADMIN_PASSWORD=replace_with_local_secret" in example
    assert "FRAUDSHIELD_GRAFANA_PORT=3000" in example
    assert 'ContainsKey("GRAFANA_ADMIN_PASSWORD")' in init
    assert "New-RandomSecret" in init
    assert "Existing .env preserved" in init
    assert 'Write-Host "$grafanaPassword"' not in init
    assert 'Service -eq "grafana"' in start
    assert "grafanaHealth" in start
    assert "http://127.0.0.1:$grafanaPort" in start
    assert "fraudshield_grafana_data" in status
    assert "docker compose down" in stop
    assert "down -v" not in stop
