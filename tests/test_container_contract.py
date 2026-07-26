"""Static Docker and Compose security contract tests."""

import uuid
from pathlib import Path

import pytest
import yaml

from fraudshield.container.smoke_test import require_single_model_inference_event


def test_dockerfile_runtime_contract() -> None:
    text = Path("Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM python:3.12-slim-bookworm") == 2
    assert "USER 10001:10001" in text
    assert "--reload" not in text
    assert '"0.0.0.0"' in text
    assert "artifacts/mlflow" not in text
    assert "data/raw" not in text
    assert "COPY artifacts/docker_build_wheels/ /build-wheels/" in text
    assert "--no-index --find-links=/build-wheels" in text
    assert "--no-build-isolation" in text
    assert "--trusted-host" not in text
    assert "--index-url" not in text
    assert "PIP_NO_VERIFY_CERTS" not in text


def test_compose_security_and_ordering_contract() -> None:
    raw = Path("compose.yaml").read_text(encoding="utf-8")
    compose = yaml.safe_load(raw)
    assert "version" not in compose
    assert set(compose["services"]) == {"postgres", "migrate", "api"}
    assert compose["services"]["postgres"]["image"] == "postgres:16-alpine"
    assert compose["services"]["api"]["read_only"] is True
    assert compose["services"]["api"]["cap_drop"] == ["ALL"]
    assert "service_completed_successfully" in raw
    assert "127.0.0.1:${FRAUDSHIELD_API_PORT:-8000}:8000" in raw
    assert "/var/run/docker.sock" not in raw


def test_smoke_log_parser_counts_only_explicit_inference_events() -> None:
    request_id = uuid.UUID("11111111-1111-4111-8111-111111111111")
    summary = (
        f'{{"idempotent_replay":true,"model_inference_invoked":false,'
        f'"request_id":"{request_id}"}}'
    )
    event = f'{{"event":"model_inference_invoked","request_id":"{request_id}"}}'

    assert require_single_model_inference_event(f"{event}\n{summary}", request_id) == 1
    with pytest.raises(RuntimeError, match="Replay invoked model inference"):
        require_single_model_inference_event(f"{event}\n{summary}\n{event}", request_id)
