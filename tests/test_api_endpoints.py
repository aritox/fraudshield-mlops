"""Tests for FastAPI routes, middleware, and safe failures."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fraudshield.api.config import load_api_config
from fraudshield.api.errors import ModelLoadError
from fraudshield.api.main import create_app
from fraudshield.persistence.database import DatabaseHealth
from fraudshield.persistence.models import Base
from fraudshield.persistence.service import PredictionPersistenceService
from test_api_config import _write_configs


def _transaction(step: int = 1, amount: float = 100.0) -> dict[str, object]:
    return {
        "step": step,
        "type": "TRANSFER",
        "amount": amount,
        "oldbalanceOrg": 100.0,
        "oldbalanceDest": 0.0,
    }


class ReadyService:
    def __init__(self) -> None:
        self.loaded = 0
        self.batch_calls = 0

    def load(self) -> None:
        self.loaded += 1

    def is_ready(self) -> bool:
        return True

    def model_info(self):
        return {
            "registered_model_name": "fraudshield-production-sgd",
            "resolved_model_version": "1",
            "alias": "champion",
            "model_family": "SGDClassifier",
            "frozen_threshold": 0.98310834,
            "raw_input_fields": [
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "oldbalanceDest",
            ],
            "output_fields": ["fraud_score", "prediction", "threshold", "risk_level"],
            "risk_levels": {
                "low": "score < 0.5",
                "medium_minimum": 0.5,
                "high_minimum": 0.98310834,
            },
            "loaded_timestamp_utc": "2026-01-01T00:00:00Z",
            "source_model_checksum": "abc123",
            "synthetic_dataset_warning": "PaySim is synthetic.",
        }

    def predict_one(self, transaction):
        return self.predict_batch([transaction])[0]

    def predict_batch(self, transactions):
        self.batch_calls += 1
        return [
            {
                "item_index": index,
                "fraud_score": 0.99 if transaction.step % 2 else 0.1,
                "prediction": 1 if transaction.step % 2 else 0,
                "threshold": 0.98310834,
                "risk_level": "high" if transaction.step % 2 else "low",
            }
            for index, transaction in enumerate(transactions)
        ]


class UnavailableService(ReadyService):
    def load(self) -> None:
        raise ModelLoadError("technical local detail")

    def is_ready(self) -> bool:
        return False


class BrokenInferenceService(ReadyService):
    def predict_one(self, transaction):
        raise RuntimeError("C:\\private\\artifact\\path")


def _client(tmp_path: Path, service, maximum_batch_size: int = 1000, **kwargs):
    config = load_api_config(
        _write_configs(tmp_path, maximum_batch_size=maximum_batch_size),
        root=tmp_path,
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        connection.execute(text("INSERT INTO alembic_version VALUES ('phase2d_001')"))
    persistence = PredictionPersistenceService(sessionmaker(bind=engine, expire_on_commit=False))

    class HealthyDatabase:
        @staticmethod
        def status():
            return DatabaseHealth(True, "PostgreSQL", "current")

    app = create_app(
        config,
        service,
        persistence_service=persistence,
        database_health=HealthyDatabase(),
        **kwargs,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_root_health_readiness_model_info_and_docs(tmp_path: Path) -> None:
    service = ReadyService()
    with _client(tmp_path, service) as client:
        assert client.get("/").json()["readiness_status"] == "ready"
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["model_alias"] == "champion"
        info = client.get("/model/info")
        assert info.status_code == 200
        assert info.json()["model_family"] == "SGDClassifier"
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
    assert service.loaded == 1


def test_single_and_batch_prediction_headers_order_and_safe_logs(
    tmp_path: Path,
    caplog,
) -> None:
    service = ReadyService()
    caplog.set_level(logging.INFO, logger="fraudshield.api")
    with _client(tmp_path, service) as client:
        single = client.post(
            "/predict",
            json=_transaction(amount=123456.789),
            headers={"X-Request-ID": "00000000-0000-4000-8000-000000000123"},
        )
        batch = client.post(
            "/predict/batch",
            json={"transactions": [_transaction(1), _transaction(2)]},
        )

    assert single.status_code == 200
    assert single.headers["X-Request-ID"] == "00000000-0000-4000-8000-000000000123"
    UUID(single.json()["prediction_id"])
    assert float(single.headers["X-Process-Time-Ms"]) >= 0
    assert single.json()["threshold"] == 0.98310834
    assert batch.status_code == 200
    assert [item["item_index"] for item in batch.json()["predictions"]] == [0, 1]
    assert [item["risk_level"] for item in batch.json()["predictions"]] == ["high", "low"]
    assert service.batch_calls == 2
    assert "123456.789" not in caplog.text
    assert "oldbalanceOrg" not in caplog.text


def test_validation_batch_limit_and_unavailable_errors(tmp_path: Path) -> None:
    with _client(tmp_path / "valid", ReadyService(), maximum_batch_size=1) as client:
        invalid = client.post("/predict", json={**_transaction(), "isFraud": 0})
        oversized = client.post(
            "/predict/batch",
            json={"transactions": [_transaction(1), _transaction(2)]},
        )
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_error"
    assert oversized.status_code == 422
    assert oversized.json()["error"] == "batch_size_exceeded"

    with _client(tmp_path / "unavailable", UnavailableService()) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        response = client.post("/predict", json=_transaction())
        assert response.status_code == 503
        assert response.json()["error"] == "model_not_ready"
        assert "technical" not in response.text


def test_unexpected_inference_failure_is_sanitized(tmp_path: Path) -> None:
    with _client(tmp_path, BrokenInferenceService()) as client:
        response = client.post("/predict", json=_transaction())

    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert "private" not in response.text
