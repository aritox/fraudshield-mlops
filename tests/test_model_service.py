"""Tests for read-only champion loading and inference."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraudshield.api.config import load_api_config
from fraudshield.api.errors import ModelLoadError
from fraudshield.api.model_service import ProductionModelService
from fraudshield.api.schemas import TransactionRequest
from fraudshield.tracking.config import load_mlflow_config
from test_api_config import _write_configs


class FakeSchema:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def input_names(self) -> list[str]:
        return list(self._names)


class FakeMetadata:
    def get_input_schema(self) -> FakeSchema:
        return FakeSchema(["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"])

    def get_output_schema(self) -> FakeSchema:
        return FakeSchema(["fraud_score", "prediction", "threshold", "risk_level"])


class FakePyFunc:
    def __init__(self) -> None:
        self.metadata = FakeMetadata()
        self.predict_calls = 0

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.predict_calls += 1
        scores = np.array([0.1, 0.99, 0.7], dtype=float)[: len(frame)]
        return pd.DataFrame(
            {
                "fraud_score": scores,
                "prediction": (scores >= 0.98310834).astype(int),
                "threshold": np.full(len(frame), 0.98310834),
                "risk_level": np.where(
                    scores >= 0.98310834,
                    "high",
                    np.where(scores >= 0.5, "medium", "low"),
                ),
            },
            index=frame.index,
        )

    def fit(self, *_args, **_kwargs):
        raise AssertionError("fit must never be called")

    def partial_fit(self, *_args, **_kwargs):
        raise AssertionError("partial_fit must never be called")


class FakeClient:
    def __init__(self, family: str = "SGDClassifier") -> None:
        self.calls = 0
        self.version = SimpleNamespace(
            name="fraudshield-production-sgd",
            version=1,
            aliases=["champion"],
            tags={
                "role": "production",
                "model_family": family,
                "threshold_source": "phase1d_validation_f2",
                "operational_threshold": "0.98310834",
                "test_used_for_selection": "false",
                "source_model_sha256": "abc123",
            },
        )

    def get_model_version_by_alias(self, name: str, alias: str):
        self.calls += 1
        assert name == "fraudshield-production-sgd"
        assert alias == "champion"
        return self.version


def _service(tmp_path: Path, family: str = "SGDClassifier"):
    api_path = _write_configs(tmp_path)
    database = tmp_path / "artifacts" / "mlflow" / "mlflow.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"synthetic registry fixture")
    (tmp_path / "artifacts" / "mlflow" / "artifacts").mkdir()
    config = load_api_config(api_path, root=tmp_path)
    mlflow_config = load_mlflow_config(root=tmp_path)
    client = FakeClient(family)
    model = FakePyFunc()
    loader_calls: list[str] = []

    def loader(uri: str):
        loader_calls.append(uri)
        return model

    service = ProductionModelService(
        config,
        mlflow_config=mlflow_config,
        client=client,
        model_loader=loader,
    )
    return service, client, model, loader_calls


def _request(step: int) -> TransactionRequest:
    return TransactionRequest(
        step=step,
        type="TRANSFER",
        amount=100.0,
        oldbalanceOrg=100.0,
        oldbalanceDest=0.0,
    )


def test_champion_loads_once_warms_and_predicts_in_order(tmp_path: Path) -> None:
    service, client, model, loader_calls = _service(tmp_path)

    service.load()
    service.load()
    predictions = service.predict_batch([_request(1), _request(2), _request(3)])

    assert service.is_ready()
    assert client.calls == 1
    assert loader_calls == ["models:/fraudshield-production-sgd@champion"]
    assert model.predict_calls == 2  # one warm-up and one batch call
    assert [item["item_index"] for item in predictions] == [0, 1, 2]
    assert [item["risk_level"] for item in predictions] == ["low", "high", "medium"]
    assert all(item["threshold"] == 0.98310834 for item in predictions)
    assert service.model_info()["model_family"] == "SGDClassifier"


def test_xgboost_family_is_rejected_safely(tmp_path: Path) -> None:
    service, _, _, _ = _service(tmp_path, family="XGBoost")

    with pytest.raises(ModelLoadError):
        service.load()

    assert not service.is_ready()
    assert service.unavailable_reason() == "Production model is unavailable"


def test_prohibited_data_access_during_load_is_blocked(tmp_path: Path) -> None:
    service, _, _, _ = _service(tmp_path)
    protected = tmp_path / "data" / "processed" / "test.parquet"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"synthetic fixture")

    def prohibited_loader(_uri: str):
        protected.read_bytes()

    service._model_loader = prohibited_loader
    with pytest.raises(ModelLoadError):
        service.load()
