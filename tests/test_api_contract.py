"""Tests for deterministic OpenAPI and API-manifest export."""

from __future__ import annotations

import json
from pathlib import Path

from fraudshield.api import export_contract as contract_module
from fraudshield.tracking.mlflow_setup import sha256_file
from test_api_config import _write_configs


def test_openapi_and_manifest_export_without_model_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_configs(tmp_path)
    monkeypatch.setattr(contract_module, "_git_commit", lambda _root: "abc123")

    manifest = contract_module.export_contract(root=tmp_path, config_path=config_path)

    openapi_path = tmp_path / "artifacts" / "api" / "openapi.json"
    manifest_path = tmp_path / "artifacts" / "api" / "api_manifest.json"
    openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_routes = {
        "/",
        "/health/live",
        "/health/ready",
        "/model/info",
        "/predict",
        "/predict/batch",
    }

    assert required_routes.issubset(openapi["paths"])
    assert "TransactionRequest" in openapi["components"]["schemas"]
    assert "BatchPredictionResponse" in openapi["components"]["schemas"]
    assert stored_manifest == manifest
    assert manifest["openapi_sha256"] == sha256_file(openapi_path)
    assert manifest["source_git_commit"] == "abc123"
    assert manifest["model_retrained"] is False
    assert manifest["parquet_data_accessed"] is False
    combined = openapi_path.read_text() + manifest_path.read_text()
    assert str(tmp_path) not in combined
    assert "password" not in combined.lower()
