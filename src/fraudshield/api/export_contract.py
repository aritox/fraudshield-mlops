"""Export the Phase 2B OpenAPI contract and reproducibility manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from fraudshield.api.config import load_api_config
from fraudshield.api.main import create_app
from fraudshield.tracking.mlflow_setup import (
    install_prohibited_data_guard,
    sha256_file,
    utc_timestamp,
    write_json,
)

OPENAPI_RELATIVE = Path("artifacts/api/openapi.json")
MANIFEST_RELATIVE = Path("artifacts/api/api_manifest.json")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def export_contract(
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Generate tracked API contracts without starting lifespan or loading a model."""

    config = load_api_config(config_path=config_path, root=root)
    repo_root = config.repository_root
    install_prohibited_data_guard(repo_root)
    application = create_app(config=config, load_model_on_startup=False)
    openapi = application.openapi()
    serialized = json.dumps(openapi, indent=2, sort_keys=True) + "\n"
    if str(repo_root) in serialized or "sqlite:///" in serialized.lower():
        raise ValueError("OpenAPI contract contains a machine-specific path or SQLite URI")
    openapi_path = repo_root / OPENAPI_RELATIVE
    openapi_path.parent.mkdir(parents=True, exist_ok=True)
    openapi_path.write_text(serialized, encoding="utf-8")

    endpoint_list = sorted(
        {
            *openapi.get("paths", {}).keys(),
            config.api.docs_url,
            config.api.redoc_url,
            config.api.openapi_url,
        }
    )
    schema_names = sorted(openapi.get("components", {}).get("schemas", {}))
    request_schemas = [
        name for name in schema_names if name in {"TransactionRequest", "BatchPredictionRequest"}
    ]
    response_schemas = [name for name in schema_names if name.endswith("Response")]
    manifest = {
        "generation_timestamp_utc": utc_timestamp(),
        "api_title": config.application.title,
        "api_version": config.application.version,
        "fastapi_version": importlib.metadata.version("fastapi"),
        "starlette_version": importlib.metadata.version("starlette"),
        "pydantic_version": importlib.metadata.version("pydantic"),
        "uvicorn_version": importlib.metadata.version("uvicorn"),
        "httpx_version": importlib.metadata.version("httpx"),
        "mlflow_version": mlflow.__version__,
        "python_version": platform.python_version(),
        "registered_production_model": config.model.registered_name,
        "required_alias": config.model.alias,
        "expected_threshold": config.model.expected_threshold,
        "endpoint_list": endpoint_list,
        "request_schema_names": request_schemas,
        "response_schema_names": response_schemas,
        "openapi_relative_path": OPENAPI_RELATIVE.as_posix(),
        "openapi_sha256": sha256_file(openapi_path),
        "source_git_commit": _git_commit(repo_root),
        "model_retrained": False,
        "metrics_recomputed": False,
        "raw_data_accessed": False,
        "parquet_data_accessed": False,
        "contract_status": "generated",
    }
    manifest_text = json.dumps(manifest, sort_keys=True)
    if str(repo_root) in manifest_text or "sqlite:///" in manifest_text.lower():
        raise ValueError("API manifest contains a machine-specific path or SQLite URI")
    write_json(repo_root / MANIFEST_RELATIVE, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(export_contract(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
