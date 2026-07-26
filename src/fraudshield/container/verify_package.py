"""Verify the immutable model package and synthetic prediction equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from fraudshield.container.package_model import (
    MANIFEST_RELATIVE,
    MODEL_URI,
    PACKAGE_RELATIVE,
    aggregate_checksum,
    package_files,
)
from fraudshield.data.config import repository_root
from fraudshield.tracking.config import load_mlflow_config
from fraudshield.tracking.mlflow_setup import (
    configure_local_mlflow,
    install_prohibited_data_guard,
    write_json,
)
from fraudshield.tracking.model_wrapper import synthetic_input_example


def verify_package_integrity(package: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_files = package_files(package)
    actual_checksum = aggregate_checksum(actual_files)
    if actual_files != manifest.get("files"):
        raise RuntimeError("Container model package file checksums differ from the manifest")
    if actual_checksum != manifest.get("exported_package_checksum"):
        raise RuntimeError("Container model package checksum differs from the manifest")
    expected = {
        "registered_model_name": "fraudshield-production-sgd",
        "alias": "champion",
        "resolved_version": "1",
        "frozen_threshold": 0.98310834,
        "model_family": "SGDClassifier",
        "models_retrained": False,
        "metrics_recomputed": False,
        "raw_data_accessed": False,
        "parquet_data_accessed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Container model manifest field is invalid: {key}")
    return manifest


def verify_package(root: Path | None = None) -> dict[str, Any]:
    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    package = repo_root / PACKAGE_RELATIVE
    manifest_path = repo_root / MANIFEST_RELATIVE
    manifest = verify_package_integrity(package, manifest_path)
    mlflow_config = load_mlflow_config(root=repo_root)
    configure_local_mlflow(mlflow_config)
    example = synthetic_input_example()
    alias_output = mlflow.pyfunc.load_model(MODEL_URI).predict(example)
    local_output = mlflow.pyfunc.load_model(str(package)).predict(example)
    if list(alias_output.columns) != list(local_output.columns):
        raise RuntimeError("Packaged model output columns differ from the champion")
    for column in ("fraud_score", "threshold"):
        if not np.allclose(alias_output[column], local_output[column], rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Packaged model {column} differs from the champion")
    for column in ("prediction", "risk_level"):
        if alias_output[column].tolist() != local_output[column].tolist():
            raise RuntimeError(f"Packaged model {column} differs from the champion")
    manifest["package_status"] = "verified"
    write_json(manifest_path, manifest)
    return {
        "status": "verified",
        "model_name": manifest["registered_model_name"],
        "model_version": manifest["resolved_version"],
        "package_checksum": manifest["exported_package_checksum"],
        "prediction_rows": len(local_output),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify_package(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
