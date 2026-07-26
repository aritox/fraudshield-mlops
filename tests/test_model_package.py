"""Immutable model package checksum contract tests."""

import json
from pathlib import Path

import pytest

from fraudshield.container.package_model import aggregate_checksum, package_files
from fraudshield.container.verify_package import verify_package_integrity


def test_package_checksum_and_manifest_validation(tmp_path: Path) -> None:
    package = tmp_path / "model"
    package.mkdir()
    (package / "MLmodel").write_text("synthetic", encoding="utf-8")
    files = package_files(package)
    manifest = {
        "registered_model_name": "fraudshield-production-sgd",
        "alias": "champion",
        "resolved_version": "1",
        "frozen_threshold": 0.98310834,
        "model_family": "SGDClassifier",
        "models_retrained": False,
        "metrics_recomputed": False,
        "raw_data_accessed": False,
        "parquet_data_accessed": False,
        "files": files,
        "exported_package_checksum": aggregate_checksum(files),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_package_integrity(package, path)["resolved_version"] == "1"
    (package / "MLmodel").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksums"):
        verify_package_integrity(package, path)
