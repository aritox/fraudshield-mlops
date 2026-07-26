"""Export the existing MLflow champion into an immutable local container package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

from fraudshield.api.config import load_api_config
from fraudshield.data.config import repository_root
from fraudshield.tracking.config import load_mlflow_config
from fraudshield.tracking.mlflow_setup import (
    configure_local_mlflow,
    install_prohibited_data_guard,
    sha256_file,
    utc_timestamp,
    write_json,
)

MODEL_URI = "models:/fraudshield-production-sgd@champion"
PACKAGE_RELATIVE = Path("artifacts/container_model/production_sgd")
MANIFEST_RELATIVE = Path("artifacts/container/model_package_manifest.json")


def package_files(path: Path) -> list[dict[str, str]]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    files.sort(key=lambda file: file.relative_to(path).as_posix())
    return [
        {"path": file.relative_to(path).as_posix(), "sha256": sha256_file(file)}
        for file in files
    ]


def aggregate_checksum(files: list[dict[str, str]]) -> str:
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
    ).stdout.strip()


def _make_mlmodel_portable(package: Path) -> None:
    path = package / "MLmodel"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["artifact_path"] = "."
    model_bundle = document["flavors"]["python_function"]["artifacts"]["model_bundle"]
    model_bundle["path"] = "artifacts/production_sgd.joblib"
    model_bundle["uri"] = "artifacts/production_sgd.joblib"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def validate_version(version: Any, expected_threshold: float) -> None:
    tags = dict(version.tags or {})
    aliases = set(version.aliases or [])
    if version.name != "fraudshield-production-sgd" or "champion" not in aliases:
        raise ValueError("Champion model identity is invalid")
    if str(version.version) != "1":
        raise ValueError("Champion must resolve to version 1")
    if tags.get("role") != "production" or tags.get("model_family") != "SGDClassifier":
        raise ValueError("Champion governance tags are invalid")
    if not tags.get("source_model_sha256"):
        raise ValueError("Champion source checksum is missing")
    if float(tags.get("operational_threshold", "nan")) != expected_threshold:
        raise ValueError("Champion threshold differs from the frozen threshold")
    if str(tags.get("test_used_for_selection", "")).lower() not in {"false", "0"}:
        raise ValueError("Champion test-selection policy is invalid")


def package_model(root: Path | None = None) -> dict[str, Any]:
    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    api_config = load_api_config(root=repo_root)
    mlflow_config = load_mlflow_config(root=repo_root)
    client, _, _ = configure_local_mlflow(mlflow_config)
    version = client.get_model_version_by_alias("fraudshield-production-sgd", "champion")
    validate_version(version, api_config.model.expected_threshold)
    target = repo_root / PACKAGE_RELATIVE
    manifest_path = repo_root / MANIFEST_RELATIVE
    if target.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Existing container model package has no manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_files = package_files(target)
        if {item["path"]: item["sha256"] for item in old_files} != {
            item["path"]: item["sha256"] for item in manifest.get("files", [])
        }:
            raise RuntimeError("Existing container model package has changed")
        _make_mlmodel_portable(target)
        files = package_files(target)
        manifest["files"] = files
        manifest["exported_package_checksum"] = aggregate_checksum(files)
        manifest["package_status"] = "exported"
        write_json(manifest_path, manifest)
        from fraudshield.container.verify_package import verify_package_integrity

        return verify_package_integrity(target, manifest_path)
    with tempfile.TemporaryDirectory(prefix="fraudshield-package-") as temporary:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=MODEL_URI,
                dst_path=temporary,
            )
        )
        if not (downloaded / "MLmodel").is_file():
            raise RuntimeError("Downloaded MLflow package is incomplete")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(downloaded, target)
    _make_mlmodel_portable(target)
    files = package_files(target)
    tags = dict(version.tags or {})
    manifest = {
        "generation_timestamp_utc": utc_timestamp(),
        "registered_model_name": "fraudshield-production-sgd",
        "alias": "champion",
        "resolved_version": str(version.version),
        "source_uri": MODEL_URI,
        "source_model_checksum": tags["source_model_sha256"],
        "exported_package_checksum": aggregate_checksum(files),
        "files": files,
        "frozen_threshold": api_config.model.expected_threshold,
        "model_family": "SGDClassifier",
        "source_git_commit": _git_commit(repo_root),
        "models_retrained": False,
        "metrics_recomputed": False,
        "raw_data_accessed": False,
        "parquet_data_accessed": False,
        "package_status": "exported",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(package_model(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
