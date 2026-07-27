"""Frozen aggregate training-reference profile generation and verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fraudshield.monitoring.config import MonitoringConfig
from fraudshield.tracking.mlflow_setup import sha256_file, write_json

TRAIN_RELATIVE_PATH = Path("data/processed/train.parquet")
PROFILE_MANIFEST_RELATIVE_PATH = Path("artifacts/monitoring/reference_manifest.json")
SELECTED_COLUMNS = ("step", "type", "amount", "oldbalanceOrg", "oldbalanceDest")
NUMERIC_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("step", "step", "identity"),
    ("log1p(amount)", "amount", "log1p"),
    ("log1p(oldbalanceOrg)", "oldbalanceOrg", "log1p"),
    ("log1p(oldbalanceDest)", "oldbalanceDest", "log1p"),
)
_GUARDED_ROOTS: set[str] = set()


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_reference_source(root: Path, source_path: Path) -> Path:
    """Require the one permitted processed training split."""

    expected = (root / TRAIN_RELATIVE_PATH).resolve()
    resolved = source_path.resolve()
    if resolved != expected:
        raise ValueError("Only data/processed/train.parquet may be used as the reference source")
    return resolved


def validate_reference_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Reject labels, identifiers, and post-transaction columns before any read."""

    selected = tuple(columns)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Reference columns must be a unique non-empty selection")
    unexpected = set(selected) - set(SELECTED_COLUMNS)
    if unexpected:
        raise ValueError("Reference export requested a prohibited column")
    return selected


def install_reference_access_guard(root: Path) -> None:
    """Block raw data and every processed split except train.parquet."""

    resolved_root = os.path.normcase(os.path.abspath(root))
    if resolved_root in _GUARDED_ROOTS:
        return
    allowed = os.path.normcase(os.path.abspath(root / TRAIN_RELATIVE_PATH))
    raw_root = os.path.normcase(os.path.abspath(root / "data" / "raw"))
    processed_root = os.path.normcase(os.path.abspath(root / "data" / "processed"))

    def inside(candidate: str, directory: str) -> bool:
        return candidate == directory or candidate.startswith(directory + os.sep)

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        candidate = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if inside(candidate, raw_root):
            raise PermissionError("Raw data access is prohibited during reference export")
        if inside(candidate, processed_root) and candidate != allowed:
            raise PermissionError("Only train.parquet is allowed during reference export")

    sys.addaudithook(audit_hook)
    _GUARDED_ROOTS.add(resolved_root)


def _batches(parquet: pq.ParquetFile, column: str) -> Iterator[pa.RecordBatch]:
    validate_reference_columns((column,))
    yield from parquet.iter_batches(batch_size=250_000, columns=[column])


def _numeric_values(
    parquet: pq.ParquetFile,
    column: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int, str]:
    chunks: list[np.ndarray] = []
    missing = 0
    provenance = hashlib.sha256()
    for batch in _batches(parquet, column):
        raw_values = np.asarray(batch.column(0).to_pandas(), dtype=np.float64)
        provenance.update(raw_values.astype("<f8", copy=False).tobytes())
        finite = np.isfinite(raw_values)
        missing += int((~finite).sum())
        values = raw_values[finite]
        if (values < 0).any():
            raise ValueError(f"Reference feature {column} contains negative values")
        chunks.append(transform(values))
    if not chunks:
        raise ValueError(f"Reference feature {column} has no values")
    combined = np.concatenate(chunks)
    if combined.size == 0 or not np.isfinite(combined).all():
        raise ValueError(f"Reference feature {column} has no finite values")
    return combined, missing, provenance.hexdigest()


def numeric_profile(
    feature_name: str,
    source_column: str,
    transformation: str,
    values: np.ndarray,
    missing_count: int,
    quantile_bins: int,
) -> dict[str, Any]:
    """Build a deterministic aggregate quantile profile."""

    quantiles = np.linspace(0.0, 1.0, quantile_bins + 1)
    raw_edges = np.quantile(values, quantiles, method="linear")
    edges = np.unique(raw_edges).astype(float).tolist()
    if len(edges) == 1:
        edges.append(edges[0])
    internal_edges = np.asarray(edges[1:-1], dtype=np.float64)
    assignments = np.searchsorted(internal_edges, values, side="right")
    counts = np.bincount(assignments, minlength=len(edges) - 1).astype(np.int64)
    proportions = counts.astype(np.float64) / float(values.size)
    return {
        "feature_name": feature_name,
        "source_column": source_column,
        "transformation": transformation,
        "bin_edges": edges,
        "reference_counts": counts.tolist(),
        "reference_proportions": proportions.tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "missing_count": missing_count,
        "non_missing_count": int(values.size),
    }


def _categorical_profile(
    parquet: pq.ParquetFile,
    source_rows: int,
) -> tuple[dict[str, Any], str]:
    counts: Counter[str] = Counter()
    missing = 0
    provenance = hashlib.sha256()
    for batch in _batches(parquet, "type"):
        for value in batch.column(0).to_pylist():
            if value is None:
                missing += 1
                provenance.update(b"\xff")
            else:
                encoded = str(value).encode("utf-8")
                provenance.update(len(encoded).to_bytes(4, "little"))
                provenance.update(encoded)
                counts[str(value)] += 1
    non_missing = sum(counts.values())
    if non_missing + missing != source_rows or non_missing == 0:
        raise ValueError("Categorical reference row count is inconsistent")
    categories = sorted(counts)
    profile = {
        "feature_name": "type",
        "source_column": "type",
        "transformation": "identity",
        "allowed_categories": categories,
        "reference_counts": [counts[item] for item in categories] + [0],
        "reference_proportions": [counts[item] / non_missing for item in categories] + [0.0],
        "unknown_bucket": "__unknown__",
        "unknown_count": 0,
        "missing_count": missing,
        "non_missing_count": non_missing,
    }
    return profile, provenance.hexdigest()


def build_reference_profile(config: MonitoringConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the five permitted training columns and return aggregates."""

    root = config.repository_root
    install_reference_access_guard(root)
    source_path = validate_reference_source(root, root / TRAIN_RELATIVE_PATH)
    if not source_path.is_file():
        raise FileNotFoundError("Training Parquet is unavailable")

    parquet = pq.ParquetFile(source_path)
    source_rows = int(parquet.metadata.num_rows)
    numeric: dict[str, dict[str, Any]] = {}
    column_checksums: dict[str, str] = {}
    step_range: tuple[int, int] | None = None
    for feature_name, source_column, transformation in NUMERIC_FEATURES:
        transform = np.log1p if transformation == "log1p" else lambda values: values
        values, missing, checksum = _numeric_values(parquet, source_column, transform)
        column_checksums[source_column] = checksum
        if values.size + missing != source_rows:
            raise ValueError("Reference feature row count is inconsistent")
        profile = numeric_profile(
            feature_name,
            source_column,
            transformation,
            values,
            missing,
            config.reference.numeric_quantile_bins,
        )
        numeric[feature_name] = profile
        if source_column == "step":
            step_range = (int(profile["minimum"]), int(profile["maximum"]))
        del values

    categorical, type_checksum = _categorical_profile(parquet, source_rows)
    column_checksums["type"] = type_checksum
    provenance_payload = json.dumps(
        column_checksums,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    profile = {
        "profile_version": config.reference.version,
        "source_split": "train",
        "numeric_quantile_bins": config.reference.numeric_quantile_bins,
        "epsilon": config.reference.epsilon,
        "source_row_count": source_rows,
        "numeric_features": numeric,
        "categorical_features": {"type": categorical},
    }
    metadata = {
        "source_row_count": source_rows,
        "source_step_minimum": step_range[0] if step_range else None,
        "source_step_maximum": step_range[1] if step_range else None,
        "source_provenance_kind": "ordered_selected_columns_sha256",
        "source_selected_columns_sha256": hashlib.sha256(provenance_payload).hexdigest(),
    }
    return profile, metadata


def export_reference(config: MonitoringConfig) -> dict[str, Any]:
    """Write the deterministic profile and its governance manifest."""

    profile, metadata = build_reference_profile(config)
    profile_path = config.reference_profile_path
    write_json(profile_path, profile)
    profile_checksum = sha256_file(profile_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.repository_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "generation_timestamp_utc": utc_timestamp(),
        "profile_version": config.reference.version,
        "source_split": "train",
        **metadata,
        "selected_columns": list(SELECTED_COLUMNS),
        "profile_sha256": profile_checksum,
        "source_git_commit": commit,
        "raw_data_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "label_accessed": False,
        "raw_rows_stored": False,
    }
    serialized = json.dumps(manifest, sort_keys=True)
    if str(config.repository_root) in serialized:
        raise ValueError("Reference manifest contains an absolute repository path")
    write_json(config.repository_root / PROFILE_MANIFEST_RELATIVE_PATH, manifest)
    return manifest


def load_reference_profile(config: MonitoringConfig) -> dict[str, Any]:
    """Load the frozen profile only after its identity and checksum are verified."""

    profile_path = config.reference_profile_path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("profile_version") != config.reference.version:
        raise ValueError("Reference profile version is invalid")
    if profile.get("source_split") != "train":
        raise ValueError("Reference profile is not train-only")
    manifest_path = config.repository_root / PROFILE_MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile_sha256") != sha256_file(profile_path):
        raise ValueError("Reference profile checksum is invalid")
    if manifest.get("label_accessed") is not False:
        raise ValueError("Reference profile label-access governance is invalid")
    return profile
