"""Create an internal chronological tuning split inside the official train split."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fraudshield.data.config import repository_root
from fraudshield.data.validate import calculate_sha256, utc_timestamp
from fraudshield.models.train_baseline import reject_test_path

TARGET_COLUMN = "isFraud"
TIME_COLUMN = "step"
SPLIT_MANIFEST_RELATIVE = "artifacts/data/split_manifest.json"
INNER_SPLIT_MANIFEST_RELATIVE = "artifacts/tuning/inner_split_manifest.json"


@dataclass(frozen=True)
class StepWindow:
    """Inclusive step range for a tuning period."""

    minimum: int
    maximum: int

    def contains(self, step: int) -> bool:
        return self.minimum <= step <= self.maximum

    def to_dict(self) -> dict[str, int]:
        return {"minimum": int(self.minimum), "maximum": int(self.maximum)}


@dataclass(frozen=True)
class InnerSplit:
    """Internal chronological split derived from train.parquet."""

    train_path: Path
    boundary_step: int
    fit_window: StepWindow
    tuning_window: StepWindow
    fit_rows: int
    tuning_rows: int
    fit_frauds: int
    tuning_frauds: int
    manifest: dict[str, Any]
    manifest_path: Path


def _iter_train_batches(path: Path, batch_size: int, columns: list[str]):
    reject_test_path(path)
    parquet_file = pq.ParquetFile(path)
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=columns)


def count_rows_by_step(train_path: Path, batch_size: int) -> tuple[Counter[int], Counter[int]]:
    """Count rows and frauds per complete step without loading train.parquet fully."""

    reject_test_path(train_path)
    row_counts: Counter[int] = Counter()
    fraud_counts: Counter[int] = Counter()
    for batch in _iter_train_batches(train_path, batch_size, [TIME_COLUMN, TARGET_COLUMN]):
        frame = batch.to_pandas()
        grouped_rows = frame.groupby(TIME_COLUMN).size()
        grouped_frauds = frame.groupby(TIME_COLUMN)[TARGET_COLUMN].sum()
        row_counts.update({int(step): int(count) for step, count in grouped_rows.items()})
        fraud_counts.update({int(step): int(count) for step, count in grouped_frauds.items()})
    if not row_counts:
        raise ValueError("Training split is empty")
    return row_counts, fraud_counts


def select_inner_boundary(step_counts: Counter[int], fit_fraction: float) -> int:
    """Select the whole-step boundary closest to the requested fitting row fraction."""

    steps = sorted(step_counts)
    if len(steps) < 2:
        raise ValueError("At least two complete steps are required for inner tuning")
    total_rows = sum(step_counts.values())
    target_rows = total_rows * fit_fraction
    cumulative = 0
    candidates: list[tuple[float, int]] = []
    for step in steps[:-1]:
        cumulative += step_counts[step]
        candidates.append((abs(cumulative - target_rows), step))
    return min(candidates)[1]


def _period_counts(
    row_counts: Counter[int],
    fraud_counts: Counter[int],
    window: StepWindow,
) -> tuple[int, int]:
    rows = sum(count for step, count in row_counts.items() if window.contains(step))
    frauds = sum(count for step, count in fraud_counts.items() if window.contains(step))
    return int(rows), int(frauds)


def _build_manifest(
    root: Path,
    train_path: Path,
    split_manifest_path: Path,
    method: str,
    boundary_step: int,
    fit_window: StepWindow,
    tuning_window: StepWindow,
    fit_rows: int,
    tuning_rows: int,
    fit_frauds: int,
    tuning_frauds: int,
) -> dict[str, Any]:
    total_fit = fit_rows
    total_tuning = tuning_rows
    no_step_overlap = fit_window.maximum < tuning_window.minimum
    chronological = fit_window.maximum < tuning_window.minimum
    return {
        "created_at_utc": utc_timestamp(),
        "source_train_path": train_path.relative_to(root).as_posix(),
        "source_split_manifest_sha256": calculate_sha256(split_manifest_path),
        "method": method,
        "selected_step_boundary": int(boundary_step),
        "inner_fit_step_range": fit_window.to_dict(),
        "inner_tuning_step_range": tuning_window.to_dict(),
        "inner_fit_row_count": int(fit_rows),
        "inner_tuning_row_count": int(tuning_rows),
        "inner_fit_fraud_count": int(fit_frauds),
        "inner_tuning_fraud_count": int(tuning_frauds),
        "inner_fit_fraud_percentage": float(fit_frauds / total_fit * 100) if total_fit else 0.0,
        "inner_tuning_fraud_percentage": (
            float(tuning_frauds / total_tuning * 100) if total_tuning else 0.0
        ),
        "no_step_overlap": bool(no_step_overlap),
        "chronological_ordering_check": bool(chronological),
        "test_set_accessed": False,
        "status": "passed" if no_step_overlap and chronological else "failed",
    }


def create_inner_split(
    root: Path | None = None,
    train_path: Path | None = None,
    batch_size: int = 250_000,
    fit_fraction: float = 0.80,
    method: str = "chronological_whole_step",
) -> InnerSplit:
    """Create and persist the Phase 1D internal tuning split manifest."""

    repo_root = root or repository_root()
    relative_train_path = train_path or Path("data/processed/train.parquet")
    if relative_train_path.is_absolute():
        raise ValueError("train_path must be relative to the repository root")
    if relative_train_path.name != "train.parquet":
        raise ValueError("inner tuning must read only train.parquet")
    if not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must be between 0 and 1")
    if method != "chronological_whole_step":
        raise ValueError("inner tuning method must be chronological_whole_step")

    absolute_train_path = repo_root / relative_train_path
    reject_test_path(absolute_train_path)
    if not absolute_train_path.exists():
        raise FileNotFoundError(f"Training split not found: {absolute_train_path}")

    split_manifest_path = repo_root / SPLIT_MANIFEST_RELATIVE
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Source split manifest not found: {split_manifest_path}")

    row_counts, fraud_counts = count_rows_by_step(absolute_train_path, batch_size)
    steps = sorted(row_counts)
    boundary_step = select_inner_boundary(row_counts, fit_fraction)
    fit_window = StepWindow(minimum=steps[0], maximum=boundary_step)
    tuning_window = StepWindow(
        minimum=min(step for step in steps if step > boundary_step),
        maximum=steps[-1],
    )
    fit_rows, fit_frauds = _period_counts(row_counts, fraud_counts, fit_window)
    tuning_rows, tuning_frauds = _period_counts(row_counts, fraud_counts, tuning_window)
    if fit_rows <= 0 or tuning_rows <= 0:
        raise ValueError("Inner fit and tuning periods must both be non-empty")
    if fit_frauds <= 0 or tuning_frauds <= 0:
        raise ValueError("Inner fit and tuning periods must both contain fraud examples")
    if fit_window.maximum >= tuning_window.minimum:
        raise ValueError("Inner tuning split is not chronological")

    manifest = _build_manifest(
        root=repo_root,
        train_path=absolute_train_path,
        split_manifest_path=split_manifest_path,
        method=method,
        boundary_step=boundary_step,
        fit_window=fit_window,
        tuning_window=tuning_window,
        fit_rows=fit_rows,
        tuning_rows=tuning_rows,
        fit_frauds=fit_frauds,
        tuning_frauds=tuning_frauds,
    )
    manifest_path = repo_root / INNER_SPLIT_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return InnerSplit(
        train_path=relative_train_path,
        boundary_step=boundary_step,
        fit_window=fit_window,
        tuning_window=tuning_window,
        fit_rows=fit_rows,
        tuning_rows=tuning_rows,
        fit_frauds=fit_frauds,
        tuning_frauds=tuning_frauds,
        manifest=manifest,
        manifest_path=manifest_path,
    )
