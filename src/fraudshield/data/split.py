"""Create leakage-aware chronological PaySim train/validation/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from fraudshield.data.config import load_data_config, raw_data_directory, repository_root
from fraudshield.data.download import find_matching_csv
from fraudshield.data.validate import calculate_sha256, utc_timestamp

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_OUTPUTS = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for chronological whole-step splitting."""

    time_column: str
    target_column: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    chunk_size: int
    output_directory: Path
    compression: str
    random_seed: int

    @property
    def requested_fractions(self) -> dict[str, float]:
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "test": self.test_fraction,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "target_column": self.target_column,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "chunk_size": self.chunk_size,
            "output_directory": self.output_directory.as_posix(),
            "compression": self.compression,
            "random_seed": self.random_seed,
        }


@dataclass
class SplitResult:
    """Structured output from split generation."""

    manifest: dict[str, Any]
    manifest_path: Path
    passed: bool
    reused: bool


def load_split_config(config_path: Path | None = None, root: Path | None = None) -> SplitConfig:
    """Load and validate split settings."""

    repo_root = root or repository_root()
    resolved_config_path = config_path or repo_root / "configs" / "split.yaml"

    with resolved_config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    required_keys = {
        "time_column",
        "target_column",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "chunk_size",
        "output_directory",
        "compression",
        "random_seed",
    }
    missing_keys = sorted(required_keys - raw_config.keys())
    if missing_keys:
        raise ValueError(f"Missing required split config keys: {', '.join(missing_keys)}")

    train_fraction = float(raw_config["train_fraction"])
    validation_fraction = float(raw_config["validation_fraction"])
    test_fraction = float(raw_config["test_fraction"])
    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    chunk_size = int(raw_config["chunk_size"])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output_directory = Path(str(raw_config["output_directory"]))
    if output_directory.is_absolute():
        raise ValueError("output_directory must be relative to the repository root")

    return SplitConfig(
        time_column=str(raw_config["time_column"]),
        target_column=str(raw_config["target_column"]),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        chunk_size=chunk_size,
        output_directory=output_directory,
        compression=str(raw_config["compression"]),
        random_seed=int(raw_config["random_seed"]),
    )


def _read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _manifest_matches(
    manifest: dict[str, Any] | None,
    config: SplitConfig,
    root: Path,
    source_path: Path,
    source_checksum: str,
) -> bool:
    if manifest is None or manifest.get("overall_split_status") != "passed":
        return False
    if manifest.get("source_sha256_checksum") != source_checksum:
        return False
    if manifest.get("source_dataset_relative_path") != source_path.relative_to(root).as_posix():
        return False
    if manifest.get("split_config") != config.to_manifest_dict():
        return False

    checks = manifest.get("checks", {})
    if not all(
        bool(checks.get(key))
        for key in (
            "total_row_conservation",
            "total_fraud_conservation",
            "no_step_overlap",
            "chronological_ordering_check",
        )
    ):
        return False

    for split_name in SPLIT_NAMES:
        split_manifest = manifest.get("splits", {}).get(split_name, {})
        output_path = root / str(split_manifest.get("output_file", ""))
        if not output_path.exists():
            return False
    return True


def _remove_generated_outputs(root: Path, config: SplitConfig, manifest_path: Path) -> None:
    output_directory = root / config.output_directory
    for filename in SPLIT_OUTPUTS.values():
        output_path = output_directory / filename
        if output_path.exists():
            output_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()


def _count_steps(
    csv_path: Path,
    config: SplitConfig,
) -> tuple[Counter[int], Counter[int], int, int]:
    step_counts: Counter[int] = Counter()
    fraud_counts: Counter[int] = Counter()
    total_rows = 0
    total_fraud = 0

    for chunk in pd.read_csv(
        csv_path,
        chunksize=config.chunk_size,
        usecols=[config.time_column, config.target_column],
    ):
        total_rows += int(len(chunk))
        total_fraud += int(chunk[config.target_column].sum())
        grouped_rows = chunk.groupby(config.time_column).size()
        grouped_fraud = chunk.groupby(config.time_column)[config.target_column].sum()
        step_counts.update({int(step): int(count) for step, count in grouped_rows.items()})
        fraud_counts.update({int(step): int(count) for step, count in grouped_fraud.items()})

    return step_counts, fraud_counts, total_rows, total_fraud


def _closest_boundary(steps: list[int], cumulative_counts: dict[int, int], target: float) -> int:
    return min(steps, key=lambda step: (abs(cumulative_counts[step] - target), step))


def _assign_steps(
    step_counts: Counter[int],
    total_rows: int,
    config: SplitConfig,
) -> tuple[dict[int, str], dict[str, tuple[int, int]], dict[str, int]]:
    steps = sorted(step_counts)
    if len(steps) < 3:
        raise ValueError(
            "At least three unique time steps are required for three chronological splits"
        )

    cumulative_counts: dict[int, int] = {}
    running_total = 0
    for step in steps:
        running_total += step_counts[step]
        cumulative_counts[step] = running_total

    train_boundary = _closest_boundary(
        steps[:-2],
        cumulative_counts,
        total_rows * config.train_fraction,
    )
    validation_target = total_rows * (config.train_fraction + config.validation_fraction)
    validation_candidates = [step for step in steps[1:-1] if step > train_boundary]
    if not validation_candidates:
        raise ValueError("Could not select a non-empty validation split")
    validation_boundary = _closest_boundary(
        validation_candidates,
        cumulative_counts,
        validation_target,
    )

    assignments: dict[int, str] = {}
    for step in steps:
        if step <= train_boundary:
            assignments[step] = "train"
        elif step <= validation_boundary:
            assignments[step] = "validation"
        else:
            assignments[step] = "test"

    ranges = {
        split_name: (
            min(step for step, assigned in assignments.items() if assigned == split_name),
            max(step for step, assigned in assignments.items() if assigned == split_name),
        )
        for split_name in SPLIT_NAMES
    }
    boundaries = {"train_validation": train_boundary, "validation_test": validation_boundary}
    return assignments, ranges, boundaries


def _write_split_chunk(
    writers: dict[str, pq.ParquetWriter],
    output_paths: dict[str, Path],
    split_name: str,
    chunk: pd.DataFrame,
    compression: str,
) -> None:
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if split_name not in writers:
        writers[split_name] = pq.ParquetWriter(
            output_paths[split_name],
            table.schema,
            compression=compression,
        )
    writers[split_name].write_table(table)


def _write_splits(
    csv_path: Path,
    config: SplitConfig,
    root: Path,
    assignments: dict[int, str],
) -> tuple[dict[str, int], dict[str, int], dict[str, set[int]]]:
    output_directory = root / config.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_directory / filename for name, filename in SPLIT_OUTPUTS.items()}
    writers: dict[str, pq.ParquetWriter] = {}
    row_counts = Counter({name: 0 for name in SPLIT_NAMES})
    fraud_counts = Counter({name: 0 for name in SPLIT_NAMES})
    split_steps: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}

    try:
        for chunk in pd.read_csv(csv_path, chunksize=config.chunk_size):
            chunk["_fraudshield_split"] = chunk[config.time_column].map(
                lambda step: assignments[int(step)]
            )
            for split_name in SPLIT_NAMES:
                split_chunk = chunk.loc[chunk["_fraudshield_split"] == split_name].drop(
                    columns=["_fraudshield_split"]
                )
                if split_chunk.empty:
                    continue
                row_counts[split_name] += int(len(split_chunk))
                fraud_counts[split_name] += int(split_chunk[config.target_column].sum())
                split_steps[split_name].update(
                    int(step) for step in split_chunk[config.time_column].unique()
                )
                _write_split_chunk(
                    writers=writers,
                    output_paths=output_paths,
                    split_name=split_name,
                    chunk=split_chunk,
                    compression=config.compression,
                )
    finally:
        for writer in writers.values():
            writer.close()

    return dict(row_counts), dict(fraud_counts), split_steps


def _split_overlap_ok(split_steps: dict[str, set[int]]) -> bool:
    seen: set[int] = set()
    for split_name in SPLIT_NAMES:
        overlap = seen.intersection(split_steps[split_name])
        if overlap:
            return False
        seen.update(split_steps[split_name])
    return True


def _chronological_ok(ranges: dict[str, tuple[int, int]]) -> bool:
    return (
        ranges["train"][1] < ranges["validation"][0]
        and ranges["validation"][1] < ranges["test"][0]
    )


def _build_manifest(
    root: Path,
    csv_path: Path,
    source_checksum: str,
    config: SplitConfig,
    boundaries: dict[str, int],
    ranges: dict[str, tuple[int, int]],
    row_counts: dict[str, int],
    fraud_counts: dict[str, int],
    split_steps: dict[str, set[int]],
    total_rows: int,
    total_fraud: int,
) -> dict[str, Any]:
    output_directory = root / config.output_directory
    actual_fractions = {
        name: (row_counts[name] / total_rows if total_rows else 0.0) for name in SPLIT_NAMES
    }
    total_row_conservation = sum(row_counts.values()) == total_rows
    total_fraud_conservation = sum(fraud_counts.values()) == total_fraud
    no_step_overlap = _split_overlap_ok(split_steps)
    chronological_ordering_check = _chronological_ok(ranges)
    class_coverage_check = all(
        fraud_counts[name] > 0 and (row_counts[name] - fraud_counts[name]) > 0
        for name in SPLIT_NAMES
    )
    passed = all(
        (
            total_row_conservation,
            total_fraud_conservation,
            no_step_overlap,
            chronological_ordering_check,
            class_coverage_check,
        )
    )

    splits: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        output_path = output_directory / SPLIT_OUTPUTS[split_name]
        rows = row_counts[split_name]
        fraud = fraud_counts[split_name]
        splits[split_name] = {
            "step_minimum": int(ranges[split_name][0]),
            "step_maximum": int(ranges[split_name][1]),
            "row_count": int(rows),
            "fraud_count": int(fraud),
            "non_fraud_count": int(rows - fraud),
            "fraud_percentage": float((fraud / rows * 100) if rows else 0.0),
            "actual_fraction": float(actual_fractions[split_name]),
            "output_file": output_path.relative_to(root).as_posix(),
            "output_file_size_bytes": (
                int(output_path.stat().st_size) if output_path.exists() else 0
            ),
        }

    return {
        "source_dataset_relative_path": csv_path.relative_to(root).as_posix(),
        "source_sha256_checksum": source_checksum,
        "split_timestamp_utc": utc_timestamp(),
        "splitting_method": "chronological_whole_step",
        "requested_split_fractions": config.requested_fractions,
        "actual_split_fractions": {name: float(value) for name, value in actual_fractions.items()},
        "split_config": config.to_manifest_dict(),
        "random_seed": int(config.random_seed),
        "time_column": config.time_column,
        "target_column": config.target_column,
        "selected_step_boundaries": {
            "train_validation": int(boundaries["train_validation"]),
            "validation_test": int(boundaries["validation_test"]),
        },
        "splits": splits,
        "checks": {
            "total_row_conservation": bool(total_row_conservation),
            "total_fraud_conservation": bool(total_fraud_conservation),
            "no_step_overlap": bool(no_step_overlap),
            "chronological_ordering_check": bool(chronological_ordering_check),
            "class_coverage_check": bool(class_coverage_check),
        },
        "overall_split_status": "passed" if passed else "failed",
    }


def _print_summary(manifest: dict[str, Any], reused: bool) -> None:
    print("Temporal split summary")
    print(f"Status: {manifest['overall_split_status']}")
    print(f"Mode: {'reused' if reused else 'generated'}")
    boundaries = manifest["selected_step_boundaries"]
    print(
        "Selected step boundaries: "
        f"train/validation <= {boundaries['train_validation']}; "
        f"validation/test <= {boundaries['validation_test']}"
    )
    print("Rows, actual percentages, fraud counts, and fraud rates:")
    for split_name in SPLIT_NAMES:
        split = manifest["splits"][split_name]
        print(
            f"  {split_name}: rows={split['row_count']}, "
            f"actual={split['actual_fraction'] * 100:.4f}%, "
            f"fraud={split['fraud_count']}, "
            f"fraud_rate={split['fraud_percentage']:.6f}%, "
            f"path={split['output_file']}"
        )
    checks = manifest["checks"]
    print("Conservation checks:")
    print(f"  rows: {checks['total_row_conservation']}")
    print(f"  fraud: {checks['total_fraud_conservation']}")
    print(f"  no step overlap: {checks['no_step_overlap']}")
    print(f"  chronological ordering: {checks['chronological_ordering_check']}")


def create_splits(
    root: Path | None = None,
    config: SplitConfig | None = None,
    force: bool = False,
) -> SplitResult:
    """Create or reuse chronological split Parquet files."""

    repo_root = root or repository_root()
    split_config = config or load_split_config(root=repo_root)
    data_config = load_data_config(root=repo_root)
    csv_path = find_matching_csv(
        raw_data_directory(data_config, repo_root),
        data_config.expected_columns,
    )
    if csv_path is None:
        raise FileNotFoundError("No validated PaySim CSV was found under the raw data directory")

    if split_config.time_column not in data_config.expected_columns:
        raise ValueError(f"time_column is not in the source schema: {split_config.time_column}")
    if split_config.target_column != data_config.target_column:
        raise ValueError("split target_column must match the data target_column")

    manifest_path = repo_root / "artifacts" / "data" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_checksum = calculate_sha256(csv_path)
    existing_manifest = _read_manifest(manifest_path)

    if not force and _manifest_matches(
        existing_manifest,
        split_config,
        repo_root,
        csv_path,
        source_checksum,
    ):
        assert existing_manifest is not None
        return SplitResult(
            manifest=existing_manifest,
            manifest_path=manifest_path,
            passed=True,
            reused=True,
        )

    if force:
        _remove_generated_outputs(repo_root, split_config, manifest_path)

    output_directory = repo_root / split_config.output_directory
    if output_directory.exists():
        for filename in SPLIT_OUTPUTS.values():
            stale_path = output_directory / filename
            if stale_path.exists():
                stale_path.unlink()
    else:
        output_directory.mkdir(parents=True, exist_ok=True)

    step_counts, source_fraud_counts, total_rows, total_fraud = _count_steps(csv_path, split_config)
    assignments, ranges, boundaries = _assign_steps(step_counts, total_rows, split_config)
    row_counts, fraud_counts, split_steps = _write_splits(
        csv_path=csv_path,
        config=split_config,
        root=repo_root,
        assignments=assignments,
    )
    if sum(source_fraud_counts.values()) != total_fraud:
        raise RuntimeError("Internal fraud count mismatch during split planning")

    manifest = _build_manifest(
        root=repo_root,
        csv_path=csv_path,
        source_checksum=source_checksum,
        config=split_config,
        boundaries=boundaries,
        ranges=ranges,
        row_counts=row_counts,
        fraud_counts=fraud_counts,
        split_steps=split_steps,
        total_rows=total_rows,
        total_fraud=total_fraud,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return SplitResult(
        manifest=manifest,
        manifest_path=manifest_path,
        passed=manifest["overall_split_status"] == "passed",
        reused=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chronological PaySim split Parquet files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only generated split Parquet files and the split manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = create_splits(force=args.force)
        _print_summary(result.manifest, reused=result.reused)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
