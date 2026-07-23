"""Training-only exploratory analysis for PaySim splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pyarrow.parquet as pq

from fraudshield.data.config import repository_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BATCH_SIZE = 250_000
MAX_NON_FRAUD_SAMPLE = 200_000
RANDOM_SEED = 42
TARGET_COLUMN = "isFraud"
FLAGGED_COLUMN = "isFlaggedFraud"
TYPE_COLUMN = "type"
TIME_COLUMN = "step"
AMOUNT_COLUMN = "amount"
BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
PLOT_FILENAMES = [
    "01_class_distribution.png",
    "02_transactions_by_type.png",
    "03_fraud_rate_by_type.png",
    "04_transactions_over_time.png",
    "05_fraud_rate_over_time.png",
    "06_amount_distribution_log_scale.png",
    "07_flagged_vs_actual_fraud.png",
]


@dataclass
class RunningStats:
    """Streaming min, max, mean, and sample standard deviation."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: pd.Series) -> None:
        for value in pd.to_numeric(values.dropna(), errors="coerce").dropna():
            number = float(value)
            self.count += 1
            delta = number - self.mean
            self.mean += delta / self.count
            delta2 = number - self.mean
            self.m2 += delta * delta2
            self.minimum = number if self.minimum is None else min(self.minimum, number)
            self.maximum = number if self.maximum is None else max(self.maximum, number)

    def to_dict(self, include_std: bool) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {
            "count": int(self.count),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": float(self.mean) if self.count else None,
        }
        if include_std:
            result["standard_deviation"] = (
                float((self.m2 / (self.count - 1)) ** 0.5) if self.count > 1 else 0.0
            )
        return result


@dataclass
class EdaResult:
    """Structured output from training-only EDA."""

    summary: dict[str, Any]
    summary_path: Path
    plot_paths: list[Path]


def _int_mapping(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _rate_mapping(counts: Counter[Any], fraud_counts: Counter[Any]) -> dict[str, float]:
    return {
        str(key): float((fraud_counts.get(key, 0) / value * 100) if value else 0.0)
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _with_sample_keys(frame: pd.DataFrame, sequence_start: int, seed: int) -> pd.DataFrame:
    sample_columns = [TIME_COLUMN, TYPE_COLUMN, AMOUNT_COLUMN, "nameOrig", "nameDest"]
    keyed = frame.copy()
    stable_values = keyed[sample_columns].astype(str)
    keyed["_fraudshield_sample_key"] = pd.util.hash_pandas_object(
        stable_values,
        index=False,
        hash_key=f"{seed:016d}",
    )
    keyed["_fraudshield_sample_sequence"] = range(sequence_start, sequence_start + len(keyed))
    return keyed


def deterministic_training_sample(
    train_path: Path,
    max_non_fraud: int = MAX_NON_FRAUD_SAMPLE,
    seed: int = RANDOM_SEED,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Return all fraud rows plus a deterministic bounded non-fraud sample."""

    parquet_file = pq.ParquetFile(train_path)
    fraud_frames: list[pd.DataFrame] = []
    selected_non_fraud = pd.DataFrame()
    sequence = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        fraud_chunk = chunk.loc[chunk[TARGET_COLUMN] == 1]
        if not fraud_chunk.empty:
            fraud_frames.append(fraud_chunk)
        non_fraud_chunk = chunk.loc[chunk[TARGET_COLUMN] == 0]
        if non_fraud_chunk.empty:
            continue
        keyed_non_fraud = _with_sample_keys(non_fraud_chunk, sequence, seed)
        sequence += len(keyed_non_fraud)
        selected_non_fraud = pd.concat(
            [selected_non_fraud, keyed_non_fraud],
            ignore_index=True,
        )
        selected_non_fraud = (
            selected_non_fraud.sort_values(
                ["_fraudshield_sample_key", "_fraudshield_sample_sequence"],
                kind="mergesort",
            )
            .head(max_non_fraud)
            .reset_index(drop=True)
        )

    frames = []
    if fraud_frames:
        frames.append(pd.concat(fraud_frames, ignore_index=True))
    if not selected_non_fraud.empty:
        frames.append(
            selected_non_fraud.drop(
                columns=["_fraudshield_sample_key", "_fraudshield_sample_sequence"]
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_summary(train_path: Path, root: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(train_path)
    total_rows = 0
    fraud_count = 0
    transaction_counts: Counter[str] = Counter()
    fraud_by_type: Counter[str] = Counter()
    transaction_by_step: Counter[int] = Counter()
    fraud_by_step: Counter[int] = Counter()
    missing_counts: Counter[str] = Counter()
    flagged_vs_fraud: Counter[str] = Counter()
    zero_balance_counts: Counter[str] = Counter()
    amount_stats = {0: RunningStats(), 1: RunningStats()}
    balance_stats: dict[str, dict[int, RunningStats]] = {
        column: {0: RunningStats(), 1: RunningStats()} for column in BALANCE_COLUMNS
    }

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        chunk = batch.to_pandas()
        total_rows += int(len(chunk))
        fraud_count += int(chunk[TARGET_COLUMN].sum())
        missing_counts.update({column: int(value) for column, value in chunk.isna().sum().items()})

        transaction_counts.update(str(value) for value in chunk[TYPE_COLUMN].dropna())
        fraud_rows = chunk.loc[chunk[TARGET_COLUMN] == 1]
        fraud_by_type.update(str(value) for value in fraud_rows[TYPE_COLUMN].dropna())

        transaction_by_step.update(int(value) for value in chunk[TIME_COLUMN].dropna())
        fraud_by_step.update(int(value) for value in fraud_rows[TIME_COLUMN].dropna())

        crosstab = chunk.groupby([FLAGGED_COLUMN, TARGET_COLUMN]).size()
        for (flagged, fraud), count in crosstab.items():
            flagged_vs_fraud[f"flagged_{int(flagged)}__fraud_{int(fraud)}"] += int(count)

        for target_value in (0, 1):
            class_chunk = chunk.loc[chunk[TARGET_COLUMN] == target_value]
            amount_stats[target_value].update(class_chunk[AMOUNT_COLUMN])
            for column in BALANCE_COLUMNS:
                balance_stats[column][target_value].update(class_chunk[column])

        for column in BALANCE_COLUMNS:
            zero_balance_counts[column] += int((chunk[column] == 0).sum())

    non_fraud_count = total_rows - fraud_count
    fraud_percentage = (fraud_count / total_rows * 100) if total_rows else 0.0
    return {
        "source_training_file": train_path.relative_to(root).as_posix(),
        "total_training_rows": int(total_rows),
        "fraud_count": int(fraud_count),
        "non_fraud_count": int(non_fraud_count),
        "fraud_percentage": float(fraud_percentage),
        "transaction_count_by_type": _int_mapping(transaction_counts),
        "fraud_count_by_type": _int_mapping(fraud_by_type),
        "fraud_rate_by_type": _rate_mapping(transaction_counts, fraud_by_type),
        "transaction_count_by_step": _int_mapping(transaction_by_step),
        "fraud_count_by_step": _int_mapping(fraud_by_step),
        "fraud_rate_by_step": _rate_mapping(transaction_by_step, fraud_by_step),
        "amount_statistics_by_class": {
            str(target): stats.to_dict(include_std=True) for target, stats in amount_stats.items()
        },
        "balance_statistics_by_class": {
            column: {
                str(target): stats.to_dict(include_std=False)
                for target, stats in class_stats.items()
            }
            for column, class_stats in balance_stats.items()
        },
        "flagged_fraud_crosstab": _int_mapping(flagged_vs_fraud),
        "zero_balance_frequencies": {
            column: {
                "count": int(count),
                "percentage": float((count / total_rows * 100) if total_rows else 0.0),
            }
            for column, count in sorted(zero_balance_counts.items())
        },
        "missing_values_by_column": _int_mapping(missing_counts),
    }


def _save_bar(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_line(
    x_values: list[int],
    y_values: list[float],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, y_values, color="#0f766e", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _create_plots(summary: dict[str, Any], sample: pd.DataFrame, plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = [plots_dir / filename for filename in PLOT_FILENAMES]

    _save_bar(
        ["Non-fraud", "Fraud"],
        [summary["non_fraud_count"], summary["fraud_count"]],
        "Training Class Distribution",
        "Transactions",
        plot_paths[0],
    )
    _save_bar(
        list(summary["transaction_count_by_type"].keys()),
        list(summary["transaction_count_by_type"].values()),
        "Training Transactions by Type",
        "Transactions",
        plot_paths[1],
    )
    _save_bar(
        list(summary["fraud_rate_by_type"].keys()),
        list(summary["fraud_rate_by_type"].values()),
        "Training Fraud Rate by Type",
        "Fraud rate (%)",
        plot_paths[2],
    )

    steps = [int(step) for step in summary["transaction_count_by_step"]]
    ordered_steps = sorted(steps)
    _save_line(
        ordered_steps,
        [summary["transaction_count_by_step"][str(step)] for step in ordered_steps],
        "Training Transactions over Time",
        "Transactions",
        plot_paths[3],
    )
    _save_line(
        ordered_steps,
        [summary["fraud_rate_by_step"][str(step)] for step in ordered_steps],
        "Training Fraud Rate over Time",
        "Fraud rate (%)",
        plot_paths[4],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    if not sample.empty:
        for target_value, label, color in (
            (0, "Non-fraud sample", "#2563eb"),
            (1, "Fraud", "#dc2626"),
        ):
            values = pd.to_numeric(
                sample.loc[sample[TARGET_COLUMN] == target_value, AMOUNT_COLUMN],
                errors="coerce",
            ).dropna()
            positive_values = values.loc[values > 0]
            if not positive_values.empty:
                ax.hist(
                    positive_values,
                    bins=50,
                    alpha=0.55,
                    label=label,
                    color=color,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Training Amount Distribution by Class")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Transactions (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_paths[5], dpi=140)
    plt.close(fig)

    crosstab = summary["flagged_fraud_crosstab"]
    labels = [
        "flagged_0__fraud_0",
        "flagged_0__fraud_1",
        "flagged_1__fraud_0",
        "flagged_1__fraud_1",
    ]
    _save_bar(
        labels,
        [crosstab.get(label, 0) for label in labels],
        "Training Rule-Flagged vs Actual Fraud",
        "Transactions",
        plot_paths[6],
    )

    return plot_paths


def create_training_eda(
    root: Path | None = None,
    max_non_fraud_sample: int = MAX_NON_FRAUD_SAMPLE,
) -> EdaResult:
    """Create training-only EDA summary and plots."""

    repo_root = root or repository_root()
    train_path = repo_root / "data" / "processed" / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split was not found: {train_path}")

    artifact_root = repo_root / "artifacts" / "eda"
    plots_dir = artifact_root / "plots"
    artifact_root.mkdir(parents=True, exist_ok=True)

    summary = _collect_summary(train_path, repo_root)
    sample = deterministic_training_sample(
        train_path=train_path,
        max_non_fraud=max_non_fraud_sample,
        seed=RANDOM_SEED,
    )
    sample_fraud_count = int((sample[TARGET_COLUMN] == 1).sum()) if not sample.empty else 0
    sample_non_fraud_count = int((sample[TARGET_COLUMN] == 0).sum()) if not sample.empty else 0
    summary["deterministic_sample"] = {
        "random_seed": int(RANDOM_SEED),
        "total_rows": int(len(sample)),
        "fraud_rows": sample_fraud_count,
        "non_fraud_rows": sample_non_fraud_count,
        "max_non_fraud_rows": int(max_non_fraud_sample),
    }

    plot_paths = _create_plots(summary, sample, plots_dir)
    summary["generated_plots"] = [path.relative_to(repo_root).as_posix() for path in plot_paths]
    summary_path = artifact_root / "train_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return EdaResult(summary=summary, summary_path=summary_path, plot_paths=plot_paths)


def _print_summary(result: EdaResult, root: Path) -> None:
    summary = result.summary
    print("Training-only EDA summary")
    print(f"Rows: {summary['total_training_rows']}")
    print(f"Fraud count: {summary['fraud_count']}")
    print(f"Fraud percentage: {summary['fraud_percentage']:.6f}%")
    print(f"Deterministic sample rows: {summary['deterministic_sample']['total_rows']}")
    print(f"Summary: {result.summary_path.relative_to(root).as_posix()}")
    print("Plots:")
    for path in result.plot_paths:
        print(f"  {path.relative_to(root).as_posix()}")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Generate training-only PaySim EDA artifacts."
    ).parse_args()


def main() -> int:
    parse_args()
    root = repository_root()
    try:
        result = create_training_eda(root=root)
        _print_summary(result, root)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
