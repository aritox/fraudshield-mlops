
"""Configuration and path helpers for dataset ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Dataset ingestion and validation settings."""

    dataset_name: str
    kaggle_handle: str
    target_column: str
    chunk_size: int
    raw_data_directory: Path
    expected_columns: list[str]


def repository_root() -> Path:
    """Return the repository root for the installed source tree."""

    return Path(__file__).resolve().parents[3]


def load_data_config(config_path: Path | None = None, root: Path | None = None) -> DataConfig:
    """Load the dataset configuration from YAML."""

    repo_root = root or repository_root()
    resolved_config_path = config_path or repo_root / "configs" / "data.yaml"

    with resolved_config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    required_keys = {
        "dataset_name",
        "kaggle_handle",
        "target_column",
        "chunk_size",
        "raw_data_directory",
        "expected_columns",
    }
    missing_keys = sorted(required_keys - raw_config.keys())
    if missing_keys:
        joined_keys = ", ".join(missing_keys)
        raise ValueError(f"Missing required data config keys: {joined_keys}")

    raw_data_directory = Path(str(raw_config["raw_data_directory"]))
    if raw_data_directory.is_absolute():
        raise ValueError("raw_data_directory must be relative to the repository root")

    expected_columns = raw_config["expected_columns"]
    if not isinstance(expected_columns, list) or not all(
        isinstance(column, str) for column in expected_columns
    ):
        raise ValueError("expected_columns must be a list of column names")

    chunk_size = int(raw_config["chunk_size"])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    return DataConfig(
        dataset_name=str(raw_config["dataset_name"]),
        kaggle_handle=str(raw_config["kaggle_handle"]),
        target_column=str(raw_config["target_column"]),
        chunk_size=chunk_size,
        raw_data_directory=raw_data_directory,
        expected_columns=expected_columns,
    )


def raw_data_directory(config: DataConfig, root: Path | None = None) -> Path:
    """Return the configured raw data directory under the repository root."""

    repo_root = root or repository_root()
    return repo_root / config.raw_data_directory
