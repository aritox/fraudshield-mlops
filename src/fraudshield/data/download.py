"""Download the public Kaggle PaySim dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from fraudshield.data.config import DataConfig, load_data_config, raw_data_directory

PAYSIM_DOWNLOAD_DIRECTORY = "paysim_download"


def _csv_header(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def find_matching_csv(raw_dir: Path, expected_columns: list[str]) -> Path | None:
    """Return an existing CSV whose header matches the configured schema."""

    if not raw_dir.exists():
        return None

    matches: list[Path] = []
    for csv_path in sorted(raw_dir.rglob("*.csv")):
        try:
            if _csv_header(csv_path) == expected_columns:
                matches.append(csv_path)
        except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
            continue

    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_size)


def _format_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _print_result(csv_path: Path, root: Path, status: str) -> None:
    relative_path = csv_path.relative_to(root)
    print(f"CSV path: {relative_path}")
    print(f"Filename: {csv_path.name}")
    print(f"File size: {_format_size_mb(csv_path):.2f} MB")
    print(f"Status: {status}")


def _is_authentication_error(error: Exception) -> bool:
    message = str(error).lower()
    auth_markers = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "credential",
        "credentials",
        "authenticate",
        "authentication",
        "kaggle.json",
    )
    return any(marker in message for marker in auth_markers)


def paysim_download_directory(config: DataConfig, root: Path) -> Path:
    """Return the generated PaySim download directory under the raw data root."""

    return raw_data_directory(config, root) / PAYSIM_DOWNLOAD_DIRECTORY


def _replace_download_directory(download_dir: Path) -> None:
    if not download_dir.exists():
        return
    if download_dir.is_dir():
        shutil.rmtree(download_dir)
    else:
        download_dir.unlink()


def _csv_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.csv"))


def download_dataset(config: DataConfig, root: Path, force: bool = False) -> Path:
    """Download or reuse the configured dataset and return the CSV path."""

    raw_dir = raw_data_directory(config, root)
    download_dir = paysim_download_directory(config, root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_csv = find_matching_csv(raw_dir, config.expected_columns)
    if existing_csv is not None and not force:
        _print_result(existing_csv, root, "reused")
        return existing_csv

    if force:
        _replace_download_directory(download_dir)

    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        kagglehub.dataset_download(
            config.kaggle_handle,
            output_dir=str(download_dir),
            force_download=force,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "kagglehub is not installed. Reinstall the project with development dependencies."
        ) from error
    except Exception as error:
        if _is_authentication_error(error):
            raise RuntimeError(
                "Kaggle authentication is required. Configure Kaggle credentials locally "
                "without committing or printing any token, then retry."
            ) from error
        raise RuntimeError(f"Dataset download failed: {error}") from error

    downloaded_csv = find_matching_csv(download_dir, config.expected_columns)
    if downloaded_csv is None:
        discovered_csvs = _csv_files(download_dir)
        if discovered_csvs:
            csv_names = ", ".join(path.relative_to(root).as_posix() for path in discovered_csvs)
            raise RuntimeError(
                "CSV files were downloaded, but none matched the expected PaySim schema: "
                f"{csv_names}"
            )
        raise RuntimeError(
            f"No CSV was found under {download_dir} after download."
        )

    _print_result(downloaded_csv, root, "downloaded")
    return downloaded_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kaggle PaySim dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if a valid CSV exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]

    try:
        config = load_data_config(root=root)
        download_dataset(config=config, root=root, force=args.force)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
