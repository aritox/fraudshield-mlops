from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fraudshield.data.config import DataConfig
from fraudshield.data.download import download_dataset

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def make_config() -> DataConfig:
    return DataConfig(
        dataset_name="paysim",
        kaggle_handle="ealaxi/paysim1",
        target_column="isFraud",
        chunk_size=2,
        raw_data_directory=Path("data/raw"),
        expected_columns=EXPECTED_COLUMNS,
    )


def write_csv(csv_path: Path, columns: list[str] | None = None) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected_columns = columns or EXPECTED_COLUMNS
    row = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.5,
        "nameOrig": "C1",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 899.5,
        "nameDest": "M1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        file.write(",".join(selected_columns) + "\n")
        file.write(",".join(str(row.get(column, "")) for column in selected_columns) + "\n")
    return csv_path


def test_gitkeep_does_not_block_download_and_uses_generated_output_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / ".gitkeep").touch()
    calls: list[dict[str, object]] = []

    def dataset_download(handle: str, output_dir: str, force_download: bool) -> None:
        calls.append(
            {
                "handle": handle,
                "output_dir": output_dir,
                "force_download": force_download,
            }
        )
        write_csv(Path(output_dir) / "nested" / "PS_20174392719_1491204439457_log.csv")

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=dataset_download),
    )

    csv_path = download_dataset(config=make_config(), root=tmp_path)

    assert csv_path.relative_to(tmp_path).as_posix() == (
        "data/raw/paysim_download/nested/PS_20174392719_1491204439457_log.csv"
    )
    assert (raw_dir / ".gitkeep").exists()
    assert calls == [
        {
            "handle": "ealaxi/paysim1",
            "output_dir": str(tmp_path / "data" / "raw" / "paysim_download"),
            "force_download": False,
        }
    ]


def test_valid_nested_csv_is_reused_without_downloading(tmp_path: Path, monkeypatch) -> None:
    expected_csv = write_csv(tmp_path / "data" / "raw" / "existing" / "paysim.csv")

    def dataset_download(handle: str, output_dir: str, force_download: bool) -> None:
        raise AssertionError("download should not be called")

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=dataset_download),
    )

    csv_path = download_dataset(config=make_config(), root=tmp_path)

    assert csv_path == expected_csv


def test_invalid_csv_is_ignored_before_download(tmp_path: Path, monkeypatch) -> None:
    write_csv(tmp_path / "data" / "raw" / "unrelated.csv", columns=["not", "paysim"])

    def dataset_download(handle: str, output_dir: str, force_download: bool) -> None:
        write_csv(Path(output_dir) / "valid.csv")

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=dataset_download),
    )

    csv_path = download_dataset(config=make_config(), root=tmp_path)

    assert csv_path.relative_to(tmp_path).as_posix() == "data/raw/paysim_download/valid.csv"


def test_force_replaces_only_generated_download_directory(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    generated_dir = raw_dir / "paysim_download"
    raw_dir.mkdir(parents=True)
    (raw_dir / ".gitkeep").touch()
    (raw_dir / "notes.txt").write_text("keep", encoding="utf-8")
    write_csv(raw_dir / "unrelated.csv", columns=["not", "paysim"])
    (generated_dir / "old").mkdir(parents=True)
    (generated_dir / "old" / "stale.txt").write_text("remove", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def dataset_download(handle: str, output_dir: str, force_download: bool) -> None:
        calls.append({"output_dir": output_dir, "force_download": force_download})
        write_csv(Path(output_dir) / "fresh" / "paysim.csv")

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=dataset_download),
    )

    csv_path = download_dataset(config=make_config(), root=tmp_path, force=True)

    assert csv_path.relative_to(tmp_path).as_posix() == "data/raw/paysim_download/fresh/paysim.csv"
    assert calls == [
        {
            "output_dir": str(tmp_path / "data" / "raw" / "paysim_download"),
            "force_download": True,
        }
    ]
    assert (raw_dir / ".gitkeep").exists()
    assert (raw_dir / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert (raw_dir / "unrelated.csv").exists()
    assert not (generated_dir / "old" / "stale.txt").exists()
