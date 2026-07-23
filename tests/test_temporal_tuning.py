from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fraudshield.models.temporal_tuning import create_inner_split
from fraudshield.models.train_baseline import reject_test_path


def _rows() -> list[dict[str, object]]:
    rows = []
    for step in range(1, 11):
        for index in range(2):
            fraud = int(step in {3, 7, 10} and index == 0)
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if fraud else "PAYMENT",
                    "amount": 100.0 + step,
                    "oldbalanceOrg": 500.0,
                    "oldbalanceDest": 10.0,
                    "isFraud": fraud,
                    "isFlaggedFraud": 0,
                    "nameOrig": f"C{step}{index}",
                    "nameDest": f"M{step}{index}",
                    "newbalanceOrig": 0.0,
                    "newbalanceDest": 0.0,
                }
            )
    return rows


def _write_train_fixture(root: Path) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(_rows()).to_parquet(processed / "train.parquet", index=False)
    artifacts = root / "artifacts" / "data"
    artifacts.mkdir(parents=True)
    (artifacts / "split_manifest.json").write_text(
        json.dumps({"splitting_method": "chronological_whole_step"}) + "\n",
        encoding="utf-8",
    )


def test_inner_split_uses_complete_non_overlapping_chronological_steps(tmp_path: Path) -> None:
    _write_train_fixture(tmp_path)

    result = create_inner_split(root=tmp_path, batch_size=3, fit_fraction=0.8)

    assert result.fit_window.maximum < result.tuning_window.minimum
    assert result.manifest["no_step_overlap"] is True
    assert result.manifest["chronological_ordering_check"] is True
    assert result.fit_window.maximum == result.boundary_step
    assert result.fit_rows + result.tuning_rows == 20
    assert result.fit_frauds == 2
    assert result.tuning_frauds == 1
    assert result.manifest_path.exists()
    assert result.manifest["test_set_accessed"] is False


def test_test_parquet_is_rejected_by_phase1d_guard() -> None:
    try:
        reject_test_path(Path("data/processed/test.parquet"))
    except ValueError as error:
        assert "sealed test split" in str(error)
    else:
        raise AssertionError("test.parquet was not rejected")
