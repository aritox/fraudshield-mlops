"""Focused frozen-reference and PSI configuration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fraudshield.monitoring.config import load_monitoring_config


def _config(tmp_path: Path, **updates) -> Path:
    raw = yaml.safe_load(Path("configs/monitoring.yaml").read_text(encoding="utf-8"))
    for section, values in updates.items():
        raw[section].update(values)
    path = tmp_path / "configs" / "monitoring.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_monitoring_config_loads_focused_values(tmp_path: Path) -> None:
    config = load_monitoring_config(_config(tmp_path), root=tmp_path)
    assert config.monitoring.interval_seconds == 60
    assert config.monitoring.window_hours == 24
    assert config.monitoring.minimum_events == 20
    assert config.monitoring.minimum_labeled_events == 20
    assert config.monitoring.metrics_host == "0.0.0.0"
    assert config.monitoring.metrics_port == 8001
    assert config.reference.source_split == "train"
    assert config.reference.numeric_quantile_bins == 10
    assert config.reference.epsilon == pytest.approx(0.000001)
    assert config.reference_profile_path == (
        tmp_path / "artifacts/monitoring/reference_profile.json"
    ).resolve()
    assert config.drift.stable_below == 0.10
    assert config.drift.moderate_below == config.drift.significant_at_or_above == 0.25


@pytest.mark.parametrize(
    ("section", "updates"),
    [
        ("monitoring", {"window_hours": 0}),
        ("monitoring", {"interval_seconds": 0}),
        ("monitoring", {"minimum_events": 0}),
        ("monitoring", {"minimum_labeled_events": 0}),
        ("monitoring", {"metrics_host": ""}),
        ("monitoring", {"metrics_port": 65536}),
        ("reference", {"source_split": "validation"}),
        ("reference", {"source_split": "test"}),
        ("reference", {"source_split": "raw"}),
        ("reference", {"profile_path": "data/raw/reference.json"}),
        ("reference", {"profile_path": "data/processed/validation/reference.json"}),
        ("reference", {"profile_path": "C:/private/reference.json"}),
        ("reference", {"numeric_quantile_bins": 0}),
        ("reference", {"epsilon": 0}),
        ("reference", {"epsilon": float("inf")}),
        ("drift", {"stable_below": 0.25}),
        ("drift", {"significant_at_or_above": 0.30}),
    ],
)
def test_monitoring_config_rejects_unsafe_values(
    tmp_path: Path,
    section: str,
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        load_monitoring_config(_config(tmp_path, **{section: updates}), root=tmp_path)


def test_tracked_monitoring_config_contains_no_secret_or_absolute_path() -> None:
    path = Path("configs/monitoring.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    serialized = json.dumps(raw).lower()
    assert "password" not in serialized
    assert "credential" not in serialized
    assert "c:/users/" not in serialized
    assert "c:\\users\\" not in serialized
