"""Strict configuration for the frozen training reference and PSI engine."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fraudshield.data.config import repository_root

CONFIG_RELATIVE_PATH = Path("configs/monitoring.yaml")


@dataclass(frozen=True)
class MonitoringWindowConfig:
    window_hours: int
    minimum_events: int


@dataclass(frozen=True)
class ReferenceConfig:
    profile_path: Path
    version: str
    source_split: str
    numeric_quantile_bins: int
    epsilon: float


@dataclass(frozen=True)
class DriftConfig:
    stable_below: float
    moderate_below: float
    significant_at_or_above: float


@dataclass(frozen=True)
class MonitoringConfig:
    monitoring: MonitoringWindowConfig
    reference: ReferenceConfig
    drift: DriftConfig
    repository_root: Path
    config_path: Path

    @property
    def reference_profile_path(self) -> Path:
        return (self.repository_root / self.reference.profile_path).resolve()


def _section(raw: dict[str, Any], name: str, expected_keys: set[str]) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict) or set(section) != expected_keys:
        raise ValueError(f"Monitoring configuration section {name!r} is invalid")
    return section


def _validate_safe_content(raw: dict[str, Any], root: Path) -> None:
    serialized = json.dumps(raw, sort_keys=True)
    lowered = serialized.lower()
    if any(token in lowered for token in ("password", "credential", "api_key", "secret")):
        raise ValueError("Monitoring configuration must not contain credentials")
    if str(root).lower() in lowered:
        raise ValueError("Monitoring configuration must not contain an absolute repository path")


def load_monitoring_config(
    config_path: Path | None = None,
    root: Path | None = None,
) -> MonitoringConfig:
    """Load and validate the focused Phase 2D.2 monitoring configuration."""

    repo_root = (root or repository_root()).resolve()
    path = (config_path or repo_root / CONFIG_RELATIVE_PATH).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("Monitoring configuration must be inside the repository") from error

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) != {"monitoring", "reference", "drift"}:
        raise ValueError("Monitoring configuration sections are invalid")
    _validate_safe_content(raw, repo_root)

    monitoring = _section(raw, "monitoring", {"window_hours", "minimum_events"})
    reference = _section(
        raw,
        "reference",
        {"profile_path", "version", "source_split", "numeric_quantile_bins", "epsilon"},
    )
    drift = _section(
        raw,
        "drift",
        {"stable_below", "moderate_below", "significant_at_or_above"},
    )

    window_hours = int(monitoring["window_hours"])
    minimum_events = int(monitoring["minimum_events"])
    if window_hours <= 0 or minimum_events <= 0:
        raise ValueError("Monitoring window and minimum event count must be positive")

    if reference["source_split"] != "train":
        raise ValueError("Reference source split must be exactly train")
    profile_path = Path(str(reference["profile_path"]))
    if profile_path.is_absolute():
        raise ValueError("Reference profile path must be repository-relative")
    if {part.lower() for part in profile_path.parts} & {"raw", "validation", "test"}:
        raise ValueError("Reference profile path must not target raw, validation, or test data")
    resolved_profile = (repo_root / profile_path).resolve()
    try:
        resolved_profile.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("Reference profile path escapes the repository") from error
    version = str(reference["version"]).strip()
    if not version or any(character.isspace() for character in version):
        raise ValueError("Reference profile version must be non-empty and contain no whitespace")
    quantile_bins = int(reference["numeric_quantile_bins"])
    epsilon = float(reference["epsilon"])
    if quantile_bins <= 0:
        raise ValueError("Reference quantile bin count must be positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Reference epsilon must be finite and positive")

    stable = float(drift["stable_below"])
    moderate = float(drift["moderate_below"])
    significant = float(drift["significant_at_or_above"])
    if not all(math.isfinite(value) for value in (stable, moderate, significant)):
        raise ValueError("PSI thresholds must be finite")
    if not 0 <= stable < moderate or significant != moderate:
        raise ValueError("PSI thresholds must satisfy stable < moderate = significant")

    return MonitoringConfig(
        monitoring=MonitoringWindowConfig(window_hours, minimum_events),
        reference=ReferenceConfig(
            profile_path=profile_path,
            version=version,
            source_split="train",
            numeric_quantile_bins=quantile_bins,
            epsilon=epsilon,
        ),
        drift=DriftConfig(stable, moderate, significant),
        repository_root=repo_root,
        config_path=path,
    )
