"""Database configuration and secret-safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraudshield.persistence.config import load_database_config


def test_database_configuration_and_environment_override(monkeypatch) -> None:
    config = load_database_config()
    monkeypatch.setenv(
        "FRAUDSHIELD_DATABASE_URL",
        "postgresql+psycopg://app:secret@localhost:5432/example",
    )
    url = config.url()
    assert url.drivername == "postgresql+psycopg"
    assert url.database == "example"
    assert "secret" not in str(url)
    assert config.pool.pool_size == 5
    assert config.persistence.required_for_predictions is True


def test_missing_password_and_invalid_override_are_safe() -> None:
    config = load_database_config()
    with pytest.raises(ValueError, match="credentials") as missing:
        config.url({"POSTGRES_USER": "app", "POSTGRES_DB": "fraudshield"})
    assert "password" not in str(missing.value).lower()
    with pytest.raises(ValueError, match="must use"):
        config.url({"FRAUDSHIELD_DATABASE_URL": "sqlite:///private.db"})


def test_tracked_configuration_has_no_absolute_path_or_password() -> None:
    text = Path("configs/database.yaml").read_text(encoding="utf-8")
    assert "C:\\Users" not in text
    assert "replace_with" not in text
    assert "password:" not in text.lower()
