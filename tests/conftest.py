"""Phase 2C test safety guard and isolated database fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fraudshield.persistence.models import Base
from fraudshield.tracking.mlflow_setup import install_prohibited_data_guard

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
install_prohibited_data_guard(REPOSITORY_ROOT)


@pytest.fixture
def audit_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()
