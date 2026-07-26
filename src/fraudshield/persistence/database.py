"""SQLAlchemy engine, sessions, transactions, and safe health checks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from fraudshield.persistence.config import DatabaseConfig


@dataclass(frozen=True)
class DatabaseRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]


EXPECTED_ALEMBIC_REVISION = "phase2c_001"


@dataclass(frozen=True)
class DatabaseHealth:
    healthy: bool
    database_type: str
    migration_status: str
    error_code: str | None = None


class DatabaseHealthService:
    def __init__(self, runtime: DatabaseRuntime) -> None:
        self.runtime = runtime

    def status(self) -> DatabaseHealth:
        try:
            with self.runtime.engine.connect() as connection:
                database_type = (
                    "PostgreSQL" if connection.dialect.name == "postgresql" else "SQLite"
                )
                try:
                    revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
                except Exception:
                    return DatabaseHealth(
                        healthy=False,
                        database_type=database_type,
                        migration_status="not_current",
                        error_code="migration_not_current",
                    )
                if revision != EXPECTED_ALEMBIC_REVISION:
                    return DatabaseHealth(
                        healthy=False,
                        database_type=database_type,
                        migration_status="not_current",
                        error_code="migration_not_current",
                    )
                return DatabaseHealth(
                    healthy=True,
                    database_type=database_type,
                    migration_status="current",
                )
        except Exception:
            return DatabaseHealth(
                healthy=False,
                database_type="PostgreSQL",
                migration_status="unknown",
                error_code="database_not_ready",
            )


def create_database_runtime(
    config: DatabaseConfig, url: URL | str | None = None
) -> DatabaseRuntime:
    resolved = url or config.url()
    is_sqlite = str(resolved).startswith("sqlite")
    options: dict[str, Any] = {"future": True, "pool_pre_ping": config.pool.pool_pre_ping}
    if not is_sqlite:
        options.update(
            pool_size=config.pool.pool_size,
            max_overflow=config.pool.max_overflow,
            pool_timeout=config.pool.pool_timeout_seconds,
            pool_recycle=config.pool.pool_recycle_seconds,
            connect_args={"connect_timeout": config.pool.connect_timeout_seconds},
        )
    engine = create_engine(resolved, **options)
    return DatabaseRuntime(
        engine=engine,
        session_factory=sessionmaker(bind=engine, expire_on_commit=False, future=True),
    )


@contextmanager
def transaction(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        with session.begin():
            yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connectivity(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
