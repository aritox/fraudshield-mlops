"""Alembic migration and application-startup contract tests."""

from pathlib import Path


def test_initial_revision_and_metadata_intent_match() -> None:
    migration = Path("alembic/versions/phase2c_001_create_prediction_audit_schema.py").read_text(
        encoding="utf-8"
    )
    assert 'revision = "phase2c_001"' in migration
    assert "down_revision = None" in migration
    for table in ("prediction_requests", "prediction_events", "prediction_outcomes"):
        assert f'"{table}"' in migration
    assert migration.index('op.drop_table("prediction_outcomes")') < migration.index(
        'op.drop_table("prediction_events")'
    )


def test_application_has_no_schema_creation_fallback() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("src/fraudshield/api").glob("*.py")
    )
    assert "create_all" not in source
    assert "alembic upgrade" not in source
    assert "password" not in Path("alembic.ini").read_text(encoding="utf-8").lower()
