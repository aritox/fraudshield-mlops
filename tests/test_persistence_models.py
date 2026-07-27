"""Prediction audit model contract tests."""

from sqlalchemy import inspect

from fraudshield.persistence.models import Base


def test_expected_tables_columns_constraints_relationships_and_indexes() -> None:
    assert set(Base.metadata.tables) == {
        "prediction_requests",
        "prediction_events",
        "prediction_outcomes",
        "monitoring_runs",
        "monitoring_metrics",
    }
    requests = Base.metadata.tables["prediction_requests"]
    events = Base.metadata.tables["prediction_events"]
    outcomes = Base.metadata.tables["prediction_outcomes"]
    assert requests.primary_key.columns.keys() == ["request_id"]
    assert events.primary_key.columns.keys() == ["prediction_id"]
    assert outcomes.primary_key.columns.keys() == ["prediction_id"]
    assert {index.name for index in events.indexes} >= {
        "ix_prediction_events_request_id",
        "ix_prediction_events_risk_level",
        "ix_prediction_events_transaction_type",
    }
    assert any(
        set(constraint.columns.keys()) == {"request_id", "item_index"}
        for constraint in events.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    )


def test_sqlite_materializes_portable_uuid_schema(audit_session_factory) -> None:
    inspector = inspect(audit_session_factory.kw["bind"])
    assert set(inspector.get_table_names()) == set(Base.metadata.tables)
    assert inspector.get_foreign_keys("prediction_events")[0]["options"]["ondelete"] == "CASCADE"
