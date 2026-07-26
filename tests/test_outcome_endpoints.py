"""Delayed outcome API tests."""

import uuid
from pathlib import Path

from test_api_endpoints import ReadyService, _client, _transaction


def test_outcome_create_replay_conflict_and_audit(tmp_path: Path) -> None:
    with _client(tmp_path, ReadyService()) as client:
        prediction = client.post("/predict", json=_transaction()).json()
        payload = {
            "outcomes": [
                {
                    "prediction_id": prediction["prediction_id"],
                    "actual_fraud": 1,
                    "source": "synthetic-review",
                }
            ]
        }
        created = client.post("/outcomes", json=payload)
        replay = client.post("/outcomes", json=payload)
        conflicting = client.post(
            "/outcomes",
            json={"outcomes": [{**payload["outcomes"][0], "actual_fraud": 0}]},
        )
        audit = client.get(f"/predictions/{prediction['prediction_id']}")
    assert created.status_code == 200
    assert created.json()["outcomes"][0]["replayed"] is False
    assert replay.json()["outcomes"][0]["replayed"] is True
    assert conflicting.status_code == 409
    assert audit.json()["outcome"]["actual_fraud"] == 1


def test_unknown_outcome_batch_is_atomic(tmp_path: Path) -> None:
    with _client(tmp_path, ReadyService()) as client:
        prediction = client.post("/predict", json=_transaction()).json()
        response = client.post(
            "/outcomes",
            json={
                "outcomes": [
                    {
                        "prediction_id": prediction["prediction_id"],
                        "actual_fraud": 1,
                        "source": "synthetic",
                    },
                    {
                        "prediction_id": str(uuid.uuid4()),
                        "actual_fraud": 0,
                        "source": "synthetic",
                    },
                ]
            },
        )
        audit = client.get(f"/predictions/{prediction['prediction_id']}")
    assert response.status_code == 404
    assert audit.json()["outcome"] is None
