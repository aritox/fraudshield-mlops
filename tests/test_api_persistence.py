"""API persistence and idempotency behavior tests."""

from pathlib import Path

from test_api_endpoints import ReadyService, _client, _transaction


def test_prediction_ids_replay_conflict_and_audit(tmp_path: Path) -> None:
    request_id = "11111111-1111-4111-8111-111111111111"
    service = ReadyService()
    with _client(tmp_path, service) as client:
        first = client.post("/predict", json=_transaction(), headers={"X-Request-ID": request_id})
        replay = client.post("/predict", json=_transaction(), headers={"X-Request-ID": request_id})
        conflict = client.post(
            "/predict", json=_transaction(amount=101.0), headers={"X-Request-ID": request_id}
        )
        audit = client.get(f"/predictions/{first.json()['prediction_id']}")
    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    assert replay.json() == first.json()
    assert service.batch_calls == 1
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "idempotency_conflict"
    assert audit.status_code == 200
    assert audit.json()["model_version"] == "1"


def test_non_uuid_request_id_is_rejected(tmp_path: Path) -> None:
    with _client(tmp_path, ReadyService()) as client:
        response = client.post(
            "/predict", json=_transaction(), headers={"X-Request-ID": "request-123"}
        )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
