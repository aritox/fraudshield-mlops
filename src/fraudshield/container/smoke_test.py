"""Run the synthetic local Compose smoke test and export Phase 2C manifests."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import dotenv_values

from fraudshield.container.package_model import MANIFEST_RELATIVE
from fraudshield.data.config import repository_root
from fraudshield.persistence.config import load_database_config
from fraudshield.persistence.database import create_database_runtime
from fraudshield.persistence.repository import PredictionRepository
from fraudshield.persistence.verify_database import verify_database
from fraudshield.tracking.mlflow_setup import install_prohibited_data_guard, write_json

REPORT_RELATIVE = Path("artifacts/container/compose_smoke_report.json")
CONTAINER_MANIFEST_RELATIVE = Path("artifacts/container/container_manifest.json")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _compose(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *arguments],
        cwd=root,
        capture_output=True,
        check=check,
        text=True,
    )


def _http(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    request_id: uuid.UUID | None = None,
) -> tuple[int, dict[str, Any] | None, dict[str, str]]:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if request_id is not None:
        headers["X-Request-ID"] = str(request_id)
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error:
        response = error
    content = response.read()
    parsed = (
        json.loads(content) if content and "json" in response.headers.get_content_type() else None
    )
    return response.status, parsed, {key.lower(): value for key, value in response.headers.items()}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_single_model_inference_event(logs: str, request_id: uuid.UUID) -> int:
    """Require one explicit inference event, ignoring request-summary boolean fields."""

    marker = '"event":"model_inference_invoked"'
    count = sum(
        str(request_id) in line and marker in line
        for line in logs.splitlines()
    )
    _require(count == 1, "Replay invoked model inference")
    return count


def _wait_http(url: str, expected_status: int, *, attempts: int = 60) -> None:
    for _ in range(attempts):
        try:
            status, _, _ = _http(url)
            if status == expected_status:
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for HTTP {expected_status}")


def _wait_service(root: Path, service: str, state: str, *, attempts: int = 60) -> None:
    for _ in range(attempts):
        result = _compose(root, "ps", "--all", "--format", "json", check=False)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        record = next((item for item in records if item.get("Service") == service), None)
        if record is not None:
            status = str(record.get("State", "")).lower()
            health = str(record.get("Health", "")).lower()
            if state == "healthy" and status == "running" and health == "healthy":
                return
            if (
                state == "exited-zero"
                and status == "exited"
                and int(record.get("ExitCode", 1)) == 0
            ):
                return
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {service} to become {state}")


@contextmanager
def _database_url(root: Path, values: dict[str, str | None]):
    password = values.get("POSTGRES_PASSWORD")
    user = values.get("POSTGRES_USER")
    database = values.get("POSTGRES_DB")
    port = values.get("FRAUDSHIELD_POSTGRES_PORT", "5432")
    _require(
        bool(password and user and database and port), "Required local database values are missing"
    )
    previous = os.environ.get("FRAUDSHIELD_DATABASE_URL")
    os.environ["FRAUDSHIELD_DATABASE_URL"] = (
        f"postgresql+psycopg://{quote(str(user), safe='')}:{quote(str(password), safe='')}"
        f"@127.0.0.1:{port}/{quote(str(database), safe='')}"
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FRAUDSHIELD_DATABASE_URL", None)
        else:
            os.environ["FRAUDSHIELD_DATABASE_URL"] = previous


def _row_counts(root: Path) -> dict[str, int]:
    config = load_database_config(root=root)
    runtime = create_database_runtime(config)
    try:
        with runtime.session_factory() as session, session.begin():
            return PredictionRepository().row_counts(session)
    finally:
        runtime.engine.dispose()


def _service_health(root: Path) -> dict[str, dict[str, Any]]:
    result = _compose(root, "ps", "--all", "--format", "json")
    return {
        item["Service"]: {
            "state": item.get("State"),
            "health": item.get("Health") or "not_applicable",
            "exit_code": item.get("ExitCode"),
        }
        for item in (json.loads(line) for line in result.stdout.splitlines() if line.strip())
    }


def _export_container_manifest(
    root: Path,
    service_health: dict[str, Any],
    postgres_port: str,
) -> dict[str, Any]:
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", "fraudshield-api:phase2c"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    image = json.loads(inspect_result.stdout)[0]
    model = json.loads((root / MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    manifest = {
        "generation_timestamp_utc": _timestamp(),
        "base_python_image": "python:3.12-slim-bookworm",
        "api_image_identifier": image["Id"],
        "api_image_size_bytes": image["Size"],
        "image_creation_timestamp_utc": image["Created"],
        "packaged_model_name": model["registered_model_name"],
        "packaged_model_version": model["resolved_version"],
        "packaged_model_checksum": model["exported_package_checksum"],
        "exposed_container_port": 8000,
        "published_host_binding": "127.0.0.1:8000",
        "postgresql_host_binding": f"127.0.0.1:{postgres_port}",
        "compose_service_names": ["postgres", "migrate", "api"],
        "non_root": image["Config"].get("User") == "10001:10001",
        "health_check_status": service_health,
        "raw_data_included": False,
        "parquet_data_included": False,
        "source_joblib_included": False,
        "credentials_included": False,
    }
    write_json(root / CONTAINER_MANIFEST_RELATIVE, manifest)
    return manifest


def run_smoke_test(root: Path | None = None) -> dict[str, Any]:
    repo_root = (root or repository_root()).resolve()
    install_prohibited_data_guard(repo_root)
    environment = {key: value for key, value in dotenv_values(repo_root / ".env").items()}
    api_port = environment.get("FRAUDSHIELD_API_PORT", "8000")
    base_url = f"http://127.0.0.1:{api_port}"
    single_request_id = uuid.uuid4()
    batch_request_id = uuid.uuid4()
    failure_request_id = uuid.uuid4()
    single_body = {
        "step": 24,
        "type": "TRANSFER",
        "amount": 1500.0,
        "oldbalanceOrg": 1500.0,
        "oldbalanceDest": 0.0,
    }
    batch_body = {
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 25.0,
                "oldbalanceOrg": 100.0,
                "oldbalanceDest": 50.0,
            },
            {
                "step": 24,
                "type": "TRANSFER",
                "amount": 1500.0,
                "oldbalanceOrg": 1500.0,
                "oldbalanceDest": 0.0,
            },
            {
                "step": 48,
                "type": "CASH_OUT",
                "amount": 425.0,
                "oldbalanceOrg": 425.0,
                "oldbalanceDest": 10.0,
            },
        ]
    }
    with _database_url(repo_root, environment):
        _wait_service(repo_root, "postgres", "healthy")
        _wait_service(repo_root, "migrate", "exited-zero")
        _wait_service(repo_root, "api", "healthy")
        health_results = {}
        for path in ("/health/live", "/health/db", "/health/ready", "/model/info", "/docs"):
            status, body, _ = _http(base_url + path)
            _require(status == 200, f"{path} did not return HTTP 200")
            health_results[path] = status
            if path == "/model/info":
                _require(
                    body is not None and body["model_family"] == "SGDClassifier", "Wrong model"
                )

        before = _row_counts(repo_root)
        status, single, headers = _http(
            base_url + "/predict", method="POST", body=single_body, request_id=single_request_id
        )
        _require(status == 200 and single is not None, "Single prediction failed")
        _require(single["threshold"] == 0.98310834, "Production threshold changed")
        _require(single["model_version"] == "1", "Production model version changed")
        _require(single["model_name"] == "fraudshield-production-sgd", "Wrong model name")
        _require("x-process-time-ms" in headers, "Latency response header is missing")
        after_single = _row_counts(repo_root)
        _require(
            after_single["prediction_requests"] == before["prediction_requests"] + 1,
            "Single request row was not persisted",
        )
        _require(
            after_single["prediction_events"] == before["prediction_events"] + 1,
            "Single prediction event was not persisted",
        )

        status, replay, replay_headers = _http(
            base_url + "/predict", method="POST", body=single_body, request_id=single_request_id
        )
        _require(status == 200 and replay == single, "Idempotent replay changed the response")
        _require(replay_headers.get("x-idempotent-replay") == "true", "Replay header missing")
        after_replay = _row_counts(repo_root)
        _require(after_replay == after_single, "Replay increased database row counts")

        conflict_body = dict(single_body, amount=1501.0)
        status, conflict, _ = _http(
            base_url + "/predict",
            method="POST",
            body=conflict_body,
            request_id=single_request_id,
        )
        _require(
            status == 409 and conflict and conflict["error"] == "idempotency_conflict",
            "Conflict failed",
        )
        after_conflict = _row_counts(repo_root)
        _require(after_conflict == after_single, "Conflict changed database rows")

        status, batch, _ = _http(
            base_url + "/predict/batch", method="POST", body=batch_body, request_id=batch_request_id
        )
        _require(status == 200 and batch is not None, "Batch prediction failed")
        _require(batch["prediction_count"] == 3, "Batch result count is not three")
        prediction_ids = [item["prediction_id"] for item in batch["predictions"]]
        _require(len(set(prediction_ids)) == 3, "Batch prediction IDs are not unique")
        after_batch = _row_counts(repo_root)
        _require(
            after_batch["prediction_requests"] == after_single["prediction_requests"] + 1,
            "Batch request was not atomic",
        )
        _require(
            after_batch["prediction_events"] == after_single["prediction_events"] + 3,
            "Batch events were not atomic",
        )

        selected_prediction_id = single["prediction_id"]
        status, audit, _ = _http(base_url + f"/predictions/{selected_prediction_id}")
        _require(status == 200 and audit is not None, "Prediction audit lookup failed")
        _require(audit["fraud_score"] == single["fraud_score"], "Audit score does not match")

        outcome_body = {
            "outcomes": [
                {
                    "prediction_id": selected_prediction_id,
                    "actual_fraud": 1,
                    "source": "phase2c-synthetic-smoke",
                }
            ]
        }
        status, outcome, _ = _http(base_url + "/outcomes", method="POST", body=outcome_body)
        _require(status == 200 and outcome is not None, "Outcome creation failed")
        _require(outcome["outcomes"][0]["replayed"] is False, "First outcome was a replay")
        status, outcome_replay, _ = _http(base_url + "/outcomes", method="POST", body=outcome_body)
        _require(status == 200 and outcome_replay is not None, "Outcome replay failed")
        _require(outcome_replay["outcomes"][0]["replayed"] is True, "Outcome replay not identified")
        conflicting_outcome = json.loads(json.dumps(outcome_body))
        conflicting_outcome["outcomes"][0]["actual_fraud"] = 0
        status, outcome_conflict, _ = _http(
            base_url + "/outcomes", method="POST", body=conflicting_outcome
        )
        _require(
            status == 409 and outcome_conflict and outcome_conflict["error"] == "outcome_conflict",
            "Outcome conflict failed",
        )
        counts_before_restart = _row_counts(repo_root)

        _compose(repo_root, "stop", "postgres")
        _wait_http(base_url + "/health/ready", 503, attempts=30)
        status, unavailable, _ = _http(
            base_url + "/predict", method="POST", body=single_body, request_id=failure_request_id
        )
        _require(
            status == 503 and unavailable and unavailable["error"] == "persistence_unavailable",
            "Unavailable database did not block prediction",
        )

        _compose(repo_root, "stop", "api")
        _compose(repo_root, "start", "postgres")
        _wait_service(repo_root, "postgres", "healthy")
        _compose(repo_root, "start", "api")
        _wait_service(repo_root, "api", "healthy")
        _wait_http(base_url + "/health/ready", 200)
        counts_after_restart = _row_counts(repo_root)
        _require(counts_after_restart == counts_before_restart, "Database rows did not persist")
        status, audit_after_restart, _ = _http(base_url + f"/predictions/{selected_prediction_id}")
        _require(status == 200 and audit_after_restart is not None, "Audit did not survive restart")
        _require(audit_after_restart["outcome"] is not None, "Outcome did not survive restart")

        logs = _compose(repo_root, "logs", "--no-color", "api").stdout
        _require(
            logs.count(f'"request_id":"{single_request_id}"') >= 1, "Request ID missing from logs"
        )
        inference_count = require_single_model_inference_event(logs, single_request_id)
        forbidden_log_values = [
            '"amount"',
            "oldbalanceOrg",
            "oldbalanceDest",
            "postgresql+psycopg://",
            str(environment["POSTGRES_PASSWORD"]),
        ]
        _require(
            not any(value and value in logs for value in forbidden_log_values),
            "Sensitive value found in API logs",
        )

        database = verify_database(repo_root)
        services = _service_health(repo_root)
        container_manifest = _export_container_manifest(
            repo_root,
            services,
            str(environment.get("FRAUDSHIELD_POSTGRES_PORT") or "5432"),
        )
        report = {
            "timestamp_utc": _timestamp(),
            "service_health": services,
            "migration_revision": database["alembic_revision"],
            "api_liveness_readiness": health_results,
            "synthetic_request_ids": {
                "single": str(single_request_id),
                "batch": str(batch_request_id),
                "database_failure": str(failure_request_id),
            },
            "prediction_ids": [selected_prediction_id, *prediction_ids],
            "database_row_counts": counts_after_restart,
            "idempotent_replay_result": {
                "status": "passed",
                "same_prediction_id": replay["prediction_id"] == selected_prediction_id,
                "header": replay_headers["x-idempotent-replay"],
                "model_inference_invocations": inference_count,
                "row_counts_unchanged": after_replay == after_single,
            },
            "conflicting_retry_result": {"status": "passed", "http_status": 409},
            "batch_atomicity_result": {"status": "passed", "prediction_count": 3},
            "outcome_persistence_result": {
                "status": "passed",
                "identical_replay": True,
                "conflict_http_status": 409,
            },
            "database_failure_result": {
                "status": "passed",
                "readiness_http_status": 503,
                "prediction_http_status": 503,
            },
            "volume_persistence_result": {
                "status": "passed",
                "row_counts_preserved": counts_after_restart == counts_before_restart,
                "audit_record_preserved": True,
            },
            "log_safety_result": "passed",
            "container_image_identifier": container_manifest["api_image_identifier"],
            "no_real_dataset_used": True,
            "smoke_test_status": "passed",
        }
        write_json(repo_root / REPORT_RELATIVE, report)
        return report


def main() -> None:
    result = run_smoke_test()
    print(
        json.dumps(
            {
                "status": result["smoke_test_status"],
                "prediction_ids": result["prediction_ids"],
                "database_row_counts": result["database_row_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
