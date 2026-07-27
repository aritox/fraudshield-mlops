"""Run one production-monitoring calculation from persisted PostgreSQL records."""

from __future__ import annotations

import argparse
import json

from fraudshield.data.config import repository_root
from fraudshield.monitoring.config import load_monitoring_config
from fraudshield.monitoring.service import MonitoringRunResult, MonitoringService
from fraudshield.persistence.config import load_database_config
from fraudshield.persistence.database import DatabaseRuntime, create_database_runtime
from fraudshield.tracking.mlflow_setup import install_prohibited_data_guard


def build_monitoring_service() -> tuple[MonitoringService, DatabaseRuntime]:
    root = repository_root().resolve()
    install_prohibited_data_guard(root)
    monitoring_config = load_monitoring_config(root=root)
    database_config = load_database_config(root=root)
    runtime = create_database_runtime(database_config)
    return MonitoringService(runtime.session_factory, monitoring_config), runtime


def result_summary(result: MonitoringRunResult) -> dict[str, object]:
    return {
        "run_id": str(result.run_id),
        "status": result.status,
        "overall_drift_status": result.overall_drift_status,
        "event_count": result.event_count,
        "labeled_count": result.labeled_count,
        "performance_available": result.metric("performance_available") == 1.0,
        "reference_version": result.reference_version,
        "window_start": result.window_start.isoformat(),
        "window_end": result.window_end.isoformat(),
    }


def run_once() -> MonitoringRunResult:
    service, runtime = build_monitoring_service()
    try:
        return service.run_once()
    finally:
        runtime.engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run exactly one monitoring window")
    arguments = parser.parse_args()
    if not arguments.once:
        parser.error("Only --once is supported by this command; use monitoring.worker continuously")
    print(json.dumps(result_summary(run_once()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
