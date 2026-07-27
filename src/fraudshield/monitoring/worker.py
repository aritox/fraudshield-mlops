"""Continuous bounded-loop monitoring worker with internal Prometheus metrics."""

from __future__ import annotations

import json
import logging
import signal
import threading

from prometheus_client import start_http_server

from fraudshield.monitoring.run import build_monitoring_service, result_summary
from fraudshield.monitoring.worker_metrics import WorkerMetrics

LOGGER = logging.getLogger("fraudshield.monitoring.worker")


def run_worker(stop_event: threading.Event | None = None) -> None:
    stop = stop_event or threading.Event()
    service, runtime = build_monitoring_service()
    config = service.config.monitoring
    metrics = WorkerMetrics()
    server, thread = start_http_server(
        port=config.metrics_port,
        addr=config.metrics_host,
        registry=metrics.registry,
    )
    try:
        while not stop.wait(config.interval_seconds):
            try:
                result = service.run_once()
                metrics.record(result)
                LOGGER.info(json.dumps({"event": "monitoring_run", **result_summary(result)}))
            except Exception as error:
                metrics.record_failure()
                LOGGER.error(
                    json.dumps(
                        {
                            "event": "monitoring_run_failed",
                            "error_type": type(error).__name__,
                        }
                    )
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run_worker(stop_event)


if __name__ == "__main__":
    main()
