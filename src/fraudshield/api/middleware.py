"""Request correlation, latency headers, and safe structured logging."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach correlation and latency metadata without logging request bodies."""

    def __init__(self, app, logger: logging.Logger | None = None) -> None:
        super().__init__(app)
        self.logger = logger or logging.getLogger("fraudshield.api")

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        generated = str(uuid.uuid4())
        if incoming:
            try:
                request_id = str(uuid.UUID(incoming))
            except ValueError:
                response = JSONResponse(
                    status_code=422,
                    content={
                        "error": "validation_error",
                        "message": "X-Request-ID must be a UUID",
                        "request_id": generated,
                    },
                )
                response.headers["X-Request-ID"] = generated
                response.headers["X-Process-Time-Ms"] = "0.000"
                return response
        else:
            request_id = generated
        started = time.perf_counter()
        request.state.request_id = request_id
        request.state.batch_size = None
        request.state.prediction_count = None
        request.state.high_risk_prediction_count = None
        request.state.model_alias = None
        request.state.model_version = None
        request.state.idempotent_replay = False
        request.state.model_inference_invoked = False
        response = await call_next(request)
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        if request.state.idempotent_replay:
            response.headers["X-Idempotent-Replay"] = "true"
        event = {
            "timestamp_utc": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time_ms": round(elapsed_ms, 3),
            "batch_size": request.state.batch_size,
            "prediction_count": request.state.prediction_count,
            "high_risk_prediction_count": request.state.high_risk_prediction_count,
            "model_alias": request.state.model_alias,
            "model_version": request.state.model_version,
            "idempotent_replay": request.state.idempotent_replay,
            "model_inference_invoked": request.state.model_inference_invoked,
        }
        self.logger.info(json.dumps(event, separators=(",", ":"), sort_keys=True))
        return response
