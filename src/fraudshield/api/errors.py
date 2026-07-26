"""Sanitized API exceptions and error handlers."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("fraudshield.api")


class ApiError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.status_code = status_code


class ModelNotReadyError(ApiError):
    def __init__(self) -> None:
        super().__init__("model_not_ready", "Production model is unavailable", 503)


class BatchSizeExceededError(ApiError):
    def __init__(self, maximum: int) -> None:
        super().__init__(
            "batch_size_exceeded",
            f"Batch contains more than the allowed {maximum} transactions",
            422,
        )


class InferenceError(ApiError):
    def __init__(self) -> None:
        super().__init__("inference_error", "Prediction could not be completed", 500)


class ModelLoadError(RuntimeError):
    """Internal startup error; never return its detail to an API caller."""


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def _payload(
    request: Request,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": code,
        "message": message,
        "request_id": _request_id(request),
    }
    if details is not None:
        payload["details"] = details
    return payload


def install_error_handlers(app: FastAPI) -> None:
    """Install consistent safe JSON error responses."""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {
                "location": [
                    str(item) if not isinstance(item, int) else item
                    for item in error["loc"]
                ],
                "message": error["msg"],
                "error_type": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_payload(
                request,
                "validation_error",
                "Request validation failed",
                details,
            ),
        )

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(request, exc.code, exc.safe_message),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.error(
            json.dumps(
                {
                    "event": "unhandled_api_error",
                    "request_id": _request_id(request),
                    "exception_type": type(exc).__name__,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return JSONResponse(
            status_code=500,
            content=_payload(request, "internal_error", "Internal service error"),
        )
