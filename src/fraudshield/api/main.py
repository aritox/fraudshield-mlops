"""FastAPI application exposing the registered FraudShield SGD champion."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request

from fraudshield.api.config import ApiConfig, load_api_config
from fraudshield.api.errors import (
    BatchSizeExceededError,
    ModelLoadError,
    ModelNotReadyError,
    install_error_handlers,
)
from fraudshield.api.middleware import RequestContextMiddleware
from fraudshield.api.model_service import ProductionModelService
from fraudshield.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    LivenessResponse,
    ModelInfoResponse,
    ReadinessResponse,
    RootResponse,
    SinglePredictionResponse,
    TransactionRequest,
)

LOGGER = logging.getLogger("fraudshield.api")


def _event(name: str, **values: Any) -> str:
    return json.dumps({"event": name, **values}, separators=(",", ":"), sort_keys=True)


def create_app(
    config: ApiConfig | None = None,
    model_service: ProductionModelService | None = None,
    *,
    load_model_on_startup: bool | None = None,
) -> FastAPI:
    """Create an API application with injectable dependencies for tests and export."""

    api_config = config or load_api_config()
    service = model_service or ProductionModelService(api_config)
    should_load = (
        api_config.model.load_on_startup
        if load_model_on_startup is None
        else load_model_on_startup
    )
    logging.basicConfig(level=getattr(logging, api_config.logging.level))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model_service = service
        if should_load:
            try:
                service.load()
                info = service.model_info()
                LOGGER.info(
                    _event(
                        "api_startup_ready",
                        model_alias=info["alias"],
                        model_version=info["resolved_model_version"],
                    )
                )
            except ModelLoadError:
                LOGGER.error(_event("api_startup_not_ready", reason="model_load_failed"))
        else:
            LOGGER.info(_event("api_startup_model_loading_disabled"))
        yield
        LOGGER.info(_event("api_shutdown"))

    application = FastAPI(
        title=api_config.application.title,
        description=api_config.application.description,
        version=api_config.application.version,
        docs_url=api_config.api.docs_url,
        redoc_url=api_config.api.redoc_url,
        openapi_url=api_config.api.openapi_url,
        lifespan=lifespan,
    )
    application.state.api_config = api_config
    application.state.model_service = service
    application.add_middleware(RequestContextMiddleware, logger=LOGGER)
    install_error_handlers(application)

    @application.get("/", response_model=RootResponse)
    async def root() -> RootResponse:
        return RootResponse(
            service=api_config.application.title,
            api_version=api_config.application.version,
            docs_url=api_config.api.docs_url,
            redoc_url=api_config.api.redoc_url,
            openapi_url=api_config.api.openapi_url,
            readiness_status="ready" if service.is_ready() else "not_ready",
        )

    @application.get("/health/live", response_model=LivenessResponse)
    async def liveness() -> LivenessResponse:
        return LivenessResponse(
            status="alive",
            service=api_config.application.title,
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            ),
        )

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ErrorResponse}},
    )
    async def readiness() -> ReadinessResponse:
        if not service.is_ready():
            raise ModelNotReadyError()
        info = service.model_info()
        return ReadinessResponse(
            status="ready",
            model_name=info["registered_model_name"],
            model_version=info["resolved_model_version"],
            model_alias=info["alias"],
        )

    @application.get(
        "/model/info",
        response_model=ModelInfoResponse,
        responses={503: {"model": ErrorResponse}},
    )
    async def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(**service.model_info())

    @application.post(
        "/predict",
        response_model=SinglePredictionResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict(
        transaction: TransactionRequest,
        request: Request,
    ) -> SinglePredictionResponse:
        if not service.is_ready():
            raise ModelNotReadyError()
        started = time.perf_counter()
        result = service.predict_one(transaction)
        info = service.model_info()
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        request.state.batch_size = 1
        request.state.prediction_count = 1
        request.state.high_risk_prediction_count = int(result["risk_level"] == "high")
        request.state.model_alias = info["alias"]
        request.state.model_version = info["resolved_model_version"]
        return SinglePredictionResponse(
            request_id=request.state.request_id,
            fraud_score=result["fraud_score"],
            prediction=result["prediction"],
            threshold=result["threshold"],
            risk_level=result["risk_level"],
            model_name=info["registered_model_name"],
            model_version=info["resolved_model_version"],
            model_alias=info["alias"],
            processing_time_ms=elapsed_ms,
        )

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict_batch(
        batch: BatchPredictionRequest,
        request: Request,
    ) -> BatchPredictionResponse:
        if not service.is_ready():
            raise ModelNotReadyError()
        batch_size = len(batch.transactions)
        request.state.batch_size = batch_size
        if batch_size > api_config.inference.maximum_batch_size:
            raise BatchSizeExceededError(api_config.inference.maximum_batch_size)
        started = time.perf_counter()
        results = service.predict_batch(batch.transactions)
        info = service.model_info()
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        request.state.prediction_count = len(results)
        request.state.high_risk_prediction_count = sum(
            item["risk_level"] == "high" for item in results
        )
        request.state.model_alias = info["alias"]
        request.state.model_version = info["resolved_model_version"]
        return BatchPredictionResponse(
            request_id=request.state.request_id,
            predictions=results,
            prediction_count=len(results),
            model_name=info["registered_model_name"],
            model_version=info["resolved_model_version"],
            model_alias=info["alias"],
            processing_time_ms=elapsed_ms,
        )

    return application


app = create_app()
