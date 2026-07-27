"""FastAPI application exposing persisted production SGD predictions."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

from fraudshield.api.config import ApiConfig, load_api_config
from fraudshield.api.errors import (
    BatchSizeExceededError,
    DatabaseNotReadyError,
    IdempotencyConflictApiError,
    MigrationNotCurrentError,
    ModelLoadError,
    ModelNotReadyError,
    OutcomeConflictApiError,
    PersistenceUnavailableApiError,
    PredictionNotFoundApiError,
    install_error_handlers,
)
from fraudshield.api.middleware import RequestContextMiddleware
from fraudshield.api.model_service import ProductionModelService
from fraudshield.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DatabaseHealthResponse,
    ErrorResponse,
    LivenessResponse,
    ModelInfoResponse,
    OutcomeBatchRequest,
    OutcomeBatchResponse,
    OutcomeResponse,
    PredictionAuditResponse,
    ReadinessResponse,
    RootResponse,
    SinglePredictionResponse,
    TransactionRequest,
)
from fraudshield.monitoring.api_metrics import ApiMetrics, HttpMetricsMiddleware
from fraudshield.persistence.config import load_database_config
from fraudshield.persistence.database import DatabaseHealthService, create_database_runtime
from fraudshield.persistence.schemas import (
    ModelIdentity,
    OutcomeValue,
    ScoredValue,
    TransactionValues,
)
from fraudshield.persistence.service import (
    IdempotencyConflictError,
    OutcomeConflictError,
    PersistenceUnavailableError,
    PredictionNotFoundError,
    PredictionPersistenceService,
)

LOGGER = logging.getLogger("fraudshield.api")


def _event(name: str, **values: Any) -> str:
    return json.dumps({"event": name, **values}, separators=(",", ":"), sort_keys=True)


def _transaction_values(transaction: TransactionRequest) -> TransactionValues:
    return TransactionValues(
        step=transaction.step,
        transaction_type=transaction.type.value,
        amount=transaction.amount,
        oldbalance_origin=transaction.oldbalanceOrg,
        oldbalance_destination=transaction.oldbalanceDest,
    )


def create_app(
    config: ApiConfig | None = None,
    model_service: ProductionModelService | None = None,
    persistence_service: PredictionPersistenceService | None = None,
    database_health: DatabaseHealthService | None = None,
    api_metrics: ApiMetrics | None = None,
    *,
    load_model_on_startup: bool | None = None,
) -> FastAPI:
    """Create the API with injectable model and persistence dependencies."""

    api_config = config or load_api_config()
    service = model_service or ProductionModelService(api_config)
    metrics = api_metrics or ApiMetrics()
    runtime = None
    maximum_outcome_batch_size = 1000
    if persistence_service is None or database_health is None:
        try:
            database_config = load_database_config(root=api_config.repository_root)
            runtime = create_database_runtime(database_config)
            persistence_service = persistence_service or PredictionPersistenceService(
                runtime.session_factory
            )
            database_health = database_health or DatabaseHealthService(runtime)
            maximum_outcome_batch_size = database_config.persistence.maximum_outcome_batch_size
        except (OSError, ValueError):
            runtime = None
    should_load = (
        api_config.model.load_on_startup if load_model_on_startup is None else load_model_on_startup
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
                        "api_startup_model_ready",
                        model_alias=info["alias"],
                        model_version=info["resolved_model_version"],
                    )
                )
                metrics.set_model_readiness(True, info)
            except ModelLoadError:
                metrics.set_model_readiness(False)
                LOGGER.error(_event("api_startup_not_ready", reason="model_load_failed"))
        else:
            LOGGER.info(_event("api_startup_model_loading_disabled"))
        yield
        if runtime is not None:
            runtime.engine.dispose()
        LOGGER.info(_event("api_shutdown"))

    application = FastAPI(
        title=api_config.application.title,
        description=(
            f"{api_config.application.description}. Local demonstration only; authentication "
            "and authorization are required before non-local deployment."
        ),
        version=api_config.application.version,
        docs_url=api_config.api.docs_url,
        redoc_url=api_config.api.redoc_url,
        openapi_url=api_config.api.openapi_url,
        lifespan=lifespan,
    )
    application.state.api_config = api_config
    application.state.model_service = service
    application.state.persistence_service = persistence_service
    application.state.database_health = database_health
    application.state.api_metrics = metrics
    application.add_middleware(RequestContextMiddleware, logger=LOGGER)
    application.add_middleware(
        HttpMetricsMiddleware,
        metrics=metrics,
        excluded_paths={"/metrics"},
    )
    install_error_handlers(application)

    def health_or_error() -> Any:
        if database_health is None:
            metrics.set_database_readiness(False)
            raise DatabaseNotReadyError()
        health = database_health.status()
        if not health.healthy:
            metrics.set_database_readiness(False)
            if health.error_code == "migration_not_current":
                raise MigrationNotCurrentError()
            raise DatabaseNotReadyError()
        metrics.set_database_readiness(True)
        return health

    def persistence_or_error() -> PredictionPersistenceService:
        if persistence_service is None:
            metrics.persistence_failures.inc()
            raise PersistenceUnavailableApiError()
        if database_health is not None:
            health = database_health.status()
            if health.error_code == "migration_not_current":
                raise MigrationNotCurrentError()
            if not health.healthy:
                metrics.persistence_failures.inc()
                raise PersistenceUnavailableApiError()
        return persistence_service

    @application.get("/", response_model=RootResponse)
    async def root() -> RootResponse:
        database_ready = database_health is not None and database_health.status().healthy
        return RootResponse(
            service=api_config.application.title,
            api_version=api_config.application.version,
            docs_url=api_config.api.docs_url,
            redoc_url=api_config.api.redoc_url,
            openapi_url=api_config.api.openapi_url,
            readiness_status="ready" if service.is_ready() and database_ready else "not_ready",
        )

    @application.get("/health/live", response_model=LivenessResponse)
    async def liveness() -> LivenessResponse:
        return LivenessResponse(
            status="alive",
            service=api_config.application.title,
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )

    @application.get(
        "/health/db",
        response_model=DatabaseHealthResponse,
        responses={503: {"model": ErrorResponse}},
    )
    async def database_readiness() -> DatabaseHealthResponse:
        health = health_or_error()
        return DatabaseHealthResponse(
            status="healthy",
            database_type="PostgreSQL",
            migration_status=health.migration_status,
        )

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ErrorResponse}},
    )
    async def readiness() -> ReadinessResponse:
        if not service.is_ready():
            raise ModelNotReadyError()
        health_or_error()
        info = service.model_info()
        return ReadinessResponse(
            status="ready",
            model_name=info["registered_model_name"],
            model_version=info["resolved_model_version"],
            model_alias=info["alias"],
            database_type="PostgreSQL",
            migration_status="current",
        )

    @application.get(
        "/model/info", response_model=ModelInfoResponse, responses={503: {"model": ErrorResponse}}
    )
    async def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(**service.model_info())

    @application.get("/metrics", include_in_schema=True)
    async def prometheus_metrics() -> Response:
        model_ready = service.is_ready()
        metrics.set_model_readiness(
            model_ready,
            service.model_info() if model_ready else None,
        )
        database_ready = database_health is not None and database_health.status().healthy
        metrics.set_database_readiness(database_ready)
        return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    def persisted_prediction(
        transactions: list[TransactionRequest], request: Request, endpoint: str
    ) -> Any:
        if not service.is_ready():
            raise ModelNotReadyError()
        persistence = persistence_or_error()
        info = service.model_info()
        model = ModelIdentity(
            name=info["registered_model_name"],
            version=info["resolved_model_version"],
            alias=info["alias"],
            threshold=api_config.model.expected_threshold,
        )
        values = [_transaction_values(item) for item in transactions]

        def score() -> list[ScoredValue]:
            request.state.model_inference_invoked = True
            LOGGER.info(_event("model_inference_invoked", request_id=request.state.request_id))
            predictions = (
                [service.predict_one(transactions[0])]
                if endpoint == "/predict"
                else service.predict_batch(transactions)
            )
            return [
                ScoredValue(
                    item_index=item["item_index"],
                    fraud_score=item["fraud_score"],
                    prediction=item["prediction"],
                    risk_level=item["risk_level"],
                )
                for item in predictions
            ]

        try:
            result = persistence.persist_predictions(
                request_id=uuid.UUID(request.state.request_id),
                endpoint=endpoint,
                transactions=values,
                model=model,
                scorer=score,
            )
        except IdempotencyConflictError as error:
            metrics.idempotency_conflicts.inc()
            raise IdempotencyConflictApiError() from error
        except PersistenceUnavailableError as error:
            metrics.persistence_failures.inc()
            raise PersistenceUnavailableApiError() from error
        request.state.idempotent_replay = result.replayed
        if result.replayed:
            metrics.idempotent_replays.inc()
        else:
            for transaction, prediction in zip(transactions, result.predictions, strict=True):
                metrics.observe_prediction(
                    prediction=prediction.prediction,
                    risk_level=prediction.risk_level,
                    transaction_type=transaction.type.value,
                    model_version=result.model.version,
                    fraud_score=prediction.fraud_score,
                )
        request.state.batch_size = len(transactions)
        request.state.prediction_count = len(result.predictions)
        request.state.high_risk_prediction_count = sum(
            item.risk_level == "high" for item in result.predictions
        )
        request.state.model_alias = result.model.alias
        request.state.model_version = result.model.version
        return result

    @application.post(
        "/predict",
        response_model=SinglePredictionResponse,
        responses={
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict(
        transaction: TransactionRequest, request: Request
    ) -> SinglePredictionResponse:
        result = persisted_prediction([transaction], request, "/predict")
        item = result.predictions[0]
        return SinglePredictionResponse(
            request_id=str(result.request_id),
            prediction_id=item.prediction_id,
            fraud_score=item.fraud_score,
            prediction=item.prediction,
            threshold=item.threshold,
            risk_level=item.risk_level,
            model_name=result.model.name,
            model_version=result.model.version,
            model_alias=result.model.alias,
            processing_time_ms=result.processing_time_ms,
        )

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict_batch(
        batch: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        if len(batch.transactions) > api_config.inference.maximum_batch_size:
            raise BatchSizeExceededError(api_config.inference.maximum_batch_size)
        result = persisted_prediction(batch.transactions, request, "/predict/batch")
        return BatchPredictionResponse(
            request_id=str(result.request_id),
            predictions=[vars(item) for item in result.predictions],
            prediction_count=len(result.predictions),
            model_name=result.model.name,
            model_version=result.model.version,
            model_alias=result.model.alias,
            processing_time_ms=result.processing_time_ms,
        )

    @application.get(
        "/predictions/{prediction_id}",
        response_model=PredictionAuditResponse,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def prediction_audit(prediction_id: uuid.UUID) -> PredictionAuditResponse:
        persistence = persistence_or_error()
        try:
            record = persistence.get_audit(prediction_id)
        except PredictionNotFoundError as error:
            raise PredictionNotFoundApiError() from error
        except PersistenceUnavailableError as error:
            metrics.persistence_failures.inc()
            raise PersistenceUnavailableApiError() from error
        outcome = OutcomeResponse(**vars(record.outcome)) if record.outcome else None
        values = vars(record) | {"outcome": outcome}
        return PredictionAuditResponse(**values)

    @application.post(
        "/outcomes",
        response_model=OutcomeBatchResponse,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def submit_outcomes(batch: OutcomeBatchRequest) -> OutcomeBatchResponse:
        if len(batch.outcomes) > maximum_outcome_batch_size:
            raise BatchSizeExceededError(maximum_outcome_batch_size)
        persistence = persistence_or_error()
        values = [
            OutcomeValue(
                prediction_id=item.prediction_id,
                actual_fraud=item.actual_fraud,
                observed_at=(item.observed_at or datetime.now(UTC)).astimezone(UTC),
                source=item.source,
            )
            for item in batch.outcomes
        ]
        try:
            stored = persistence.submit_outcomes(values)
        except PredictionNotFoundError as error:
            raise PredictionNotFoundApiError() from error
        except OutcomeConflictError as error:
            raise OutcomeConflictApiError() from error
        except PersistenceUnavailableError as error:
            metrics.persistence_failures.inc()
            raise PersistenceUnavailableApiError() from error
        return OutcomeBatchResponse(
            outcomes=[OutcomeResponse(**vars(item)) for item in stored],
            outcome_count=len(stored),
        )

    return application


app = create_app()
