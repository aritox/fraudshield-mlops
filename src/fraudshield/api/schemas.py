"""Strict request and response schemas for FraudShield inference."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionType(StrEnum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransactionRequest(StrictSchema):
    step: Annotated[int, Field(strict=True, ge=1)]
    type: TransactionType
    amount: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    oldbalanceOrg: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    oldbalanceDest: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @field_validator("amount", "oldbalanceOrg", "oldbalanceDest", mode="before")
    @classmethod
    def reject_non_numeric_values(cls, value: Any) -> Any:
        if isinstance(value, (bool, str)):
            raise ValueError("must be a JSON number, not a boolean or string")
        return value


class BatchPredictionRequest(StrictSchema):
    transactions: Annotated[list[TransactionRequest], Field(min_length=1)]


class SinglePredictionResponse(StrictSchema):
    request_id: str
    prediction_id: UUID
    fraud_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prediction: Literal[0, 1]
    threshold: float
    risk_level: RiskLevel
    model_name: str
    model_version: str
    model_alias: Literal["champion"]
    processing_time_ms: Annotated[float, Field(ge=0)]


class BatchPredictionItem(StrictSchema):
    prediction_id: UUID
    item_index: Annotated[int, Field(ge=0)]
    fraud_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prediction: Literal[0, 1]
    threshold: float
    risk_level: RiskLevel


class BatchPredictionResponse(StrictSchema):
    request_id: str
    predictions: list[BatchPredictionItem]
    prediction_count: Annotated[int, Field(ge=0)]
    model_name: str
    model_version: str
    model_alias: Literal["champion"]
    processing_time_ms: Annotated[float, Field(ge=0)]


class RootResponse(StrictSchema):
    service: str
    api_version: str
    docs_url: str
    redoc_url: str
    openapi_url: str
    readiness_status: Literal["ready", "not_ready"]


class LivenessResponse(StrictSchema):
    status: Literal["alive"]
    service: str
    timestamp_utc: str


class ReadinessResponse(StrictSchema):
    status: Literal["ready"]
    model_name: str
    model_version: str
    model_alias: Literal["champion"]
    database_type: Literal["PostgreSQL"]
    migration_status: Literal["current"]


class DatabaseHealthResponse(StrictSchema):
    status: Literal["healthy"]
    database_type: Literal["PostgreSQL"]
    migration_status: Literal["current"]


class OutcomeSubmission(StrictSchema):
    prediction_id: UUID
    actual_fraud: Literal[0, 1]
    observed_at: datetime | None = None
    source: Annotated[str, Field(min_length=1, max_length=128)]

    @field_validator("actual_fraud", mode="before")
    @classmethod
    def actual_fraud_is_strict_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("actual_fraud must be the integer 0 or 1")
        return value

    @field_validator("observed_at")
    @classmethod
    def observed_at_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at must include a timezone")
        return value

    @field_validator("source")
    @classmethod
    def source_is_nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source must not be blank")
        return normalized


class OutcomeBatchRequest(StrictSchema):
    outcomes: Annotated[list[OutcomeSubmission], Field(min_length=1)]


class OutcomeResponse(StrictSchema):
    prediction_id: UUID
    actual_fraud: Literal[0, 1]
    observed_at: datetime
    source: str
    created_at: datetime
    updated_at: datetime
    replayed: bool


class OutcomeBatchResponse(StrictSchema):
    outcomes: list[OutcomeResponse]
    outcome_count: Annotated[int, Field(ge=1)]


class PredictionAuditResponse(StrictSchema):
    prediction_id: UUID
    request_id: UUID
    item_index: Annotated[int, Field(ge=0)]
    endpoint: str
    model_name: str
    model_version: str
    model_alias: str
    threshold: float
    step: Annotated[int, Field(ge=1)]
    transaction_type: TransactionType
    amount: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    oldbalance_origin: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    oldbalance_destination: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    fraud_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prediction: Literal[0, 1]
    risk_level: RiskLevel
    request_created_at: datetime
    request_completed_at: datetime | None
    prediction_created_at: datetime
    outcome: OutcomeResponse | None


class ModelInfoResponse(StrictSchema):
    registered_model_name: str
    resolved_model_version: str
    alias: Literal["champion"]
    model_family: str
    frozen_threshold: float
    raw_input_fields: list[str]
    output_fields: list[str]
    risk_levels: dict[str, float | str]
    loaded_timestamp_utc: str
    source_model_checksum: str | None
    synthetic_dataset_warning: str


class ErrorDetail(StrictSchema):
    location: list[str | int]
    message: str
    error_type: str


class ErrorResponse(StrictSchema):
    error: str
    message: str
    request_id: str
    details: list[ErrorDetail] | None = None
