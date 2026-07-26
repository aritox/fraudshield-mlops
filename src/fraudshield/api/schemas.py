"""Strict request and response schemas for FraudShield inference."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

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
    fraud_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prediction: Literal[0, 1]
    threshold: float
    risk_level: RiskLevel
    model_name: str
    model_version: str
    model_alias: Literal["champion"]
    processing_time_ms: Annotated[float, Field(ge=0)]


class BatchPredictionItem(StrictSchema):
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
