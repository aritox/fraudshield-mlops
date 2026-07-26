"""Ordinary Python values exchanged by persistence boundaries."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str
    alias: str
    threshold: float


@dataclass(frozen=True)
class TransactionValues:
    step: int
    transaction_type: str
    amount: float
    oldbalance_origin: float
    oldbalance_destination: float


@dataclass(frozen=True)
class ScoredValue:
    item_index: int
    fraud_score: float
    prediction: int
    risk_level: str


@dataclass(frozen=True)
class PersistedPrediction:
    prediction_id: uuid.UUID
    item_index: int
    fraud_score: float
    prediction: int
    threshold: float
    risk_level: str


@dataclass(frozen=True)
class PredictionResult:
    request_id: uuid.UUID
    predictions: list[PersistedPrediction]
    model: ModelIdentity
    processing_time_ms: float
    replayed: bool


@dataclass(frozen=True)
class OutcomeValue:
    prediction_id: uuid.UUID
    actual_fraud: int
    observed_at: datetime
    source: str


@dataclass(frozen=True)
class StoredOutcome:
    prediction_id: uuid.UUID
    actual_fraud: int
    observed_at: datetime
    source: str
    created_at: datetime
    updated_at: datetime
    replayed: bool


@dataclass(frozen=True)
class AuditRecord:
    prediction_id: uuid.UUID
    request_id: uuid.UUID
    item_index: int
    endpoint: str
    model_name: str
    model_version: str
    model_alias: str
    threshold: float
    step: int
    transaction_type: str
    amount: float
    oldbalance_origin: float
    oldbalance_destination: float
    fraud_score: float
    prediction: int
    risk_level: Literal["low", "medium", "high"]
    request_created_at: datetime
    request_completed_at: datetime | None
    prediction_created_at: datetime
    outcome: StoredOutcome | None
