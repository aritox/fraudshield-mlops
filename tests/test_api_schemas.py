"""Tests for strict API request validation."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from fraudshield.api.schemas import BatchPredictionRequest, TransactionRequest


def _transaction(**updates):
    values = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 100.0,
        "oldbalanceOrg": 100.0,
        "oldbalanceDest": 0.0,
    }
    values.update(updates)
    return values


def test_valid_transaction_and_batch() -> None:
    transaction = TransactionRequest.model_validate(_transaction())
    batch = BatchPredictionRequest.model_validate({"transactions": [_transaction()]})

    assert transaction.step == 10
    assert transaction.type.value == "TRANSFER"
    assert len(batch.transactions) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in _transaction().items() if key != "amount"},
        _transaction(isFraud=0),
        _transaction(nameOrig="C1"),
        _transaction(amount=-1),
        _transaction(amount=math.nan),
        _transaction(oldbalanceOrg=math.inf),
        _transaction(type="UNKNOWN"),
        _transaction(step=True),
        _transaction(amount=False),
        _transaction(amount="100.0"),
    ],
)
def test_invalid_transaction_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TransactionRequest.model_validate(payload)


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({"transactions": []})
