"""Leakage-safe baseline features for pre-transaction fraud scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

RAW_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
)
TARGET_COLUMN: Final[str] = "isFraud"
FORBIDDEN_RAW_COLUMNS: Final[tuple[str, ...]] = (
    TARGET_COLUMN,
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
    "newbalanceOrig",
    "newbalanceDest",
)
TRANSACTION_TYPES: Final[tuple[str, ...]] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "step",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "log_amount",
    "log_oldbalance_origin",
    "log_oldbalance_destination",
    "log_amount_to_origin_balance",
    "log_amount_to_destination_balance",
    "origin_balance_zero_before",
    "destination_balance_zero_before",
    "amount_exceeds_origin_balance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
)


@dataclass(frozen=True)
class BaselineFeatureTransformer:
    """Deterministic stateless transformer for the first modeling baseline."""

    allow_unknown_types: bool = False

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a float32 model matrix using only allowed raw inputs."""

        missing_columns = [column for column in RAW_INPUT_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

        step = pd.to_numeric(frame["step"], errors="coerce").to_numpy(dtype=np.float64)
        amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype=np.float64)
        oldbalance_origin = pd.to_numeric(
            frame["oldbalanceOrg"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        oldbalance_destination = pd.to_numeric(
            frame["oldbalanceDest"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        numeric_inputs = {
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalance_origin,
            "oldbalanceDest": oldbalance_destination,
        }
        invalid_numeric = [
            column for column, values in numeric_inputs.items() if not np.isfinite(values).all()
        ]
        if invalid_numeric:
            raise ValueError(
                "Feature columns contain missing or non-finite values: "
                f"{invalid_numeric}"
            )
        if (
            (amount < 0).any()
            or (oldbalance_origin < 0).any()
            or (oldbalance_destination < 0).any()
        ):
            raise ValueError("Amount and pre-transaction balance features must be non-negative")

        transaction_type = frame["type"].astype(str).to_numpy()
        unknown_types = sorted(set(transaction_type).difference(TRANSACTION_TYPES))
        if unknown_types and not self.allow_unknown_types:
            raise ValueError(f"Unknown transaction types: {', '.join(unknown_types)}")

        hour_of_day = np.mod(step - 1, 24)
        feature_columns = [
            step,
            hour_of_day,
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24),
            np.log1p(amount),
            np.log1p(oldbalance_origin),
            np.log1p(oldbalance_destination),
            np.log1p(amount / (oldbalance_origin + 1)),
            np.log1p(amount / (oldbalance_destination + 1)),
            (oldbalance_origin == 0).astype(np.float64),
            (oldbalance_destination == 0).astype(np.float64),
            (amount > oldbalance_origin).astype(np.float64),
        ]
        feature_columns.extend(
            (transaction_type == transaction).astype(np.float64)
            for transaction in TRANSACTION_TYPES
        )

        features = np.column_stack(feature_columns).astype(np.float32, copy=False)
        if not np.isfinite(features).all():
            raise ValueError("Generated feature matrix contains NaN or infinite values")
        if features.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError(
                "Generated feature matrix does not match the documented feature order"
            )
        assert_no_forbidden_feature_names()
        return features


def feature_names() -> list[str]:
    """Return the fixed model feature order."""

    assert_no_forbidden_feature_names()
    return list(FEATURE_NAMES)


def expected_raw_input_columns() -> list[str]:
    """Return raw columns accepted by the baseline feature transformer."""

    return list(RAW_INPUT_COLUMNS)


def forbidden_raw_columns() -> list[str]:
    """Return raw columns that must never become model features."""

    return list(FORBIDDEN_RAW_COLUMNS)


def assert_no_forbidden_feature_names() -> None:
    """Reject a feature policy where forbidden raw columns become model features."""

    forbidden = set(FORBIDDEN_RAW_COLUMNS)
    violations = [name for name in FEATURE_NAMES if name in forbidden]
    if violations:
        raise AssertionError(f"Forbidden raw columns in feature names: {violations}")
