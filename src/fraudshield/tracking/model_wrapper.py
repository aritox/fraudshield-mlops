"""MLflow PyFunc wrappers for frozen FraudShield model bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from fraudshield.features.baseline import (
    TRANSACTION_TYPES,
    BaselineFeatureTransformer,
    expected_raw_input_columns,
    feature_names,
    forbidden_raw_columns,
)

WRAPPER_VERSION = "phase2a-pyfunc-v1"
OUTPUT_COLUMNS = ("fraud_score", "prediction", "threshold", "risk_level")


def synthetic_input_example() -> pd.DataFrame:
    """Return a small non-sensitive inference example."""

    return pd.DataFrame(
        {
            "step": [1, 24, 48],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [25.0, 1500.0, 450.0],
            "oldbalanceOrg": [100.0, 1500.0, 900.0],
            "oldbalanceDest": [50.0, 0.0, 200.0],
        }
    )


def production_signature() -> ModelSignature:
    """Return the fixed raw-input and prediction-output schema."""

    inputs = Schema(
        [
            ColSpec("long", "step"),
            ColSpec("string", "type"),
            ColSpec("double", "amount"),
            ColSpec("double", "oldbalanceOrg"),
            ColSpec("double", "oldbalanceDest"),
        ]
    )
    outputs = Schema(
        [
            ColSpec("double", "fraud_score"),
            ColSpec("long", "prediction"),
            ColSpec("double", "threshold"),
            ColSpec("string", "risk_level"),
        ]
    )
    return ModelSignature(inputs=inputs, outputs=outputs)


def benchmark_signature() -> ModelSignature:
    """Return the benchmark wrapper schema."""

    return ModelSignature(
        inputs=production_signature().inputs,
        outputs=Schema([ColSpec("double", "fraud_score")]),
    )


def _validated_input(model_input: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(model_input, pd.DataFrame):
        raise TypeError("FraudShield PyFunc input must be a pandas DataFrame")
    columns = list(model_input.columns)
    forbidden = sorted(set(columns).intersection(forbidden_raw_columns()))
    if forbidden:
        raise ValueError(f"Forbidden model input columns supplied: {', '.join(forbidden)}")
    required = expected_raw_input_columns()
    missing = [name for name in required if name not in columns]
    unexpected = [name for name in columns if name not in required]
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected input columns: {', '.join(unexpected)}")

    frame = model_input.loc[:, required].copy(deep=True)
    numeric_columns = ["step", "amount", "oldbalanceOrg", "oldbalanceDest"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric inputs must be finite")
    if (frame[["amount", "oldbalanceOrg", "oldbalanceDest"]].to_numpy() < 0).any():
        raise ValueError("Amount and pre-transaction balances must be non-negative")
    step = frame["step"].to_numpy(dtype=np.float64)
    if (step < 1).any() or not np.equal(step, np.floor(step)).all():
        raise ValueError("step must contain positive integers")
    frame["step"] = step.astype(np.int64)

    types = frame["type"]
    if types.isna().any() or not types.map(lambda value: isinstance(value, str)).all():
        raise ValueError("type values must be strings")
    unknown = sorted(set(types).difference(TRANSACTION_TYPES))
    if unknown:
        raise ValueError(f"Unknown transaction types: {', '.join(unknown)}")
    return frame


def _load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("model bundle must be a dictionary")
    bundle_features = bundle.get("feature_names", bundle.get("ordered_feature_names", []))
    if list(bundle_features) != feature_names():
        raise ValueError("model bundle feature order does not match the frozen policy")
    return bundle


class ProductionSGDPyFuncModel(mlflow.pyfunc.PythonModel):
    """Inference-only wrapper for the frozen production SGD bundle."""

    def __init__(self, medium_threshold: float, operational_threshold: float) -> None:
        self.medium_threshold = float(medium_threshold)
        self.operational_threshold = float(operational_threshold)
        self._bundle: dict[str, Any] | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._bundle = _load_bundle(context.artifacts["model_bundle"])
        bundle_threshold = float(
            self._bundle.get("operational_threshold", self._bundle.get("selected_threshold"))
        )
        if bundle_threshold != self.operational_threshold:
            raise ValueError("model bundle threshold differs from the frozen wrapper threshold")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        if self._bundle is None:
            raise RuntimeError("production model bundle is not loaded")
        frame = _validated_input(model_input)
        features = BaselineFeatureTransformer().transform(frame)
        scaled = self._bundle["scaler"].transform(features)
        scores = np.asarray(self._bundle["model"].predict_proba(scaled)[:, 1], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("model produced non-finite fraud scores")
        predictions = (scores >= self.operational_threshold).astype(np.int64)
        levels = np.where(
            scores >= self.operational_threshold,
            "high",
            np.where(scores >= self.medium_threshold, "medium", "low"),
        )
        return pd.DataFrame(
            {
                "fraud_score": scores,
                "prediction": predictions,
                "threshold": np.full(len(scores), self.operational_threshold),
                "risk_level": levels,
            },
            index=model_input.index,
        )


class BenchmarkXGBoostPyFuncModel(mlflow.pyfunc.PythonModel):
    """Inference-only wrapper for the frozen XGBoost benchmark bundle."""

    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._bundle = _load_bundle(context.artifacts["model_bundle"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        if self._bundle is None:
            raise RuntimeError("benchmark model bundle is not loaded")
        frame = _validated_input(model_input)
        matrix = pd.DataFrame(
            BaselineFeatureTransformer().transform(frame),
            columns=feature_names(),
            index=frame.index,
        )
        scores = np.asarray(self._bundle["model"].predict_proba(matrix)[:, 1], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("benchmark produced non-finite fraud scores")
        return pd.DataFrame({"fraud_score": scores}, index=model_input.index)
