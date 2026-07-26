"""Read-only service for the registered production SGD champion."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from fraudshield.api.config import ApiConfig
from fraudshield.api.errors import (
    BatchSizeExceededError,
    InferenceError,
    ModelLoadError,
    ModelNotReadyError,
)
from fraudshield.api.schemas import TransactionRequest
from fraudshield.features.baseline import expected_raw_input_columns
from fraudshield.tracking.config import MlflowConfig, load_mlflow_config
from fraudshield.tracking.mlflow_setup import (
    configure_local_mlflow,
    install_prohibited_data_guard,
)
from fraudshield.tracking.model_wrapper import OUTPUT_COLUMNS, synthetic_input_example

LOGGER = logging.getLogger("fraudshield.api")
VALID_RISK_LEVELS = {"low", "medium", "high"}


def _false_tag(value: str | None) -> bool:
    return str(value).lower() in {"false", "0"}


class ProductionModelService:
    """Load the production alias once and provide validated inference methods."""

    def __init__(
        self,
        config: ApiConfig,
        *,
        mlflow_config: MlflowConfig | None = None,
        client: MlflowClient | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.mlflow_config = mlflow_config
        self._client = client
        self._model_loader = model_loader or mlflow.pyfunc.load_model
        self._model: Any | None = None
        self._model_version: str | None = None
        self._model_tags: dict[str, str] = {}
        self._loaded_timestamp: str | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Resolve, validate, load, and warm the production champion once."""

        with self._load_lock:
            if self._model is not None:
                return
            try:
                root = self.config.repository_root
                install_prohibited_data_guard(root)
                client = self._client
                if self.config.model.uri == "models:/fraudshield-production-sgd@champion":
                    tracking_config = self.mlflow_config or load_mlflow_config(root=root)
                    self._validate_local_registry_paths(root, tracking_config)
                    if client is None:
                        client, _, _ = configure_local_mlflow(tracking_config)
                    version = client.get_model_version_by_alias(
                        self.config.model.registered_name,
                        self.config.model.alias,
                    )
                else:
                    from fraudshield.container.verify_package import verify_package_integrity

                    package = Path(self.config.model.uri)
                    manifest_path = self.config.model.package_manifest
                    if manifest_path is None:
                        raise ValueError("Packaged model manifest is not configured")
                    manifest = verify_package_integrity(package, manifest_path)
                    version = SimpleNamespace(
                        name=manifest["registered_model_name"],
                        version=manifest["resolved_version"],
                        aliases=[manifest["alias"]],
                        tags={
                            "role": "production",
                            "model_family": manifest["model_family"],
                            "threshold_source": "phase1d_validation_f2",
                            "operational_threshold": str(manifest["frozen_threshold"]),
                            "test_used_for_selection": "false",
                            "source_model_sha256": manifest["source_model_checksum"],
                        },
                    )
                self._validate_model_version(version)
                model = self._model_loader(self.config.model.uri)
                self._validate_signature(model)
                warmup = synthetic_input_example()
                output = model.predict(warmup)
                self._validate_output(output, warmup.index)
                self._client = client
                self._model = model
                self._model_version = str(version.version)
                self._model_tags = dict(version.tags)
                self._loaded_timestamp = (
                    datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
                )
                self._load_error = None
            except Exception as error:
                self._load_error = "Production model is unavailable"
                LOGGER.error(
                    "model_startup_failed exception_type=%s",
                    type(error).__name__,
                )
                raise ModelLoadError("Production model startup validation failed") from error

    @staticmethod
    def _validate_local_registry_paths(root: Path, config: MlflowConfig) -> None:
        database = root / config.storage.backend_database
        artifact_root = root / config.storage.artifact_root
        if not database.is_file() or not artifact_root.is_dir():
            raise FileNotFoundError("Local MLflow registry state is unavailable")

    def _validate_model_version(self, version: Any) -> None:
        if version.name != self.config.model.registered_name:
            raise ValueError("Champion resolved to an unexpected registered model")
        aliases = set(version.aliases or [])
        if self.config.model.alias not in aliases:
            raise ValueError("Resolved model version does not own the champion alias")
        tags = version.tags or {}
        if tags.get("role") != "production":
            raise ValueError("Champion is not tagged for production")
        if tags.get("model_family") != self.config.model.expected_family:
            raise ValueError("Champion model family is not SGDClassifier")
        if tags.get("threshold_source") != "phase1d_validation_f2":
            raise ValueError("Champion threshold source is not the frozen validation F2 policy")
        if float(tags.get("operational_threshold", "nan")) != self.config.model.expected_threshold:
            raise ValueError("Champion operational threshold differs from the frozen threshold")
        if not _false_tag(tags.get("test_used_for_selection")):
            raise ValueError("Champion does not preserve the test-selection governance policy")

    @staticmethod
    def _validate_signature(model: Any) -> None:
        input_schema = model.metadata.get_input_schema()
        output_schema = model.metadata.get_output_schema()
        if input_schema is None or input_schema.input_names() != expected_raw_input_columns():
            raise ValueError("Champion input signature does not match the API contract")
        if output_schema is None or output_schema.input_names() != list(OUTPUT_COLUMNS):
            raise ValueError("Champion output signature does not match the API contract")

    def _validate_output(self, output: Any, expected_index: pd.Index) -> pd.DataFrame:
        if not isinstance(output, pd.DataFrame):
            raise ValueError("Champion output must be a pandas DataFrame")
        if list(output.columns) != list(OUTPUT_COLUMNS):
            raise ValueError("Champion output columns differ from the frozen contract")
        if not output.index.equals(expected_index):
            raise ValueError("Champion output row order differs from input order")
        scores = output["fraud_score"].to_numpy(dtype=np.float64)
        thresholds = output["threshold"].to_numpy(dtype=np.float64)
        predictions = output["prediction"].to_numpy()
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Champion returned invalid fraud scores")
        if (
            not np.isfinite(thresholds).all()
            or not np.equal(thresholds, self.config.model.expected_threshold).all()
        ):
            raise ValueError("Champion returned a non-frozen threshold")
        if not set(predictions.tolist()).issubset({0, 1}):
            raise ValueError("Champion returned invalid binary predictions")
        if not set(output["risk_level"].astype(str)).issubset(VALID_RISK_LEVELS):
            raise ValueError("Champion returned invalid risk levels")
        return output

    def is_ready(self) -> bool:
        return self._model is not None

    def unavailable_reason(self) -> str:
        return self._load_error or "Production model is unavailable"

    def model_info(self) -> dict[str, Any]:
        if not self.is_ready():
            raise ModelNotReadyError()
        return {
            "registered_model_name": self.config.model.registered_name,
            "resolved_model_version": self._model_version,
            "alias": self.config.model.alias,
            "model_family": self.config.model.expected_family,
            "frozen_threshold": self.config.model.expected_threshold,
            "raw_input_fields": expected_raw_input_columns(),
            "output_fields": list(OUTPUT_COLUMNS),
            "risk_levels": {
                "low": f"score < {self.config.inference.medium_risk_threshold}",
                "medium_minimum": self.config.inference.medium_risk_threshold,
                "high_minimum": self.config.inference.high_risk_threshold,
            },
            "loaded_timestamp_utc": self._loaded_timestamp,
            "source_model_checksum": self._model_tags.get("source_model_sha256"),
            "synthetic_dataset_warning": (
                "PaySim is synthetic; fraud_score is not a calibrated real-world probability."
            ),
        }

    def predict_one(self, transaction: TransactionRequest) -> dict[str, Any]:
        return self.predict_batch([transaction])[0]

    def predict_batch(
        self,
        transactions: Sequence[TransactionRequest],
    ) -> list[dict[str, Any]]:
        if not self.is_ready() or self._model is None:
            raise ModelNotReadyError()
        if len(transactions) > self.config.inference.maximum_batch_size:
            raise BatchSizeExceededError(self.config.inference.maximum_batch_size)
        if not transactions:
            raise InferenceError()
        try:
            rows = [transaction.model_dump(mode="json") for transaction in transactions]
            frame = pd.DataFrame(rows, columns=expected_raw_input_columns())
            output = self._model.predict(frame)
            validated = self._validate_output(output, frame.index)
            return [
                {
                    "item_index": int(index),
                    "fraud_score": float(row.fraud_score),
                    "prediction": int(row.prediction),
                    "threshold": float(row.threshold),
                    "risk_level": str(row.risk_level),
                }
                for index, row in enumerate(validated.itertuples(index=False))
            ]
        except (ModelNotReadyError, BatchSizeExceededError):
            raise
        except Exception as error:
            LOGGER.error("inference_failed exception_type=%s", type(error).__name__)
            raise InferenceError() from error
