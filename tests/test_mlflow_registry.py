"""Tests for local MLflow registry aliases and duplicate prevention."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from fraudshield.tracking.config import load_mlflow_config
from fraudshield.tracking.mlflow_setup import (
    VERSION_KEY_TAG,
    configure_local_mlflow,
    ensure_experiment,
    ensure_registered_model,
    model_versions_with_key,
    stable_key,
)
from test_mlflow_config import _write_config


class ConstantPyFunc(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        del context, params
        return pd.DataFrame({"fraud_score": [0.25] * len(model_input)})


def test_registration_aliases_and_duplicate_version_lookup(tmp_path: Path) -> None:
    config = load_mlflow_config(_write_config(tmp_path), root=tmp_path)
    client, _, artifact_uri = configure_local_mlflow(config)
    experiment_id = ensure_experiment(
        client,
        config.experiment_names.development,
        artifact_uri,
    )
    ensure_registered_model(client, config.registered_models.production)
    ensure_registered_model(client, config.registered_models.benchmark)

    version_keys = {}
    versions = {}
    for role, name, alias in (
        (
            "production",
            config.registered_models.production,
            config.registry_aliases.production,
        ),
        ("benchmark", config.registered_models.benchmark, config.registry_aliases.benchmark),
    ):
        with mlflow.start_run(experiment_id=experiment_id):
            model_info = mlflow.pyfunc.log_model(
                name=f"{role}_model",
                python_model=ConstantPyFunc(),
                registered_model_name=name,
                input_example=pd.DataFrame({"value": [1.0]}),
            )
        version = str(model_info.registered_model_version)
        key = stable_key(role, "source-model", "configuration", "wrapper")
        client.set_model_version_tag(name, version, VERSION_KEY_TAG, key)
        client.set_registered_model_alias(name, alias, version)
        version_keys[role] = key
        versions[role] = version

    assert len(
        model_versions_with_key(
            client,
            config.registered_models.production,
            version_keys["production"],
        )
    ) == 1
    assert len(
        model_versions_with_key(
            client,
            config.registered_models.benchmark,
            version_keys["benchmark"],
        )
    ) == 1
    assert (
        str(
            client.get_model_version_by_alias(
                config.registered_models.production,
                config.registry_aliases.production,
            ).version
        )
        == versions["production"]
    )
    assert (
        str(
            client.get_model_version_by_alias(
                config.registered_models.benchmark,
                config.registry_aliases.benchmark,
            ).version
        )
        == versions["benchmark"]
    )


def test_registry_snapshot_shape_uses_relative_configuration(tmp_path: Path) -> None:
    config = load_mlflow_config(_write_config(tmp_path), root=tmp_path)
    tracked = config.tracked_settings()

    assert tracked["storage"]["backend_database"] == "artifacts/mlflow/mlflow.db"
    assert tracked["storage"]["artifact_root"] == "artifacts/mlflow/artifacts"
    assert str(tmp_path) not in str(tracked)
