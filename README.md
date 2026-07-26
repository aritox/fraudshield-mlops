# FraudShield MLOps

FraudShield is a production-style fraud detection MLOps project built with free and open-source Python tooling. The current phase is **Phase 1D: time-aware tuning and stronger model comparison**.

## Dataset

This phase uses the public Kaggle PaySim dataset:

- Dataset name: PaySim
- Kaggle handle: `ealaxi/paysim1`
- Target column: `isFraud`

PaySim is a synthetic mobile-money transaction dataset generated from simulator behavior. It is useful for repeatable fraud detection workflow development, but it is not real customer banking data.

## Project Layout

```text
fraudshield-mlops/
├── artifacts/              # Generated manifests and validation reports, not committed
├── configs/                # Versioned configuration files
├── data/
│   ├── raw/                # Original datasets, not committed
│   ├── interim/            # Intermediate transformed data, not committed
│   └── processed/          # Final modeling datasets, not committed
├── notebooks/              # Local notebooks, not required for Phase 1A
├── src/fraudshield/
│   ├── data/               # Data download and validation code
│   ├── features/           # Reserved for later feature engineering
│   ├── models/             # Reserved for later model training
│   └── monitoring/         # Reserved for later monitoring
└── tests/                  # Automated tests
```

## Setup

Use the existing Python 3.12 virtual environment for this project.

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Kaggle credentials must be configured locally before downloading from Kaggle. Do not commit or print Kaggle tokens. Raw datasets, generated reports, credentials, environment files, and secrets are excluded from Git.

## Download

Download the PaySim CSV into `data/raw`:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.data.download
```

Force a fresh download:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.data.download --force
```

The downloader reuses an existing schema-matching CSV unless `--force` is supplied. It prints the final CSV path, filename, file size, and whether the file was downloaded or reused.

## Validation

Validate the complete CSV in chunks without loading the full dataset into memory:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.data.validate
```

Validation checks:

- Exact expected PaySim columns
- Binary values in `isFraud` and `isFlaggedFraud`
- Numeric values in amount and balance columns
- Missing values
- Negative transaction amounts
- Valid transaction type values

Generated reports:

- `artifacts/data/dataset_manifest.json`
- `artifacts/data/data_quality_report.json`

The manifest includes dataset identity, filename, relative path, file size, SHA-256 checksum, UTC validation timestamp, row count, and column count. The quality report includes validation status, fraud counts, fraud percentage, flagged fraud count, missing-value counts, transaction amount range, transaction counts by type, and fraud counts by transaction type.

## Phase 1B -- Temporal splitting and exploratory analysis

Transaction data must be split chronologically because random splits can train on patterns from transactions that occur after validation or test examples. That leaks future information into development and makes model results look stronger than a real-time deployment would allow. The test set is a final holdout and should not be used for EDA, feature decisions, model selection, or threshold tuning.

FraudShield uses a 70/15/15 chronological design based on the `step` column. Entire time steps stay together so no simulated time point appears in more than one split. The split is chronological and not stratified because preserving future simulation is more important than forcing identical fraud rates across train, validation, and test.

Training EDA reads only `data/processed/train.parquet`. Validation and test row and target counts may appear in the split manifest, but EDA charts and feature statistics do not inspect validation or test feature distributions.

The real-time baseline excludes leakage-prone or audit-only columns:

- `isFlaggedFraud` is an existing rule-based signal and is excluded so the ML system can be evaluated independently.
- `nameOrig` and `nameDest` are high-cardinality identifiers and must not be used directly as model inputs. Future identifier-derived features must use only transactions before prediction time.
- `newbalanceOrig` and `newbalanceDest` describe balances after the transaction and are excluded from a pre-transaction model.

Create or reuse chronological split files:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.data.split
```

Force regeneration of only the generated split Parquet files and split manifest:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.data.split --force
```

Generate training-only EDA artifacts:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.analysis.eda
```

Generated Phase 1B files and reports:

- `configs/split.yaml`
- `configs/feature_policy.yaml`
- `data/processed/train.parquet`
- `data/processed/validation.parquet`
- `data/processed/test.parquet`
- `artifacts/data/split_manifest.json`
- `artifacts/eda/train_eda_summary.json`
- `artifacts/eda/plots/*.png`

## Quality Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m ruff check .
```

## Phase 1C -- Leakage-safe baseline modeling

Phase 1C trains the first real-time, pre-transaction fraud baseline. The model uses only fields available before the transaction is completed: `step`, `type`, `amount`, `oldbalanceOrg`, and `oldbalanceDest`.

Post-transaction balances are excluded because `newbalanceOrig` and `newbalanceDest` describe account state after the transaction and would leak future information into a real-time prediction. Raw account identifiers are excluded because `nameOrig` and `nameDest` are high-cardinality IDs; direct use would encourage memorization instead of general fraud behavior. `isFlaggedFraud` remains audit-only because it is an existing rule-based signal, and this baseline evaluates the ML model independently from that rule.

Scaling is fitted with `StandardScaler.partial_fit` on training batches only, then reused unchanged for validation scoring. Class weights are calculated from training class counts only. SMOTE is not used because this baseline is incremental, validation must preserve the real chronological distribution, and synthetic oversampling would make the first benchmark harder to interpret.

PR-AUC is the primary model-selection metric because fraud is rare and accuracy is dominated by non-fraud transactions. F2 is used for operational threshold selection because missing fraud is more costly than reviewing extra alerts in this baseline. Model selection chooses the candidate by validation PR-AUC; threshold selection then chooses the operating cutoff for the chosen model using validation F2.

Validation results are development results, not final test performance. The final chronological test split remains sealed during Phase 1C: `data/processed/test.parquet` is not read, counted, scored, or used for thresholds.

Train or reuse the Phase 1C baseline:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.models.train_baseline
```

Replace only Phase 1C generated model artifacts, metrics, manifest, and plots:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.models.train_baseline --force
```

Generated Phase 1C files:

- `configs/modeling.yaml`
- `artifacts/models/baseline_champion.joblib` ignored by Git
- `artifacts/modeling/baseline_metrics.json`
- `artifacts/modeling/model_manifest.json`
- `artifacts/modeling/plots/*.png`

## Phase 1D -- Time-aware tuning and stronger models

Phase 1D tunes stronger candidates without using the official validation split for hyperparameter search. The search creates an internal chronological whole-step split inside `data/processed/train.parquet`: earlier training steps are used for fitting and later training steps are used for tuning. This keeps every complete simulated time step in one period and avoids random cross-validation that would mix future behavior into model selection.

The official validation split is used only after the best SGD and XGBoost configurations are frozen in `artifacts/tuning/frozen_candidate_configs.json`. The final test split remains sealed: `data/processed/test.parquet` is not opened, counted, scored, plotted, or used for threshold selection. Phase 1D manifests record `test_set_accessed: false`.

SGD logistic tuning tests moderate positive-class weights (`1` through `50`) because the fully balanced Phase 1C-style fraud weight of about `612` saturated probabilities and hurt model usefulness. XGBoost uses deterministic resource-aware sampling so local training does not need to materialize all non-fraud training rows at once: every fraud row is retained, non-fraud rows are sampled reproducibly, and sample weights make sampled non-fraud rows represent the original non-fraud population.

The three evaluation stages are distinct:

- Inner tuning: model and hyperparameter search inside the official training split only.
- Official validation: final development comparison after configurations are frozen.
- Final test evaluation: still sealed and not part of Phase 1D.

PR-AUC remains the primary selection metric because fraud is rare and ranking quality matters more than accuracy. F2 is used for operating-threshold selection because recall is more important than precision at this stage. A Phase 1D candidate must beat the Phase 1C validation PR-AUC by at least the configured tolerance before replacing the baseline; otherwise the simpler Phase 1C baseline remains champion. Probability calibration is not performed yet.

Run or reuse Phase 1D:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.models.tune_models
```

Replace only Phase 1D generated tuning, model, and plot artifacts:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.models.tune_models --force
```

Generated Phase 1D files:

- `configs/tuning.yaml`
- `artifacts/tuning/inner_split_manifest.json`
- `artifacts/tuning/tuning_results.json`
- `artifacts/tuning/tuning_manifest.json`
- `artifacts/tuning/frozen_candidate_configs.json`
- `artifacts/tuning/plots/*.png`
- `artifacts/modeling/phase1d_validation_metrics.json`
- `artifacts/modeling/phase1d_model_manifest.json`
- `artifacts/modeling/phase1d_plots/*.png`
- `artifacts/models/phase1d_champion.joblib` ignored by Git

## Phase 1E -- Production SGD and final holdout evaluation

Phase 1E promotes the frozen Phase 1D SGD candidate for production use. SGD was
selected instead of XGBoost using operational and governance criteria: it has
credible validation performance, incremental `partial_fit` training, low
inference latency, a small model, interpretable coefficients, simple deployment,
and straightforward monitoring. XGBoost remains a benchmark because its near-
perfect PaySim result depends substantially on deterministic synthetic balance
rules.

The production model is trained only on the official training split. The official
validation split is used to verify reproducibility and preserve the frozen F2
threshold. The final test split is evaluated only once after the model, features,
hyperparameters, and threshold are frozen. Test results cannot change the model,
threshold, features, or training configuration. Future production monitoring is a
separate activity that tracks drift, alert volume, precision, recall, amount
capture, and latency without treating the score as a guaranteed calibrated
probability.

The lifecycle boundaries are:

- Training: fit the scaler and SGD model on chronological training data only.
- Validation/model development: verify the frozen candidate and threshold on the
  official validation split.
- Final holdout evaluation: score the sealed test split once and record immutable
  results.
- Future production monitoring: observe live behavior and govern any later
  retraining as a new time-aware modeling cycle.

Run promotion:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.models.promote_sgd
```

Use `--force` only to replace the Phase 1E production bundle and governance
manifest when the frozen configuration or source checksums require it. The
production configuration is `configs/production.yaml`; the bundle is ignored by
Git at `artifacts/models/production_sgd.joblib`.

Run the one-time final evaluation only after promotion succeeds:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.models.evaluate_final_test
```

If a completed marker exists, the evaluator reuses the recorded results when all
checksums match. Re-evaluation requires the explicit
`--acknowledge-final-holdout-rerun` option. Generated Phase 1E artifacts include
the production decision and manifest under `artifacts/governance/`, final metrics
and plots under `artifacts/evaluation/`, and the model card under
`artifacts/model_card/`. PaySim is synthetic, and SGD scores are ranking scores,
not automatically calibrated real-world fraud probabilities.

## Phase 2A -- Local MLflow tracking and model registry

Phase 2A imports the existing FraudShield reports and packages the already-fitted
model bundles in MLflow. MLflow records experiments and runs containing parameters,
metrics, tags, and source artifacts. Its registry adds named models, immutable model
versions, and aliases that identify the currently governed version without using
deprecated model stages.

All tracking is local and uses free, open-source software. Experiment and registry
metadata are stored in SQLite at `artifacts/mlflow/mlflow.db`; packaged artifacts are
stored under `artifacts/mlflow/artifacts/`. Both runtime locations are ignored by Git.
No cloud account, Databricks workspace, external API, or paid service is required.

The registered production model is `fraudshield-production-sgd` with alias
`champion`. Its custom PyFunc interface accepts the five leakage-safe raw fields and
returns `fraud_score`, a thresholded prediction, the frozen threshold, and a low,
medium, or high risk level. The score is a ranking score, not a calibrated fraud
probability. The XGBoost model is registered separately as
`fraudshield-xgboost-benchmark` with alias `challenger`; it is not production-approved
because the Phase 1D audit found substantial dependence on synthetic PaySim balance
rules.

Development validation metrics and the one-time final-holdout metrics are imported
verbatim from existing JSON reports. Phase 2A does not open raw data or Parquet files,
rescore data, recompute metrics, train models, or change thresholds. Stable checksum
keys prevent duplicate equivalent runs and model versions when commands are rerun.

Import or reuse the existing runs and model versions:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.tracking.log_existing_runs
```

Verify experiments, aliases, duplicate prevention, stored metrics, and production
PyFunc inference:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.tracking.verify_registry
```

Start the local UI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_mlflow.ps1
```

The UI is available at `http://127.0.0.1:5000`. Tracked exports are
`artifacts/mlflow/registry_snapshot.json` and
`artifacts/mlflow/mlflow_manifest.json`; the SQLite database and model artifact store
remain local runtime state.

## Phase 2B -- FastAPI real-time inference service

Phase 2B exposes the registered production SGD champion as a local REST API.
FastAPI defines the HTTP routes and OpenAPI documentation, Pydantic strictly
validates JSON requests and responses, and Uvicorn runs the ASGI application.
The service resolves `models:/fraudshield-production-sgd@champion` from the local
MLflow registry and loads its PyFunc model once during application startup.

Liveness and readiness have separate meanings. Liveness reports whether the API
process is running. Readiness reports whether the champion passed registry,
signature, threshold, and warm-up validation. If loading fails, liveness remains
healthy while readiness and prediction routes return a sanitized HTTP 503.

`POST /predict` scores one transaction. `POST /predict/batch` validates up to 1,000
transactions and scores the complete batch in one PyFunc call while preserving
input order. Responses call the model output `fraud_score`: it is a ranking score,
not a calibrated real-world fraud probability. The XGBoost challenger is never used
by the production API because its PaySim performance depends heavily on synthetic
simulator rules.

Every response includes a request ID and processing-time header. Application logs
contain safe route, status, latency, count, alias, and version metadata, but never
transaction amounts, balances, request bodies, or artifact paths. The current
service binds only to `127.0.0.1`; authentication, PostgreSQL prediction logging,
monitoring, and hosted deployment belong to later phases.

Export the API contract without loading the model or starting a server:

```powershell
.\.venv\Scripts\python.exe -m fraudshield.api.export_contract
```

Start the local API:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1
```

Local addresses:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

Synthetic single request:

```json
{
  "step": 24,
  "type": "TRANSFER",
  "amount": 1500.0,
  "oldbalanceOrg": 1500.0,
  "oldbalanceDest": 0.0
}
```

Example response shape:

```json
{
  "request_id": "7cb4972e-319f-4c9c-9c9a-235e63497989",
  "fraud_score": 0.9982,
  "prediction": 1,
  "threshold": 0.98310834,
  "risk_level": "high",
  "model_name": "fraudshield-production-sgd",
  "model_version": "1",
  "model_alias": "champion",
  "processing_time_ms": 2.4
}
```

Synthetic batch request:

```json
{
  "transactions": [
    {
      "step": 1,
      "type": "PAYMENT",
      "amount": 25.0,
      "oldbalanceOrg": 100.0,
      "oldbalanceDest": 50.0
    },
    {
      "step": 24,
      "type": "TRANSFER",
      "amount": 1500.0,
      "oldbalanceOrg": 1500.0,
      "oldbalanceDest": 0.0
    }
  ]
}
```

The batch response contains an ordered `predictions` list with `item_index`,
`fraud_score`, `prediction`, `threshold`, and `risk_level`, plus request, model,
count, and processing metadata. Tracked Phase 2B contracts are
`artifacts/api/openapi.json` and `artifacts/api/api_manifest.json`.
