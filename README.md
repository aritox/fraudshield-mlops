# FraudShield MLOps

FraudShield is a production-style fraud detection MLOps project built with free and open-source Python tooling. The current phase is **Phase 1A: reproducible PaySim dataset ingestion and schema validation**.

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

Phase 1A intentionally does not include exploratory analysis, feature engineering, model training, MLflow, FastAPI, Docker, Airflow, Spark, or monitoring.
