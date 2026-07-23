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
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Kaggle credentials must be configured locally before downloading from Kaggle. Do not commit or print Kaggle tokens. Raw datasets, generated reports, credentials, environment files, and secrets are excluded from Git.

## Download

Download the PaySim CSV into `data/raw`:

```powershell
python -m fraudshield.data.download
```

Force a fresh download:

```powershell
python -m fraudshield.data.download --force
```

The downloader reuses an existing schema-matching CSV unless `--force` is supplied. It prints the final CSV path, filename, file size, and whether the file was downloaded or reused.

## Validation

Validate the complete CSV in chunks without loading the full dataset into memory:

```powershell
python -m fraudshield.data.validate
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

## Quality Checks

```powershell
python -m pytest -v
python -m ruff check .
```

Phase 1A intentionally does not include exploratory analysis, feature engineering, model training, MLflow, FastAPI, Docker, Airflow, Spark, or monitoring.
