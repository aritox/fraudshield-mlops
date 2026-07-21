# FraudShield MLOps

FraudShield is a production-style, end-to-end fraud detection MLOps platform for building, validating, and operating machine learning models with free and open-source Python tooling.

The project is organized around the lifecycle of a fraud detection system: data ingestion, feature engineering, model training, evaluation, reproducible packaging, and future monitoring workflows. It uses a `src` package layout, isolated Python 3.12 environment, and development tooling suitable for collaborative ML engineering.

## Goals

- Build reliable fraud detection models from structured transaction data.
- Keep data, configuration, code, notebooks, and generated artifacts separated.
- Support reproducible local development without paid APIs, subscriptions, cloud services, or proprietary datasets.
- Provide a foundation for later additions such as experiment tracking, model serving, orchestration, and monitoring.

## Project Layout

```text
fraudshield-mlops/
├── artifacts/              # Generated model and evaluation outputs
├── configs/                # Configuration files for experiments and pipelines
├── data/
│   ├── raw/                # Original datasets, not committed to Git
│   ├── interim/            # Intermediate transformed data, not committed to Git
│   └── processed/          # Final modeling datasets, not committed to Git
├── notebooks/              # Exploratory analysis and reports
├── src/
│   └── fraudshield/
│       ├── data/           # Data loading and validation code
│       ├── features/       # Feature engineering logic
│       ├── models/         # Training, evaluation, and inference code
│       ├── monitoring/     # Future drift and quality monitoring code
│       └── __init__.py
└── tests/                  # Automated tests
```

## Local Development

This project targets Python 3.12 only. Create and activate the virtual environment before installing dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Register the notebook kernel:

```powershell
python -m ipykernel install --user --name fraudshield --display-name "Python (FraudShield)"
```

## Data Policy

Datasets are intentionally excluded from version control. Keep raw, interim, and processed data under `data/`, and commit only the `.gitkeep` placeholders required to preserve the directory structure.

No fraud dataset has been downloaded for this initial scaffold.

## Tooling

- Package management: `pip` with editable installs
- Core ML stack: pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn
- Data formats: PyArrow
- Validation and settings: Pydantic, pydantic-settings, python-dotenv, PyYAML
- Development: pytest, pytest-cov, Ruff, JupyterLab, IPython kernel

## Quality Checks

```powershell
python -m pytest
python -m ruff check src tests
```
