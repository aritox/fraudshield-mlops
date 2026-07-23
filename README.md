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
