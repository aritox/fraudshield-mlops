# Production SGD Model Card

## Model Overview

Production model: `production_sgd_logistic`, an incremental `SGDClassifier` logistic model.
PaySim is synthetic and does not represent real customers or real-world fraud probabilities.

## Intended Use

Rank transactions for fraud review using pre-transaction fields and the frozen
operational threshold.

## Out-of-Scope Use

Do not use this model as a calibrated probability, for identity decisions, demographic decisions,
or outside a monitored fraud-review workflow.

## Dataset And Splits

PaySim is synthetic. The chronological design keeps complete steps together:

- Training: steps 1-323, 4463587 rows, 3643 frauds.
- Validation: official development holdout, 943289 rows, 560 frauds.
- Final test: steps 378-743, evaluated only after the SGD pipeline was frozen.

## Features

step, hour_of_day, hour_sin, hour_cos, log_amount, log_oldbalance_origin, log_oldbalance_destination, log_amount_to_origin_balance, log_amount_to_destination_balance, origin_balance_zero_before, destination_balance_zero_before, amount_exceeds_origin_balance, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER

Forbidden features: `isFraud`, `isFlaggedFraud`, `nameOrig`, `nameDest`,
`newbalanceOrig`, `newbalanceDest`.

## Hyperparameters And Threshold

- Loss: `log_loss`; penalty: `l2`; alpha: `0.00001`.
- Epochs: `3`; positive sample weight: `5.0`; random seed: `42`.
- Operational threshold: `0.98310834`, selected by validation F2.

## Performance

Validation PR-AUC: `0.540633`; ROC-AUC: `0.995932`.

Final-test PR-AUC: `0.702244`; ROC-AUC: `0.993768`.

Final-test precision: `0.802373`; recall: `0.539651`;
F1: `0.645296`; F2: `0.577467`;
fraud amount recall: `0.731731`.

Top-k final-test metrics:

- 0.1%: reviewed 956, recall 0.234414, fraud amount recall 0.372144
- 0.5%: reviewed 4779, recall 0.683541, fraud amount recall 0.840623
- 1.0%: reviewed 9558, recall 0.798753, fraud amount recall 0.916396

## Model Choice And Limitations

SGD was selected instead of XGBoost for incremental training, low latency, small size,
interpretability, deployment simplicity, monitoring simplicity, and lower dependence on PaySim
simulator shortcuts. XGBoost remains a benchmark. Its near-perfect PaySim results rely heavily
on deterministic synthetic balance rules. SGD scores are ranking scores and are not guaranteed to
be calibrated real-world fraud probabilities.

## Fairness, Monitoring, And Retraining

PaySim does not provide the demographic information needed for a meaningful fairness assessment.
Fairness and subgroup performance must be assessed before real-world use. Monitor score drift,
alert volume, precision, recall, fraud amount capture, missing inputs, and latency. Retraining
requires a new time-aware development cycle, frozen decision record, validation threshold review,
and a separately governed holdout.

## Reproducibility

The production artifact was trained only on the official training split. The test set was evaluated
only after the model, features, hyperparameters, and threshold were frozen. Artifact SHA-256:
`2d3bd1ff56e0c2a159d2ba395366dbf6d06e934559a1049c0ae6026a9ff29542`.
