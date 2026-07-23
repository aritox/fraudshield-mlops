from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.models.sanity_audit import _feature_hashes, _metric_matches, _rule_definitions


def test_feature_hashes_are_deterministic_and_support_rounding() -> None:
    features = np.array([[1.00004, 2.0], [1.00005, 2.0]], dtype=np.float32)

    exact = _feature_hashes(features)
    rounded = _feature_hashes(features, decimals=3)

    assert exact == _feature_hashes(features)
    assert exact[0] != exact[1]
    assert rounded[0] == rounded[1]


def test_rule_definitions_use_only_pretransaction_fields() -> None:
    frame = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [100.0, 50.0],
            "oldbalanceOrg": [100.0, 0.0],
            "oldbalanceDest": [0.0, 10.0],
        }
    )

    rules = _rule_definitions(frame)

    assert rules["oldbalanceOrg_equals_amount"].tolist() == [True, False]
    assert rules["amount_ge_oldbalanceOrg"].tolist() == [True, True]
    assert rules["transfer_or_cashout_and_oldbalanceOrg_equals_amount"].tolist() == [
        True,
        False,
    ]
    assert rules["transfer_and_oldbalanceDest_zero"].tolist() == [True, False]


def test_metric_match_helper_detects_compact_metric_equality() -> None:
    metrics = {
        "average_precision": 0.5,
        "roc_auc": 0.75,
        "selected_threshold": 0.2,
        "selected_threshold_metrics": {
            "precision": 0.4,
            "recall": 0.8,
            "f1": 0.5333333333,
            "f_beta": 0.6666666667,
            "specificity": 0.9,
            "false_positive_rate": 0.1,
            "alert_rate": 0.2,
            "fraud_amount_recall": 0.7,
            "confusion_matrix": {
                "true_negative": 9,
                "false_positive": 1,
                "false_negative": 2,
                "true_positive": 8,
            },
        },
        "top_k": [{"top_k_percentage": 1.0, "recall": 1.0}],
    }

    assert _metric_matches(metrics, metrics)["all_match"] is True
