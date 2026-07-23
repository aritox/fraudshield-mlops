from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.features.baseline import (
    BaselineFeatureTransformer,
    feature_names,
    forbidden_raw_columns,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 24, 25],
            "type": ["CASH_IN", "PAYMENT", "TRANSFER"],
            "amount": [10.0, 50.0, 125.0],
            "oldbalanceOrg": [100.0, 0.0, 100.0],
            "oldbalanceDest": [0.0, 25.0, 500.0],
            "isFraud": [0, 0, 1],
            "isFlaggedFraud": [0, 0, 0],
            "nameOrig": ["C1", "C2", "C3"],
            "nameDest": ["M1", "M2", "C4"],
            "newbalanceOrig": [90.0, 0.0, 0.0],
            "newbalanceDest": [10.0, 75.0, 625.0],
        }
    )


def test_fixed_feature_order_and_forbidden_columns_are_excluded() -> None:
    names = feature_names()

    assert names == [
        "step",
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "log_amount",
        "log_oldbalance_origin",
        "log_oldbalance_destination",
        "log_amount_to_origin_balance",
        "log_amount_to_destination_balance",
        "origin_balance_zero_before",
        "destination_balance_zero_before",
        "amount_exceeds_origin_balance",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]
    assert set(names).isdisjoint(forbidden_raw_columns())


def test_transform_is_deterministic_finite_and_float32() -> None:
    transformer = BaselineFeatureTransformer()
    first = transformer.transform(make_frame())
    second = transformer.transform(make_frame())

    np.testing.assert_allclose(first, second)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert first.shape == (3, len(feature_names()))


def test_cyclical_time_features_and_one_hot_encoding() -> None:
    features = BaselineFeatureTransformer().transform(make_frame())
    names = feature_names()

    assert features[0, names.index("hour_of_day")] == 0
    assert features[1, names.index("hour_of_day")] == 23
    assert features[2, names.index("hour_of_day")] == 0
    assert features[0, names.index("hour_sin")] == pytest.approx(0.0)
    assert features[0, names.index("hour_cos")] == pytest.approx(1.0)
    assert features[0, names.index("type_CASH_IN")] == 1
    assert features[1, names.index("type_PAYMENT")] == 1
    assert features[2, names.index("type_TRANSFER")] == 1
    assert features[:, names.index("type_CASH_OUT")].sum() == 0


def test_missing_required_column_is_rejected() -> None:
    frame = make_frame().drop(columns=["amount"])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        BaselineFeatureTransformer().transform(frame)


def test_unknown_transaction_type_is_rejected() -> None:
    frame = make_frame()
    frame.loc[0, "type"] = "WIRE"

    with pytest.raises(ValueError, match="Unknown transaction types"):
        BaselineFeatureTransformer().transform(frame)


def test_negative_values_are_rejected() -> None:
    frame = make_frame()
    frame.loc[0, "oldbalanceDest"] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        BaselineFeatureTransformer().transform(frame)
