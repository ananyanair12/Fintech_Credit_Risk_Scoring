import numpy as np

from src.features.build_features import FEATURES, FeatureBuilder
from tests.helpers import synthetic_raw


def test_output_shape_and_no_nans():
    df = synthetic_raw()
    out = FeatureBuilder().fit(df).transform(df)
    assert list(out.columns) == FEATURES
    assert len(out) == len(df)
    assert not out.isna().any().any()


def test_sentinel_and_bad_age_handled():
    df = synthetic_raw()
    fb = FeatureBuilder().fit(df)
    out = fb.transform(df)
    assert out["past_due_30_59"].max() <= 10          # 98 code removed / capped
    assert out["age"].min() >= 18                     # age 0 imputed
    assert out["past_due_sentinel"].sum() > 0
    assert out["income_missing"].sum() == df["monthly_income"].isna().sum()


def test_caps_come_from_train_only():
    df = synthetic_raw()
    fb = FeatureBuilder().fit(df.iloc[:2000])
    extreme = df.iloc[2000:].copy()
    extreme["revolving_utilization"] = 1e6
    out = fb.transform(extreme)
    assert np.isclose(out["revolving_utilization"].max(), fb.caps_["revolving_utilization"])


def test_single_row_with_missing_income():
    df = synthetic_raw()
    fb = FeatureBuilder().fit(df)
    row = df.iloc[[5]].copy()
    row["monthly_income"] = None
    out = fb.transform(row)
    assert out["income_missing"].iloc[0] == 1.0
    assert not out.isna().any().any()

def test_debt_ratio_is_amount_when_income_missing():
    df = synthetic_raw()
    fb = FeatureBuilder().fit(df)
    row = df.iloc[[5]].copy()
    row["monthly_income"] = None
    row["debt_ratio"] = 4000.0
    out = fb.transform(row)
    assert out["log_debt_amount_no_income"].iloc[0] > 8      # amount kept as its own feature
    assert out["debt_ratio"].iloc[0] < 10                    # ratio column not polluted

def test_debt_ratio_is_amount_when_income_zero():
    df = synthetic_raw()
    fb = FeatureBuilder().fit(df)
    row = df.iloc[[5]].copy()
    row["monthly_income"] = 0.0
    row["debt_ratio"] = 4000.0
    out = fb.transform(row)
    assert out["log_debt_amount_no_income"].iloc[0] > 8
    assert out["debt_ratio"].iloc[0] < 10