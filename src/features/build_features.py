"""Feature engineering as a scikit-learn transformer.

Everything that is *learned* (medians, caps) is fit on the training split only, and the
fitted object is saved inside the model pipeline, so training and serving apply identical
transformations (no train/serve skew, no leakage).

Design decisions (also documented in the README):
- MonthlyIncome (~20% missing) and NumberOfDependents (~2.6% missing): median imputation
  plus an explicit "was missing" flag, because missingness itself can carry risk signal.
- Past-due columns contain the placeholder codes 96 and 98 -> not real counts. They are set
  to missing, imputed, and flagged with `past_due_sentinel`.
- Implausible ages (<18 or >100) are treated as missing and imputed.
- DebtRatio quirk: when MonthlyIncome is missing or zero, the dataset stores the absolute monthly debt
  AMOUNT in DebtRatio (a ratio needs income). About 92% of rows with DebtRatio > 5 have missing
  income. We split the column: `debt_ratio` keeps only true ratios (amounts are replaced by the
  train median), and `log_debt_amount_no_income` carries the amount for the no-income rows.
- Heavy-tailed columns are capped at the train 99th percentile to limit outlier influence.
- Derived features encode ratios/aggregates that credit analysts already use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "past_due_90",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
]
PAST_DUE_COLS = ["past_due_30_59", "past_due_60_89", "past_due_90"]
SENTINEL_CODES = (96, 98)
CAP_COLS = [
    "revolving_utilization",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "real_estate_loans",
    "dependents",
]
MEDIAN_COLS = ["age", "monthly_income", "dependents", "debt_ratio"] + PAST_DUE_COLS
PAST_DUE_CAP = 10
AGE_BINS = [0, 25, 35, 45, 55, 65, 200]

ENGINEERED = [
    "income_missing",
    "dependents_missing",
    "past_due_sentinel",
    "total_past_due",
    "any_past_due",
    "severe_delinquency",
    "income_per_dependent",
    "log_income",
    "utilization_x_delinquency",
    "age_band",
    "log_debt_amount_no_income",
]
FEATURES = RAW_FEATURES + ENGINEERED


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Clean raw applicant columns and derive model features."""

    def __init__(self, cap_quantile: float = 0.99):
        self.cap_quantile = cap_quantile

    # -- helpers ---------------------------------------------------------------------
    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        """Coerce to numeric and turn invalid values into NaN (no imputation yet)."""
        X = X[RAW_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)

        # DebtRatio holds a dollar amount (not a ratio) when income is missing or zero -> split it
        no_income = X["monthly_income"].isna() | (X["monthly_income"] <= 0)
        amount = X["debt_ratio"].where(no_income, 0.0).clip(lower=0.0)
        X["log_debt_amount_no_income"] = np.log1p(amount).fillna(0.0)
        X["debt_ratio"] = X["debt_ratio"].where(~no_income, np.nan)

        sentinel = np.zeros(len(X), dtype=bool)
        for col in PAST_DUE_COLS:
            mask = X[col].isin(SENTINEL_CODES)
            sentinel |= mask.to_numpy()
            X.loc[mask, col] = np.nan
        X["past_due_sentinel"] = sentinel.astype(float)
        bad_age = (X["age"] < 18) | (X["age"] > 100)
        X.loc[bad_age, "age"] = np.nan
        return X

    # -- sklearn API -----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        Xc = self._clean(X)
        self.medians_ = {
            c: float(np.nan_to_num(Xc[c].median(), nan=0.0)) for c in MEDIAN_COLS
        }
        self.caps_ = {
            c: float(Xc[c].quantile(self.cap_quantile)) for c in CAP_COLS
        }
        self.feature_names_ = list(FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "caps_"):
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        Xc = self._clean(X)

        # missing-value flags (before imputation)
        Xc["income_missing"] = Xc["monthly_income"].isna().astype(float)
        Xc["dependents_missing"] = Xc["dependents"].isna().astype(float)

        # imputation and capping with statistics learned on the training split
        for col, med in self.medians_.items():
            Xc[col] = Xc[col].fillna(med)
        for col, cap in self.caps_.items():
            Xc[col] = Xc[col].clip(upper=cap)
        for col in PAST_DUE_COLS:
            Xc[col] = Xc[col].clip(upper=PAST_DUE_CAP)

        # derived features
        Xc["total_past_due"] = Xc[PAST_DUE_COLS].sum(axis=1)
        Xc["any_past_due"] = (Xc["total_past_due"] > 0).astype(float)
        Xc["severe_delinquency"] = (Xc["past_due_90"] > 0).astype(float)
        Xc["income_per_dependent"] = Xc["monthly_income"] / (Xc["dependents"] + 1.0)
        Xc["log_income"] = np.log1p(Xc["monthly_income"])
        Xc["utilization_x_delinquency"] = Xc["revolving_utilization"] * (
            1.0 + Xc["total_past_due"]
        )
        Xc["age_band"] = pd.cut(
            Xc["age"], bins=AGE_BINS, labels=False, right=True
        ).astype(float)

        return Xc[FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES, dtype=object)