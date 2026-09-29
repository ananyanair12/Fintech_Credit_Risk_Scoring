"""Synthetic applicant data for tests (same schema as Give Me Some Credit)."""
import numpy as np
import pandas as pd


def synthetic_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "revolving_utilization": rng.beta(0.6, 1.5, n),
            "age": rng.integers(21, 85, n).astype(float),
            "past_due_30_59": rng.poisson(0.3, n).astype(float),
            "debt_ratio": rng.gamma(2.0, 0.25, n),
            "monthly_income": rng.lognormal(8.6, 0.6, n),
            "open_credit_lines": rng.poisson(8, n).astype(float),
            "past_due_90": rng.poisson(0.1, n).astype(float),
            "real_estate_loans": rng.poisson(1, n).astype(float),
            "past_due_60_89": rng.poisson(0.1, n).astype(float),
            "dependents": rng.poisson(0.8, n).astype(float),
        }
    )
    logit = (
        -4.2 + 2.2 * df["revolving_utilization"] + 0.6 * df["past_due_30_59"]
        + 1.0 * df["past_due_90"] + 0.7 * df["past_due_60_89"] - 0.02 * (df["age"] - 45)
    )
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # inject the real-data quirks
    df.loc[rng.random(n) < 0.2, "monthly_income"] = np.nan
    df.loc[rng.random(n) < 0.03, "dependents"] = np.nan
    df.loc[rng.random(n) < 0.005, "past_due_30_59"] = 98
    df.loc[0, "age"] = 0
    return df