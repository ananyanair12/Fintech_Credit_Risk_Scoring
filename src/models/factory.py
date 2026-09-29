"""Model factory: builds full sklearn Pipelines (features -> [scaler] -> classifier).

xgboost / lightgbm are imported lazily so the logistic-regression path works without them.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.features.build_features import FeatureBuilder

KINDS = ("logreg", "xgboost", "lightgbm")

DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    },
}


def make_classifier(kind: str, params: dict):
    if kind == "logreg":
        # class_weight="balanced" is the documented imbalance strategy for the baseline
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            tree_method="hist",
            eval_metric="auc",
            n_jobs=-1,
            random_state=SEED,
            **params,
        )
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1, **params)
    raise ValueError(f"Unknown model kind: {kind}")


def build_pipeline(kind: str, params: dict | None = None):
    """Return (pipeline, resolved_params)."""
    params = {**DEFAULT_PARAMS[kind], **(params or {})}
    steps = [("features", FeatureBuilder())]
    if kind == "logreg":
        steps.append(("scaler", StandardScaler().set_output(transform="pandas")))
    steps.append(("clf", make_classifier(kind, params)))
    return Pipeline(steps), params