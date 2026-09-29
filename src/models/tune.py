"""Hyper-parameter search (Optuna) for the gradient-boosted models.

Tuning runs on pre-computed feature matrices for speed; the objective is validation AUC.
`scale_pos_weight` is part of the search space, so the class-imbalance handling for the
boosted models is chosen by validation performance and recorded in MLflow.
"""
from __future__ import annotations

import optuna
from sklearn.metrics import roc_auc_score

from src.config import SEED
from src.models.factory import make_classifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(kind: str, trial: optuna.Trial, spw_max: float) -> dict:
    if kind == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, spw_max),
        }
    if kind == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, spw_max),
        }
    raise ValueError(f"Tuning not supported for {kind}")


def tune(kind, X_train, y_train, X_val, y_val, n_trials: int = 20):
    """Return (best_params, best_val_auc)."""
    spw_max = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(kind, trial, spw_max)
        clf = make_classifier(kind, params)
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)