"""Train and compare Logistic Regression, XGBoost and LightGBM. Every run goes to MLflow.

Usage:  python -m src.models.train [--trials 20]
Outputs: artifacts/candidates/*.joblib and artifacts/model_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    CANDIDATES,
    COMPARISON_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    TARGET,
)
from src.data.split import load_split
from src.features.build_features import FeatureBuilder
from src.models.factory import build_pipeline
from src.models.metrics import summarize
from src.models.tune import tune


def _xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy()


def _tune_logreg(X_train, y_train, X_val, y_val):
    """Tiny grid over the inverse regularisation strength C."""
    best_c, best_auc = None, -1.0
    for c in (0.01, 0.1, 1.0, 10.0):
        pipe, _ = build_pipeline("logreg", {"C": c})
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_c, best_auc = c, auc
    return {"C": best_c}, best_auc


def run_candidate(name, kind, params, extra, data) -> dict:
    X_train, y_train, X_val, y_val = data
    with mlflow.start_run(run_name=name) as run:
        pipe, params = build_pipeline(kind, params)
        mlflow.set_tags({"model_kind": kind, **{k: str(v) for k, v in extra.items()}})
        mlflow.log_params({"model": kind, **params, "n_train": len(X_train)})

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        val_scores = pipe.predict_proba(X_val)[:, 1]
        train_scores = pipe.predict_proba(X_train)[:, 1]
        val_m = summarize(y_val, val_scores)
        train_auc = float(roc_auc_score(y_train, train_scores))

        mlflow.log_metrics(
            {f"val_{k}": v for k, v in val_m.items()}
            | {"train_auc": train_auc, "train_time_s": train_time}
        )

        CANDIDATES.mkdir(parents=True, exist_ok=True)
        path = CANDIDATES / f"{name}.joblib"
        joblib.dump(pipe, path)
        mlflow.log_artifact(str(path), artifact_path="model")

        print(
            f"  {name:<18} val AUC {val_m['auc']:.4f} | Gini {val_m['gini']:.4f} | "
            f"KS {val_m['ks']:.4f} | {train_time:.1f}s"
        )
        return {
            "run_name": name,
            "model": kind,
            **{f"val_{k}": v for k, v in val_m.items()},
            "train_auc": train_auc,
            "train_time_s": round(train_time, 2),
            "params": json.dumps(params),
            "mlflow_run_id": run.info.run_id,
        }


def main(n_trials: int = 20) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, y_train = _xy(load_split("train"))
    X_val, y_val = _xy(load_split("val"))
    data = (X_train, y_train, X_val, y_val)

    print(f"Training on {len(X_train):,} rows, validating on {len(X_val):,} rows")
    rows = []

    print("\nBaselines / defaults")
    rows.append(run_candidate("logreg_baseline", "logreg", {}, {"stage": "baseline"}, data))
    rows.append(run_candidate("xgboost_default", "xgboost", {}, {"stage": "default"}, data))
    rows.append(run_candidate("lightgbm_default", "lightgbm", {}, {"stage": "default"}, data))

    print("\nTuning")
    lr_params, _ = _tune_logreg(X_train, y_train, X_val, y_val)
    rows.append(run_candidate("logreg_tuned", "logreg", lr_params, {"stage": "tuned"}, data))

    # Boosted models are tuned on pre-computed features (fast); final fit uses full pipeline
    fb = FeatureBuilder().fit(X_train)
    Xtr_f, Xva_f = fb.transform(X_train), fb.transform(X_val)
    for kind in ("xgboost", "lightgbm"):
        best_params, best_auc = tune(kind, Xtr_f, y_train, Xva_f, y_val, n_trials)
        print(f"  {kind} tuning: best val AUC {best_auc:.4f} over {n_trials} trials")
        rows.append(
            run_candidate(
                f"{kind}_tuned", kind, best_params,
                {"stage": "tuned", "optuna_trials": n_trials}, data,
            )
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["val_gini", "val_auc"], ascending=False)
        .reset_index(drop=True)
    )
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    print("\nModel comparison (validation set):")
    cols = ["run_name", "val_auc", "val_gini", "val_ks", "train_time_s"]
    print(comparison[cols].round(4).to_string(index=False))
    print(f"\nSaved {COMPARISON_PATH}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per model")
    main(parser.parse_args().trials)