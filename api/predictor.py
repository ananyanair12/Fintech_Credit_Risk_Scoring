"""Loads the model bundle once and turns applicant dicts into score + SHAP explanation."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.explain.shap_utils import explain_rows, make_explainer
from src.features.build_features import RAW_FEATURES


class Predictor:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle = joblib.load(path)
        self.pipe = self.bundle["pipeline"]
        self.explainer = make_explainer(self.bundle)
        self.threshold = float(self.bundle["threshold"])

    def info(self) -> dict:
        b = self.bundle
        return {
            "model_name": b["model_name"],
            "model_kind": b["model_kind"],
            "threshold": self.threshold,
            "cost_ratio": b["cost_ratio"],
            "trained_at": b["trained_at"],
            "val_metrics": b["val_metrics"],
            "test_metrics": b["test_metrics"],
        }

    def predict(self, features: dict, top_n: int = 5) -> dict:
        t0 = time.perf_counter()
        df = pd.DataFrame([features], columns=RAW_FEATURES)
        score = float(self.pipe.predict_proba(df)[0, 1])

        sv, base, feats = explain_rows(self.bundle, self.explainer, df)
        order = np.argsort(-np.abs(sv[0]))[:top_n]
        factors = [
            {
                "feature": feats.columns[i],
                "value": float(feats.iloc[0, i]),
                "shap": float(sv[0, i]),
                "direction": "increases_risk" if sv[0, i] > 0 else "decreases_risk",
            }
            for i in order
        ]
        return {
            "risk_score": round(score, 6),
            "decision": "decline" if score >= self.threshold else "approve",
            "threshold": round(self.threshold, 6),
            "shap_explanation": {"base_value": base, "top_factors": factors},
            "model_version": self.bundle["model_name"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }