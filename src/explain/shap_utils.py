"""SHAP explainability: global importance, example waterfalls, and per-request explanations.

Values are in log-odds space of the model output (positive = pushes risk UP).

Usage:  python -m src.explain.shap_utils
Outputs: artifacts/shap_global.csv and docs/images/shap_*.png
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    DOCS_IMAGES,
    MODEL_PATH,
    SEED,
    SHAP_GLOBAL_PATH,
    TARGET,
)


def make_explainer(bundle: dict):
    clf = bundle["pipeline"][-1]
    if bundle["model_kind"] == "logreg":
        return shap.LinearExplainer(clf, bundle["background"])
    return shap.TreeExplainer(clf)


def shap_values_for(explainer, X_transformed: pd.DataFrame):
    """Return (values [n_rows, n_features], base_value) for the positive (default) class."""
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):  # older SHAP: one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[:, :, 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return sv, base


def explain_rows(bundle: dict, explainer, X_raw: pd.DataFrame):
    """Return (shap_values, base_value, feature_frame) for raw applicant rows."""
    pipe = bundle["pipeline"]
    feats = pipe.named_steps["features"].transform(X_raw)  # unscaled, human-readable
    Xt = pipe[:-1].transform(X_raw)  # what the classifier actually sees
    sv, base = shap_values_for(explainer, Xt)
    return sv, base, feats


# ------------------------------------------------------------------------------ main
def main(sample_size: int = 3000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.split import load_split

    bundle = joblib.load(MODEL_PATH)
    explainer = make_explainer(bundle)
    val = load_split("val")
    sample = val.sample(min(sample_size, len(val)), random_state=SEED)
    X = sample.drop(columns=[TARGET])

    sv, base, feats = explain_rows(bundle, explainer, X)
    names = list(feats.columns)

    global_df = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    SHAP_GLOBAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    global_df.to_csv(SHAP_GLOBAL_PATH, index=False)
    print("Top features by mean |SHAP|:")
    print(global_df.head(10).round(4).to_string(index=False))

    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

    # Global summary (beeswarm)
    shap.summary_plot(sv, feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(DOCS_IMAGES / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Three example applicants: clearly approved, clearly declined, borderline
    scores = bundle["pipeline"].predict_proba(X)[:, 1]
    picks = {
        "approved": int(np.argmin(scores)),
        "declined": int(np.argmax(scores)),
        "borderline": int(np.argmin(np.abs(scores - bundle["threshold"]))),
    }
    for label, i in picks.items():
        expl = shap.Explanation(
            values=sv[i], base_values=base, data=feats.iloc[i].to_numpy(), feature_names=names
        )
        shap.plots.waterfall(expl, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(DOCS_IMAGES / f"shap_{label}.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  {label:<10} score {scores[i]:.3f}")

    print(f"\nSaved {SHAP_GLOBAL_PATH} and SHAP plots in {DOCS_IMAGES}")


if __name__ == "__main__":
    main()