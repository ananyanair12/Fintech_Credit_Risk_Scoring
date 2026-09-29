"""Select the production model, choose a cost-based threshold, evaluate once on test.

Usage:  python -m src.models.evaluate
Outputs: artifacts/model.joblib (bundle), artifacts/metadata.json, docs/images/*.png
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src.config import (  # noqa: E402
    CANDIDATES,
    COMPARISON_PATH,
    COST_RATIO,
    DOCS_IMAGES,
    METADATA_PATH,
    MODEL_PATH,
    SEED,
    TARGET,
)
from src.data.split import load_split  # noqa: E402
from src.features.build_features import FEATURES, RAW_FEATURES  # noqa: E402
from src.models.metrics import summarize  # noqa: E402


# ------------------------------------------------------------------ threshold logic
def confusion_at(y, s, t, cost_ratio) -> dict:
    """Decline (flag as default) when score >= t. Cost = cost_ratio*FN + FP."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(s) >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {
        "threshold": float(t),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "approval_rate": float(1.0 - pred.mean()),
        "false_negatives": fn,
        "false_positives": fp,
        "cost": float(cost_ratio * fn + fp),
    }


def choose_threshold(y, s, cost_ratio: float, n_grid: int = 300):
    """Scan thresholds and pick the one that minimises expected cost. Returns (best, curve)."""
    s = np.asarray(s)
    grid = np.unique(np.quantile(s, np.linspace(0.0, 0.999, n_grid)))
    curve = pd.DataFrame([confusion_at(y, s, t, cost_ratio) for t in grid])
    best = curve.loc[curve["cost"].idxmin()].to_dict()
    return best, curve


# ------------------------------------------------------------------------- plotting
def _save(fig, name: str) -> None:
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(DOCS_IMAGES / name, dpi=130)
    plt.close(fig)


def plot_roc(y_val, s_val, y_test, s_test) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, y, s in (("Validation", y_val, s_val), ("Test", y_test, s_test)):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{label} (AUC {summarize(y, s)['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend()
    _save(fig, "roc_curve.png")


def plot_pr(y, s, t) -> None:
    prec, rec, thr = precision_recall_curve(y, s)
    idx = int(np.argmin(np.abs(thr - t)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec)
    ax.scatter(rec[idx], prec[idx], color="red", zorder=3, label=f"Chosen threshold {t:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (validation)")
    ax.legend()
    _save(fig, "pr_curve.png")


def plot_cost(curve, best, cost_ratio) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(curve["threshold"], curve["cost"])
    ax.axvline(best["threshold"], color="red", ls="--", label=f"Min cost @ {best['threshold']:.3f}")
    ax.set(
        xlabel="Decline threshold (score >=)",
        ylabel=f"Cost = {cost_ratio:g} x FN + FP",
        title="Expected cost vs threshold (validation)",
    )
    ax.legend()
    _save(fig, "threshold_cost.png")


def plot_ks(y, s) -> None:
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, mask in (("Non-default (good)", y == 0), ("Default (bad)", y == 1)):
        vals = np.sort(s[mask])
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), label=label)
    ax.set(xlabel="Risk score", ylabel="Cumulative share",
           title=f"KS separation (test) = {summarize(y, s)['ks']:.3f}")
    ax.legend()
    _save(fig, "ks_plot.png")


# ------------------------------------------------------------------------- latency
def measure_latency(pipe, X, n: int = 200) -> dict:
    times = []
    for i in range(min(n, len(X))):
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        pipe.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


# ------------------------------------------------------------------------------ main
def main() -> None:
    comparison = pd.read_csv(COMPARISON_PATH)
    best_row = comparison.iloc[0]  # sorted by validation Gini
    name, kind = best_row["run_name"], best_row["model"]
    print(f"Selected model: {name} (val Gini {best_row['val_gini']:.4f})")

    pipe = joblib.load(CANDIDATES / f"{name}.joblib")
    val, test, train = load_split("val"), load_split("test"), load_split("train")
    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET].to_numpy()
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].to_numpy()

    s_val = pipe.predict_proba(X_val)[:, 1]
    s_test = pipe.predict_proba(X_test)[:, 1]

    # Threshold chosen on VALIDATION; test set is only used to report the final numbers.
    best, curve = choose_threshold(y_val, s_val, COST_RATIO)
    t = best["threshold"]
    val_at_t = confusion_at(y_val, s_val, t, COST_RATIO)
    test_at_t = confusion_at(y_test, s_test, t, COST_RATIO)
    approve_all_cost = float(COST_RATIO * (y_test == 1).sum())
    test_at_t["cost_reduction_vs_approve_all"] = 1.0 - test_at_t["cost"] / approve_all_cost

    test_metrics = summarize(y_test, s_test)
    val_metrics = summarize(y_val, s_val)
    latency = measure_latency(pipe, X_test)

    plot_roc(y_val, s_val, y_test, s_test)
    plot_pr(y_val, s_val, t)
    plot_cost(curve, best, COST_RATIO)
    plot_ks(y_test, s_test)

    background = pipe[:-1].transform(train.drop(columns=[TARGET]).sample(200, random_state=SEED))
    bundle = {
        "pipeline": pipe,
        "model_name": name,
        "model_kind": kind,
        "threshold": float(t),
        "cost_ratio": COST_RATIO,
        "raw_features": RAW_FEATURES,
        "feature_names": FEATURES,
        "background": background,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "model_name": name,
        "model_kind": kind,
        "threshold": float(t),
        "cost_ratio": COST_RATIO,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_at_threshold": val_at_t,
        "test_at_threshold": test_at_t,
        "inference_latency_ms": latency,
        "default_rate_test": float(y_test.mean()),
        "trained_at": bundle["trained_at"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\nTest-set performance (evaluated once):")
    print(f"  AUC {test_metrics['auc']:.4f} | Gini {test_metrics['gini']:.4f} | KS {test_metrics['ks']:.4f}")
    print(f"\nThreshold (cost ratio FN:FP = {COST_RATIO:g}:1) chosen on validation: {t:.4f}")
    print(f"  precision {test_at_t['precision']:.3f} | recall {test_at_t['recall']:.3f} | "
          f"approval rate {test_at_t['approval_rate']:.3f}")
    print(f"  cost reduction vs approving everyone: {test_at_t['cost_reduction_vs_approve_all']:.1%}")
    print(f"Single-row model latency: p50 {latency['p50_ms']:.1f} ms | p95 {latency['p95_ms']:.1f} ms")
    print(f"\nSaved {MODEL_PATH}, {METADATA_PATH}, and plots in {DOCS_IMAGES}")


if __name__ == "__main__":
    main()