"""Credit-risk evaluation metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def gini(y_true, score) -> float:
    """Gini = 2*AUC - 1 (rank-ordering power, the standard credit-scoring metric)."""
    return float(2.0 * roc_auc_score(y_true, score) - 1.0)


def ks_statistic(y_true, score) -> float:
    """Max separation between the cumulative score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def summarize(y_true, score) -> dict:
    auc = float(roc_auc_score(y_true, score))
    return {
        "auc": auc,
        "gini": 2.0 * auc - 1.0,
        "ks": ks_statistic(y_true, score),
        "avg_precision": float(average_precision_score(y_true, score)),
    }