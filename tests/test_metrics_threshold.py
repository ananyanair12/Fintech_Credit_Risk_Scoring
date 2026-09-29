import numpy as np

from src.models.evaluate import choose_threshold, confusion_at
from src.models.metrics import gini, ks_statistic


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert gini(y, s) == 1.0
    assert ks_statistic(y, s) == 1.0


def test_random_scores_have_low_gini():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert abs(gini(y, rng.random(5000))) < 0.05


def test_threshold_moves_lower_when_false_negatives_cost_more():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.07).astype(int)
    s = np.clip(0.07 + 0.25 * y + rng.normal(0, 0.15, 20000), 0, 1)
    t_low_cost, _ = choose_threshold(y, s, cost_ratio=2)
    t_high_cost, _ = choose_threshold(y, s, cost_ratio=30)
    assert t_high_cost["threshold"] <= t_low_cost["threshold"]


def test_confusion_counts():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1])
    r = confusion_at(y, s, 0.5, cost_ratio=10)
    assert r["false_negatives"] == 1 and r["false_positives"] == 1
    assert r["cost"] == 11