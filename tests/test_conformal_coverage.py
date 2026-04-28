"""Tests for split conformal prediction coverage guarantees."""
import sys
import numpy as np
import pytest

sys.path.insert(0, '../src')
from conformal import (
    compute_nonconformity_scores,
    compute_quantile,
    predict_set,
    empirical_coverage,
)


class _FixedProbaModel:
    """Stub model returning fixed probability outputs."""
    def __init__(self, probas):
        self._probas = np.array(probas)

    def predict_proba(self, X):
        n = len(X)
        p1 = self._probas[:n]
        return np.column_stack([1 - p1, p1])


def test_coverage_at_alpha_01():
    """Empirical coverage should be >= 1 - alpha (marginal guarantee)."""
    rng = np.random.default_rng(42)
    n_cal = 1000
    n_test = 500

    # Simulate well-calibrated probabilities
    y_cal = rng.integers(0, 2, n_cal)
    proba_cal = np.clip(y_cal * 0.7 + (1 - y_cal) * 0.3 + rng.normal(0, 0.15, n_cal), 0.01, 0.99)

    y_test = rng.integers(0, 2, n_test)
    proba_test = np.clip(y_test * 0.7 + (1 - y_test) * 0.3 + rng.normal(0, 0.15, n_test), 0.01, 0.99)

    cal_model = _FixedProbaModel(proba_cal)
    test_model = _FixedProbaModel(proba_test)

    scores = compute_nonconformity_scores(cal_model, np.zeros((n_cal, 1)), y_cal)
    q_hat = compute_quantile(scores, alpha=0.1)
    pred_sets = predict_set(test_model, np.zeros((n_test, 1)), q_hat)
    cov = empirical_coverage(pred_sets, y_test)

    assert cov >= 0.87, f"Coverage {cov:.3f} is below acceptable lower bound 0.87"
    assert cov <= 1.0, f"Coverage {cov:.3f} exceeds 1.0"


def test_larger_alpha_smaller_sets():
    """Higher alpha (less coverage) should produce smaller or equal prediction sets."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = np.clip(0.6 * y + 0.4 * (1 - y) + rng.normal(0, 0.1, n), 0.01, 0.99)
    model = _FixedProbaModel(p)

    scores = compute_nonconformity_scores(model, np.zeros((n, 1)), y)
    q_01 = compute_quantile(scores, alpha=0.1)
    q_02 = compute_quantile(scores, alpha=0.2)

    sets_01 = predict_set(model, np.zeros((n, 1)), q_01)
    sets_02 = predict_set(model, np.zeros((n, 1)), q_02)

    assert sets_01['set_size'].mean() >= sets_02['set_size'].mean(), \
        "Higher alpha should yield smaller prediction sets"


def test_quantile_monotone():
    """Quantile should be monotonically non-decreasing in (1-alpha)."""
    scores = np.random.default_rng(7).uniform(0, 1, 200)
    q_01 = compute_quantile(scores, alpha=0.1)
    q_02 = compute_quantile(scores, alpha=0.2)
    assert q_01 >= q_02, "q_hat(alpha=0.1) should be >= q_hat(alpha=0.2)"


def test_set_size_between_0_and_2():
    """Prediction set size must be 0, 1, or 2."""
    rng = np.random.default_rng(1)
    n = 200
    p = rng.uniform(0.1, 0.9, n)
    model = _FixedProbaModel(p)
    pred_sets = predict_set(model, np.zeros((n, 1)), q_hat=0.4)
    assert pred_sets['set_size'].between(0, 2).all()
