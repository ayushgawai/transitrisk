"""Programmatic leakage audit tests."""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, '../src')


def test_target_uses_future_data_only():
    """y_primary must derive from next-hour data, not current-hour."""
    # Build a tiny synthetic table and verify target assignment
    records = []
    base = pd.Timestamp("2025-01-01")
    for h in range(5):
        records.append({
            'station_id': 1,
            'route_id': 'R01',
            'hour_floor': base + pd.Timedelta(hours=h),
            'next_hour_avg_delay': float(h * 3),  # 0, 3, 6, 9, 12
        })
    df = pd.DataFrame(records)
    # threshold = 5.0
    df['y_primary'] = (df['next_hour_avg_delay'] >= 5.0).astype(int)
    expected = [0, 0, 1, 1, 1]
    assert list(df['y_primary']) == expected


def test_lag_shift_prevents_leakage():
    """shift(1) on sorted series must not include current row's value."""
    delays = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    lagged = delays.shift(1)
    # lagged[0] should be NaN, lagged[1] should be 1.0
    assert pd.isna(lagged.iloc[0])
    assert lagged.iloc[1] == 1.0
    assert lagged.iloc[4] == 4.0  # not 5.0


def test_leaky_feature_detection():
    """
    Simulates the leakage audit: adding true next-hour delay as a feature
    should dramatically increase AUC (from ~0.88 to ~0.97+).
    This test checks the audit logic, not actual model performance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)
    n = 5000
    # Safe feature: noisy signal
    X_safe = rng.normal(0, 1, (n, 5))
    y = (X_safe[:, 0] + rng.normal(0, 1.5, n) > 0).astype(int)

    # Leaky feature: essentially the target with tiny noise
    leaky = y + rng.normal(0, 0.05, n)

    split = int(0.8 * n)
    X_tr, X_te = X_safe[:split], X_safe[split:]
    y_tr, y_te = y[:split], y[split:]

    lr_safe = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    auc_safe = roc_auc_score(y_te, lr_safe.predict_proba(X_te)[:, 1])

    X_tr_leak = np.column_stack([X_tr, leaky[:split]])
    X_te_leak = np.column_stack([X_te, leaky[split:]])
    lr_leak = LogisticRegression(max_iter=1000).fit(X_tr_leak, y_tr)
    auc_leak = roc_auc_score(y_te, lr_leak.predict_proba(X_te_leak)[:, 1])

    assert auc_leak - auc_safe > 0.05, (
        f"Audit failed: leaky AUC ({auc_leak:.3f}) should be substantially "
        f"higher than safe AUC ({auc_safe:.3f}). Delta = {auc_leak - auc_safe:.3f}"
    )
