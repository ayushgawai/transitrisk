"""Tests for feature engineering correctness."""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, '../src')
from features import add_temporal_features, add_lag_features, CANONICAL_FEATURES


def make_sample_df(n_stations=2, n_routes=2, n_hours=50):
    """Build a minimal modeling table for testing."""
    rng = np.random.default_rng(42)
    records = []
    base = pd.Timestamp("2025-01-01")
    for s in range(n_stations):
        for r in range(n_routes):
            for h in range(n_hours):
                records.append({
                    'station_id': s,
                    'route_id': f'R0{r+1}',
                    'hour_floor': base + pd.Timedelta(hours=h),
                    'mean_delay_current': rng.uniform(0, 15),
                    'share_delayed_5_current': rng.uniform(0, 1),
                    'trip_count': rng.integers(1, 20),
                    'mean_headway_current': rng.uniform(3, 18),
                    'mean_demand_current': rng.uniform(20, 200),
                    'mean_precip_mm': rng.uniform(0, 5),
                    'mean_wind_kph': rng.uniform(0, 40),
                    'std_delay_current': rng.uniform(0, 5),
                })
    return pd.DataFrame(records)


def test_temporal_features_no_nan():
    df = make_sample_df()
    df = add_temporal_features(df)
    temporal_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                     'is_weekend', 'is_peak_morning', 'is_peak_evening', 'month']
    for col in temporal_cols:
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].notna().all(), f"NaN found in {col}"


def test_lag_features_no_future_leakage():
    """Lag features must not use data from the current or future timestep."""
    df = make_sample_df(n_stations=1, n_routes=1, n_hours=30)
    df = add_lag_features(df)
    # lag_1h should be NaN for the first row in each group (no prior data)
    first_rows = df.groupby(['station_id', 'route_id']).head(1)
    assert first_rows['lag_1h_mean_delay'].isna().all(), \
        "First row lag should be NaN (no prior hour data)"


def test_lag_1h_values_match():
    """lag_1h_mean_delay at row t should equal mean_delay_current at row t-1."""
    df = make_sample_df(n_stations=1, n_routes=1, n_hours=10)
    df_sorted = df.sort_values('hour_floor').reset_index(drop=True)
    df_sorted = add_lag_features(df_sorted)
    # Row index 1 should have lag_1h == row 0's mean_delay_current
    assert abs(df_sorted.loc[1, 'lag_1h_mean_delay'] - df_sorted.loc[0, 'mean_delay_current']) < 1e-9


def test_canonical_features_count():
    assert len(CANONICAL_FEATURES) == 38
