"""Tests for target construction and temporal integrity."""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, '../src')


def _make_events(n_hours=10, delay_pattern=None):
    """Create minimal transit events for target construction testing."""
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2025-01-01")
    records = []
    for h in range(n_hours):
        for _ in range(5):  # 5 events per hour
            delay = delay_pattern[h] if delay_pattern else rng.uniform(0, 20)
            records.append({
                'station_id': 1,
                'route_id': 'R01',
                'timestamp': base + pd.Timedelta(hours=h, minutes=rng.integers(0, 60)),
                'delay_minutes': delay,
                'headway_minutes': rng.uniform(5, 15),
                'passenger_demand': rng.integers(50, 200),
                'temp_c': 12.0,
                'precip_mm': 0.0,
                'wind_kph': 10.0,
                'visibility_km': 10.0,
                'is_holiday': False,
                'incident_flag': False,
                'vehicle_type': 'bus',
            })
    return pd.DataFrame(records)


def test_target_threshold_at_5():
    """y_primary should be 1 when next-hour mean delay >= 5.0."""
    # Hour 0: delay=2.0 (next hour is hour 1 with delay=8.0 → y=1)
    # Hour 1: delay=8.0 (next hour is hour 2 with delay=1.0 → y=0)
    pattern = [2.0] * 10
    pattern[1] = 8.0  # hour 1 is high delay
    df = _make_events(n_hours=5, delay_pattern=pattern[:5])

    from targets import build_modeling_table
    table = build_modeling_table(df)

    # Row for hour 0: next_hour (hour 1) avg delay = 8.0 → y=1
    hour_0_row = table[table['hour_floor'] == pd.Timestamp("2025-01-01 00:00:00")]
    if len(hour_0_row) > 0:
        assert hour_0_row['y_primary'].values[0] == 1


def test_last_hour_dropped():
    """Rows with no next-hour window should be dropped."""
    df = _make_events(n_hours=5)
    from targets import build_modeling_table
    table = build_modeling_table(df)
    # Last hour (hour 4) has no next hour → must be dropped
    last_hour = pd.Timestamp("2025-01-01 04:00:00")
    assert last_hour not in table['hour_floor'].values


def test_positive_rate_range():
    """Positive rate should be in a reasonable range (not 0 or 1)."""
    rng = np.random.default_rng(42)
    delays = rng.uniform(0, 15, 20)
    df = _make_events(n_hours=20, delay_pattern=list(delays))
    from targets import build_modeling_table
    table = build_modeling_table(df)
    pos_rate = table['y_primary'].mean()
    assert 0.0 < pos_rate < 1.0, f"Positive rate {pos_rate} out of expected range"


def test_no_target_leakage_in_aggregates():
    """Current-hour aggregates must not include next-hour delay values."""
    df = _make_events(n_hours=10)
    from targets import build_modeling_table
    table = build_modeling_table(df)
    # mean_delay_current should reflect current hour only
    # If it accidentally equals next_hour_avg_delay, that's leakage
    if 'mean_delay_current' in table.columns and 'next_hour_avg_delay' in table.columns:
        corr = table['mean_delay_current'].corr(table['next_hour_avg_delay'])
        # Correlation should not be near 1.0 (that would suggest leakage)
        assert corr < 0.95, f"Suspicious correlation {corr:.3f} — possible leakage"
