# Leakage Audit

## Protocol

All lag features use `shift(n)` where `n >= 1`, ensuring no feature value is computed from data at or after `prediction_time = hour_floor + 1h`.

## Programmatic Audit (run in notebook 03)

1. Train logistic regression on **safe features** (all 38 canonical features) → record AUC_safe
2. Add deliberately leaky feature: `current_hour_avg_delay` (the same value we're predicting on) → record AUC_leaky  
3. Assert `AUC_leaky - AUC_safe > 0.05` (proves audit detects leakage)
4. Confirm production feature set excludes `current_hour_avg_delay`

**Expected results:**
- AUC_safe ≈ 0.88–0.91 (XGBoost on safe features)
- AUC_leaky ≈ 0.97–0.99 (with leaky feature)
- Delta ≈ 0.08–0.10 (confirms audit sensitivity)

## Feature temporal guarantees

| Feature | Data used | Latest timestamp | Safe? |
|---------|-----------|-----------------|-------|
| lag_1h_mean_delay | hour t-1 | t-1 | ✓ |
| rolling_6h_mean_delay | hours t-6 to t-1 | t-1 | ✓ |
| same_hour_yesterday_mean_delay | t-24h | t-24 | ✓ |
| mean_delay_current | current hour t | t | ✓ (current hour stats only, not next hour) |
| precip_lag_1h | hour t-1 | t-1 | ✓ |
| peak_x_precip | current hour | t | ✓ |

## Target construction

`y_primary` is computed from `next_hour_avg_delay` — data from hour `t+1`. This is the target only, never used as a feature.
