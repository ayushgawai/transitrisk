from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

CANONICAL_FEATURES: list[str] = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_peak_morning", "is_peak_evening", "month",
    "station_id_target_encoded", "route_id_target_encoded",
    "station_busyness_quartile", "station_route_busyness_quartile",
    "trip_count", "mean_delay_current", "std_delay_current",
    "share_delayed_5_current", "mean_headway_current", "mean_demand_current",
    "lag_1h_mean_delay", "lag_1h_share_delayed_5",
    "lag_2h_mean_delay", "lag_3h_mean_delay", "lag_3h_share_delayed_5",
    "rolling_6h_mean_delay", "rolling_6h_std_delay",
    "same_hour_yesterday_mean_delay", "same_hour_last_week_mean_delay",
    "lag_1h_trip_count",
    "mean_temp_c", "mean_precip_mm", "mean_wind_kph", "mean_visibility_km",
    "precip_lag_1h", "precip_rolling_3h_sum", "wind_lag_1h",
    "peak_x_precip", "short_headway_x_demand", "weekend_x_precip",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["hour_floor"].dt.hour
    dow = df["hour_floor"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_peak_morning"] = hour.isin([7, 8, 9]).astype(int)
    df["is_peak_evening"] = hour.isin([16, 17, 18, 19]).astype(int)
    df["month"] = df["hour_floor"].dt.month

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["station_id", "route_id", "hour_floor"]).reset_index(drop=True)
    grp = df.groupby(["station_id", "route_id"], sort=False)

    md  = grp["mean_delay_current"]
    sd5 = grp["share_delayed_5_current"]
    tc  = grp["trip_count"]

    df["lag_1h_mean_delay"]           = md.shift(1)
    df["lag_1h_share_delayed_5"]      = sd5.shift(1)
    df["lag_2h_mean_delay"]           = md.shift(2)
    df["lag_3h_mean_delay"]           = md.shift(3)
    df["lag_3h_share_delayed_5"]      = sd5.shift(3)
    df["rolling_6h_mean_delay"]       = md.shift(1).transform(lambda x: x.rolling(6, min_periods=2).mean())
    df["rolling_6h_std_delay"]        = md.shift(1).transform(lambda x: x.rolling(6, min_periods=2).std())
    df["lag_1h_trip_count"]           = tc.shift(1)
    df["same_hour_yesterday_mean_delay"]  = md.shift(24)
    df["same_hour_last_week_mean_delay"]  = md.shift(168)
    return df


def add_weather_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["station_id", "route_id", "hour_floor"]).reset_index(drop=True)
    grp = df.groupby(["station_id", "route_id"], sort=False)

    pm = grp["mean_precip_mm"]
    wk = grp["mean_wind_kph"]

    df["precip_lag_1h"]         = pm.shift(1)
    df["precip_rolling_3h_sum"] = pm.shift(1).transform(lambda x: x.rolling(3, min_periods=1).sum())
    df["wind_lag_1h"]           = wk.shift(1)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_peak = (df["is_peak_morning"] | df["is_peak_evening"]).astype(int)
    df["peak_x_precip"] = is_peak * df["mean_precip_mm"]
    df["short_headway_x_demand"] = (df["mean_headway_current"] < 5).astype(int) * df["mean_demand_current"]
    df["weekend_x_precip"] = df["is_weekend"] * df["mean_precip_mm"]
    return df


def add_target_encoding(
    df: pd.DataFrame,
    target_col: str,
    train_mask: pd.Series,
    n_folds: int = 5,
) -> pd.DataFrame:
    df = df.copy()
    global_mean = df.loc[train_mask, target_col].mean()

    for col in ["station_id", "route_id"]:
        enc_col = f"{col}_target_encoded"
        df[enc_col] = global_mean

        train_df = df[train_mask].copy()
        kf = KFold(n_splits=n_folds, shuffle=False)
        train_indices = np.where(train_mask)[0]

        encoded_train = np.full(train_mask.sum(), global_mean)

        for fold_train_idx, fold_val_idx in kf.split(train_df):
            fold_train_rows = train_df.iloc[fold_train_idx]
            fold_val_rows = train_df.iloc[fold_val_idx]

            means = fold_train_rows.groupby(col)[target_col].mean()
            val_vals = fold_val_rows[col].map(means).fillna(global_mean).values
            encoded_train[fold_val_idx] = val_vals

        df.loc[train_mask, enc_col] = encoded_train

        # Test/val rows: use global train mean per category
        test_val_mask = ~train_mask
        train_means = train_df.groupby(col)[target_col].mean()
        df.loc[test_val_mask, enc_col] = df.loc[test_val_mask, col].map(train_means).fillna(global_mean).values

    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    station_mean_trips = df.groupby("station_id")["trip_count"].transform("mean")
    df["station_busyness_quartile"] = pd.qcut(
        station_mean_trips, q=4, labels=[1, 2, 3, 4], duplicates="drop"
    ).astype(float)

    sr_mean_trips = df.groupby(["station_id", "route_id"])["trip_count"].transform("mean")
    df["station_route_busyness_quartile"] = pd.qcut(
        sr_mean_trips, q=4, labels=[1, 2, 3, 4], duplicates="drop"
    ).astype(float)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    train_mask: pd.Series,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_weather_lag_features(df)
    df = add_interaction_features(df)
    df = add_target_encoding(df, target_col="y_primary", train_mask=train_mask)
    df = add_spatial_features(df)

    cols = feature_cols if feature_cols is not None else CANONICAL_FEATURES

    # Fill boundary NaN lag values with 0; fill spatial quartile NaN with median
    lag_cols = [c for c in cols if "lag" in c or "rolling" in c]
    df[lag_cols] = df[lag_cols].fillna(0.0)
    spatial_cols = [c for c in cols if "quartile" in c or "encoded" in c]
    for c in spatial_cols:
        df[c] = df[c].fillna(df[c].median())

    X = df[cols].fillna(0.0).copy()
    return X, cols
