from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_parquet(data_dir / "data" / "raw" / "transit_events.parquet")


def clean(df: pd.DataFrame, log: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaning_log: dict[str, Any] = {}
    df = df.copy()

    # Drop NaT timestamps
    nat_mask = df["timestamp"].isna()
    nat_count = nat_mask.sum()
    df = df[~nat_mask].reset_index(drop=True)
    cleaning_log["dropped_nat_timestamps"] = int(nat_count)

    # Exact-row hash deduplication
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    cleaning_log["dropped_exact_duplicates"] = before_dedup - len(df)

    # Filter future timestamps: max valid = START + 90 days = 2025-04-01
    max_valid_date = pd.Timestamp("2025-04-01")
    future_mask = df["timestamp"] > max_valid_date
    cleaning_log["dropped_future_timestamps"] = int(future_mask.sum())
    df = df[~future_mask].reset_index(drop=True)

    # Fix negative delays
    neg_mask = df["delay_minutes"] < 0
    cleaning_log["fixed_negative_delays"] = int(neg_mask.sum())
    df.loc[neg_mask, "delay_minutes"] = 0.0

    # Normalize station_id: convert corrupt string values back to int
    df["station_id"] = (
        df["station_id"]
        .astype(str)
        .str.upper()
        .str.strip()
        .astype(int)
    )

    # Winsorize delay_minutes at 99.9th percentile
    pct_999 = df["delay_minutes"].quantile(0.999)
    cleaning_log["winsorize_delay_threshold"] = float(pct_999)
    df["delay_minutes"] = df["delay_minutes"].clip(upper=pct_999)

    # Median impute precip_mm; add indicator
    df["precip_missing"] = df["precip_mm"].isna().astype(int)
    cleaning_log["imputed_precip_count"] = int(df["precip_missing"].sum())
    precip_median = df["precip_mm"].median()
    df["precip_mm"] = df["precip_mm"].fillna(precip_median)

    # KNN impute passenger_demand; add indicator
    df["demand_missing"] = df["passenger_demand"].isna().astype(int)
    cleaning_log["imputed_demand_count"] = int(df["demand_missing"].sum())

    if df["demand_missing"].sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        demand_col_idx = df.columns.get_loc("passenger_demand")
        numeric_cols = ["passenger_demand", "mean_demand_proxy"]

        # Use available numeric context for KNN
        knn_features = ["passenger_demand", "delay_minutes", "headway_minutes", "temp_c", "precip_mm"]
        knn_data = df[knn_features].copy()
        imputed = imputer.fit_transform(knn_data)
        df["passenger_demand"] = np.round(imputed[:, 0]).astype(np.int64)

    if log:
        for k, v in cleaning_log.items():
            print(f"  [{k}] {v}")

    return df, cleaning_log


def save_cleaned(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    processed_dir = output_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "transit_events_cleaned.parquet", index=False)
