from __future__ import annotations

import pandas as pd
import numpy as np


def build_modeling_table(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    df = cleaned_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_floor"] = df["timestamp"].dt.floor("h")

    # Per-hour vehicle type mode helper
    def mode_str(s: pd.Series) -> str:
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else s.iloc[0]

    # Aggregate current-hour stats
    count_col = "event_id" if "event_id" in df.columns else df.columns[0]
    agg = (
        df.groupby(["station_id", "route_id", "hour_floor"])
        .agg(
            trip_count=(count_col, "count"),
            mean_delay_current=("delay_minutes", "mean"),
            std_delay_current=("delay_minutes", "std"),
            share_delayed_5_current=("delay_minutes", lambda x: (x >= 5).mean()),
            mean_headway_current=("headway_minutes", "mean"),
            mean_demand_current=("passenger_demand", "mean"),
            mean_temp_c=("temp_c", "mean"),
            mean_precip_mm=("precip_mm", "mean"),
            mean_wind_kph=("wind_kph", "mean"),
            mean_visibility_km=("visibility_km", "mean"),
            is_holiday=("is_holiday", "max"),
            incident_flag=("incident_flag", "max"),
            vehicle_type=("vehicle_type", mode_str),
        )
        .reset_index()
    )

    agg["std_delay_current"] = agg["std_delay_current"].fillna(0.0)

    # Build next-hour targets: join each row with the row one hour ahead
    agg_sorted = agg.sort_values(["station_id", "route_id", "hour_floor"]).copy()
    agg_sorted["next_hour_floor"] = agg_sorted["hour_floor"] + pd.Timedelta(hours=1)

    # For y_tertiary: fraction of next-hour trips with delay >= 10 min
    next_hour_frac_delayed_10 = (
        df.groupby(["station_id", "route_id", "hour_floor"])
        .apply(lambda x: (x["delay_minutes"] >= 10).mean(), include_groups=False)
        .reset_index(name="next_frac_delayed_10")
    )

    # Merge next-hour avg delay
    next_lookup = agg_sorted[["station_id", "route_id", "hour_floor", "mean_delay_current"]].rename(
        columns={"hour_floor": "next_hour_floor", "mean_delay_current": "next_hour_avg_delay"}
    )

    modeling = agg_sorted.merge(
        next_lookup,
        on=["station_id", "route_id", "next_hour_floor"],
        how="inner",
    )

    # Merge next-hour frac delayed 10
    next_frac_lookup = next_hour_frac_delayed_10.rename(
        columns={"hour_floor": "next_hour_floor"}
    )
    modeling = modeling.merge(
        next_frac_lookup,
        on=["station_id", "route_id", "next_hour_floor"],
        how="left",
    )

    # Build targets
    modeling["y_primary"] = (modeling["next_hour_avg_delay"] >= 5.0).astype(int)
    modeling["y_secondary"] = modeling["next_hour_avg_delay"]
    modeling["y_tertiary"] = (modeling["next_frac_delayed_10"] >= 0.20).astype(int)

    modeling = modeling.drop(columns=["next_hour_floor", "next_frac_delayed_10"])

    return modeling.reset_index(drop=True)
