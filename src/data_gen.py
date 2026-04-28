from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
START_DATE = "2025-01-01"
RANDOM_SEED = 42
N_STATIONS = 60
N_ROUTES = 12
N_DAYS = 90
TARGET_EVENTS = 1_247_832

HOLIDAYS = {
    pd.Timestamp("2025-01-01"),
    pd.Timestamp("2025-01-20"),
    pd.Timestamp("2025-02-17"),
    pd.Timestamp("2025-04-18"),
    pd.Timestamp("2025-05-26"),
    pd.Timestamp("2025-07-04"),
    pd.Timestamp("2025-09-01"),
    pd.Timestamp("2025-11-27"),
}

DEMAND_CURVE: dict[int, float] = {
    0: 0.10, 1: 0.05, 2: 0.04, 3: 0.04, 4: 0.05, 5: 0.15,
    6: 0.45, 7: 0.80, 8: 1.00, 9: 0.75, 10: 0.60, 11: 0.65,
    12: 0.70, 13: 0.65, 14: 0.60, 15: 0.70, 16: 0.85, 17: 0.95,
    18: 0.90, 19: 0.70, 20: 0.50, 21: 0.35, 22: 0.20, 23: 0.12,
}

DOW_FACTORS: dict[int, float] = {0: 1.08, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.10, 5: 0.85, 6: 0.80}

VEHICLE_TYPES = ["bus", "train", "light_rail"]

STATION_NAMES = [
    "Central Station", "Market Square", "University Ave", "Airport Terminal",
    "Harbor View", "Downtown Core", "Riverside", "Westgate", "Eastside Hub",
    "Northpark", "South Terminal", "Convention Center", "City Hall",
    "Financial District", "Old Town", "Lakefront", "Stadium",
    "Medical Center", "Tech Park", "Midtown", "Grand Ave",
    "Civic Center", "Bayfront", "Main Street", "Union Station",
    "Museum District", "Botanical Garden", "Navy Yard", "Chinatown",
    "Little Italy", "Arts District", "Fashion Quarter", "Waterfront",
    "Capitol Hill", "Sunset Blvd", "Highlands", "Greenway",
    "Fairgrounds", "Coliseum", "Industrial Park", "Warehouse District",
    "Entertainment Zone", "Brewery District", "Food Market", "Park Place",
    "Crystal Mall", "Ocean Drive", "Skyline Plaza", "Heritage Walk",
    "Innovation Hub", "Gateway", "Crossroads", "Metroplex",
    "Elm Street", "Oak Grove", "Pine Ridge", "Cedar Falls",
    "Maple Ave", "Birch Lane", "Willow Creek",
]


# ---------------------------------------------------------------------------
# Delay generation
# ---------------------------------------------------------------------------

def generate_delay(
    base, peak_factor, weather_factor, demand_pressure,
    headway_compounding, dow_factor, holiday_factor, incident_flag, rng
):
    incident_shock = rng.uniform(8, 25) if incident_flag else 0
    noise = rng.normal(0, 3.5)
    delay = base * peak_factor * weather_factor * demand_pressure * headway_compounding * dow_factor * holiday_factor
    delay = max(0, delay + incident_shock + noise)
    return min(delay, 90.0)


# ---------------------------------------------------------------------------
# Factor helpers (vectorized)
# ---------------------------------------------------------------------------

def _peak_factor_vec(hours: np.ndarray) -> np.ndarray:
    pf = np.ones(len(hours))
    pf[np.isin(hours, [7, 8, 17, 18])] = 1.85
    pf[np.isin(hours, [6, 9, 16, 19])] = 1.35
    pf[np.isin(hours, [22, 23, 0, 1, 2, 3, 4, 5])] = 0.65
    return pf


def _weather_factor_vec(precip: np.ndarray, wind: np.ndarray, vis: np.ndarray) -> np.ndarray:
    wf = np.where(
        precip < 0.5, 1.0,
        np.where(
            precip < 2.0, 1.0 + 0.02 * precip,
            np.where(
                precip < 5.0, 1.04 + 0.08 * (precip - 2.0),
                1.28 + 0.15 * np.power(np.maximum(0.0, precip - 5.0), 1.2),
            ),
        ),
    )
    wf = wf * (1.0 + 0.015 * np.maximum(0, wind - 25))
    wf = np.where(vis < 1.0, wf * 1.20, wf)
    return wf


def _demand_pressure_vec(demand: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    ratio = demand / capacity
    return 1.0 + 0.35 * np.maximum(0, ratio - 0.7) ** 2


def _headway_compounding_vec(headway: np.ndarray) -> np.ndarray:
    return 1.0 + 0.45 * np.exp(-headway / 7.0)


# ---------------------------------------------------------------------------
# Build static lookup tables
# ---------------------------------------------------------------------------

def _build_station_table(rng: np.random.Generator) -> pd.DataFrame:
    ranks = np.arange(1, N_STATIONS + 1, dtype=float)
    raw_weights = 1.0 / ranks ** 1.3
    weights = raw_weights / raw_weights.sum()

    lats = rng.uniform(37.5, 37.9, N_STATIONS)
    lons = rng.uniform(-122.5, -122.0, N_STATIONS)
    capacities = rng.integers(50, 301, N_STATIONS)

    return pd.DataFrame({
        "station_id": np.arange(1, N_STATIONS + 1),
        "station_name": STATION_NAMES[:N_STATIONS],
        "weight": weights,
        "capacity": capacities,
        "latitude": np.round(lats, 5),
        "longitude": np.round(lons, 5),
    })


def _build_route_table(rng: np.random.Generator) -> pd.DataFrame:
    base_delays = rng.uniform(1.0, 4.0, N_ROUTES)
    base_headways = rng.uniform(3, 18, N_ROUTES)
    vtypes = [VEHICLE_TYPES[i % 3] for i in range(N_ROUTES)]
    return pd.DataFrame({
        "route_id": [f"R{i:02d}" for i in range(1, N_ROUTES + 1)],
        "base_delay": base_delays,
        "base_headway": base_headways,
        "vehicle_type": vtypes,
    })


def _build_station_route_map(rng: np.random.Generator) -> pd.DataFrame:
    rng2 = np.random.default_rng(42)
    pairs = []
    for sid in range(1, N_STATIONS + 1):
        n_routes = rng2.integers(2, 5)
        route_indices = rng2.choice(N_ROUTES, size=n_routes, replace=False)
        for ri in route_indices:
            pairs.append((sid, f"R{ri + 1:02d}"))
    return pd.DataFrame(pairs, columns=["station_id", "route_id"])


# ---------------------------------------------------------------------------
# Weather generation (per day × station)
# ---------------------------------------------------------------------------

def _generate_weather(rng: np.random.Generator, n: int, day_of_year: np.ndarray) -> dict[str, np.ndarray]:
    temp_c = 12 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + rng.normal(0, 3, n)
    raw_precip = rng.uniform(0, 1, n)
    precip_mm = np.where(
        raw_precip < 0.70,
        0.0,
        rng.lognormal(mean=0.5, sigma=0.9, size=n),
    )
    wind_kph = rng.gamma(shape=2.5, scale=6.0, size=n)
    visibility_km = np.maximum(0.2, 10 - 0.4 * precip_mm + rng.normal(0, 1, n))
    return {"temp_c": temp_c, "precip_mm": precip_mm, "wind_kph": wind_kph, "visibility_km": visibility_km}


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _generate_events(
    station_df: pd.DataFrame,
    route_df: pd.DataFrame,
    sr_map: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    start = pd.Timestamp(START_DATE)
    end = start + pd.Timedelta(days=N_DAYS)
    holiday_dates = {h.date() for h in HOLIDAYS if start <= h < end}

    route_base_delay = dict(zip(route_df["route_id"], route_df["base_delay"]))
    route_base_headway = dict(zip(route_df["route_id"], route_df["base_headway"]))
    route_vehicle = dict(zip(route_df["route_id"], route_df["vehicle_type"]))
    station_weight = dict(zip(station_df["station_id"], station_df["weight"]))
    station_capacity = dict(zip(station_df["station_id"], station_df["capacity"]))

    sr_pairs = list(zip(sr_map["station_id"], sr_map["route_id"]))
    n_pairs = len(sr_pairs)

    # Pair weights = sum of station weights for each route served
    pair_weights = np.array([station_weight[sid] for sid, _ in sr_pairs])
    pair_weights /= pair_weights.sum()

    total_minutes = N_DAYS * 24 * 60
    # We'll generate exactly TARGET_EVENTS distributed across all minutes
    # Strategy: for each day-hour block, sample events proportional to demand_curve + pair_weights

    # Precompute how many events per hour-block
    hours = np.arange(24)
    demand_arr = np.array([DEMAND_CURVE[h] for h in hours])
    demand_arr /= demand_arr.sum()

    events_per_day = TARGET_EVENTS / N_DAYS
    events_per_hour = (demand_arr * events_per_day).astype(int)
    # Fix rounding to exactly hit target
    deficit = TARGET_EVENTS - int(events_per_day * N_DAYS)

    all_records: list[dict] = []
    event_id = 1

    for day_idx in range(N_DAYS):
        current_date = start + pd.Timedelta(days=day_idx)
        day_of_year = current_date.day_of_year
        dow = current_date.dayofweek
        is_holiday = current_date.date() in holiday_dates
        dow_f = DOW_FACTORS[dow]
        holiday_f = 0.75 if is_holiday else 1.0

        for hour in range(24):
            n_events = int(events_per_hour[hour])
            if day_idx == 0 and hour == 0:
                n_events += TARGET_EVENTS - int(events_per_day * N_DAYS)

            if n_events == 0:
                continue

            # Sample station-route pairs for this hour block
            pair_indices = rng.choice(n_pairs, size=n_events, p=pair_weights)

            # Generate minute offsets within the hour
            minutes_offset = rng.integers(0, 60, n_events)
            base_ts = current_date + pd.Timedelta(hours=hour)

            timestamps = [base_ts + pd.Timedelta(minutes=int(m)) for m in minutes_offset]

            # Generate weather for all events in block (one draw per event)
            doy_arr = np.full(n_events, day_of_year, dtype=float)
            wx = _generate_weather(rng, n_events, doy_arr)

            peak_f = _peak_factor_vec(np.full(n_events, hour))

            # Peak and raining flag for incident probability
            is_peak = hour in [7, 8, 17, 18]
            incident_p = 0.012 * (1.5 if (is_peak and np.any(wx["precip_mm"] > 0.5)) else 1.0)
            incident_flags = rng.uniform(0, 1, n_events) < incident_p

            for i in range(n_events):
                sid, rid = sr_pairs[pair_indices[i]]
                cap = station_capacity[sid]
                sw = station_weight[sid]
                base_hw = route_base_headway[rid]
                base_delay = route_base_delay[rid]

                headway = max(1.5, base_hw * rng.lognormal(0, 0.2))
                demand = int(sw * 100 * DEMAND_CURVE[hour] * rng.lognormal(0, 0.15))

                wx_f = _weather_factor_vec(
                    np.array([wx["precip_mm"][i]]),
                    np.array([wx["wind_kph"][i]]),
                    np.array([wx["visibility_km"][i]]),
                )[0]
                dp = _demand_pressure_vec(np.array([demand], dtype=float), np.array([cap], dtype=float))[0]
                hc = _headway_compounding_vec(np.array([headway]))[0]

                delay = generate_delay(
                    base_delay, peak_f[i], wx_f, dp, hc,
                    dow_f, holiday_f, incident_flags[i], rng
                )

                sched_ts = timestamps[i]
                actual_ts = sched_ts + pd.Timedelta(minutes=delay)

                all_records.append({
                    "event_id": event_id,
                    "timestamp": timestamps[i],
                    "station_id": sid,
                    "station_name": station_df.loc[station_df["station_id"] == sid, "station_name"].iloc[0],
                    "route_id": rid,
                    "scheduled_time": sched_ts,
                    "actual_time": actual_ts,
                    "delay_minutes": delay,
                    "headway_minutes": headway,
                    "passenger_demand": demand,
                    "temp_c": wx["temp_c"][i],
                    "precip_mm": wx["precip_mm"][i],
                    "wind_kph": wx["wind_kph"][i],
                    "visibility_km": wx["visibility_km"][i],
                    "is_holiday": is_holiday,
                    "incident_flag": bool(incident_flags[i]),
                    "vehicle_type": route_vehicle[rid],
                })
                event_id += 1

    return pd.DataFrame(all_records)


def _generate_events_fast(
    station_df: pd.DataFrame,
    route_df: pd.DataFrame,
    sr_map: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Vectorized generation - builds arrays then constructs DataFrame at end."""
    start = pd.Timestamp(START_DATE)
    end = start + pd.Timedelta(days=N_DAYS)
    holiday_dates = {h.date() for h in HOLIDAYS if start <= h < end}

    route_base_delay = dict(zip(route_df["route_id"], route_df["base_delay"]))
    route_base_headway = dict(zip(route_df["route_id"], route_df["base_headway"]))
    route_vehicle_map = dict(zip(route_df["route_id"], route_df["vehicle_type"]))
    station_weight_map = dict(zip(station_df["station_id"], station_df["weight"]))
    station_capacity_map = dict(zip(station_df["station_id"], station_df["capacity"]))
    station_name_map = dict(zip(station_df["station_id"], station_df["station_name"]))

    sr_pairs = list(zip(sr_map["station_id"], sr_map["route_id"]))
    n_pairs = len(sr_pairs)
    pair_station_ids = np.array([p[0] for p in sr_pairs])
    pair_route_ids = [p[1] for p in sr_pairs]

    pair_weights = np.array([station_weight_map[sid] for sid in pair_station_ids])
    pair_weights /= pair_weights.sum()

    demand_arr_raw = np.array([DEMAND_CURVE[h] for h in range(24)])
    demand_arr = demand_arr_raw / demand_arr_raw.sum()

    # Events per (day, hour)
    events_per_day_float = TARGET_EVENTS / N_DAYS
    events_per_hour_float = demand_arr * events_per_day_float
    events_per_hour_base = events_per_hour_float.astype(int)
    remainder = TARGET_EVENTS - int(events_per_hour_base.sum() * N_DAYS)

    chunks = []
    total_so_far = 0

    for day_idx in range(N_DAYS):
        current_date = start + pd.Timedelta(days=day_idx)
        day_of_year = current_date.day_of_year
        dow = current_date.dayofweek
        is_holiday = current_date.date() in holiday_dates
        dow_f = DOW_FACTORS[dow]
        holiday_f = 0.75 if is_holiday else 1.0

        for hour in range(24):
            n_ev = int(events_per_hour_base[hour])
            if day_idx == N_DAYS - 1 and hour == 23:
                n_ev += TARGET_EVENTS - total_so_far - n_ev

            if n_ev <= 0:
                continue

            pair_idx = rng.choice(n_pairs, size=n_ev, p=pair_weights)
            sids = pair_station_ids[pair_idx]
            rids = [pair_route_ids[i] for i in pair_idx]

            minute_offsets = rng.integers(0, 60, n_ev)
            base_ns = pd.Timestamp(current_date + pd.Timedelta(hours=hour)).value
            timestamps_ns = base_ns + (minute_offsets * 60 * 10**9).astype(np.int64)

            doy_arr = np.full(n_ev, day_of_year, dtype=float)
            temp_c = 12 + 10 * np.sin(2 * np.pi * (doy_arr - 80) / 365) + rng.normal(0, 3, n_ev)
            raw_precip_roll = rng.uniform(0, 1, n_ev)
            lognorm_precip = rng.lognormal(mean=0.5, sigma=0.9, size=n_ev)
            precip_mm = np.where(raw_precip_roll < 0.70, 0.0, lognorm_precip)
            wind_kph = rng.gamma(shape=2.5, scale=6.0, size=n_ev)
            visibility_km = np.maximum(0.2, 10 - 0.4 * precip_mm + rng.normal(0, 1, n_ev))

            base_delays_arr = np.array([route_base_delay[r] for r in rids])
            base_headways_arr = np.array([route_base_headway[r] for r in rids])
            capacities_arr = np.array([station_capacity_map[s] for s in sids], dtype=float)
            weights_arr = np.array([station_weight_map[s] for s in sids])

            headways = np.maximum(1.5, base_headways_arr * rng.lognormal(0, 0.2, n_ev))
            demands = (weights_arr * 100 * DEMAND_CURVE[hour] * rng.lognormal(0, 0.15, n_ev)).astype(int)

            peak_f = _peak_factor_vec(np.full(n_ev, hour))
            wx_f = _weather_factor_vec(precip_mm, wind_kph, visibility_km)
            dp = _demand_pressure_vec(demands.astype(float), capacities_arr)
            hc = _headway_compounding_vec(headways)

            is_peak = hour in [7, 8, 17, 18]
            incident_p = 0.012 * (1.5 if (is_peak and np.any(precip_mm > 0.5)) else 1.0)
            incident_flags = rng.uniform(0, 1, n_ev) < incident_p

            incident_shocks = np.where(incident_flags, rng.uniform(8, 25, n_ev), 0.0)
            noise = rng.normal(0, 3.5, n_ev)

            raw_delay = (
                base_delays_arr * peak_f * wx_f * dp * hc * dow_f * holiday_f
                + incident_shocks + noise
            )
            delays = np.clip(np.maximum(0, raw_delay), 0, 90.0)

            actual_ts_ns = timestamps_ns + (delays * 60 * 10**9).astype(np.int64)

            n_start = total_so_far + 1
            event_ids = np.arange(n_start, n_start + n_ev, dtype=np.int64)

            chunk = pd.DataFrame({
                "event_id": event_ids,
                "timestamp": pd.to_datetime(timestamps_ns, unit="ns"),
                "station_id": sids,
                "station_name": [station_name_map[s] for s in sids],
                "route_id": rids,
                "scheduled_time": pd.to_datetime(timestamps_ns, unit="ns"),
                "actual_time": pd.to_datetime(actual_ts_ns, unit="ns"),
                "delay_minutes": delays,
                "headway_minutes": headways,
                "passenger_demand": demands,
                "temp_c": temp_c,
                "precip_mm": precip_mm,
                "wind_kph": wind_kph,
                "visibility_km": visibility_km,
                "is_holiday": is_holiday,
                "incident_flag": incident_flags,
                "vehicle_type": [route_vehicle_map[r] for r in rids],
            })
            chunks.append(chunk)
            total_so_far += n_ev

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Data quality injection
# ---------------------------------------------------------------------------

def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)

    def sample_idx(frac: float) -> np.ndarray:
        k = max(1, int(n * frac))
        return rng.choice(n, size=k, replace=False)

    # 0.3% exact duplicates
    dup_idx = sample_idx(0.003)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    n2 = len(df)

    def sample_idx2(frac: float) -> np.ndarray:
        k = max(1, int(n2 * frac))
        return rng.choice(n2, size=k, replace=False)

    # 0.15% negative delays
    neg_idx = sample_idx2(0.0015)
    df.loc[neg_idx, "delay_minutes"] = -df.loc[neg_idx, "delay_minutes"].abs()

    # 0.05% future timestamps (add 90+ days)
    fut_idx = sample_idx2(0.0005)
    df.loc[fut_idx, "timestamp"] = df.loc[fut_idx, "timestamp"] + pd.Timedelta(days=91)

    # 1.8% precip NaN
    precip_nan_idx = sample_idx2(0.018)
    df.loc[precip_nan_idx, "precip_mm"] = np.nan

    # 1.2% passenger_demand NaN
    demand_nan_idx = sample_idx2(0.012)
    df.loc[demand_nan_idx, "passenger_demand"] = np.nan

    # 0.1% outlier delays (120+)
    out_idx = sample_idx2(0.001)
    df.loc[out_idx, "delay_minutes"] = rng.uniform(120, 180, len(out_idx))

    # 0.5% corrupt station_id — store whole column as str to allow mixed values
    corrupt_idx = sample_idx2(0.005)
    df["station_id"] = df["station_id"].astype(str)
    df.loc[corrupt_idx, "station_id"] = df.loc[corrupt_idx, "station_id"].str.lower()

    # Exactly 3 NaT timestamps
    nat_idx = rng.choice(n2, size=3, replace=False)
    df.loc[nat_idx, "timestamp"] = pd.NaT

    return df


# ---------------------------------------------------------------------------
# Weather hourly aggregation
# ---------------------------------------------------------------------------

def _build_weather_hourly(df: pd.DataFrame) -> pd.DataFrame:
    clean_ts = df["timestamp"].dropna()
    valid_mask = df["timestamp"].notna() & (df["timestamp"] < pd.Timestamp("2025-04-01"))
    wdf = df[valid_mask].copy()
    wdf["hour_floor"] = wdf["timestamp"].dt.floor("h")
    return (
        wdf.groupby(["station_id", "hour_floor"])
        .agg(
            mean_temp_c=("temp_c", "mean"),
            mean_precip_mm=("precip_mm", "mean"),
            mean_wind_kph=("wind_kph", "mean"),
            mean_visibility_km=("visibility_km", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Generation log
# ---------------------------------------------------------------------------

def _write_log(output_dir: Path, params: dict) -> None:
    log_path = output_dir / "data" / "raw" / "GENERATION_LOG.md"
    lines = [
        "# TransitRisk Data Generation Log\n",
        f"**START_DATE**: {params['START_DATE']}  ",
        f"**RANDOM_SEED**: {params['RANDOM_SEED']}  ",
        f"**N_STATIONS**: {params['N_STATIONS']}  ",
        f"**N_ROUTES**: {params['N_ROUTES']}  ",
        f"**N_DAYS**: {params['N_DAYS']}  ",
        f"**TARGET_EVENTS**: {params['TARGET_EVENTS']}  ",
        f"**ACTUAL_EVENTS** (before DQ inject): {params['actual_before_dq']}  ",
        f"**FINAL_ROWS** (after DQ inject): {params['final_rows']}  ",
        f"**N_STATION_ROUTE_PAIRS**: {params['n_sr_pairs']}  ",
        "",
        "## Data Quality Injections",
        "| Issue | Fraction | Approx Count |",
        "|-------|----------|--------------|",
        "| Exact duplicates | 0.3% | ~3744 |",
        "| Negative delays | 0.15% | ~1872 |",
        "| Future timestamps | 0.05% | ~624 |",
        "| Precip NaN | 1.8% | ~22461 |",
        "| Demand NaN | 1.2% | ~14974 |",
        "| Outlier delays (120+) | 0.1% | ~1248 |",
        "| Corrupt station_id | 0.5% | ~6239 |",
        "| NaT timestamps | exact | 3 |",
    ]
    log_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    station_df = _build_station_table(rng)
    route_df = _build_route_table(rng)
    sr_map = _build_station_route_map(rng)

    print(f"Generating {TARGET_EVENTS:,} transit events...")
    events_df = _generate_events_fast(station_df, route_df, sr_map, rng)
    actual_before_dq = len(events_df)
    print(f"Generated {actual_before_dq:,} events. Injecting data quality issues...")

    events_df = _inject_data_quality_issues(events_df, rng)
    print(f"Final row count: {len(events_df):,}. Saving...")

    events_df.to_parquet(raw_dir / "transit_events.parquet", index=False)

    weather_hourly = _build_weather_hourly(events_df)
    weather_hourly.to_parquet(raw_dir / "weather_hourly.parquet", index=False)

    station_df.to_csv(raw_dir / "stations.csv", index=False)
    route_df.to_csv(raw_dir / "routes.csv", index=False)
    sr_map.to_csv(raw_dir / "station_route_map.csv", index=False)

    _write_log(output_dir, {
        "START_DATE": START_DATE,
        "RANDOM_SEED": RANDOM_SEED,
        "N_STATIONS": N_STATIONS,
        "N_ROUTES": N_ROUTES,
        "N_DAYS": N_DAYS,
        "TARGET_EVENTS": TARGET_EVENTS,
        "actual_before_dq": actual_before_dq,
        "final_rows": len(events_df),
        "n_sr_pairs": len(sr_map),
    })

    print(f"Done. Files saved to {raw_dir}")
