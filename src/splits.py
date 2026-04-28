from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "hour_floor",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    unique_dates = df[date_col].sort_values().unique()
    n = len(unique_dates)

    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])

    train_mask = df[date_col].isin(train_dates)
    val_mask = df[date_col].isin(val_dates)
    test_mask = df[date_col].isin(test_dates)

    return train_mask, val_mask, test_mask


def save_indices(
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": train_mask[train_mask].index.tolist(),
        "val": val_mask[val_mask].index.tolist(),
        "test": test_mask[test_mask].index.tolist(),
    }
    output_path.write_text(json.dumps(payload))


def load_indices(path: str | Path) -> tuple[list[int], list[int], list[int]]:
    data = json.loads(Path(path).read_text())
    return data["train"], data["val"], data["test"]


def get_timeseries_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)
