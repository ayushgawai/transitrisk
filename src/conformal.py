from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_nonconformity_scores(
    model: Any,
    X_cal: np.ndarray | pd.DataFrame,
    y_cal: np.ndarray | pd.Series,
) -> np.ndarray:
    proba = model.predict_proba(X_cal)[:, 1]
    y = np.asarray(y_cal)
    scores = np.where(y == 1, 1 - proba, proba)
    return scores


def compute_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))


def predict_set(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    q_hat: float,
) -> pd.DataFrame:
    proba = model.predict_proba(X_test)[:, 1]
    include_1 = (1 - proba) <= q_hat
    include_0 = proba <= q_hat
    set_size = include_1.astype(int) + include_0.astype(int)

    return pd.DataFrame({
        "pred_1": include_1,
        "pred_0": include_0,
        "set_size": set_size,
        "proba": proba,
    })


def empirical_coverage(
    prediction_sets_df: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> float:
    y = np.asarray(y_test)
    in_set = np.where(
        y == 1,
        prediction_sets_df["pred_1"].values,
        prediction_sets_df["pred_0"].values,
    )
    return float(in_set.mean())


def coverage_by_stratum(
    prediction_sets_df: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    strata_series: pd.Series,
) -> pd.DataFrame:
    y = np.asarray(y_test)
    in_set = np.where(
        y == 1,
        prediction_sets_df["pred_1"].values,
        prediction_sets_df["pred_0"].values,
    )

    strata = strata_series.values if hasattr(strata_series, "values") else strata_series
    rows = []
    for val in np.unique(strata):
        mask = strata == val
        rows.append({
            "stratum": val,
            "coverage": float(in_set[mask].mean()),
            "mean_set_size": float(prediction_sets_df["set_size"].values[mask].mean()),
        })

    return pd.DataFrame(rows)
