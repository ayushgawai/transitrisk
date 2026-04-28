from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

C_FN = 5
C_FP = 1


def expected_cost(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    c_fn: int = C_FN,
    c_fp: int = C_FP,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return float((c_fn * fn + c_fp * fp) / len(y_true))


def find_cost_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    c_fn: int = C_FN,
    c_fp: int = C_FP,
    n_thresholds: int = 200,
) -> float:
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    costs = [expected_cost(y_true, y_proba, t, c_fn, c_fp) for t in thresholds]
    return float(thresholds[np.argmin(costs)])


def find_f1_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    y_true = np.asarray(y_true)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1s)])


def threshold_sensitivity(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    cost_ratios: list[int] | None = None,
) -> pd.DataFrame:
    if cost_ratios is None:
        cost_ratios = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    rows = []
    for ratio in cost_ratios:
        t = find_cost_optimal_threshold(y_true, y_proba, c_fn=ratio, c_fp=1)
        cost = expected_cost(y_true, y_proba, t, c_fn=ratio, c_fp=1)
        rows.append({"cost_ratio": ratio, "optimal_threshold": t, "expected_cost": cost})

    return pd.DataFrame(rows)


def evaluate_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float | list]:
    y_true = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }
