from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(y)
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba)),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_all_models(
    models_dict: dict[str, Any],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for name, model in models_dict.items():
        metrics = compute_all_metrics(model, X_test, y_test, threshold=threshold)
        metrics["model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("model")
    return df


def slice_metrics(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    strata_series: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y = np.asarray(y_test)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    strata = strata_series.values if hasattr(strata_series, "values") else np.asarray(strata_series)

    rows = []
    for val in np.unique(strata):
        mask = strata == val
        ys, ps, yps = y[mask], proba[mask], y_pred[mask]

        if len(np.unique(ys)) < 2:
            pr_auc = float("nan")
            roc_auc = float("nan")
        else:
            pr_auc = float(average_precision_score(ys, ps))
            roc_auc = float(roc_auc_score(ys, ps))

        rows.append({
            "stratum": val,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1": float(f1_score(ys, yps, zero_division=0)),
            "n_samples": int(mask.sum()),
            "positive_rate": float(ys.mean()),
        })

    return pd.DataFrame(rows)


def regression_benchmark(
    X_train: np.ndarray | pd.DataFrame,
    y_train_continuous: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test_continuous: np.ndarray | pd.Series,
) -> dict[str, float]:
    y_tr = np.asarray(y_train_continuous)
    y_te = np.asarray(y_test_continuous)

    lr = LinearRegression()
    lr.fit(X_train, y_tr)
    lr_r2 = float(r2_score(y_te, lr.predict(X_test)))

    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb.fit(X_train, y_tr)
    hgb_r2 = float(r2_score(y_te, hgb.predict(X_test)))

    return {"linear_regression_r2": lr_r2, "hgb_regression_r2": hgb_r2}


def persistence_baseline(
    df_test: pd.DataFrame,
    target_col: str = "y_primary",
) -> dict[str, float]:
    # Predict next-hour high-delay if current share_delayed_5_current > 0.3
    y_true = df_test[target_col].values
    y_pred = (df_test["share_delayed_5_current"] > 0.3).astype(int).values

    if len(np.unique(y_true)) < 2:
        roc = float("nan")
    else:
        roc = float(roc_auc_score(y_true, y_pred))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc,
    }
