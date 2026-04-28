from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

RANDOM_STATE = 42


class _CalibratedWrapper:
    """Wraps a pre-fitted model with a calibrated probability layer."""
    def __init__(self, base_model, calibrator, classes_):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = classes_

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw.reshape(-1, 1))
        cal = np.clip(cal, 0, 1)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_model(
    model: Any,
    X_cal: np.ndarray | pd.DataFrame,
    y_cal: np.ndarray | pd.Series,
    method: str = "isotonic",
) -> Any:
    raw_proba = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    y_cal = np.asarray(y_cal)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_proba.ravel(), y_cal)
    else:
        from sklearn.linear_model import LogisticRegression
        cal = LogisticRegression(C=1.0, max_iter=1000)
        cal.fit(raw_proba, y_cal)
        # Wrap to expose predict as proba
        _lr = cal
        class _PlattWrap:
            def predict(self, X):
                return _lr.predict_proba(X)[:, 1]
        cal = _PlattWrap()
    return _CalibratedWrapper(model, cal, np.array([0, 1]))


def brier_score(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> float:
    proba = model.predict_proba(X)[:, 1]
    return float(brier_score_loss(y, proba))


def reliability_data(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)[:, 1]
    fraction_positives, mean_predicted_probs = calibration_curve(y, proba, n_bins=n_bins)
    return mean_predicted_probs, fraction_positives


def compare_calibration(
    model: Any,
    X_cal: np.ndarray | pd.DataFrame,
    y_cal: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, Any]:
    uncal_brier = brier_score(model, X_test, y_test)

    platt_model = calibrate_model(model, X_cal, y_cal, method="sigmoid")
    platt_brier = brier_score(platt_model, X_test, y_test)

    isotonic_model = calibrate_model(model, X_cal, y_cal, method="isotonic")
    isotonic_brier = brier_score(isotonic_model, X_test, y_test)

    scores = {
        "sigmoid": platt_brier,
        "isotonic": isotonic_brier,
    }
    best_method = min(scores, key=scores.__getitem__)

    return {
        "uncalibrated_brier": uncal_brier,
        "platt_brier": platt_brier,
        "isotonic_brier": isotonic_brier,
        "best_method": best_method,
    }
