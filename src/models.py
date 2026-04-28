from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_model_configs() -> dict[str, tuple[Any, dict[str, Any]]]:
    configs: dict[str, tuple[Any, dict[str, Any]]] = {}

    # Logistic Regression
    configs["logreg"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1)),
        ]),
        {
            "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["saga"],
            "clf__class_weight": [None, "balanced"],
        },
    )

    # Naive Bayes
    configs["nb"] = (
        GaussianNB(),
        {
            "var_smoothing": np.logspace(-12, -6, 20),
        },
    )

    # KNN
    configs["knn"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_jobs=-1)),
        ]),
        {
            "clf__n_neighbors": [5, 10, 15, 25, 50],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    )

    # Decision Tree
    configs["dt"] = (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 5, 10, 20],
            "class_weight": [None, "balanced"],
        },
    )

    # Random Forest
    configs["rf"] = (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
        },
    )

    # XGBoost
    configs["xgb"] = (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
            "scale_pos_weight": [1, 2, 3, 5],
        },
    )

    # SVM RBF
    configs["svm_rbf"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE, kernel="rbf")),
        ]),
        {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "clf__class_weight": [None, "balanced"],
        },
    )

    return configs


def tune_model(
    name: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cv: Any,
    n_iter: int = 25,
    scoring: str = "average_precision",
) -> Any:
    configs = get_model_configs()
    estimator, param_grid = configs[name]

    search = RandomizedSearchCV(
        estimator,
        param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def fit_svm_subsample(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    param_grid: dict[str, Any],
    cv: Any,
    n_rows: int = 30_000,
) -> Any:
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(n_rows, len(y_train)), random_state=RANDOM_STATE)
    sub_idx, _ = next(sss.split(X_train, y_train))

    X_sub = X_train.iloc[sub_idx] if hasattr(X_train, "iloc") else X_train[sub_idx]
    y_sub = y_train.iloc[sub_idx] if hasattr(y_train, "iloc") else y_train[sub_idx]

    configs = get_model_configs()
    estimator, _ = configs["svm_rbf"]

    search = RandomizedSearchCV(
        estimator,
        param_grid,
        n_iter=25,
        cv=cv,
        scoring="average_precision",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_sub, y_sub)
    return search.best_estimator_


def save_model(model: Any, name: str, models_dir: str | Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"{name}.joblib")


def load_model(name: str, models_dir: str | Path) -> Any:
    return joblib.load(Path(models_dir) / f"{name}.joblib")
