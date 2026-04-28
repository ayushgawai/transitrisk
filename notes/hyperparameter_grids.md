# Hyperparameter Search Grids

All models use `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`, 25 iterations, scoring=`average_precision` (PR-AUC), `random_state=42`.

## Logistic Regression
```python
{
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],  # required for l1
    'max_iter': [1000],
}
```

## Gaussian Naive Bayes
```python
{
    'var_smoothing': np.logspace(-9, -3, 13)
}
```

## k-Nearest Neighbors
```python
{
    'n_neighbors': [5, 10, 20, 50],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}
```

## Decision Tree
```python
{
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 10, 50],
    'min_samples_leaf': [1, 5, 20],
    'criterion': ['gini', 'entropy'],
}
```

## Random Forest
```python
{
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 0.5],
    'min_samples_leaf': [1, 5],
}
```

## XGBoost
```python
{
    'n_estimators': [300, 500, 800],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'min_child_weight': [1, 5],
}
```

## SVM (RBF kernel) — trained on 30k subsample
```python
{
    'C': [0.5, 1.0, 5.0],
    'gamma': ['scale', 0.01, 0.1],
}
```
