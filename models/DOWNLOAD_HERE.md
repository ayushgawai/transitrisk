# Download Required Model Files Here

Download these three files from the shared Google Drive link and place them in this folder (`models/`):

| File | Size | Model |
|---|---|---|
| `knn.joblib` | 27 MB | k-Nearest Neighbours |
| `rf.joblib` | 9.8 MB | Random Forest |
| `svm_rbf.joblib` | 6 MB | SVM-RBF |

**These models are already here (from GitHub, no download needed):**
- `nb.joblib` — Naive Bayes
- `logreg.joblib` — Logistic Regression
- `dt.joblib` — Decision Tree
- `xgb.joblib` — XGBoost (uncalibrated)
- `xgb_calibrated.joblib` — XGBoost Calibrated ← **main model used in dashboard**

The dashboard works with just `xgb_calibrated.joblib`. The other models are needed for the full model comparison in the Stress Explorer tab.
