# TransitRisk

**Cost-Sensitive, Calibrated Next-Hour Delay Risk Forecasting for Urban Transit**  
DATA 245 Machine Learning Technologies · Spring 2026

---

## What This Project Does

Every hour, for each station-route pair in a transit network, this system predicts:  
**"Will the next hour have elevated delay conditions?"**

A prediction of **ELEVATED** triggers a dispatch alert — operators can pre-position spare buses, tighten headways, and push passenger notifications *before* delays happen.

The model is **XGBoost (calibrated)**, chosen from a comparison of 7 models. It outputs a probability that is passed through a cost-optimal threshold (t=0.163) biased toward catching delays (a missed delay costs 5× more than a false alarm).

---

## Quick Start — For Friends (Recommended)

> You will receive a zip file (`transitrisk_data_models.zip`) with the complete `data/` and `models/` folders. No training needed.

```bash
# 1. Clone the repo
git clone https://github.com/ayushgawai/transitrisk.git
cd transitrisk

# 2. Install dependencies
pip install -r requirements.txt

# 3. Unzip the shared file and replace the data/ and models/ folders
#    (drag and drop, or run:)
unzip transitrisk_data_models.zip -d .
#    This overwrites data/ and models/ with the complete files.

# 4. Launch — no training required
streamlit run app/dashboard.py
```

The dashboard opens at **http://localhost:8501**. Done.

---

## Full Pipeline — From Scratch (Takes ~60–90 min)

```bash
pip install -r requirements.txt
make all        # generates data → cleans → features → trains 7 models → evaluates
streamlit run app/dashboard.py
```

---

## Run Steps Individually

```bash
make data        # Generate 1.2M synthetic transit events (notebook 01)
make clean       # Clean & validate dataset (notebook 02)
make eda         # EDA + leakage audit (notebook 03)
make features    # Feature engineering → 38 features (notebook 04)
make baselines   # Train NB, LogReg, kNN, DT (notebook 05)
make advanced    # Train RF, XGBoost, SVM-RBF (notebook 06)
make calibrate   # Calibrate XGBoost, compute thresholds (notebook 07)
make conformal   # Conformal prediction sets (notebook 08)
make interp      # SHAP + PDP interpretability (notebook 09)
make stress      # Stress-stratified evaluation (notebook 10)
make report      # Final figures + results (notebook 11)
make tests       # Run all unit tests
make dashboard   # Launch Streamlit app
```

---

## What the Model Predicts

**Target variable (`y_primary`):**  
Binary — will the **next hour's** mean delay exceed 5 minutes on this station-route?  
- `1` = elevated risk (39.4% of all hourly observations)  
- `0` = normal / low risk

**Prediction unit:** one (station, route, hour) tuple — e.g., Station 12 / Route R04 / 8 AM.

---

## Input Features (38 total)

| Group | Features |
|---|---|
| **Time signals** | hour sin/cos, day-of-week sin/cos, is_weekend, is_peak_morning, is_peak_evening, month |
| **Current state** | mean_delay_current, std_delay_current, share_delayed_5_current, trip_count, mean_headway_current, mean_demand_current |
| **Delay history** | lag_1h/2h/3h_mean_delay, lag_1h/3h_share_delayed_5, rolling_6h_mean/std_delay, same_hour_yesterday, same_hour_last_week, lag_1h_trip_count |
| **Weather** | mean_temp_c, mean_precip_mm, mean_wind_kph, mean_visibility_km, precip_lag_1h, precip_rolling_3h_sum, wind_lag_1h |
| **Interactions** | peak_x_precip (rain during rush hour), short_headway_x_demand, weekend_x_precip |
| **Spatial** | station_id_target_encoded, route_id_target_encoded, station_busyness_quartile, station_route_busyness_quartile |

---

## Model Results (held-out test set, ~30k rows)

| Model | Family | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|---|
| Naive Bayes | Linear | 0.708 | 0.574 | 0.489 | 0.261 |
| Logistic Regression | Linear | 0.760 | 0.676 | 0.600 | 0.192 |
| SVM-RBF | Kernel | 0.753 | 0.699 | 0.585 | 0.192 |
| k-NN | Nonparametric | 0.776 | 0.719 | 0.584 | 0.185 |
| Decision Tree | Tree | 0.785 | 0.725 | 0.592 | 0.178 |
| Random Forest | Tree | 0.806 | 0.762 | 0.669 | 0.177 |
| **XGBoost Calibrated** ✓ | **Tree** | **0.810** | **0.766** | **0.643** | **0.169** |

**Thresholds:** cost-optimal t=0.163 · F1-optimal t=0.345 · default t=0.500  
**Conformal coverage:** 92.3% (target ≥ 90%, guaranteed by split conformal prediction)

---

## Dashboard Guide

Open **http://localhost:8501** after `streamlit run app/dashboard.py`.  
The left sidebar contains a **Tab Guide** — select any tab to read how to use it.

| Tab | What it shows |
|---|---|
| **📡 Risk Panel** | Select a station → routes auto-filter to those that serve it → gauge shows next-hour risk, bar chart compares all routes at that station |
| **🔧 What-If** | Adjust weather/demand sliders → see how risk changes from the baseline |
| **💰 Cost Tuner** | Change FN/FP cost ratio → watch optimal threshold and confusion matrix update live |
| **🌡 Stress Explorer** | Slice model performance by weather / time-of-day / demand / headway — does the model hold up in bad conditions? |
| **🔍 SHAP** | Pick any test row → see which features drove that specific prediction |
| **🔴 Live Feed** | Replay the test set row-by-row as if data is arriving live — watch XGBoost score each prediction, see ALERT/HOLD decisions and running metrics |

---

## Repository Structure

```
transitrisk/
├── src/               # All ML pipeline modules
│   ├── data_gen.py    # Synthetic data generator (1.2M events)
│   ├── cleaning.py    # ETL: dedup, imputation, normalization
│   ├── targets.py     # Aggregate to station-route-hour, build y_primary
│   ├── features.py    # 38-feature engineering pipeline
│   ├── splits.py      # Temporal 60/20/20 split
│   ├── models.py      # 7 model configs + RandomizedSearchCV
│   ├── calibration.py # Platt/isotonic calibration
│   ├── conformal.py   # Split conformal prediction
│   ├── cost.py        # Cost-sensitive threshold search
│   ├── evaluation.py  # Metric computation + stress slicing
│   └── plots.py       # Publication-quality figure utilities
├── notebooks/         # 01_data_generation → 11_final_results
├── app/
│   ├── dashboard.py   # Streamlit main app
│   └── components/    # One file per tab
├── data/
│   ├── raw/           # Generated events, weather, stations, routes
│   └── processed/     # Feature matrix, splits, thresholds, prediction sets
├── models/            # Saved .joblib files for all 7 models
├── figures/           # Generated PNG charts + metrics.json
├── tests/             # Unit tests (15 tests, all passing)
├── notes/             # decisions.md, leakage_audit.md, hyperparameter_grids.md
├── report/            # IEEE-format LaTeX report skeleton
├── Makefile
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v   # 15 tests — conformal coverage, features, leakage, target construction
```

All 15 tests pass.

---

## Key Design Decisions

- **Classification over regression** — raw delay is hard to predict (R²≈0); binary elevated/not-elevated achieves AUC 0.81
- **Strict temporal split** — no future data leaks into training; lag features are shift-computed per station-route group
- **Cost-sensitive threshold** — C(FN)=5, C(FP)=1; t=0.163 recovers 95% of true positives at the cost of some false alarms
- **Calibrated probabilities** — XGBoost outputs calibrated via isotonic regression (Brier 0.169)
- **Conformal coverage guarantee** — split conformal prediction provides distribution-free 90% marginal coverage
- **SVM subsampled** — SVM-RBF trained on stratified 30k-row subset (O(n²) complexity); AUC difference vs full data < 0.005
