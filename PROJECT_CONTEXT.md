# TransitRisk — Project Context

> This file is written for AI agents and collaborators who need full context fast.
> It covers every major decision made, why it was made, and what the project is actually doing.

---

## What This Project Is

A supervised machine learning system that predicts — **one hour in advance** — whether a specific transit route at a specific station will experience elevated delay conditions. The output drives a binary operational decision: **DISPATCH ALERT** (pre-position spare bus, tighten headways, push passenger notification) or **HOLD** (no action needed).

This is a **final project for DATA 245: Machine Learning Technologies, Spring 2026**.

---

## The Problem Being Solved

Transit agencies are reactive. A route goes late at 8:15 AM. Dispatchers notice around 8:45. By then passengers are already delayed and the corrective bus takes another 20–30 minutes to arrive. The window where action is still useful has passed.

This system buys back that window. By scoring each route at the end of every hour, it gives operators ~60 minutes of lead time to act *before* the delay materialises.

---

## Why Classification, Not Regression

The obvious framing would be "predict next-hour mean delay in minutes." We tried it. Raw delay is dominated by noise — R² is near zero on this dataset. Individual trip delays are highly stochastic (traffic incidents, driver behavior, passenger boarding times). But the *probability that conditions will be broadly elevated* is learnable.

**Reframing as binary classification** — elevated (mean delay ≥ 5 min) or not — produced ROC-AUC 0.81. This is the core novelty of the problem formulation.

The 5-minute threshold was chosen because it is the standard service-quality threshold used in transit literature and corresponds to the point where passengers measurably change behavior (miss connections, complain, switch modes).

---

## Data

### Source
Fully synthetic, generated with controlled coefficients to mimic a real urban transit network. Synthetic data was chosen because real GTFS feeds are proprietary and inconsistent across agencies.

### Scale
- **1,247,832** individual trip departure events
- **90 days** of simulation (approx. one quarter)
- **60 stations**, **12 routes**, **3 vehicle types** (bus, light rail, train)
- Weather injected hourly (temperature, precipitation, wind, visibility)
- Deliberate data quality issues injected (duplicates, negative delays, missing values, corrupt station IDs) — then cleaned

### Why 153K rows for modeling, not 1.2M
The 1.2M rows are individual trip events. The model predicts at the **(station, route, hour)** granularity — one prediction per route per hour. So we aggregate: group by station × route × hour, compute mean delay, trip count, demand, etc. This produces **153,004 station-route-hour rows** — the actual modeling unit.

### Temporal Split
- **Train: 60%** (earliest dates)
- **Val: 20%** (middle)
- **Test: 20%** (most recent)

Strictly chronological — no shuffling. This is critical: any shuffle would leak future information into training via the lag features. Cross-validation uses `TimeSeriesSplit(n_splits=5)` for the same reason.

---

## Features (38 total)

### Why These Features

| Group | Count | Purpose |
|---|---|---|
| Time signals | 8 | Capture cyclic patterns — rush hour, weekend, month seasonality. Encoded as sin/cos so Monday and Sunday aren't numerically "far apart." |
| Current operational state | 6 | What is happening right now — current delay, variance, fraction already late, trip count, headway, demand |
| Delay history / lags | 10 | Is delay building or clearing? Lag 1h/2h/3h, rolling 6h mean/std, same hour yesterday, same hour last week |
| Weather | 7 | Rain, wind, visibility, temperature — plus lagged/rolling versions because precipitation impact is delayed |
| Interaction features | 3 | `peak_x_precip` (rain during rush hour is multiplicatively worse), `short_headway_x_demand`, `weekend_x_precip` |
| Spatial encoding | 4 | Target-encoded station/route IDs (historical delay risk per location), busyness quartiles |

### Leakage Prevention
Lag features are computed with `.shift(1)` **per (station, route) group**, not globally. A common mistake is shifting the entire dataframe, which would use the previous row regardless of whether it belongs to the same route. We have a dedicated unit test (`test_leakage.py`) that verifies no future data enters any feature.

---

## Target Variable

**`y_primary`**: Binary. `1` if the next hour's mean delay on this (station, route) exceeds 5 minutes, else `0`.

- Positive rate: **39.4%** — reasonably balanced, not a rare-event problem
- Consistent across train (39.3%) and test (39.4%) — temporal split did not skew the distribution
- Two secondary targets exist (`y_secondary` at 10 min, `y_tertiary` at 2 min) but were not used in final modeling

---

## Models Chosen and Why

Seven models across three families were compared. This was required by the course spec to demonstrate breadth.

| Model | Family | Why included |
|---|---|---|
| Naive Bayes | Linear/probabilistic | Baseline; fast; interpretable; known to underfit with correlated features |
| Logistic Regression | Linear | Strong linear baseline; well-calibrated out of the box; useful for comparison |
| SVM-RBF | Kernel | Non-linear decision boundary; O(n²) complexity so trained on stratified 30k subsample |
| kNN | Non-parametric | No assumptions; good at local patterns; slow at inference |
| Decision Tree | Tree | Interpretable; tends to overfit without pruning |
| Random Forest | Ensemble tree | Strong, robust, handles interactions well |
| XGBoost | Gradient boosted tree | Best performer; handles lag/interaction features naturally; fast inference |

### Why XGBoost Won
- Highest ROC-AUC (0.809) and PR-AUC (0.759) on held-out test set
- Lowest Brier score (0.169) — best probability calibration before post-processing
- Gradient boosting naturally captures the non-linear relationships between lag features and delay risk
- After isotonic regression calibration: all 5 calibration bins within 0.013 of perfect

### SVM Note
SVM-RBF was trained on a 30k-row stratified subsample due to O(n²) complexity. Testing showed AUC difference vs full data < 0.005. This is documented in `notes/decisions.md`.

---

## ML Techniques Used (Beyond Basic Classification)

### 1. Calibrated Probabilities
Raw XGBoost probabilities are not well-calibrated (the model knows which direction but not the exact magnitude). We apply **isotonic regression** post-processing. This ensures P=0.70 actually means ~70% of those cases have elevated delays. Required for the threshold to be operationally meaningful.

Verified: all 5 calibration quantile bins within 0.013 of actual positive rate.

### 2. Cost-Sensitive Threshold
Default threshold of 0.5 is suboptimal when misclassification costs are asymmetric. Here:
- **False Negative** (miss a real delay) → passengers stranded, no warning → cost **5 units**
- **False Positive** (unnecessary alert) → one spare bus repositioned → cost **1 unit**

We search for the threshold `t*` that minimises `(5×FN + 1×FP) / N`. Result: `t* = 0.163` instead of 0.5. This catches **95.2% of real delays** (vs ~70% at t=0.5) and produces **44.3% lower expected cost**.

### 3. Split Conformal Prediction
Provides distribution-free coverage guarantees. For any `α`, the prediction set is guaranteed to contain the true label with probability ≥ `1−α`, regardless of data distribution. No parametric assumptions.

Results: all three alpha levels confirmed on held-out test set:
- α=0.05 → actual coverage 95.5% (target 95%)
- α=0.10 → actual coverage 91.8% (target 90%)
- α=0.20 → actual coverage 80.5% (target 80%)

The prediction set gives three outcomes: {Low Risk only}, {High Risk only}, or {Low, High} (uncertain). This uncertainty quantification is useful in practice — operators know when the model is confident vs when the situation is ambiguous.

### 4. Stress-Stratified Evaluation
Standard aggregate AUC hides failures. We slice model performance by:
- **Weather:** clear / light rain / moderate rain / heavy rain
- **Time of day:** morning peak / midday / evening peak / late night
- **Demand level:** Q1–Q4 quartiles
- **Headway:** frequent / normal / sparse

This reveals whether the model degrades in its most important operating conditions (e.g., heavy rain during peak hours).

### 5. Temporal Leakage Audit
Dedicated test suite (`tests/test_leakage.py`) verifies:
- Lag features never include information from the current or future hour
- Target variable is always constructed from the *next* hour's data, not current
- No station-level information from the test period appears in training

---

## Dashboard — What Each Tab Shows

The Streamlit dashboard (`app/dashboard.py`) is the primary deliverable for demonstration. It runs at `http://localhost:8501`.

| Tab | What it shows | Key use case |
|---|---|---|
| **📡 Risk Panel** | Select a station → routes auto-filter to only those served by that station → gauge shows current risk probability → bar chart compares all routes at once → 24h history with actual elevated-hour markers | "Which routes at Station 15 are high risk right now?" |
| **🔧 What-If** | Sliders for precipitation, wind, demand multiplier, headway → risk comparison before/after | "What happens to risk if it rains 15mm during morning peak?" |
| **💰 Cost Tuner** | Drag C(FN)/C(FP) ratio → cost curve updates → optimal threshold shifts → confusion matrix updates | "If missing a delay costs 8× more than a false alarm, what threshold should we use?" |
| **🌡 Stress Explorer** | Pick a slice axis → table of AUC/PR-AUC/F1 per stratum → example TP/FP/FN predictions | "Does the model hold up in heavy rain? Late night?" |
| **🔍 SHAP** | Pick any test row → see which features drove that prediction (Z-score fallback if SHAP not pre-computed) | "Why did the model predict high risk for this specific hour?" |
| **🔴 Live Feed** | Replay test set row-by-row as if data is arriving live → gauge animates → ALERT/HOLD decision per row → running Precision/Recall/F1 | Demonstrates the complete inference pipeline end-to-end |

### Important Dashboard Note — Risk Panel
Not every station serves every route (there are 176 valid station-route combinations out of a possible 720). The route dropdown **cascades based on the selected station** — it only shows routes that actually serve that station. This was a bug in an earlier version where selecting station 1 + route R01 returned zero rows because that combination doesn't exist.

---

## What's Novel (For Academic Justification)

1. **Problem reformulation** — binary classification over regression. Regression on raw delays gives R²≈0. Binary threshold classification at AUC 0.81 is tractable and actionable.

2. **Asymmetric cost framework** — the threshold is not 0.5. It is derived from a real-world cost ratio. This is a deliberate design choice, not a hyperparameter to tune by feel.

3. **Distribution-free uncertainty quantification** — conformal prediction sets with provable coverage guarantees. Most applied ML papers use confidence intervals from model outputs, which have no formal guarantees.

4. **Temporal integrity** — strict temporal split + per-group lag computation + unit-tested leakage prevention. Many time-series ML papers have subtle data leakage that inflates reported metrics. We audit and prove this doesn't happen here.

5. **Stress-stratified evaluation** — going beyond aggregate AUC to understand conditional performance. A model that scores 0.81 overall but 0.65 in heavy rain is not actually useful for transit operations.

---

## Numbers to Know

| Metric | Value |
|---|---|
| Best model | XGBoost Calibrated |
| ROC-AUC (test) | 0.809 |
| PR-AUC (test) | 0.759 |
| Brier score | 0.169 |
| Cost-optimal threshold | 0.163 |
| Recall at t=0.163 | 95.2% |
| Expected cost reduction vs t=0.5 | 44.3% |
| Conformal coverage (α=0.10) | 91.8% (target ≥90%) |
| Training rows | 91,879 |
| Validation rows | 30,501 |
| Test rows | 30,624 |
| Features | 38 |
| Unit tests | 15 (all passing) |

---

## Repository Structure

```
transitrisk/
├── src/               # Core ML pipeline (11 modules)
├── notebooks/         # 11 Jupyter notebooks (01 data → 11 figures)
├── app/               # Streamlit dashboard + 6 tab components
├── data/
│   ├── raw/           # Small CSVs on GitHub; large parquets on GDrive
│   └── processed/     # Indices, thresholds, prediction sets on GitHub; features on GDrive
├── models/            # Small models on GitHub; kNN/RF/SVM on GDrive
├── figures/           # All 6 report figures (on GitHub)
├── tests/             # 4 test files, 15 tests
├── notes/             # decisions.md, leakage_audit.md, hyperparameter_grids.md
├── report/            # IEEE LaTeX skeleton
├── Makefile           # All pipeline steps with existence guards (won't retrain if files exist)
├── requirements.txt
├── README.md          # Setup instructions including GDrive quick start
└── PROJECT_CONTEXT.md # ← this file
```

---

## Key Decisions Log

| Decision | Reason |
|---|---|
| Binary target at 5-min threshold | R²≈0 for regression; binary at AUC 0.81 is tractable |
| Temporal split, no shuffle | Lag features would leak future data if rows were shuffled |
| Cost ratio C(FN)=5, C(FP)=1 | Missed delay alert is 5× more damaging than unnecessary alert |
| SVM on 30k subsample | O(n²) complexity; AUC difference < 0.005 vs full data |
| Isotonic calibration over Platt | Isotonic is more flexible for non-sigmoid distortions; XGBoost outputs tend to be overconfident near extremes |
| Conformal on validation set | Calibration and conformal use the same held-out val set; test set is never touched until final evaluation |
| Cascade dropdowns in Risk Panel | 176 valid station-route combos out of 720 possible; independent dropdowns caused silent empty results |
