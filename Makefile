.PHONY: all data clean_data eda features models baselines advanced \
        calibrate conformal interpret stress report_figs eval \
        dashboard tests install dirs help

PYTHON     := python
NB_EXECUTE := jupyter nbconvert --to notebook --execute --inplace \
              --ExecutePreprocessor.timeout=3600

# ── sentinel files (existence = step already done) ────────────────────────────
RAW_EVENTS   := data/raw/transit_events.parquet
CLEAN_EVENTS := data/processed/transit_events_cleaned.parquet
FEATURES     := data/processed/X_features.parquet
MODELING     := data/processed/modeling_table.parquet
MODEL_XGB    := models/xgb_calibrated.joblib
MODEL_RF     := models/rf.joblib
MODEL_KNN    := models/knn.joblib

# ──────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  TransitRisk — Make targets"
	@echo ""
	@echo "  FAST START (if you downloaded the GDrive package):"
	@echo "    make dashboard        — launch Streamlit immediately"
	@echo ""
	@echo "  FULL PIPELINE (runs only missing steps):"
	@echo "    make data             — generate synthetic events   (nb 01, ~5 min)"
	@echo "    make clean_data       — clean & validate            (nb 02, ~2 min)"
	@echo "    make eda              — EDA + leakage audit         (nb 03, ~2 min)"
	@echo "    make features         — 38-feature engineering      (nb 04, ~5 min)"
	@echo "    make models           — train all 7 models          (nb 05-06, ~60 min)"
	@echo "    make eval             — calibration, conformal, etc (nb 07-11, ~10 min)"
	@echo "    make all              — full pipeline end to end"
	@echo ""
	@echo "  OTHER:"
	@echo "    make tests            — run 15 unit tests"
	@echo "    make install          — pip install -r requirements.txt"
	@echo ""

# ── install ───────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── dirs ──────────────────────────────────────────────────────────────────────
dirs:
	mkdir -p data/raw data/processed models figures

# ── data generation (skip if raw events already exist) ────────────────────────
data:
	@if [ -f $(RAW_EVENTS) ]; then \
		echo "✓ $(RAW_EVENTS) already exists — skipping generation."; \
		echo "  Delete it and re-run to regenerate."; \
	else \
		echo "→ Generating synthetic transit events ..."; \
		$(NB_EXECUTE) notebooks/01_data_generation.ipynb; \
	fi

# ── cleaning (skip if cleaned file already exists) ────────────────────────────
clean_data:
	@if [ -f $(CLEAN_EVENTS) ]; then \
		echo "✓ $(CLEAN_EVENTS) already exists — skipping cleaning."; \
	else \
		echo "→ Cleaning dataset ..."; \
		$(NB_EXECUTE) notebooks/02_data_cleaning.ipynb; \
	fi

# ── EDA (always run — read-only analysis, no side effects) ───────────────────
eda:
	$(NB_EXECUTE) notebooks/03_eda_and_audit.ipynb

# ── feature engineering (skip if features already exist) ─────────────────────
features:
	@if [ -f $(FEATURES) ] && [ -f $(MODELING) ]; then \
		echo "✓ Feature matrix already exists — skipping engineering."; \
	else \
		echo "→ Engineering features ..."; \
		$(NB_EXECUTE) notebooks/04_feature_engineering.ipynb; \
	fi

# ── baseline models (skip if all baseline models already saved) ───────────────
baselines:
	@if [ -f models/nb.joblib ] && [ -f models/logreg.joblib ] && \
	    [ -f models/knn.joblib ] && [ -f models/dt.joblib ]; then \
		echo "✓ Baseline models already exist — skipping training."; \
		echo "  Delete models/*.joblib to retrain."; \
	else \
		echo "→ Training baseline models (NB, LogReg, kNN, DT) ..."; \
		$(NB_EXECUTE) notebooks/05_modeling_baselines.ipynb; \
	fi

# ── advanced models (skip if RF + XGBoost + SVM already saved) ───────────────
advanced:
	@if [ -f $(MODEL_RF) ] && [ -f models/xgb.joblib ] && [ -f models/svm_rbf.joblib ]; then \
		echo "✓ Advanced models already exist — skipping training."; \
		echo "  Delete models/*.joblib to retrain."; \
	else \
		echo "→ Training advanced models (RF, XGBoost, SVM-RBF) — ~60 min ..."; \
		$(NB_EXECUTE) notebooks/06_modeling_advanced.ipynb; \
	fi

models: baselines advanced

# ── calibration (skip if calibrated model already saved) ─────────────────────
calibrate:
	@if [ -f $(MODEL_XGB) ]; then \
		echo "✓ Calibrated model already exists — skipping calibration."; \
	else \
		echo "→ Calibrating XGBoost ..."; \
		$(NB_EXECUTE) notebooks/07_calibration_and_threshold.ipynb; \
	fi

conformal:
	$(NB_EXECUTE) notebooks/08_conformal_prediction.ipynb

interpret:
	$(NB_EXECUTE) notebooks/09_interpretability.ipynb

stress:
	$(NB_EXECUTE) notebooks/10_stress_stratified_eval.ipynb

report_figs:
	$(NB_EXECUTE) notebooks/11_final_results_and_figures.ipynb

eval: calibrate conformal interpret stress report_figs

all: data clean_data eda features models eval

# ── dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	streamlit run app/dashboard.py

# ── tests ─────────────────────────────────────────────────────────────────────
tests:
	$(PYTHON) -m pytest tests/ -v
