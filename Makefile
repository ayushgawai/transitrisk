.PHONY: all data clean features models eval dashboard tests help

PYTHON := python
NB_EXECUTE := jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600

help:
	@echo "TransitRisk — available targets:"
	@echo "  make data       — Generate synthetic dataset (notebook 01)"
	@echo "  make clean      — Clean and validate dataset (notebook 02)"
	@echo "  make eda        — EDA + leakage audit (notebook 03)"
	@echo "  make features   — Feature engineering (notebook 04)"
	@echo "  make baselines  — Train baseline models (notebook 05)"
	@echo "  make advanced   — Train RF, XGBoost, SVM (notebook 06)"
	@echo "  make models     — Train all models (baselines + advanced)"
	@echo "  make calibrate  — Calibration + threshold analysis (notebook 07)"
	@echo "  make conformal  — Split conformal prediction (notebook 08)"
	@echo "  make interpret  — SHAP + PDP + permutation (notebook 09)"
	@echo "  make stress     — Stress-stratified evaluation (notebook 10)"
	@echo "  make report     — Final figures for report (notebook 11)"
	@echo "  make eval       — Run notebooks 07-11"
	@echo "  make all        — Run all notebooks end-to-end"
	@echo "  make dashboard  — Launch Streamlit dashboard"
	@echo "  make tests      — Run unit tests"
	@echo "  make install    — Install dependencies"

install:
	pip install -r requirements.txt

data:
	$(NB_EXECUTE) notebooks/01_data_generation.ipynb

clean_data:
	$(NB_EXECUTE) notebooks/02_data_cleaning.ipynb

eda:
	$(NB_EXECUTE) notebooks/03_eda_and_audit.ipynb

features:
	$(NB_EXECUTE) notebooks/04_feature_engineering.ipynb

baselines:
	$(NB_EXECUTE) notebooks/05_modeling_baselines.ipynb

advanced:
	$(NB_EXECUTE) notebooks/06_modeling_advanced.ipynb

models: baselines advanced

calibrate:
	$(NB_EXECUTE) notebooks/07_calibration_and_threshold.ipynb

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

dashboard:
	streamlit run app/dashboard.py

tests:
	$(PYTHON) -m pytest tests/ -v

dirs:
	mkdir -p data/raw data/processed models figures
