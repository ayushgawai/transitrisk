# Decisions Log

Any deviation from PROJECT_BUILD_SPEC.md is documented here with reason.

## Active decisions

| ID | Decision | Reason | Spec reference |
|----|----------|--------|----------------|
| D-001 | SVM trained on stratified 30k subsample | O(n²-n³) complexity; verified doubling to 60k changes AUC < 0.005 | Section 6.3 |
| D-002 | SHAP computed on 5,000-row subsample | TreeSHAP is fast but plotting 60k rows is visually noisy | Section 9.1 |
| D-003 | Temporal split at date boundaries, not row count | Avoids partial-day contamination | Section 5 |
| D-004 | Lag NaN at series boundaries filled with 0 | Only affects first 1-7 hours per station-route; negligible | Section 7.1 |

## No deviations from core parameters

- noise_std = 3.5 (data_gen.py) — DO NOT CHANGE without logging here
- C_FN = 5, C_FP = 1 (cost.py) — fixed per spec
- alpha = 0.1, 0.2 (conformal.py) — fixed per spec
- RANDOM_STATE = 42 — set everywhere
