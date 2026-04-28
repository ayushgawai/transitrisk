"""Tab 4 — Stress Slice Explorer."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


_STRATA_DEFS = {
    "Weather": {
        "column": "mean_precip_mm",
        "bins": [("clear", lambda x: x < 0.5),
                 ("light_rain", lambda x: (x >= 0.5) & (x < 2.0)),
                 ("moderate_rain", lambda x: (x >= 2.0) & (x < 5.0)),
                 ("heavy_rain", lambda x: x >= 5.0)],
    },
    "Time of Day": {
        "column": "hour_floor",
        "bins": [("morning_peak", lambda x: x.dt.hour.isin([7, 8, 9])),
                 ("midday", lambda x: x.dt.hour.isin(list(range(10, 17)))),
                 ("evening_peak", lambda x: x.dt.hour.isin([17, 18, 19])),
                 ("late_night", lambda x: x.dt.hour.isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]))],
    },
    "Demand Level": {
        "column": "mean_demand_current",
        "quartile": True,
    },
    "Headway": {
        "column": "mean_headway_current",
        "bins": [("frequent (<5)", lambda x: x < 5.0),
                 ("normal (5-15)", lambda x: (x >= 5.0) & (x < 15.0)),
                 ("sparse (>15)", lambda x: x >= 15.0)],
    },
}


def render_stress_explorer(model, X_test: pd.DataFrame, y_test: np.ndarray,
                           modeling_test: pd.DataFrame, t_cost: float = 0.31) -> None:
    axis = st.selectbox("Slice by", list(_STRATA_DEFS.keys()))
    defn = _STRATA_DEFS[axis]
    col  = defn["column"]

    if "quartile" in defn:
        col_vals = modeling_test[col]
        q_labels = pd.qcut(col_vals, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        strata_list = [(lbl, (q_labels == lbl).values) for lbl in q_labels.cat.categories]
    else:
        src = modeling_test[col] if col != "hour_floor" else pd.to_datetime(modeling_test["hour_floor"])
        strata_list = [(name, fn(src).values) for name, fn in defn["bins"]]

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= t_cost).astype(int)

    rows = []
    for stratum_name, mask in strata_list:
        if mask.sum() < 10 or y_test[mask].sum() == 0:
            continue
        rows.append({
            "Stratum": stratum_name,
            "N":       int(mask.sum()),
            "Pos rate": f"{y_test[mask].mean():.2%}",
            "ROC-AUC":  f"{roc_auc_score(y_test[mask], proba[mask]):.3f}",
            "PR-AUC":   f"{average_precision_score(y_test[mask], proba[mask]):.3f}",
            "F1":       f"{f1_score(y_test[mask], y_pred[mask], zero_division=0):.3f}",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Not enough data in some strata.")

    # Show 5 example predictions from the selected stratum
    selected_stratum = st.selectbox("Show examples from stratum",
                                    [r["Stratum"] for r in rows])
    if selected_stratum:
        mask_sel = next(m for n, m in strata_list if n == selected_stratum)
        idx = np.where(mask_sel)[0]
        # Find TP, FP, FN examples
        tp_idx = [i for i in idx if y_test[i] == 1 and y_pred[i] == 1][:2]
        fp_idx = [i for i in idx if y_test[i] == 0 and y_pred[i] == 1][:2]
        fn_idx = [i for i in idx if y_test[i] == 1 and y_pred[i] == 0][:2]
        examples = []
        for label, indices in [("TP", tp_idx), ("FP", fp_idx), ("FN", fn_idx)]:
            for i in indices:
                examples.append({
                    "Type": label, "P(risk)": f"{proba[i]:.3f}",
                    "True label": y_test[i], "Pred": y_pred[i],
                })
        if examples:
            st.markdown(f"**Sample predictions — {selected_stratum}**")
            st.dataframe(pd.DataFrame(examples), use_container_width=True)
