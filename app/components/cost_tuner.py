"""Tab 3 — Cost-Threshold Tuner."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix


def render_cost_tuner(model, X_test: pd.DataFrame, y_test: np.ndarray,
                      t_f1: float = 0.42, t_cost_default: float = 0.31) -> None:
    st.markdown("Adjust the cost ratio to see how the optimal threshold and expected cost change.")

    cost_ratio = st.slider("C(FN) / C(FP) — cost of missing a delay vs false alarm",
                           min_value=1.0, max_value=10.0, value=5.0, step=0.5)

    proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold for selected ratio
    thresholds = np.linspace(0.01, 0.99, 300)
    costs = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        costs.append((cost_ratio * fn + 1 * fp) / len(y_test))

    t_star = float(thresholds[np.argmin(costs)])
    min_cost = float(np.min(costs))

    col1, col2, col3 = st.columns(3)
    col1.metric("Cost-optimal threshold", f"{t_star:.3f}")
    col2.metric("F1-optimal threshold",   f"{t_f1:.3f}")
    col3.metric("Expected cost/pred",      f"{min_cost:.4f}")

    st.markdown(f"*At C(FN)/C(FP) = {cost_ratio:.1f}: cost-optimal **{t_star:.3f}** vs "
                f"F1-optimal **{t_f1:.3f}** vs default **0.500***")

    # Cost curve plot
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines",
                                   line=dict(color="#2563eb", width=2),
                                   name="Expected cost"))
    fig_curve.add_vline(x=t_star, line_dash="dash", line_color="#d97706",
                        annotation_text=f"t*={t_star:.2f}", annotation_position="top right")
    fig_curve.add_vline(x=t_f1,   line_dash="dot",  line_color="#16a34a",
                        annotation_text=f"t_F1={t_f1:.2f}", annotation_position="top left")
    fig_curve.add_vline(x=0.5,    line_dash="dot",  line_color="#6b7280",
                        annotation_text="0.50", annotation_position="bottom right")
    fig_curve.update_layout(
        xaxis_title="Threshold", yaxis_title="Expected cost per prediction",
        height=280, margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig_curve.update_xaxes(showgrid=False)
    fig_curve.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)")
    st.plotly_chart(fig_curve, use_container_width=True, key="ct_curve")

    # Confusion matrix at cost-optimal threshold
    st.markdown("#### Confusion Matrix at Cost-Optimal Threshold")
    y_pred_cost = (proba >= t_star).astype(int)
    cm = confusion_matrix(y_test, y_pred_cost)

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred: Low", "Pred: High"],
        y=["True: Low", "True: High"],
        text=cm, texttemplate="%{text}",
        colorscale=[[0, "#eff6ff"], [1, "#2563eb"]],
        showscale=False,
    ))
    fig_cm.update_layout(
        height=260, margin=dict(t=10, b=40, l=80, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True, key="ct_cm")

    tn, fp_n, fn_n, tp_n = cm.ravel()
    cols = st.columns(4)
    cols[0].metric("True Pos",  tp_n)
    cols[1].metric("True Neg",  tn)
    cols[2].metric("False Pos", fp_n)
    cols[3].metric("False Neg", fn_n, delta=f"Cost×{cost_ratio:.0f}",
                   delta_color="inverse")
