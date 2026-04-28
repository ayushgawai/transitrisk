"""Tab 2 — What-If Simulator."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_what_if(model, X_test: pd.DataFrame, feature_names: list[str],
                   explainer=None) -> None:
    st.markdown("Adjust weather and operational conditions to see how risk changes.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Simulate Conditions")
        precip     = st.slider("Precipitation (mm/h)", 0.0, 30.0, 2.0, step=0.5)
        wind       = st.slider("Wind speed (kph)", 0.0, 60.0, 15.0, step=1.0)
        demand_mul = st.slider("Demand multiplier", 0.5, 2.0, 1.0, step=0.1)
        headway    = st.slider("Headway (minutes)", 1.0, 30.0, 10.0, step=0.5)

    # Take a representative baseline row (median of test set)
    X_base = X_test.median().to_frame().T.copy()
    X_sim  = X_base.copy()

    if "mean_precip_mm" in feature_names:
        X_sim["mean_precip_mm"] = precip
        X_sim["precip_lag_1h"] = precip * 0.8
        X_sim["precip_rolling_3h_sum"] = precip * 2.5
    if "mean_wind_kph" in feature_names:
        X_sim["mean_wind_kph"] = wind
        X_sim["wind_lag_1h"] = wind * 0.9
    if "mean_demand_current" in feature_names:
        X_sim["mean_demand_current"] = X_base["mean_demand_current"].values[0] * demand_mul
    if "mean_headway_current" in feature_names:
        X_sim["mean_headway_current"] = headway

    # Interaction features
    if "peak_x_precip" in feature_names:
        is_peak = X_sim.get("is_peak_morning", pd.Series([0])).values[0] or X_sim.get("is_peak_evening", pd.Series([0])).values[0]
        X_sim["peak_x_precip"] = int(is_peak) * precip
    if "weekend_x_precip" in feature_names:
        is_we = X_sim.get("is_weekend", pd.Series([0])).values[0]
        X_sim["weekend_x_precip"] = int(is_we) * precip

    p_base = float(model.predict_proba(X_base[feature_names])[:, 1][0])
    p_sim  = float(model.predict_proba(X_sim[feature_names])[:, 1][0])

    with col_right:
        st.markdown("#### Risk Comparison")
        cc1, cc2 = st.columns(2)
        cc1.metric("Current (median)", f"{p_base:.1%}")
        delta = p_sim - p_base
        cc2.metric("Simulated", f"{p_sim:.1%}", delta=f"{delta:+.1%}")

        # Conformal badge
        if p_sim > 0.7:
            badge = "🔴 Likely **ELEVATED**"
        elif p_sim < 0.3:
            badge = "🟢 Likely **LOW RISK**"
        else:
            badge = "🟡 **Uncertain**"
        st.markdown(f"Conformal assessment: {badge}")

    # SHAP contributions (top-3) if explainer available
    if explainer is not None:
        st.markdown("#### Top-3 Contributing Features (Simulated)")
        try:
            sv = explainer.shap_values(X_sim[feature_names])
            sv_arr = sv if not isinstance(sv, list) else sv[1]
            importance = pd.Series(sv_arr[0], index=feature_names)
            top3 = importance.abs().nlargest(3)

            fig = go.Figure(go.Bar(
                x=importance[top3.index].values,
                y=top3.index.tolist(),
                orientation="h",
                marker_color=["#dc2626" if v > 0 else "#2563eb" for v in importance[top3.index].values],
            ))
            fig.update_layout(
                xaxis_title="SHAP contribution",
                height=180, margin=dict(t=10, b=30, l=10, r=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key="wi_shap")
        except Exception:
            st.info("SHAP unavailable — run notebook 09 first.")
    else:
        # Fallback: show feature deltas
        st.markdown("#### Feature Changes")
        changes = (X_sim[feature_names] - X_base[feature_names]).T
        changes.columns = ["delta"]
        changes = changes[changes["delta"].abs() > 0].sort_values("delta", ascending=False)
        if len(changes) > 0:
            fig = go.Figure(go.Bar(
                x=changes["delta"].values,
                y=changes.index.tolist(),
                orientation="h",
                marker_color=["#dc2626" if v > 0 else "#2563eb" for v in changes["delta"].values],
            ))
            fig.update_layout(height=200, margin=dict(t=5, b=30, l=5, r=5),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="wi_delta")
