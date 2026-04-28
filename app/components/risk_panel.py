"""Tab 1 — Risk Panel.

Shows per-station risk across all routes it actually serves.
Cascading dropdowns: pick a station → only valid routes appear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _badge_html(prob: float, t_cost: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "🔴 ELEVATED — High confidence", "#ef4444"
    if prob >= t_cost:
        return "🟠 ELEVATED — Alert threshold exceeded", "#f97316"
    if prob >= 0.3:
        return "🟡 BORDERLINE — Monitor closely", "#eab308"
    return "🟢 LOW RISK — Normal conditions", "#22c55e"


def render_risk_panel(
    modeling_df: pd.DataFrame,
    X_features: pd.DataFrame,
    model,
    prediction_sets_df: pd.DataFrame,
    test_idx: list[int],
) -> None:

    # Build station → routes mapping from ALL data (not just test)
    station_routes = (
        modeling_df.groupby("station_id")["route_id"]
        .unique()
        .apply(sorted)
        .to_dict()
    )
    all_stations = sorted(station_routes.keys())

    # ── Station selector ───────────────────────────────────────────────────────
    station = st.selectbox(
        "Select Station",
        all_stations,
        format_func=lambda s: f"Station {s}",
        key="rp_station",
    )

    valid_routes = station_routes.get(station, [])

    # ── Route selector — only routes that serve this station ──────────────────
    route = st.selectbox(
        "Select Route",
        valid_routes,
        format_func=lambda r: f"Route {r}",
        key="rp_route",
    )

    # ── Filter test set to this station-route pair ─────────────────────────────
    test_modeling = modeling_df.iloc[test_idx].reset_index(drop=True)
    test_X        = X_features.iloc[test_idx].reset_index(drop=True)

    mask_sr = (test_modeling["station_id"] == station) & (test_modeling["route_id"] == route)
    df_sr   = test_modeling[mask_sr]
    X_sr    = test_X[mask_sr]

    if len(X_sr) == 0:
        st.warning(f"No test data for Station {station} · Route {route}.")
        return

    # Run model
    proba      = model.predict_proba(X_sr)[:, 1]
    latest_p   = float(proba[-1])
    last_row   = df_sr.iloc[-1]

    t_cost = 0.163   # fixed cost-optimal threshold

    # ── Conformal badge ────────────────────────────────────────────────────────
    # prediction_sets_df is indexed over the full test set in order
    test_indices_sr = np.where(mask_sr.values)[0]
    last_test_i     = int(test_indices_sr[-1])
    if prediction_sets_df is not None and last_test_i < len(prediction_sets_df):
        ps = prediction_sets_df.iloc[last_test_i]
        if ps["pred_1"] and not ps["pred_0"]:
            conf_label = "🔴  90% confident: **ELEVATED RISK**"
            conf_color = "#ef4444"
        elif ps["pred_0"] and not ps["pred_1"]:
            conf_label = "🟢  90% confident: **LOW RISK**"
            conf_color = "#22c55e"
        else:
            conf_label = "🟡  **Uncertain** — both classes possible"
            conf_color = "#eab308"
    else:
        conf_label = "⚪  Conformal set unavailable"
        conf_color = "#6b7280"

    # ── Layout ─────────────────────────────────────────────────────────────────
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        label, color = _badge_html(latest_p, t_cost)
        st.markdown(
            f"<div style='font-size:1.1rem;font-weight:600;color:{color};margin-bottom:4px'>{label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.85rem;color:{conf_color};margin-bottom:12px'>{conf_label}</div>",
            unsafe_allow_html=True,
        )

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(latest_p * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],   "color": "rgba(34,197,94,0.15)"},
                    {"range": [30, 50],  "color": "rgba(234,179,8,0.15)"},
                    {"range": [50, 70],  "color": "rgba(249,115,22,0.15)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#7c3aed", "width": 3},
                    "thickness": 0.75,
                    "value": t_cost * 100,
                },
            },
            title={"text": "P(elevated delay risk — next hour)", "font": {"size": 13}},
        ))
        fig_gauge.update_layout(
            height=270,
            margin=dict(t=50, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="rp_gauge")

    with col_info:
        st.markdown("#### Current Conditions")
        metrics = [
            ("Current avg delay",     f"{last_row.get('mean_delay_current', 0):.1f} min"),
            ("Share delayed >5 min",  f"{last_row.get('share_delayed_5_current', 0):.1%}"),
            ("Passenger demand",      f"{last_row.get('mean_demand_current', 0):.0f} pax/hr"),
            ("Headway",               f"{last_row.get('mean_headway_current', 0):.1f} min"),
            ("Precipitation",         f"{last_row.get('mean_precip_mm', 0):.1f} mm/hr"),
            ("Wind",                  f"{last_row.get('mean_wind_kph', 0):.0f} kph"),
            ("Incident flag",         "Yes" if last_row.get("incident_flag") else "No"),
            ("Vehicle type",          str(last_row.get("vehicle_type", "—"))),
        ]
        for label, val in metrics:
            c1, c2 = st.columns([2, 1])
            c1.markdown(f"<span style='font-size:0.85rem;color:#94a3b8'>{label}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='font-size:0.85rem;font-weight:600'>{val}</span>", unsafe_allow_html=True)

    # ── All-routes comparison for this station ─────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### All Routes at Station {station} — Current Risk")

    route_probs = {}
    for r in valid_routes:
        m = (test_modeling["station_id"] == station) & (test_modeling["route_id"] == r)
        if m.sum() == 0:
            continue
        X_r = test_X[m]
        p_r = model.predict_proba(X_r)[:, 1]
        route_probs[r] = float(p_r[-1])

    colors_bar = [
        "#ef4444" if v >= 0.7 else "#f97316" if v >= t_cost else "#eab308" if v >= 0.3 else "#22c55e"
        for v in route_probs.values()
    ]
    fig_bar = go.Figure(go.Bar(
        x=list(route_probs.keys()),
        y=list(route_probs.values()),
        marker_color=colors_bar,
        text=[f"{v:.0%}" for v in route_probs.values()],
        textposition="outside",
    ))
    fig_bar.add_hline(
        y=t_cost, line_dash="dash", line_color="#7c3aed", line_width=2,
        annotation_text=f"Alert threshold (t={t_cost})",
        annotation_position="right",
    )
    fig_bar.update_layout(
        xaxis_title="Route",
        yaxis_title="P(elevated risk)",
        yaxis_range=[0, 1.1],
        height=220,
        margin=dict(t=20, b=40, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig_bar.update_xaxes(showgrid=False)
    fig_bar.update_yaxes(gridcolor="rgba(148,163,184,0.2)")
    st.plotly_chart(fig_bar, use_container_width=True, key="rp_routes_bar")

    # ── 24h history for selected route ─────────────────────────────────────────
    st.markdown(f"#### Risk History — Station {station} · Route {route}")
    hours_24 = df_sr["hour_floor"].values[-24:]
    proba_24  = proba[-24:]
    truth_24  = df_sr["y_primary"].values[-24:]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hours_24, y=proba_24,
        mode="lines+markers",
        name="P(risk)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(
            size=8,
            color=["#ef4444" if p >= t_cost else "#22c55e" for p in proba_24],
            line=dict(width=1, color="#1e40af"),
        ),
        hovertemplate="Hour: %{x}<br>P(risk)=%{y:.2f}<extra></extra>",
    ))
    # Actual label markers
    elev_mask = truth_24 == 1
    fig_hist.add_trace(go.Scatter(
        x=hours_24[elev_mask], y=np.ones(elev_mask.sum()) * -0.04,
        mode="markers",
        name="Actual elevated",
        marker=dict(symbol="triangle-up", size=8, color="#ef4444"),
        hoverinfo="skip",
    ))
    fig_hist.add_hline(
        y=t_cost, line_dash="dash", line_color="#7c3aed", line_width=1.5,
        annotation_text=f"t={t_cost}", annotation_position="right",
    )
    fig_hist.update_layout(
        xaxis_title="Hour",
        yaxis_title="P(elevated risk)",
        yaxis_range=[-0.1, 1.05],
        height=250,
        margin=dict(t=10, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05, font=dict(size=10)),
    )
    fig_hist.update_xaxes(showgrid=False)
    fig_hist.update_yaxes(gridcolor="rgba(148,163,184,0.2)")
    st.plotly_chart(fig_hist, use_container_width=True, key="rp_history")

    # Test-set accuracy for this combo
    correct = ((proba >= t_cost).astype(int) == df_sr["y_primary"].values).mean()
    pos_rate = df_sr["y_primary"].mean()
    n_alerts  = (proba >= t_cost).sum()
    st.caption(
        f"Test set — {len(df_sr)} hours · {pos_rate:.1%} elevated · "
        f"{n_alerts} alerts issued · {correct:.1%} correct decisions at t={t_cost}"
    )
