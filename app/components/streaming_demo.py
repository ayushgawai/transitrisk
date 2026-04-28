"""Tab 6 — Live Inference Feed.

Simulates what a transit operations centre sees every hour:
  raw transit data → 38 features → XGBoost → P(elevated risk next hour)
  → cost-optimal threshold → DISPATCH ALERT / HOLD decision
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HIGHLIGHT_FEATURES = [
    "mean_delay_current",
    "lag_1h_mean_delay",
    "lag_3h_mean_delay",
    "share_delayed_5_current",
    "mean_precip_mm",
    "mean_demand_current",
    "mean_headway_current",
    "rolling_6h_mean_delay",
    "is_peak_morning",
    "is_peak_evening",
]


def _badge(prob: float, t_cost: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "🔴 HIGH RISK", "#ef4444"
    if prob >= t_cost:
        return "🟠 ELEVATED", "#f97316"
    if prob >= 0.3:
        return "🟡 BORDERLINE", "#eab308"
    return "🟢 LOW RISK", "#22c55e"


def _decision_badge(prob: float, t_cost: float) -> tuple[str, str]:
    if prob >= t_cost:
        return "⚡ DISPATCH ALERT", "#ef4444"
    return "✅ HOLD — No action", "#22c55e"


def _gauge(prob: float, t_cost: float, key_suffix: str = "") -> go.Figure:
    label, color = _badge(prob, t_cost)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "rgba(34,197,94,0.12)"},
                {"range": [30, 50],  "color": "rgba(234,179,8,0.12)"},
                {"range": [50, 70],  "color": "rgba(249,115,22,0.12)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#7c3aed", "width": 3},
                "thickness": 0.8,
                "value": t_cost * 100,
            },
        },
        title={"text": "P(elevated risk — next hour)", "font": {"size": 13}},
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=50, b=5, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _feature_bar(row: pd.Series, highlight_cols: list[str]) -> go.Figure:
    cols = [c for c in highlight_cols if c in row.index]
    vals = row[cols]
    normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    fig = go.Figure(go.Bar(
        x=normed.values,
        y=cols,
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.2f}" for v in vals.values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.4], showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(size=10)),
        height=260,
        margin=dict(t=5, b=5, l=5, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _running_metrics_chart(log_df: pd.DataFrame, t_cost: float) -> go.Figure:
    preds  = (log_df["prob"] >= t_cost).astype(int)
    truths = log_df["y_true"].astype(int)
    rows = []
    for i in range(1, len(log_df) + 1):
        p = preds.iloc[:i]; t = truths.iloc[:i]
        tp = ((p == 1) & (t == 1)).sum()
        fp = ((p == 1) & (t == 0)).sum()
        fn = ((p == 0) & (t == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({"n": i, "Precision": prec, "Recall": rec, "F1": f1})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color in [("Precision", "#3b82f6"), ("Recall", "#22c55e"), ("F1", "#7c3aed")]:
        fig.add_trace(go.Scatter(
            x=df["n"], y=df[col], mode="lines",
            name=col, line=dict(color=color, width=2),
        ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        xaxis_title="Predictions so far", yaxis_title="Score",
        yaxis_range=[0, 1], height=220,
        margin=dict(t=10, b=40, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.2)")
    return fig


def _format_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["station_id", "route_id", "hour_floor", "prob", "decision",
              "y_true", "correct"]].copy()
    out = out.rename(columns={
        "station_id": "Station", "route_id": "Route",
        "hour_floor": "Hour", "prob": "P(risk)",
        "decision": "Decision", "y_true": "Actual", "correct": "✓",
    })
    out["P(risk)"] = out["P(risk)"].map(lambda x: f"{x:.3f}")
    out["Actual"]  = out["Actual"].map({1: "Elevated", 0: "Low"})
    out["✓"]       = out["✓"].map({1: "✓", 0: "✗"})
    return out.iloc[::-1].reset_index(drop=True)


def render_streaming_demo(
    model,
    modeling_test: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    t_cost: float,
    feature_names: list[str],
) -> None:

    with st.expander("ℹ️  What is this? How does the model help? (click to read)", expanded=False):
        st.markdown("""
**The prediction problem:**  
Every hour, for every station-route pair, we ask *"Will the next hour have elevated delays?"*
Elevated = mean delay ≥ 5 min. This is predicted **one hour in advance** so action can be taken.

**The pipeline each hour:**
```
Current-hour transit data (trips, delays, demand, weather)
       ↓
38 engineered features (lag delays, rolling averages, weather interactions, time signals)
       ↓
XGBoost Calibrated  →  P(elevated delay risk)  ← raw probability output
       ↓
Cost threshold  t = 0.163   (FN costs 5× more than FP)
       ↓
⚡ DISPATCH ALERT  /  ✅ HOLD
```

**Why this threshold?**  
Missing a real delay (FN) → passengers stranded, no warning → costs 5 units.  
Unnecessary alert (FP) → one bus repositioned → costs 1 unit.  
So we accept more FPs to ensure we catch almost all real delays.

**What the simulation shows:**  
Replays the held-out test set row-by-row as if data is arriving live.  
Each row = one station-route-hour observation. The model scores it in real time.
""")

    st.markdown("---")
    st.markdown("### Streaming Inference Simulator")
    st.caption(
        "Replays the held-out **test set** — as if hourly transit data is arriving live. "
        "XGBoost scores each row instantly, issues ALERT or HOLD, and tracks accuracy."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        speed = st.select_slider(
            "Replay speed",
            options=["Slow (1/s)", "Normal (3/s)", "Fast (10/s)", "Instant"],
            value="Normal (3/s)",
        )
    with ctrl2:
        n_show = st.slider("Rows to replay", min_value=20, max_value=300, value=60, step=10)
    with ctrl3:
        show_features = st.toggle("Show feature inputs", value=True)

    speed_map = {"Slow (1/s)": 1.0, "Normal (3/s)": 0.33, "Fast (10/s)": 0.10, "Instant": 0.0}
    delay = speed_map[speed]

    run = st.button("▶  Run simulation", type="primary", use_container_width=False)
    st.markdown("---")

    if "stream_log" not in st.session_state:
        st.session_state["stream_log"] = pd.DataFrame()

    # ── All placeholders declared once, named uniquely ─────────────────────────
    col_gauge, col_feat = st.columns([1, 1])
    with col_gauge:
        ph_gauge = st.empty()
    with col_feat:
        ph_features = st.empty()

    row1_left, row1_right = st.columns([1, 1])
    with row1_left:
        ph_decision = st.empty()
        ph_meta     = st.empty()
    with row1_right:
        ph_conf = st.empty()

    st.markdown("#### Running Prediction Log")
    ph_log = st.empty()
    st.markdown("#### Live Metrics (Precision / Recall / F1)")
    ph_metrics = st.empty()

    # ── Show last run if not running ───────────────────────────────────────────
    if not run:
        log = st.session_state["stream_log"]
        if not log.empty:
            last = log.iloc[-1]
            ph_gauge.plotly_chart(_gauge(last["prob"], t_cost),
                                  use_container_width=True, key="sd_gauge_static")
            dec_label, dec_color = _decision_badge(last["prob"], t_cost)
            risk_label, risk_color = _badge(last["prob"], t_cost)
            ph_decision.markdown(
                f"<div style='font-size:1.4rem;font-weight:700;color:{dec_color}'>{dec_label}</div>",
                unsafe_allow_html=True,
            )
            ph_meta.caption(
                f"Last: Station {last['station_id']} · Route {last['route_id']} · "
                f"{str(last['hour_floor'])[:16]}"
            )
            ph_conf.markdown(
                f"<div style='font-size:1.1rem;font-weight:600;color:{risk_color}'>{risk_label}</div>",
                unsafe_allow_html=True,
            )
            if show_features:
                feat_row = X_test.iloc[int(last["test_row_idx"])]
                ph_features.plotly_chart(
                    _feature_bar(feat_row, _HIGHLIGHT_FEATURES),
                    use_container_width=True, key="sd_feat_static",
                )
            ph_log.dataframe(_format_log(log.tail(50)), use_container_width=True, height=260)
            ph_metrics.plotly_chart(
                _running_metrics_chart(log, t_cost),
                use_container_width=True, key="sd_metrics_static",
            )
        else:
            ph_gauge.plotly_chart(_gauge(0.0, t_cost),
                                  use_container_width=True, key="sd_gauge_empty")
            ph_decision.markdown("*Press ▶ Run simulation to start the live feed.*")
        return

    # ── LIVE simulation ────────────────────────────────────────────────────────
    st.session_state["stream_log"] = pd.DataFrame()
    n_total = len(X_test)
    indices = np.linspace(0, n_total - 1, min(n_show, n_total), dtype=int)
    log_rows: list[dict] = []

    for i, idx in enumerate(indices):
        x_row  = X_test.iloc[[idx]]
        prob   = float(model.predict_proba(x_row[feature_names])[:, 1][0])
        y_true = int(y_test[idx])
        meta   = modeling_test.iloc[idx]
        station = meta.get("station_id", "—")
        route   = meta.get("route_id", "—")
        hour_fl = meta.get("hour_floor", "—")

        risk_label, risk_color = _badge(prob, t_cost)
        dec_label, dec_color   = _decision_badge(prob, t_cost)
        correct = int((prob >= t_cost) == bool(y_true))

        log_rows.append({
            "test_row_idx": idx,
            "station_id": station, "route_id": route, "hour_floor": hour_fl,
            "prob": prob, "y_true": y_true,
            "decision": "ALERT" if prob >= t_cost else "HOLD",
            "correct": correct,
        })
        log_df = pd.DataFrame(log_rows)

        # Update all placeholders with unique keys per iteration
        ph_gauge.plotly_chart(
            _gauge(prob, t_cost),
            use_container_width=True, key=f"sd_gauge_{i}",
        )
        if show_features:
            ph_features.plotly_chart(
                _feature_bar(X_test.iloc[idx], _HIGHLIGHT_FEATURES),
                use_container_width=True, key=f"sd_feat_{i}",
            )
        truth_str   = "✓ correct" if correct else "✗ wrong"
        truth_color = "#22c55e" if correct else "#ef4444"
        ph_decision.markdown(
            f"<div style='font-size:1.4rem;font-weight:700;color:{dec_color}'>{dec_label}</div>",
            unsafe_allow_html=True,
        )
        ph_meta.caption(
            f"Station **{station}** · Route **{route}** · {str(hour_fl)[:16]}  "
            f"<span style='color:{truth_color}'>{truth_str}</span>",
            unsafe_allow_html=True,
        )
        ph_conf.markdown(
            f"<div style='font-size:1.1rem;font-weight:600;color:{risk_color}'>{risk_label}</div>"
            f"<div style='font-size:0.85rem;color:#94a3b8'>Ground truth: "
            f"{'Elevated' if y_true else 'Low'}</div>",
            unsafe_allow_html=True,
        )
        ph_log.dataframe(_format_log(log_df.tail(50)), use_container_width=True, height=260)
        if len(log_df) >= 3:
            ph_metrics.plotly_chart(
                _running_metrics_chart(log_df, t_cost),
                use_container_width=True, key=f"sd_metrics_{i}",
            )
        if delay > 0:
            time.sleep(delay)

    st.session_state["stream_log"] = log_df

    # Final summary
    preds_arr = (log_df["prob"].values >= t_cost).astype(int)
    truth_arr = log_df["y_true"].values
    acc  = (preds_arr == truth_arr).mean()
    tp   = ((preds_arr == 1) & (truth_arr == 1)).sum()
    fp   = ((preds_arr == 1) & (truth_arr == 0)).sum()
    fn   = ((preds_arr == 0) & (truth_arr == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    st.markdown("---")
    st.markdown("#### Simulation Complete")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.1%}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1",        f"{f1:.3f}")
    st.caption(
        f"Replayed {len(log_df)} predictions · cost threshold t={t_cost:.3f} · "
        f"XGBoost Calibrated (ROC-AUC 0.809 on full test set)."
    )
