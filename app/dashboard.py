"""TransitRisk — Streamlit Dashboard.

Launch:
    cd transitrisk/
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from components.cost_tuner      import render_cost_tuner
from components.risk_panel      import render_risk_panel
from components.shap_panel      import render_shap_panel
from components.stress_explorer import render_stress_explorer
from components.streaming_demo  import render_streaming_demo
from components.what_if         import render_what_if

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TransitRisk",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — theme-safe (works in dark AND light mode) ────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 16px; border-radius: 6px 6px 0 0; }
    h1 { font-size: 1.55rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.0rem !important; }
    /* Fix sidebar text always visible */
    section[data-testid="stSidebar"] * { color: inherit !important; }
    /* Metric labels visible in dark mode */
    [data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.85; }
    /* Caption text */
    .stCaption { opacity: 0.75; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGS_DIR   = ROOT / "figures"

_NOT_READY = (
    "📂 **Data not ready.** Run notebooks 01–11 first:\n\n```bash\nmake all\n```"
)


@st.cache_data(show_spinner="Loading data…")
def load_data():
    req = [DATA_DIR / "modeling_table.parquet",
           DATA_DIR / "X_features.parquet",
           DATA_DIR / "train_val_test_indices.json"]
    if any(not p.exists() for p in req):
        return None, None, None
    modeling = pd.read_parquet(DATA_DIR / "modeling_table.parquet")
    X_all    = pd.read_parquet(DATA_DIR / "X_features.parquet")
    with open(DATA_DIR / "train_val_test_indices.json") as f:
        idx = json.load(f)
    return modeling, X_all, (idx["train"], idx["val"], idx["test"])


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import joblib
    for name in ["xgb_calibrated.joblib", "xgb.joblib"]:
        p = MODELS_DIR / name
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_data(show_spinner=False)
def load_prediction_sets():
    p = DATA_DIR / "prediction_sets.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_thresholds():
    p = DATA_DIR / "thresholds.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"t_default": 0.5, "t_f1": 0.345, "t_cost": 0.163}


@st.cache_data(show_spinner=False)
def load_metrics():
    p = FIGS_DIR / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ── Load everything ────────────────────────────────────────────────────────────
modeling, X_all, indices_tuple = load_data()
model = load_model()

if modeling is None or model is None:
    st.warning(_NOT_READY)
    st.stop()

train_idx, val_idx, test_idx = indices_tuple
X_test        = X_all.iloc[test_idx].reset_index(drop=True)
y_test        = modeling["y_primary"].values[test_idx]
modeling_test = modeling.iloc[test_idx].reset_index(drop=True)
feature_names = X_all.columns.tolist()

prediction_sets = load_prediction_sets()
thresholds      = load_thresholds()
metrics         = load_metrics()

# ── Sidebar — Help Panel ───────────────────────────────────────────────────────
_HELP = {
    "📡 Risk Panel": """
### What this tab shows
Real-time next-hour delay risk for any station-route pair.

**How to use:**
1. Pick a **Station** (1–60).
2. The **Route** dropdown auto-updates to only show routes that serve that station — not all routes pass every station.
3. Read the **gauge** — the needle shows the model's probability that delays will be elevated next hour.
4. The **purple tick** on the gauge = alert threshold (t=0.163). Anything past that triggers a dispatch alert.
5. The **bar chart** compares risk across all routes at this station simultaneously.
6. The **history chart** shows the last 24 hourly predictions. Red dots on the x-axis = hours that *actually* had elevated delays.

**What "elevated risk" means:**  
The next hour's mean delay on this route exceeds 5 minutes.

**Conformal badge:**  
The "90% confident" tag is from conformal prediction — a mathematical guarantee that the set covers the true label at least 90% of the time.
""",
    "🔧 What-If": """
### What this tab shows
Sensitivity analysis — how does risk change if conditions worsen?

**How to use:**
1. Move the sliders to simulate different weather or operational conditions.
2. Watch the **risk comparison** update — it shows baseline (typical) vs your simulated scenario.
3. The **feature changes bar** shows what changed and by how much.

**Example questions you can answer:**
- "If it rains 15 mm/hr during morning peak, what happens to risk?"
- "If headways double (buses become infrequent), how much does risk increase?"
- "What if demand drops to half normal?"

**Key insight:**  
This is how operators would use the model interactively — before a storm or event — to understand how the network will behave.
""",
    "💰 Cost Tuner": """
### What this tab shows
How the alert threshold should change depending on the cost of missing a delay vs a false alarm.

**The core trade-off:**  
- **False Negative (miss):** Model says low risk, but delays actually happen → passengers stranded, no warning.  
- **False Positive (false alarm):** Model says high risk, but delays don't happen → unnecessary bus repositioning.

**How to use:**
1. Drag the **C(FN)/C(FP) slider** to set your relative cost ratio.
2. The cost curve updates → the minimum (orange line) shows the new optimal threshold.
3. The confusion matrix shows what predictions look like at that threshold.

**Default setting:**  
C(FN)=5, C(FP)=1 → we accept 5 false alarms to avoid 1 missed delay. This gives t=0.163.

**Practical meaning:**  
A city with severe delay consequences (e.g. airport shuttle service) might use C(FN)=10. A low-frequency rural bus might use C(FN)=2.
""",
    "🌡 Stress Explorer": """
### What this tab shows
Does the model hold up under different operating conditions?

**How to use:**
1. Pick a **slice axis** — Weather, Time of Day, Demand Level, or Headway.
2. The table shows ROC-AUC, PR-AUC, and F1 broken down by each sub-group.
3. Pick a stratum to see **example predictions** — True Positives, False Positives, False Negatives.

**What to look for:**
- Does AUC drop significantly during heavy rain? → model may need weather-specific retraining.
- Does performance drop during late night? → fewer trips = less signal in lag features.
- Does it vary by demand quartile? → high-demand routes may be easier to predict.

**Why this matters:**  
A model that averages 0.81 AUC but scores 0.65 during heavy rain is unreliable for its most important use case.
""",
    "🔍 SHAP": """
### What this tab shows
Which features drove the model's decision for a **specific** prediction.

**How to use:**
1. Enter a **Test row ID** (0 to 30,623) — each ID is one station-route-hour in the test set.
2. See the model's probability, the true label, and whether it was correct.
3. The **feature values table** shows the top-10 features by deviation from the population mean — i.e. what was unusual about this hour.

**Z-score column:**  
How many standard deviations from average. Z-score = 3 means that feature was extreme this hour.

**Note on SHAP:**  
Full SHAP values require running notebook 09. The fallback (current mode) shows statistical deviation — nearly as informative for understanding individual predictions.
""",
    "🔴 Live Feed": """
### What this tab shows
The full end-to-end inference pipeline running live on the test set.

**How to use:**
1. Set **replay speed** — Slow lets you read each prediction, Instant replays all at once.
2. Set **rows to replay** — more rows = more stable running metrics.
3. Toggle **Show feature inputs** to see which signals drove each decision.
4. Press **▶ Run simulation**.

**What you're watching:**
- **Gauge** — probability output from XGBoost for the current row.
- **Feature bar** — the 10 most informative inputs for this prediction (normalised to 0-1).
- **Decision badge** — DISPATCH ALERT (p ≥ 0.163) or HOLD.
- **Ground truth** — was the model right? ✓ or ✗.
- **Prediction log** — scrolling table of all decisions so far.
- **Running metrics** — Precision/Recall/F1 computed cumulatively.

**The key insight:**  
This is exactly what a transit operations centre would see — a live risk score arriving every hour for every route, with a binary action recommendation.
""",
}

with st.sidebar:
    st.markdown("## 🚌 TransitRisk")
    st.caption("DATA 245 · Spring 2026")
    st.divider()

    # Key metrics
    xgb_m = metrics.get("xgb_calibrated", metrics.get("xgb", {}))
    if xgb_m:
        st.markdown("**Best Model: XGBoost Calibrated**")
        col_a, col_b = st.columns(2)
        col_a.metric("ROC-AUC", f"{xgb_m.get('roc_auc', 0):.3f}")
        col_b.metric("PR-AUC",  f"{xgb_m.get('pr_auc', 0):.3f}")
        col_a.metric("F1",      f"{xgb_m.get('f1', 0):.3f}")
        col_b.metric("Brier",   f"{xgb_m.get('brier', 0):.3f}")

    st.divider()
    t = thresholds
    st.markdown("**Thresholds**")
    st.markdown(f"- Cost-optimal: `t = {t['t_cost']:.3f}`")
    st.markdown(f"- F1-optimal:   `t = {t['t_f1']:.3f}`")
    st.markdown(f"- Default:      `t = {t['t_default']:.2f}`")

    st.divider()
    st.markdown(f"Test set: **{len(test_idx):,}** rows")
    st.markdown(f"Positive rate: **{y_test.mean():.1%}**")
    st.markdown(f"Features: **{len(feature_names)}**")

    st.divider()
    # Per-tab help
    st.markdown("### 📖 Tab Guide")
    help_tab = st.selectbox(
        "Select tab to read its guide",
        list(_HELP.keys()),
        label_visibility="collapsed",
    )
    st.markdown(_HELP[help_tab])

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚌 TransitRisk")
st.caption(
    "Cost-Sensitive, Calibrated Next-Hour Delay Risk Forecasting · DATA 245 Spring 2026"
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Risk Panel", "🔧 What-If", "💰 Cost Tuner",
    "🌡 Stress Explorer", "🔍 SHAP", "🔴 Live Feed",
])

with tab1:
    render_risk_panel(
        modeling_df=modeling,
        X_features=X_all,
        model=model,
        prediction_sets_df=prediction_sets,
        test_idx=test_idx,
    )

with tab2:
    render_what_if(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        explainer=None,
    )

with tab3:
    render_cost_tuner(
        model=model,
        X_test=X_test,
        y_test=y_test,
        t_f1=thresholds["t_f1"],
        t_cost_default=thresholds["t_cost"],
    )

with tab4:
    render_stress_explorer(
        model=model,
        X_test=X_test,
        y_test=y_test,
        modeling_test=modeling_test,
        t_cost=thresholds["t_cost"],
    )

with tab5:
    render_shap_panel(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        explainer=None,
    )

with tab6:
    render_streaming_demo(
        model=model,
        modeling_test=modeling_test,
        X_test=X_test,
        y_test=y_test,
        t_cost=thresholds["t_cost"],
        feature_names=feature_names,
    )
