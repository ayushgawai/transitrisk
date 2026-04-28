"""Tab 5 — SHAP Explanation Panel."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_shap_panel(model, X_test: pd.DataFrame, y_test: np.ndarray,
                      feature_names: list[str], explainer=None) -> None:
    n_test = len(X_test)
    row_id = st.number_input("Test row ID", min_value=0, max_value=n_test - 1,
                              value=0, step=1)

    X_row = X_test.iloc[[row_id]]
    proba = float(model.predict_proba(X_row)[:, 1][0])
    true_label = int(y_test[row_id])

    col1, col2 = st.columns(2)
    col1.metric("P(elevated risk)", f"{proba:.3f}")
    col2.metric("True label", "Elevated" if true_label == 1 else "Low risk")

    outcome_label = ""
    if true_label == 1 and proba > 0.5:
        outcome_label = "✅ True Positive"
    elif true_label == 0 and proba <= 0.5:
        outcome_label = "✅ True Negative"
    elif true_label == 1 and proba <= 0.5:
        outcome_label = "❌ False Negative"
    else:
        outcome_label = "⚠️ False Positive"
    st.markdown(f"Classification result: **{outcome_label}**")

    if explainer is not None:
        st.markdown("#### Top-5 SHAP Contributors")
        try:
            sv = explainer.shap_values(X_row[feature_names])
            sv_arr = sv if not isinstance(sv, list) else sv[1]
            importance = pd.Series(sv_arr[0], index=feature_names)
            top5 = importance.abs().nlargest(5)
            top5_vals = importance[top5.index]

            colors = ["#dc2626" if v > 0 else "#2563eb" for v in top5_vals.values]
            fig = go.Figure(go.Bar(
                x=top5_vals.values,
                y=top5_vals.index.tolist(),
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in top5_vals.values],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_color="#374151", line_width=1)
            fig.update_layout(
                xaxis_title="SHAP contribution (→ increases risk)",
                height=300, margin=dict(t=10, b=40, l=10, r=80),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key="shap_bar")

            # Feature values vs population
            st.markdown("#### Feature Values vs Population")
            pop_means = X_test[top5.index].mean()
            compare_df = pd.DataFrame({
                "Feature": top5.index,
                "This row": X_row[top5.index].values[0],
                "Population mean": pop_means.values,
                "SHAP": top5_vals.values,
            })
            st.dataframe(compare_df.style.format({
                "This row": "{:.3f}",
                "Population mean": "{:.3f}",
                "SHAP": "{:+.4f}",
            }), use_container_width=True)

        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            _show_feature_values_fallback(X_row, X_test, feature_names)
    else:
        _show_feature_values_fallback(X_row, X_test, feature_names)


def _show_feature_values_fallback(X_row, X_test, feature_names):
    st.markdown("#### Feature Values (run notebook 09 for SHAP)")
    row_vals = X_row[feature_names].iloc[0]
    pop_means = X_test[feature_names].mean()
    deviations = ((row_vals - pop_means) / (X_test[feature_names].std() + 1e-9)).abs()
    top_devs = deviations.nlargest(10)

    display_df = pd.DataFrame({
        "Feature": top_devs.index,
        "Value": row_vals[top_devs.index].values,
        "Pop mean": pop_means[top_devs.index].values,
        "|Z-score|": top_devs.values,
    })
    st.dataframe(display_df.style.format({
        "Value": "{:.3f}", "Pop mean": "{:.3f}", "|Z-score|": "{:.2f}"
    }), use_container_width=True)
