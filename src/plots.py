from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#be185d"]
FIGURE_DPI = 150
FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "figure.dpi": FIGURE_DPI,
})


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, name: str, figures_dir: str | Path | None, dpi: int = FIGURE_DPI) -> None:
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def model_comparison_bar(
    metrics_df: pd.DataFrame,
    metric: str = "roc_auc",
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    df = metrics_df[[metric]].dropna().sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.55)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    ax.barh(df.index.astype(str), df[metric], color=colors, height=0.6)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(f"Model Comparison — {metric}", fontsize=12, pad=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    save_fig(fig, f"model_comparison_{metric}", figures_dir)
    return fig


def reliability_diagram(
    reliability_data_list: list[tuple[np.ndarray, np.ndarray]],
    model_names: list[str],
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for i, ((mean_probs, frac_pos), name) in enumerate(zip(reliability_data_list, model_names)):
        ax.plot(mean_probs, frac_pos, "o-", color=PALETTE[i % len(PALETTE)], label=name, lw=1.5, ms=5)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, "reliability_diagram", figures_dir)
    return fig


def confusion_matrix_plot(
    cm: np.ndarray | list,
    model_name: str,
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        cbar=False, linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11)
    fig.tight_layout()
    save_fig(fig, f"confusion_matrix_{model_name}", figures_dir)
    return fig


def cost_threshold_curve(
    thresholds: np.ndarray,
    costs: np.ndarray,
    t_default: float,
    t_f1: float,
    t_cost: float,
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, costs, color=PALETTE[0], lw=2)
    ax.axvline(t_default, color="gray", ls="--", lw=1.2, label=f"Default ({t_default:.2f})")
    ax.axvline(t_f1, color=PALETTE[1], ls="--", lw=1.2, label=f"F1-optimal ({t_f1:.2f})")
    ax.axvline(t_cost, color=PALETTE[2], ls="--", lw=1.2, label=f"Cost-optimal ({t_cost:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Cost")
    ax.set_title("Cost vs. Classification Threshold", fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "cost_threshold_curve", figures_dir)
    return fig


def stress_heatmap(
    slice_df_pivot: pd.DataFrame,
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, max(4, len(slice_df_pivot) * 0.4)))
    sns.heatmap(
        slice_df_pivot.astype(float), annot=True, fmt=".3f",
        cmap="RdYlGn", ax=ax, vmin=0.5, vmax=1.0,
        linewidths=0.3, linecolor="white", cbar_kws={"shrink": 0.8},
    )
    ax.set_title("AUC by Stratum", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "stress_heatmap", figures_dir)
    return fig


def conformal_coverage_plot(
    coverage_results: pd.DataFrame,
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    strata = coverage_results["stratum"].unique() if "stratum" in coverage_results.columns else ["overall"]
    width = 0.8 / max(len(strata), 1)

    for i, stratum in enumerate(strata):
        sub = coverage_results[coverage_results.get("stratum", pd.Series(["overall"] * len(coverage_results))) == stratum]
        x = np.arange(len(sub))
        ax.bar(x + i * width, sub["coverage"], width=width, color=PALETTE[i % len(PALETTE)], label=str(stratum))

    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Conformal Coverage by Stratum", fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "conformal_coverage", figures_dir)
    return fig


def shap_summary(
    shap_values: Any,
    feature_names: list[str],
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    import shap as shap_lib
    fig, ax = plt.subplots(figsize=(8, 6))
    shap_lib.summary_plot(shap_values, feature_names=feature_names, show=False, plot_type="dot")
    fig = plt.gcf()
    fig.tight_layout()
    save_fig(fig, "shap_summary", figures_dir)
    return fig


def pdp_grid(
    model: Any,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    figures_dir: str | Path | None = None,
) -> plt.Figure:
    from sklearn.inspection import PartialDependenceDisplay

    n_features = min(6, len(feature_names))
    selected = feature_names[:n_features]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, feat in enumerate(selected):
        ax = axes_flat[i]
        disp = PartialDependenceDisplay.from_estimator(
            model, X_sample, [feat], ax=ax, line_kw={"color": PALETTE[0]}
        )
        ax.set_title(feat, fontsize=9)

    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "pdp_grid", figures_dir)
    return fig
