"""Reusable Plotly visualization components with consistent themes."""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Accessible colour-blind safe palette
CHURN_COLOR = "#D55E00"  # Vermilion
STAY_COLOR = "#0072B2"   # Blue
ACCENT_GREEN = "#009E73" # Bluish Green
ACCENT_YELLOW = "#E69F00"# Orange/Yellow
NEUTRAL_GRAY = "#7F7F7F"

COLOR_MAP = {"Yes": CHURN_COLOR, "No": STAY_COLOR, "Churn": CHURN_COLOR, "Stay": STAY_COLOR}
RISK_COLORS = {"Low": ACCENT_GREEN, "Medium": ACCENT_YELLOW, "High": CHURN_COLOR}
PLOT_TEMPLATE = "plotly_white"


def apply_theme(fig: go.Figure, height: int = 420, title: Optional[str] = None) -> go.Figure:
    """Apply consistent styling, margins, fonts, and responsive layout."""
    layout_updates = {
        "template": PLOT_TEMPLATE,
        "height": height,
        "margin": dict(l=20, r=20, t=55 if title else 25, b=20),
        "font": dict(family="Segoe UI, -apple-system, Roboto, Helvetica, Arial, sans-serif", size=13),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    }
    if title:
        layout_updates["title"] = title
    fig.update_layout(**layout_updates)
    return fig


def horizontal_bar(
    s: pd.Series,
    title: str,
    xlabel: str,
    colors: Optional[pd.Series] = None,
    color_map: Optional[Dict[str, str]] = None,
    single_color: str = STAY_COLOR,
    height: int = 460,
) -> go.Figure:
    """Generate styled horizontal bar chart from a Series."""
    df_plot = pd.DataFrame({"Feature": s.index.astype(str), "Value": s.to_numpy()})
    if colors is not None:
        df_plot["Group"] = colors.to_numpy()
        fig = px.bar(
            df_plot,
            x="Value",
            y="Feature",
            orientation="h",
            color="Group",
            color_discrete_map=color_map,
            labels={"Value": xlabel},
        )
    else:
        fig = px.bar(
            df_plot,
            x="Value",
            y="Feature",
            orientation="h",
            labels={"Value": xlabel},
            color_discrete_sequence=[single_color],
        )
    fig.update_yaxes(categoryorder="array", categoryarray=df_plot["Feature"].tolist())
    return apply_theme(fig, height=height, title=title)


def confusion_matrix_plot(cm: List[List[int]], title: str = "Confusion Matrix") -> go.Figure:
    """Generate annotated confusion matrix heatmap."""
    labels = ["No Churn (0)", "Churn (1)"]
    fig = px.imshow(
        np.array(cm),
        x=[f"Predicted {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    return apply_theme(fig, height=380, title=title)


def roc_pr_curves_plot(
    tp_df: pd.DataFrame, selected_models: List[str]
) -> Tuple[go.Figure, go.Figure]:
    """Generate dual ROC and Precision-Recall interactive figures."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

    y_true = tp_df["y_true"].to_numpy()
    fig_roc = go.Figure()
    fig_pr = go.Figure()

    for m in selected_models:
        if m not in tp_df.columns:
            continue
        p = tp_df[m].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC {auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, p)
        fig_pr.add_scatter(x=recall, y=precision, mode="lines", name=m)

    # Reference lines
    fig_roc.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(dash="dash", color="gray")
    )
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    base_rate = float(np.mean(y_true))
    fig_pr.add_hline(
        y=base_rate, line_dash="dash", line_color="gray", annotation_text=f"Base Churn Rate ({base_rate:.1%})"
    )
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")

    return apply_theme(fig_roc, height=420, title="ROC Curves"), apply_theme(
        fig_pr, height=420, title="Precision-Recall Curves"
    )

