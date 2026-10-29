"""Page 1 — Project Overview."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_header, render_metric_card
from dashboard.components.charts import apply_theme, COLOR_MAP
from src.analysis.eda import compute_overview_kpis
from src.utils.helpers import format_currency, format_percentage


def render(raw: pd.DataFrame, df: pd.DataFrame, meta: dict | None):
    render_header(
        title="Telecom Customer Churn Analytics Platform",
        subtitle="End-to-end Machine Learning System for Proactive Customer Retention",
        badge_text="Executive Overview",
    )

    kpis = compute_overview_kpis(df)

    # Top KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    render_metric_card(c1, "Total Customers", f"{kpis['total_customers']:,}", help_text="Total verified customer records after cleaning.")
    render_metric_card(c2, "Churned Customers", f"{kpis['churned_customers']:,}", help_text="Historical churn cases (Churn = Yes).")
    render_metric_card(c3, "Retained Customers", f"{kpis['retained_customers']:,}", help_text="Retained customer base (Churn = No).")
    render_metric_card(c4, "Historical Churn Rate", format_percentage(kpis['churn_rate']), help_text="Overall churn prevalence in the dataset.")

    c5, c6, c7, c8 = st.columns(4)
    render_metric_card(c5, "Avg Monthly Billing", format_currency(kpis['avg_monthly_charges']), help_text="Average monthly invoice amount.")
    render_metric_card(c6, "Avg Tenure", f"{kpis['avg_tenure_months']:.1f} mos", help_text="Mean customer duration with the provider.")
    render_metric_card(c7, "Monthly Revenue at Risk", format_currency(kpis['lost_monthly_revenue']), help_text="Total recurring billing from churned customers.")
    render_metric_card(c8, "Billing Share at Risk", format_percentage(kpis['lost_monthly_share']), help_text="Proportion of monthly revenue lost to churn.")

    st.markdown("### 📋 Problem Statement & Objectives")
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown(
            """
            In the telecommunications sector, acquiring new customers costs **5x to 25x more** than retaining existing ones. 
            When customers cancel service, recurring subscription revenue is immediately lost, and customer lifetime value (CLV) is eroded.
            
            **Core Project Objectives (from `CustomerFinal.ipynb`):**
            1. **Detect Early Warning Signals:** Uncover behavioural, demographic, and contractual patterns that precipitate churn.
            2. **Deploy Accurate Binary Classification:** Train, tune, and benchmark multiple ML architectures to separate churners from loyal subscribers without data leakage.
            3. **Optimize Decision Thresholds:** Shift from arbitrary 0.5 probability thresholds to an operational threshold (**0.40**) that prioritizes recall and minimizes costly missed churners.
            4. **Explain Individual & Global Decisions:** Implement transparent SHAP tree explainers and linear log-odds decompositions so customer success managers know *why* a customer is flagged.
            """
        )

    with right_col:
        counts = df["Churn"].value_counts().rename_axis("Status").reset_index(name="Customers")
        fig_pie = px.pie(
            counts,
            names="Status",
            values="Customers",
            color="Status",
            color_discrete_map=COLOR_MAP,
            hole=0.55,
            title="Class Distribution (Imbalance Ratio: 2.76 : 1)",
        )
        st.plotly_chart(apply_theme(fig_pie, height=320), use_container_width=True)

    # Visual Pipeline Architecture
    st.markdown("### 🔄 End-to-End Machine Learning Pipeline")
    st.markdown(
        """
        <div style="background-color: rgba(128,128,128,0.06); padding: 20px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2);">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="flex: 1; min-width: 150px; background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #0072B2;">
                    <strong>1. Ingestion</strong><br><small>7,043 rows, 21 raw columns (CSV / Excel)</small>
                </div>
                <div style="flex: 1; min-width: 150px; background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #009E73;">
                    <strong>2. Cleaning</strong><br><small>Drop 11 blank TotalCharges, remove customerID</small>
                </div>
                <div style="flex: 1; min-width: 150px; background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #E69F00;">
                    <strong>3. Feature Engineering</strong><br><small>TenureGroup (4 bins), NumberOfServices (0-6)</small>
                </div>
                <div style="flex: 1; min-width: 150px; background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #D55E00;">
                    <strong>4. Preprocessing</strong><br><small>ColumnTransformer: StandardScaler + OneHotEncoder</small>
                </div>
                <div style="flex: 1; min-width: 150px; background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #7F7F7F;">
                    <strong>5. Model Tuning</strong><br><small>Stratified 80/20 split, 5-fold CV, XGBoost (Best)</small>
                </div>
                <div style="flex: 1; min-width: 150px; background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #0072B2;">
                    <strong>6. Threshold & Serving</strong><br><small>Optimal 0.40 cutoff, SHAP explainability</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

