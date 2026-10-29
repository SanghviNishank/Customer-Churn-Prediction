"""Page 4 — Exploratory Data Analysis (EDA)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_header
from dashboard.components.charts import (
    apply_theme,
    COLOR_MAP,
    CHURN_COLOR,
    STAY_COLOR,
)
from src.analysis.eda import segment_table, two_way_segment_table
from src.analysis.statistics import numeric_correlation
from src.data.load_data import ALLOWED, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def render(df: pd.DataFrame):
    render_header(
        title="Exploratory Data Analysis (EDA)",
        subtitle="Univariate, Bivariate, and Multivariate Pattern Discovery",
        badge_text="EDA & Segmentation",
    )

    # Dynamic Filter Drawer
    with st.expander("Filter Data for EDA", expanded=False):
        c1, c2, c3 = st.columns(3)
        contract_opts = ALLOWED["Contract"]
        contracts = c1.multiselect("Contract", contract_opts, default=contract_opts, key="eda_contract")
        internet_opts = ALLOWED["InternetService"]
        internet = c2.multiselect("Internet Service", internet_opts, default=internet_opts, key="eda_internet")
        tenure_range = c3.slider("Tenure Filter", 0, 72, (0, 72), key="eda_tenure")

    filtered_df = df[
        df["Contract"].isin(contracts)
        & df["InternetService"].isin(internet)
        & df["tenure"].between(tenure_range[0], tenure_range[1])
    ]

    st.caption(f"Analyzing {len(filtered_df):,} customers based on current filter selections.")
    if filtered_df.empty:
        st.warning("No records match the current filter selection.")
        return

    tabs = st.tabs([
        "Categorical Patterns",
        "Numeric Distributions",
        "Correlation Matrix",
        "Bivariate Relationships",
        "Two-Way Segment Heatmap",
    ])

    # 1. Categorical
    with tabs[0]:
        col_cat = st.selectbox(
            "Select Categorical Feature:",
            [c for c in CATEGORICAL_FEATURES if c in filtered_df.columns],
            index=13,  # Default to Contract
        )
        seg = segment_table(filtered_df, col_cat)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            fig_bar = px.bar(
                seg,
                x=col_cat,
                y="ChurnRate",
                text=seg["ChurnRate"].map("{:.1%}".format),
                color="ChurnRate",
                color_continuous_scale="OrRd",
                title=f"Churn Rate by {col_cat}",
                hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1%"},
            )
            fig_bar.update_yaxes(tickformat=".0%")
            st.plotly_chart(apply_theme(fig_bar, height=380), use_container_width=True)

        with col_right:
            fig_counts = px.bar(
                seg,
                x=col_cat,
                y="Customers",
                text="Customers",
                color_discrete_sequence=["#7F7F7F"],
                title=f"Segment Size Distribution ({col_cat})",
            )
            st.plotly_chart(apply_theme(fig_counts, height=380), use_container_width=True)

        st.dataframe(
            seg.style.format({
                "AvgMonthly": "${:,.2f}",
                "ChurnRate": "{:.1%}",
                "ShareOfCustomers": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # 2. Numeric
    with tabs[1]:
        num_col = st.selectbox("Select Numeric Feature:", NUMERIC_FEATURES)
        fig_hist = px.histogram(
            filtered_df,
            x=num_col,
            color=TARGET,
            nbins=35,
            barmode="overlay",
            opacity=0.65,
            color_discrete_map=COLOR_MAP,
            marginal="box",
            title=f"Distribution of {num_col} Stratified by Churn Status",
        )
        st.plotly_chart(apply_theme(fig_hist, height=440), use_container_width=True)
        st.dataframe(filtered_df.groupby(TARGET)[num_col].describe().round(2), use_container_width=True)

    # 3. Correlation
    with tabs[2]:
        corr = numeric_correlation(filtered_df)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix (Numerical Features & Churn Flag)",
        )
        st.plotly_chart(apply_theme(fig_corr, height=420), use_container_width=True)
        st.caption("Note: Tenure shows strong negative correlation with Churn (-0.35), while MonthlyCharges shows moderate positive correlation (+0.19).")

    # 4. Bivariate Relationships
    with tabs[3]:
        plot_type = st.radio("Plot Type:", ["Scatter Plot", "Violin Plot", "Box Plot"], horizontal=True)
        if plot_type == "Scatter Plot":
            sample_df = filtered_df.sample(min(len(filtered_df), 2000), random_state=42)
            fig_scatter = px.scatter(
                sample_df,
                x="tenure",
                y="MonthlyCharges",
                color=TARGET,
                color_discrete_map=COLOR_MAP,
                opacity=0.6,
                hover_data=["Contract", "InternetService"],
                title="Tenure vs Monthly Charges (Sample of up to 2,000 customers)",
            )
            st.plotly_chart(apply_theme(fig_scatter, height=450), use_container_width=True)
        else:
            fn = px.box if plot_type == "Box Plot" else px.violin
            y_axis = st.selectbox("Y Axis (Numeric)", ["MonthlyCharges", "TotalCharges", "tenure"])
            x_axis = st.selectbox("X Axis (Categorical)", ["Contract", "InternetService", "PaymentMethod", "TenureGroup"])
            fig_rel = fn(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color=TARGET,
                color_discrete_map=COLOR_MAP,
                title=f"{y_axis} by {x_axis} and Churn Status",
            )
            st.plotly_chart(apply_theme(fig_rel, height=450), use_container_width=True)

    # 5. Two-Way Segment Heatmap
    with tabs[4]:
        c1, c2 = st.columns(2)
        row_dim = c1.selectbox("Row Dimension", ["Contract", "InternetService", "TenureGroup", "PaymentMethod"], index=0)
        col_dim = c2.selectbox("Column Dimension", ["InternetService", "Contract", "TenureGroup", "PaymentMethod"], index=0)
        
        if row_dim == col_dim:
            st.warning("Please choose two distinct dimensions.")
        else:
            two_way = two_way_segment_table(filtered_df, row_dim, col_dim)
            fig_tw = px.imshow(
                two_way,
                text_auto=".1%",
                color_continuous_scale="OrRd",
                title=f"Cross-Segment Churn Rate: {row_dim} vs {col_dim}",
            )
            st.plotly_chart(apply_theme(fig_tw, height=420), use_container_width=True)

