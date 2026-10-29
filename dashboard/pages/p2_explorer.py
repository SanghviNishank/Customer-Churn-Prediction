"""Page 2 — Dataset Explorer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_header
from src.data.load_data import ALLOWED, ID_COL, TARGET


def render(raw: pd.DataFrame, processed: pd.DataFrame):
    render_header(
        title="Dataset Explorer & Schema Inspector",
        subtitle="Interactive Inspection of Raw and Processed Datasets",
        badge_text="Data Exploration",
    )

    dataset_mode = st.radio(
        "Select Dataset View:",
        ["Processed Analysis Dataset (7,032 rows)", "Raw Ingested Dataset (7,043 rows)"],
        horizontal=True,
    )

    is_raw = "Raw" in dataset_mode
    active_df = raw if is_raw else processed

    # Dataset Metadata Row
    n_rows, n_cols = active_df.shape
    mem_usage = active_df.memory_usage(deep=True).sum() / (1024 * 1024)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{n_rows:,}")
    m2.metric("Total Features", f"{n_cols}")
    m3.metric("Memory Footprint", f"{mem_usage:.2f} MB")
    m4.metric("Dataset Stage", "Raw Source" if is_raw else "Model Ready")

    st.markdown("### 🔍 Interactive Filters & Table")

    # Filters
    with st.expander("Filter Criteria", expanded=False):
        f1, f2, f3 = st.columns(3)
        contract_opts = ALLOWED["Contract"]
        selected_contracts = f1.multiselect("Contract Type", contract_opts, default=contract_opts)
        
        internet_opts = ALLOWED["InternetService"]
        selected_internet = f2.multiselect("Internet Service", internet_opts, default=internet_opts)

        churn_filter = f3.multiselect("Churn Status", ["No", "Yes"], default=["No", "Yes"])

        t1, t2 = st.columns(2)
        tenure_min, tenure_max = t1.slider("Tenure Range (Months)", 0, 72, (0, 72))
        monthly_min, monthly_max = t2.slider("Monthly Charges Range ($)", 18.0, 120.0, (18.0, 120.0))

    # Apply filters
    filtered_df = active_df.copy()
    if "Contract" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Contract"].isin(selected_contracts)]
    if "InternetService" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["InternetService"].isin(selected_internet)]
    if TARGET in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[TARGET].isin(churn_filter)]
    if "tenure" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["tenure"].between(tenure_min, tenure_max)]
    if "MonthlyCharges" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["MonthlyCharges"].between(monthly_min, monthly_max)]

    # Customer ID mask
    show_id = st.checkbox("Show Customer ID column (if present in dataset)", value=False)
    display_df = filtered_df.copy()
    if not show_id and ID_COL in display_df.columns:
        display_df = display_df.drop(columns=[ID_COL])
    if "ChurnFlag" in display_df.columns:
        display_df = display_df.drop(columns=["ChurnFlag"])

    st.caption(f"Showing {len(display_df):,} matching rows out of {n_rows:,} total records.")

    # Table preview
    st.dataframe(display_df.head(250), use_container_width=True)

    # Download Button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_bytes,
        file_name="filtered_telco_data.csv",
        mime="text/csv",
    )

    st.markdown("### 📊 Column Schema & Type Inspection")
    schema_tabs = st.tabs(["Column Summary", "Descriptive Statistics (Numeric)", "Descriptive Statistics (Categorical)"])

    with schema_tabs[0]:
        schema_info = pd.DataFrame({
            "Column": active_df.columns,
            "Non-Null Count": active_df.notna().sum().values,
            "Null Count": active_df.isna().sum().values,
            "Data Type": active_df.dtypes.astype(str).values,
            "Unique Values": active_df.nunique().values,
            "Sample Value": [str(active_df[col].iloc[0]) for col in active_df.columns],
        })
        st.dataframe(schema_info, use_container_width=True)

    with schema_tabs[1]:
        num_cols = active_df.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 0:
            st.dataframe(active_df[num_cols].describe().round(2).T, use_container_width=True)
        else:
            st.info("No purely numeric columns found in this view.")

    with schema_tabs[2]:
        cat_cols = active_df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            st.dataframe(active_df[cat_cols].describe().T, use_container_width=True)
        else:
            st.info("No categorical columns found in this view.")

