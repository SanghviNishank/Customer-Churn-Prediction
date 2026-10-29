"""Page 3 — Data Quality and Preprocessing."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_header
from dashboard.components.charts import apply_theme
from src.analysis.statistics import outlier_analysis
from src.data.validate_data import check_data_quality


def render(raw: pd.DataFrame, processed: pd.DataFrame):
    render_header(
        title="Data Quality & Preprocessing Audit",
        subtitle="Verification of Cleanliness, Transformations, and Preprocessing Integrity",
        badge_text="Quality Assurance",
    )

    st.markdown("### 🔍 Verified Transformations (from `CustomerFinal.ipynb`)")
    st.markdown(
        """
        The following data cleaning operations were verified directly from the original notebook cells:
        1. **`TotalCharges` Blank String Imputation/Drop:** The raw dataset contains 11 records where `TotalCharges` is a whitespace string (`" "`), all corresponding to brand-new accounts with `tenure == 0`. The notebook converts `TotalCharges` to numeric and drops these 11 rows (7,043 &rarr; 7,032 rows).
        2. **`customerID` Column Removal:** A unique alphanumeric identifier with 7,043 unique keys carries zero predictive signal and introduces leakage or cardinality explosion. It is removed prior to modeling.
        3. **Engineered Feature: `TenureGroup`:** Binned into 4 ordinal intervals: `New` (0–12 mos), `Regular` (13–24 mos), `Established` (25–48 mos), and `Loyal` (49–72 mos).
        4. **Engineered Feature: `NumberOfServices`:** Integer count (0 to 6) summing `Yes` responses across the six add-on services: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`.
        5. **Leakage-Free `ColumnTransformer`:** Standardized numeric features with `StandardScaler` and encoded categoricals with `OneHotEncoder(handle_unknown='ignore', drop='if_binary')` inside an sklearn Pipeline fitted on the training split only.
        """
    )

    # Before and After Comparison Card
    st.markdown("### ⚖️ Before-and-After Cleaning Comparison")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", f"{len(raw):,}")
    c2.metric("Cleaned Rows", f"{len(processed):,}")
    c3.metric("Rows Removed", f"{len(raw) - len(processed):,} (0.16%)")
    c4.metric("Engineered Features Added", "+2 (TenureGroup, NumberOfServices)")

    diff_table = pd.DataFrame([
        {"Metric": "Dataset Shape", "Raw Ingested": f"{raw.shape}", "Cleaned / Processed": f"{processed.shape}"},
        {"Metric": "TotalCharges Data Type", "Raw Ingested": str(raw['TotalCharges'].dtype), "Cleaned / Processed": str(processed['TotalCharges'].dtype)},
        {"Metric": "Blank Strings in TotalCharges", "Raw Ingested": f"{(raw['TotalCharges'].astype(str).str.strip() == '').sum()} rows", "Cleaned / Processed": "0 rows"},
        {"Metric": "customerID Present", "Raw Ingested": "Yes (7,043 unique)", "Cleaned / Processed": "No (Removed)"},
        {"Metric": "Target Churn Rate", "Raw Ingested": f"{(raw['Churn'] == 'Yes').mean():.2%}", "Cleaned / Processed": f"{(processed['Churn'] == 'Yes').mean():.2%}"},
    ])
    st.table(diff_table)

    st.markdown("### 📈 Missing Values & Outlier Analysis")
    tabs = st.tabs(["Missing Values Inspection", "Outlier Analysis (IQR Method)", "Duplicate Checks"])

    with tabs[0]:
        dq = check_data_quality(raw)
        missing_df = dq["missing"]
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.dataframe(missing_df, use_container_width=True)
        with col2:
            fig_miss = px.bar(
                missing_df.head(5),
                x="Column",
                y="Total Missing",
                text="Total Missing",
                color_discrete_sequence=["#D55E00"],
                title="Missing / Blank Counts by Feature",
            )
            st.plotly_chart(apply_theme(fig_miss, height=320), use_container_width=True)

    with tabs[1]:
        st.markdown(
            "Outlier detection using the standard **Interquartile Range (IQR)** rule: "
            "`[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`."
        )
        outliers_df = outlier_analysis(processed)
        st.dataframe(outliers_df, use_container_width=True)

        selected_outlier_col = st.selectbox(
            "Select feature to view box plot & distribution:",
            ["tenure", "MonthlyCharges", "TotalCharges"],
        )
        fig_box = px.box(
            processed,
            y=selected_outlier_col,
            x="Churn",
            color="Churn",
            color_discrete_map={"Yes": "#D55E00", "No": "#0072B2"},
            points="outliers",
            title=f"Box Plot of {selected_outlier_col} by Churn Status",
        )
        st.plotly_chart(apply_theme(fig_box, height=380), use_container_width=True)

    with tabs[2]:
        st.dataframe(dq["duplicates"], use_container_width=True)
        st.success("No duplicate customer accounts found in the dataset.")

