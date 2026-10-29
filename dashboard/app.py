"""Main entry point for the Customer Churn Analytics Dashboard."""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.components.sidebar import render_sidebar
from dashboard.services.data_service import get_datasets, get_model_metadata
from dashboard.pages import (
    p1_overview,
    p2_explorer,
    p3_quality,
    p4_eda,
    p5_features,
    p6_models,
    p7_insights,
    p8_prediction,
)

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_file = Path(__file__).resolve().parent / "styles" / "main.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main():
    try:
        raw_df, processed_df = get_datasets()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    meta = get_model_metadata()
    selected_page = render_sidebar(meta, len(raw_df), len(processed_df))

    if selected_page == "1. Executive Overview":
        p1_overview.render(raw_df, processed_df, meta)
    elif selected_page == "2. Dataset Explorer":
        p2_explorer.render(raw_df, processed_df)
    elif selected_page == "3. Data Quality & Preprocessing":
        p3_quality.render(raw_df, processed_df)
    elif selected_page == "4. Exploratory Data Analysis":
        p4_eda.render(processed_df)
    elif selected_page == "5. Feature Analysis":
        p5_features.render(processed_df, meta)
    elif selected_page == "6. Model Training & Evaluation":
        p6_models.render(meta)
    elif selected_page == "7. Business Insights & Reports":
        p7_insights.render(processed_df, meta)
    elif selected_page == "8. Prediction Interface":
        p8_prediction.render(processed_df, meta)


if __name__ == "__main__":
    main()

