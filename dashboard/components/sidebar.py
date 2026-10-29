"""Sidebar component for navigation and status indicators."""
from __future__ import annotations

from typing import Optional, Dict, Any
import streamlit as st

PAGES = [
    "1. Executive Overview",
    "2. Dataset Explorer",
    "3. Data Quality & Preprocessing",
    "4. Exploratory Data Analysis",
    "5. Feature Analysis",
    "6. Model Training & Evaluation",
    "7. Business Insights & Reports",
    "8. Prediction Interface",
]


def render_sidebar(meta: Optional[Dict[str, Any]], n_raw: int, n_used: int) -> str:
    """Render modern sidebar and return the selected page name."""
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 2rem;">📊</span>
            <div>
                <h3 style="margin: 0; padding: 0; font-size: 1.25rem;">Churn Analytics</h3>
                <span style="font-size: 0.8rem; color: #718096;">ML Intelligence Suite</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown("**Navigation**")
    selected_page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.divider()

    # System Status & Metadata Box
    st.sidebar.markdown("### 🔍 Model & Data Status")
    st.sidebar.markdown(f"**Dataset:** Telco Churn (`.csv`)")
    st.sidebar.markdown(f"**Records:** {n_raw:,} raw / {n_used:,} cleaned")
    st.sidebar.markdown(f"**Target:** `Churn` (Yes / No)")

    if meta:
        model_name = meta.get("selected_model", "Custom Pipeline")
        threshold = meta.get("selected_threshold", 0.40)
        st.sidebar.success(f"**Active Model:**\n{model_name}")
        st.sidebar.caption(f"Decision Threshold: `{threshold}`")
        if meta.get("xgboost_available_at_training"):
            st.sidebar.markdown('<span class="badge badge-success">XGBoost Engine</span>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("Model artifacts not loaded. Run `train_model.py`.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Antigravity ML Engine • Read-Only Integrity")
    return selected_page

