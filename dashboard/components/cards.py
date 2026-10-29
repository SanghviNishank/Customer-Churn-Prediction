"""Reusable UI cards and banner components."""
from __future__ import annotations

from typing import Optional
import streamlit as st


def render_header(title: str, subtitle: Optional[str] = None, badge_text: Optional[str] = None):
    """Render consistent page header with optional subtitle and badge."""
    c1, c2 = st.columns([4, 1])
    with c1:
        st.title(title)
        if subtitle:
            st.markdown(f"*{subtitle}*")
    with c2:
        if badge_text:
            st.markdown(f'<div style="text-align: right; padding-top: 15px;"><span class="badge badge-info">{badge_text}</span></div>', unsafe_allow_html=True)
    st.divider()


def render_metric_card(col, label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Wrapper around st.metric with consistent styling."""
    col.metric(label=label, value=value, delta=delta, help=help_text)


def render_alert(message: str, kind: str = "info"):
    """Render colored alert container."""
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)

