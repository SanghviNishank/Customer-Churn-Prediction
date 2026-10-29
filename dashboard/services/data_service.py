"""Data and model caching service for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import joblib
import pandas as pd
import streamlit as st

from src.data.load_data import load_raw, find_dataset
from src.preprocessing.preprocessing import prepare_dashboard_frame, clean_for_model
from src.utils.helpers import MODEL_DIR, PROJECT_ROOT


@st.cache_data(show_spinner="Loading dataset...")
def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw and processed datasets with Streamlit caching."""
    raw = load_raw()
    processed = prepare_dashboard_frame(raw)
    return raw, processed


@st.cache_resource(show_spinner="Loading trained machine learning model...")
def get_churn_model():
    """Load the primary selected trained pipeline."""
    model_path = MODEL_DIR / "churn_pipeline.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def get_logreg_model():
    """Load Logistic Regression pipeline for linear feature interpretations."""
    lr_path = MODEL_DIR / "logreg_pipeline.joblib"
    if not lr_path.exists():
        return None
    return joblib.load(lr_path)


@st.cache_data(show_spinner=False)
def get_model_metadata() -> Optional[Dict[str, Any]]:
    """Load metadata.json generated during model training."""
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def get_test_predictions() -> Optional[pd.DataFrame]:
    """Load test predictions CSV."""
    pred_path = MODEL_DIR / "test_predictions.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return None

