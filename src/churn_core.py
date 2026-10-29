"""Backward-compatible churn_core module integrating the modular src components."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd

from src.utils.helpers import PROJECT_ROOT, MODEL_DIR, DATA_DIR
from src.data.load_data import (
    DEFAULT_CSV,
    TARGET,
    ID_COL,
    RANDOM_STATE,
    SERVICE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RAW_INPUT_COLUMNS,
    TENURE_BINS,
    TENURE_LABELS,
    ALLOWED,
    find_dataset,
    load_raw,
)
from src.data.validate_data import validate_input
from src.preprocessing.preprocessing import (
    add_engineered_features,
    clean_for_model,
    prepare_dashboard_frame,
    model_frame,
)
from src.models.predict import (
    DEFAULT_LOW_MAX,
    DEFAULT_HIGH_MIN,
    NOTEBOOK_THRESHOLD,
    risk_category,
    predict_frame,
)

ROOT = PROJECT_ROOT
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
TEST_PRED_PATH = MODEL_DIR / "test_predictions.csv"


def load_metadata() -> Optional[Dict[str, Any]]:
    """Load metadata.json safely."""
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return None

