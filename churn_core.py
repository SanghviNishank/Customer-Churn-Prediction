"""Top-level churn_core module for backward compatibility."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.churn_core import *  # noqa: F401, F403
import src.churn_core as _impl

ROOT = root_dir
DATA_DIR = root_dir / "data"
MODEL_DIR = root_dir / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
TEST_PRED_PATH = MODEL_DIR / "test_predictions.csv"
