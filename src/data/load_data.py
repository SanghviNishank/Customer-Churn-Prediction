"""Data loading and persistence module."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from src.utils.helpers import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, logger

DEFAULT_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumberOfServices"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]

RAW_INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["New", "Regular", "Established", "Loyal"]

ALLOWED = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def find_dataset() -> Optional[Path]:
    """Find the dataset in standard paths (data/, data/raw/, etc.)."""
    candidate_paths = [
        DATA_DIR / DEFAULT_CSV,
        RAW_DATA_DIR / DEFAULT_CSV,
        DATA_DIR / "Telco_customer_churn.xlsx",
        RAW_DATA_DIR / "Telco_customer_churn.xlsx",
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return None


def load_raw(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Read the raw dataset into a DataFrame."""
    p = Path(path) if path else find_dataset()
    if p is None or not p.exists():
        raise FileNotFoundError(
            f"Dataset not found. Please ensure '{DEFAULT_CSV}' exists in '{DATA_DIR}'."
        )
    logger.info("Loading raw dataset from %s", p)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p)


def load_processed(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load cleaned/processed dataset if available, otherwise clean raw dataset."""
    target_path = Path(path) if path else (PROCESSED_DATA_DIR / "telco_cleaned.csv")
    if target_path.exists():
        logger.info("Loading processed dataset from %s", target_path)
        return pd.read_csv(target_path)
    
    # Import here to prevent circular imports
    from src.preprocessing.preprocessing import clean_for_model
    raw = load_raw()
    df = clean_for_model(raw)
    save_processed(df, target_path)
    return df


def save_processed(df: pd.DataFrame, path: Optional[Path | str] = None) -> Path:
    """Save processed dataset to disk."""
    target_path = Path(path) if path else (PROCESSED_DATA_DIR / "telco_cleaned.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info("Saved processed dataset to %s", target_path)
    return target_path

