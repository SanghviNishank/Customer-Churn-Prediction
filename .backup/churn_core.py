"""Shared logic for training, prediction and the dashboard.

Mirrors the preprocessing in the original notebook (CustomerFinal.ipynb) without
importing or modifying it:
  * TotalCharges -> numeric; rows with blank TotalCharges are dropped (as in the notebook)
  * customerID dropped
  * TenureGroup = pd.cut(tenure, [0,12,24,48,72], New/Regular/Established/Loyal)
  * NumberOfServices = count of 'Yes' in 6 add-on service columns
  * target: Yes -> 1, No -> 0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
TEST_PRED_PATH = MODEL_DIR / "test_predictions.csv"

DEFAULT_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

SERVICE_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumberOfServices"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
# Columns a user must supply (engineered ones are derived)
RAW_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["New", "Regular", "Established", "Loyal"]

# Allowed values, used for validation and form widgets. SeniorCitizen is 0/1 in the data.
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
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                      "Electronic check", "Mailed check"],
}

# Default risk bands (probability). Configurable in the dashboard.
DEFAULT_LOW_MAX = 0.30
DEFAULT_HIGH_MIN = 0.60
# Decision threshold the notebook chose for its final XGBoost model.
NOTEBOOK_THRESHOLD = 0.4


# --------------------------------------------------------------------------- data
def find_dataset() -> Path | None:
    """Return the dataset path (the notebook's CSV first, then the .xlsx named in the brief)."""
    for name in (DEFAULT_CSV, "Telco_customer_churn.xlsx"):
        p = DATA_DIR / name
        if p.exists():
            return p
    return None


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the dataset. The file on disk is never written to."""
    path = Path(path) if path else find_dataset()
    if path is None or not path.exists():
        raise FileNotFoundError(f"Dataset not found. Put '{DEFAULT_CSV}' in {DATA_DIR}.")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TenureGroup"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS,
                                include_lowest=True).astype(object)
    out["NumberOfServices"] = out[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    return out


def clean_for_model(raw: pd.DataFrame) -> pd.DataFrame:
    """Notebook cleaning: numeric TotalCharges, drop rows where it is missing, drop ID."""
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return add_engineered_features(df).reset_index(drop=True)


def prepare_dashboard_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaned frame plus a 0/1 `ChurnFlag` column for rate calculations."""
    df = clean_for_model(raw)
    df["ChurnFlag"] = (df[TARGET] == "Yes").astype(int)
    return df


# ----------------------------------------------------------------------- prediction
def validate_input(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Validate raw customer records for prediction.

    Returns (valid_rows, invalid_rows_with_reason, messages). Nothing is silently
    dropped: every invalid row is returned with a `Problem` column.
    """
    messages: list[str] = []
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        messages.append("Missing required columns: " + ", ".join(missing))
        return df.iloc[0:0], df.assign(Problem="missing required columns"), messages

    work = df.copy()
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["SeniorCitizen"] = work["SeniorCitizen"].replace(
        {"Yes": 1, "No": 0, "yes": 1, "no": 0, "YES": 1, "NO": 0})
    work["SeniorCitizen"] = pd.to_numeric(work["SeniorCitizen"], errors="coerce")

    problems = pd.Series([""] * len(work), index=work.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        problems.loc[mask] = problems.loc[mask] + text + "; "

    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        flag(work[col].isna(), f"{col} missing or not numeric")
    flag(work["tenure"].notna() & ((work["tenure"] < 0) | (work["tenure"] > 120)),
         "tenure outside 0-120")
    flag(work["MonthlyCharges"].notna() & (work["MonthlyCharges"] < 0), "MonthlyCharges negative")
    flag(work["TotalCharges"].notna() & (work["TotalCharges"] < 0), "TotalCharges negative")
    for col, allowed in ALLOWED.items():
        flag(~work[col].isin(allowed), f"{col} not one of {allowed}")

    bad = problems != ""
    valid = work.loc[~bad, RAW_INPUT_COLUMNS].copy()
    valid["SeniorCitizen"] = valid["SeniorCitizen"].astype(int)
    invalid = df.loc[bad].copy()
    invalid["Problem"] = problems[bad].str.rstrip("; ")
    if len(invalid):
        messages.append(f"{len(invalid)} row(s) failed validation and were NOT predicted "
                        "(listed below with the reason).")
    return valid, invalid, messages


def model_frame(valid: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and order columns exactly as the pipeline expects."""
    return add_engineered_features(valid)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def risk_category(p: float, low_max: float = DEFAULT_LOW_MAX,
                  high_min: float = DEFAULT_HIGH_MIN) -> str:
    return "Low" if p < low_max else ("High" if p >= high_min else "Medium")


def predict_frame(pipeline, valid: pd.DataFrame, threshold: float,
                  low_max: float = DEFAULT_LOW_MAX,
                  high_min: float = DEFAULT_HIGH_MIN) -> pd.DataFrame:
    proba = pipeline.predict_proba(model_frame(valid))[:, 1]
    out = valid.copy()
    out["ChurnProbability"] = np.round(proba, 4)
    out["PredictedChurn"] = np.where(proba >= threshold, "Yes", "No")
    out["RiskCategory"] = [risk_category(p, low_max, high_min) for p in proba]
    return out


def load_metadata() -> dict | None:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return None
