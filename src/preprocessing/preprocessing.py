"""Data preprocessing, feature engineering, and pipeline construction."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import (
    ID_COL,
    TARGET,
    SERVICE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TENURE_BINS,
    TENURE_LABELS,
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add TenureGroup and NumberOfServices exactly as in the original analysis notebook."""
    out = df.copy()
    if "tenure" in out.columns:
        out["TenureGroup"] = pd.cut(
            out["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        ).astype(object)

    # Number of active add-on services
    active_services = [c for c in SERVICE_COLUMNS if c in out.columns]
    if active_services:
        out["NumberOfServices"] = out[active_services].eq("Yes").sum(axis=1)
    return out


def clean_for_model(raw: pd.DataFrame) -> pd.DataFrame:
    """Perform exact notebook data cleaning:
    1. TotalCharges converted to numeric.
    2. Drop records where TotalCharges is missing (11 rows where tenure==0).
    3. Drop customerID.
    4. Compute engineered features.
    """
    df = raw.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return add_engineered_features(df).reset_index(drop=True)


def prepare_dashboard_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaned frame plus a binary 0/1 `ChurnFlag` column for churn rate calculations."""
    df = clean_for_model(raw)
    if TARGET in df.columns:
        df["ChurnFlag"] = (df[TARGET] == "Yes").astype(int)
    return df


def model_frame(valid: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and align columns to exact pipeline expectations."""
    engineered = add_engineered_features(valid)
    return engineered[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Build the exact ColumnTransformer from the original notebook:
    - StandardScaler on numeric features.
    - OneHotEncoder(handle_unknown='ignore', drop='if_binary') on categorical features.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_pipeline(classifier) -> Pipeline:
    """Create a complete sklearn Pipeline wrapping preprocessor and classifier."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])

