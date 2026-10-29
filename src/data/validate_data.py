"""Data validation and data quality checking module."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.data.load_data import ALLOWED, RAW_INPUT_COLUMNS, ID_COL, TARGET


def validate_input(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Validate customer records for prediction.

    Returns:
        (valid_rows, invalid_rows_with_reason, messages)
        Nothing is silently dropped: every invalid row includes a 'Problem' column.
    """
    messages: List[str] = []
    missing_cols = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        messages.append("Missing required columns: " + ", ".join(missing_cols))
        empty_valid = df.iloc[0:0].copy()
        invalid_with_prob = df.assign(Problem="missing required columns: " + ", ".join(missing_cols))
        return empty_valid, invalid_with_prob, messages

    work = df.copy()
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    
    # Standardize SeniorCitizen (handle text or numeric)
    work["SeniorCitizen"] = work["SeniorCitizen"].replace(
        {"Yes": 1, "No": 0, "yes": 1, "no": 0, "YES": 1, "NO": 0}
    ).infer_objects(copy=False)
    work["SeniorCitizen"] = pd.to_numeric(work["SeniorCitizen"], errors="coerce")

    problems = pd.Series([""] * len(work), index=work.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        problems.loc[mask] = problems.loc[mask] + text + "; "

    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        flag(work[col].isna(), f"{col} missing or not numeric")
    
    flag(
        work["tenure"].notna() & ((work["tenure"] < 0) | (work["tenure"] > 120)),
        "tenure outside 0-120",
    )
    flag(
        work["MonthlyCharges"].notna() & (work["MonthlyCharges"] < 0),
        "MonthlyCharges negative",
    )
    flag(
        work["TotalCharges"].notna() & (work["TotalCharges"] < 0),
        "TotalCharges negative",
    )
    for col, allowed in ALLOWED.items():
        flag(~work[col].isin(allowed), f"{col} not one of {allowed}")

    bad = problems != ""
    valid = work.loc[~bad, RAW_INPUT_COLUMNS].copy()
    valid["SeniorCitizen"] = valid["SeniorCitizen"].astype(int)
    invalid = df.loc[bad].copy()
    invalid["Problem"] = problems[bad].str.rstrip("; ")
    
    if len(invalid) > 0:
        messages.append(
            f"{len(invalid)} row(s) failed validation and were NOT predicted "
            "(listed below with specific reasons)."
        )
    return valid, invalid, messages


def check_data_quality(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Perform comprehensive data quality checks on a DataFrame."""
    n_rows, n_cols = df.shape
    
    # 1. Missing and whitespace inspection
    nan_counts = df.isna().sum()
    blank_counts = df.apply(lambda s: s.astype(str).str.strip().eq("").sum() if s.dtype == object else 0)
    total_missing = nan_counts + blank_counts
    missing_pct = (total_missing / n_rows * 100).round(2)
    
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "NaN Count": nan_counts.values,
        "Blank Strings": blank_counts.values,
        "Total Missing": total_missing.values,
        "Missing %": missing_pct.values,
    }).sort_values("Total Missing", ascending=False)

    # 2. Duplicate rows check
    duplicates_all = df.duplicated().sum()
    duplicates_no_id = (
        df.drop(columns=[ID_COL]).duplicated().sum() if ID_COL in df.columns else duplicates_all
    )
    
    dup_df = pd.DataFrame({
        "Check": ["Exact Full Duplicates", "Duplicates Excluding ID"],
        "Count": [duplicates_all, duplicates_no_id],
        "Percentage": [
            round(duplicates_all / n_rows * 100, 2),
            round(duplicates_no_id / n_rows * 100, 2),
        ],
    })

    # 3. Cardinality check
    card_df = pd.DataFrame({
        "Column": df.columns,
        "Unique Values": df.nunique().values,
        "Unique Ratio %": (df.nunique().values / n_rows * 100).round(2),
        "Constant / Near Constant": [
            "Yes" if (df[col].value_counts(normalize=True).iloc[0] > 0.99) else "No"
            for col in df.columns
        ],
    }).sort_values("Unique Values", ascending=True)

    return {
        "missing": missing_df,
        "duplicates": dup_df,
        "cardinality": card_df,
    }
