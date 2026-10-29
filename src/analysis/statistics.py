"""Statistical checks, outlier detection, and correlation metrics."""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.data.load_data import NUMERIC_FEATURES, TARGET


def numeric_correlation(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Calculate Pearson correlation matrix for numeric features and churn flag."""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    work = df[cols].copy()
    if include_target:
        if "ChurnFlag" in df.columns:
            work["ChurnFlag"] = df["ChurnFlag"]
        elif TARGET in df.columns:
            work["ChurnFlag"] = (df[TARGET] == "Yes").astype(int)
    return work.corr().round(3)


def outlier_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute IQR-based outlier boundaries and counts for numerical columns."""
    cols = columns or [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    results = []
    n_rows = len(df)
    
    for c in cols:
        series = pd.to_numeric(df[c], errors="coerce").dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        count = len(outliers)
        pct = (count / n_rows * 100) if n_rows > 0 else 0.0

        results.append({
            "Feature": c,
            "Min": round(float(series.min()), 2),
            "Q1 (25%)": round(q1, 2),
            "Median (50%)": round(float(series.median()), 2),
            "Q3 (75%)": round(q3, 2),
            "Max": round(float(series.max()), 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": count,
            "Outlier %": round(pct, 2),
        })

    return pd.DataFrame(results)


def categorical_association_chi2(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Compute Chi-Square test of independence and Cramer's V for a categorical feature against Churn."""
    churn_col = df[TARGET] if TARGET in df.columns else df.get("ChurnFlag")
    if churn_col is None:
        return {"chi2": 0.0, "p_value": 1.0, "cramers_v": 0.0}

    contingency = pd.crosstab(df[col], churn_col)
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    return {
        "chi2": round(float(chi2), 2),
        "p_value": float(p_val),
        "cramers_v": round(float(cramers_v), 3),
    }

