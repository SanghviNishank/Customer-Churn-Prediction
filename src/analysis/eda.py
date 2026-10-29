"""Exploratory Data Analysis (EDA) functions and aggregations."""
from __future__ import annotations

from typing import Dict, Any
import pandas as pd

from src.data.load_data import TARGET


def compute_overview_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive KPI summary metrics."""
    n_total = len(df)
    churn_flag = df["ChurnFlag"] if "ChurnFlag" in df.columns else (df[TARGET] == "Yes").astype(int)
    n_churned = int(churn_flag.sum())
    n_retained = n_total - n_churned
    churn_rate = n_churned / n_total if n_total > 0 else 0.0

    avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 0.0
    avg_tenure = float(df["tenure"].mean()) if "tenure" in df.columns else 0.0
    
    # Revenue calculations
    total_monthly_rev = float(df["MonthlyCharges"].sum()) if "MonthlyCharges" in df.columns else 0.0
    lost_monthly_rev = (
        float(df.loc[churn_flag == 1, "MonthlyCharges"].sum())
        if "MonthlyCharges" in df.columns
        else 0.0
    )
    lost_monthly_share = lost_monthly_rev / total_monthly_rev if total_monthly_rev > 0 else 0.0

    return {
        "total_customers": n_total,
        "churned_customers": n_churned,
        "retained_customers": n_retained,
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly,
        "avg_tenure_months": avg_tenure,
        "total_monthly_revenue": total_monthly_rev,
        "lost_monthly_revenue": lost_monthly_rev,
        "lost_monthly_share": lost_monthly_share,
    }


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Generate segmentation metrics (customers, churn count, churn rate, avg monthly) for a column."""
    churn_series = df["ChurnFlag"].astype(int) if "ChurnFlag" in df.columns else (df[TARGET] == "Yes").astype(int)
    work = df.copy()
    work["_churn_calc"] = churn_series.values
    
    grouped = work.groupby(col, observed=True).agg(
        Customers=("_churn_calc", "size"),
        Churned=("_churn_calc", "sum"),
        AvgMonthly=("MonthlyCharges", "mean"),
    ).reset_index()

    grouped["ChurnRate"] = grouped["Churned"] / grouped["Customers"]
    grouped["ShareOfCustomers"] = grouped["Customers"] / grouped["Customers"].sum()
    return grouped


def two_way_segment_table(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    """Generate pivot table of churn rates across two categorical features."""
    churn_series = df["ChurnFlag"].astype(int) if "ChurnFlag" in df.columns else (df[TARGET] == "Yes").astype(int)
    work = df.copy()
    work["_churn_calc"] = churn_series.values
    return work.pivot_table(
        index=row_col,
        columns=col_col,
        values="_churn_calc",
        aggfunc="mean",
        observed=True,
    )
