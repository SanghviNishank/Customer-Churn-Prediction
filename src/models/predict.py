"""Inference, probability scoring, risk categorisation, and retention guidance."""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from src.preprocessing.preprocessing import model_frame

DEFAULT_LOW_MAX = 0.30
DEFAULT_HIGH_MIN = 0.60
NOTEBOOK_THRESHOLD = 0.40


def risk_category(
    p: float,
    low_max: float = DEFAULT_LOW_MAX,
    high_min: float = DEFAULT_HIGH_MIN,
) -> str:
    """Classify probability into Low, Medium, or High risk tier."""
    if p < low_max:
        return "Low"
    elif p >= high_min:
        return "High"
    return "Medium"


def predict_frame(
    pipeline,
    valid_df: pd.DataFrame,
    threshold: float = NOTEBOOK_THRESHOLD,
    low_max: float = DEFAULT_LOW_MAX,
    high_min: float = DEFAULT_HIGH_MIN,
) -> pd.DataFrame:
    """Score valid customer records using the fitted pipeline."""
    X_input = model_frame(valid_df)
    proba = pipeline.predict_proba(X_input)[:, 1]

    out = valid_df.copy()
    out["ChurnProbability"] = np.round(proba, 4)
    out["PredictedChurn"] = np.where(proba >= threshold, "Yes", "No")
    out["RiskCategory"] = [risk_category(p, low_max, high_min) for p in proba]
    return out


def generate_retention_observations(
    rec: pd.Series, df: pd.DataFrame, proba: float
) -> List[str]:
    """Generate empirical retention observations derived from historical churn segments.
    These reflect correlations in the dataset rather than causal guarantees.
    """
    churn_flag = df["ChurnFlag"].astype(int) if "ChurnFlag" in df.columns else (df["Churn"] == "Yes").astype(int)
    base_rate = float(churn_flag.mean())
    notes: List[str] = []

    def segment_stats(mask: pd.Series) -> tuple[float, int]:
        n = int(mask.sum())
        r = float(churn_flag.loc[mask].mean()) if n > 0 else 0.0
        return r, n

    # 1. Contract Type
    if rec.get("Contract") == "Month-to-month":
        r_m2m, n_m2m = segment_stats(df["Contract"] == "Month-to-month")
        r_long, _ = segment_stats(df["Contract"] != "Month-to-month")
        notes.append(
            f"Month-to-month contract: In this dataset, monthly customers churn at {r_m2m:.1%} (n={n_m2m:,}) "
            f"vs {r_long:.1%} for long-term contracts. Recommended Action: Consider an annual contract "
            "incentive or 'Lock & Save' discount promotion."
        )

    # 2. Internet Service (Fiber Optic)
    if rec.get("InternetService") == "Fiber optic":
        r_fo, n_fo = segment_stats(df["InternetService"] == "Fiber optic")
        notes.append(
            f"Fiber optic internet: Churn rate is {r_fo:.1%} (n={n_fo:,}) vs {base_rate:.1%} baseline. "
            "Recommended Action: Check service reliability, technical support history, or bundle satisfaction."
        )

    # 3. Add-on Services (Online Security & Tech Support)
    if rec.get("InternetService") != "No" and rec.get("OnlineSecurity") == "No":
        r_sec, n_sec = segment_stats((df["OnlineSecurity"] == "No") & (df["InternetService"] != "No"))
        notes.append(
            f"No Online Security: Internet subscribers without security add-ons churn at {r_sec:.1%} (n={n_sec:,}). "
            "Recommended Action: Offer complimentary 60-day security trial."
        )

    if rec.get("InternetService") != "No" and rec.get("TechSupport") == "No":
        r_ts, n_ts = segment_stats((df["TechSupport"] == "No") & (df["InternetService"] != "No"))
        notes.append(
            f"No Tech Support: Internet subscribers without tech support churn at {r_ts:.1%} (n={n_ts:,}). "
            "Recommended Action: Promote onboarding technical assistance."
        )

    # 4. Payment Method
    if rec.get("PaymentMethod") == "Electronic check":
        r_ec, n_ec = segment_stats(df["PaymentMethod"] == "Electronic check")
        notes.append(
            f"Payment by Electronic check: Churn rate is {r_ec:.1%} (n={n_ec:,}). "
            "Recommended Action: Incentivise automated credit card or bank auto-pay with a billing credit."
        )

    # 5. Tenure / New Customers
    tenure_val = pd.to_numeric(rec.get("tenure", 0), errors="coerce")
    if tenure_val <= 12:
        r_new, n_new = segment_stats(df["tenure"] <= 12)
        notes.append(
            f"New customer tenure (<= 12 months): High risk window with {r_new:.1%} churn rate (n={n_new:,}). "
            "Recommended Action: Deploy proactive customer success outreach in months 1-3."
        )

    if not notes:
        notes.append("No high-risk segments identified for this customer profile.")

    notes.append(
        "Note: These observations represent statistical associations from the training data, "
        "not guaranteed causal effects. Retention strategies should be tested with controlled A/B experiments."
    )
    return notes
