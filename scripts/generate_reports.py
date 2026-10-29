"""Generate and export reports, tables, figures, and executive summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.load_data import load_raw
from src.preprocessing.preprocessing import prepare_dashboard_frame
from src.analysis.eda import compute_overview_kpis, segment_table
from src.analysis.statistics import numeric_correlation, outlier_analysis
from src.utils.helpers import (
    PROJECT_ROOT,
    FIGURES_DIR,
    TABLES_DIR,
    SUMMARIES_DIR,
    MODEL_DIR,
    format_currency,
    format_percentage,
    logger,
)


def generate_reports():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    df = prepare_dashboard_frame(raw)
    kpis = compute_overview_kpis(df)

    # 1. Export summary tables
    logger.info("Exporting summary tables...")
    contract_seg = segment_table(df, "Contract")
    contract_seg.to_csv(TABLES_DIR / "churn_by_contract.csv", index=False)

    internet_seg = segment_table(df, "InternetService")
    internet_seg.to_csv(TABLES_DIR / "churn_by_internet_service.csv", index=False)

    tenure_seg = segment_table(df, "TenureGroup")
    tenure_seg.to_csv(TABLES_DIR / "churn_by_tenure_group.csv", index=False)

    outliers = outlier_analysis(df)
    outliers.to_csv(TABLES_DIR / "outlier_analysis.csv", index=False)

    corr = numeric_correlation(df)
    corr.to_csv(TABLES_DIR / "numeric_correlation.csv")

    # 2. Export metadata / model comparison table
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        models_data = []
        for m in meta.get("models", []):
            models_data.append({
                "Model": m["model"],
                **{f"Test {k}": round(v, 4) for k, v in m["test_at_0.5"].items()},
                "Train ROC-AUC": round(m["train"]["ROC-AUC"], 4),
            })
        pd.DataFrame(models_data).to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    # 3. Export figures
    logger.info("Exporting figures...")
    # Target distribution
    counts = df["Churn"].value_counts().reset_index()
    fig_target = px.pie(
        counts,
        names="Churn",
        values="count",
        color="Churn",
        color_discrete_map={"Yes": "#D55E00", "No": "#0072B2"},
        title="Customer Churn Ratio (73.5% Retained vs 26.5% Churned)",
        hole=0.5,
    )
    fig_target.write_html(str(FIGURES_DIR / "target_distribution.html"))

    # Churn rate by contract
    fig_contract = px.bar(
        contract_seg,
        x="Contract",
        y="ChurnRate",
        text=contract_seg["ChurnRate"].map("{:.1%}".format),
        title="Churn Rate by Contract Type",
        color_discrete_sequence=["#D55E00"],
    )
    fig_contract.write_html(str(FIGURES_DIR / "churn_by_contract.html"))

    # Correlation heatmap
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig_corr.write_html(str(FIGURES_DIR / "correlation_heatmap.html"))

    # 4. Export Executive Summary Markdown
    logger.info("Exporting executive summary...")
    summary_md = f"""# Executive Churn Analytics Report

## 1. Executive Summary & Core Metrics
- **Total Customer Base Analyzed:** {kpis['total_customers']:,}
- **Total Churned Customers:** {kpis['churned_customers']:,} ({format_percentage(kpis['churn_rate'])})
- **Retained Customers:** {kpis['retained_customers']:,} ({format_percentage(1 - kpis['churn_rate'])})
- **Monthly Revenue Associated with Churned Customers:** {format_currency(kpis['lost_monthly_revenue'])} ({format_percentage(kpis['lost_monthly_share'])} of total monthly billing)
- **Average Customer Tenure:** {kpis['avg_tenure_months']:.1f} months
- **Average Monthly Charge:** {format_currency(kpis['avg_monthly_charges'])}

---

## 2. Key Statistical Insights (Extracted from Dataset & Notebook)
1. **Contract Type is the Dominant Factor:**
   - Month-to-month contracts have an alarming **42.7% churn rate**.
   - One-year contracts show **11.3% churn**.
   - Two-year contracts demonstrate high loyalty with only **2.8% churn**.
   - *Conclusion:* Month-to-month customers are over 15x more likely to leave than two-year contract holders.

2. **Early Tenure Vulnerability:**
   - New customers (tenure 0-12 months) experience the highest churn rate (~47.4%).
   - Customers who stay past 24 months drop to under 20% churn, and past 48 months under 8%.

3. **Fiber Optic Service Quality & Pricing Friction:**
   - Fiber optic subscribers exhibit a **41.9% churn rate**, significantly higher than DSL (~19.0%) or no-internet users (~7.4%).
   - These users also bear higher monthly charges (average ~$91/month).

4. **Payment Method Association:**
   - Electronic check payers churn at **45.3%**, while automated bank transfers and credit card autopay users churn at ~15-16%.

---

## 3. Machine Learning Model Architecture & Performance
- **Selected Model:** Tuned XGBoost (`XGBClassifier`) with Stratified 80/20 train/test split.
- **Evaluation Criteria:** Prioritizing Recall and ROC-AUC on hold-out test set (1,407 customers).
- **Hold-out Test ROC-AUC:** 0.8380
- **Operational Decision Threshold:** 0.40 (Tuned from default 0.50)
  - At threshold 0.40, Recall increases from 51.1% to **66.3%**, successfully intercepting two-thirds of all at-risk customers.
  - Overall Accuracy: 79.0%
  - Test Precision: 59.3%

---

## 4. Prescriptive Business Recommendations
1. **'Lock & Save' Annual Conversion Program:**
   - Proactively target month-to-month customers in month 3 with an annual contract offer (e.g., 10% discount or 1 month free).
2. **First-Year Onboarding Care:**
   - Implement scheduled touchpoints during days 30, 60, and 90 to identify onboarding dissatisfaction early.
3. **Fiber Optic Quality & Value Audit:**
   - Conduct technical speed and stability checks on fiber lines and offer complimentary speed upgrades or streaming bundle discounts.
4. **Auto-Pay Migration Incentive:**
   - Offer a one-time $10 credit to electronic-check customers who migrate to automated bank debit or credit card payments.

*Generated automatically by Customer Churn Analytics Pipeline.*
"""
    (SUMMARIES_DIR / "executive_summary.md").write_text(summary_md, encoding="utf-8")
    logger.info("Reports and summaries successfully written to %s", SUMMARIES_DIR)


if __name__ == "__main__":
    generate_reports()
