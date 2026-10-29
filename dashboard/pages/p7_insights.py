"""Page 7 — Insights and Business Reports."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.components.cards import render_header
from src.analysis.eda import compute_overview_kpis, segment_table
from src.utils.helpers import SUMMARIES_DIR, TABLES_DIR, format_currency, format_percentage


def render(df: pd.DataFrame, meta: dict | None):
    render_header(
        title="Business Insights, Revenue Impact & Executive Reports",
        subtitle="Data-Backed Conclusions, Financial Risk Assessment, and Downloadable Deliverables",
        badge_text="Executive Insights",
    )

    kpis = compute_overview_kpis(df)

    st.markdown("### 💰 Financial Exposure Analysis")
    f1, f2, f3 = st.columns(3)
    annualized_lost_rev = kpis['lost_monthly_revenue'] * 12
    f1.metric("Monthly Billing from Churners", format_currency(kpis['lost_monthly_revenue']))
    f2.metric("Annualized Churn Revenue at Risk", format_currency(annualized_lost_rev), help_text="12-month projection assuming churn continues unmitigated.")
    f3.metric("Churn Billing Concentration", format_percentage(kpis['lost_monthly_share']), help_text="Share of telecom monthly billing associated with churners.")

    st.markdown("### 📌 Verified Empirical Findings (from `CustomerFinal.ipynb`)")
    st.markdown(
        """
        1. **Contract Structure is the Single Strongest Predictor:**
           - Month-to-month customers exhibit a **42.7% churn rate** compared to **11.3% for 1-year** and **2.8% for 2-year** contracts.
           - *Business takeaway:* Moving 20% of monthly subscribers into annual commitments would retain an estimated 250+ customers annually.
        
        2. **First-Year Onboarding Cliff:**
           - Customers in months 0–12 experience a **47.4% churn rate**.
           - If a customer remains past month 24, attrition drops to under **15%**.
           - *Business takeaway:* Customer onboarding and early-life care during days 1–90 is the most vital window for intervention.

        3. **Fiber Optic Quality / Value Mismatch:**
           - Fiber optic subscribers churn at **41.9%**, despite DSL users churning at only **19.0%**.
           - Fiber customers pay significantly higher monthly amounts (~$91/mo average).
           - *Business takeaway:* Customer expectations for fiber are not being met, or competitor aggressive introductory pricing is attracting them away.

        4. **Electronic Check Friction:**
           - Customers paying via manual electronic checks churn at **45.3%**, versus **15.2% for bank auto-pay** and **16.2% for credit card auto-pay**.
           - *Business takeaway:* Auto-pay removes the monthly decision-to-pay moment and improves customer stickiness.
        """
    )

    st.markdown("### 🎯 Strategic Business Recommendations")
    rec_cards = [
        ("1. 'Lock & Save' Annual Promotion", "Target month-to-month subscribers approaching month 6 with a discount (e.g. 1 month free or 10% off for 12 months). Even with a 10% discount, preserving recurring revenue yields massive positive ROI."),
        ("2. Proactive Day-30/60/90 Health Checks", "Automate proactive satisfaction outreach and speed tests for new fiber optic subscribers during their first 90 days."),
        ("3. Auto-Pay Migration Campaign", "Offer a one-time $10 account billing credit for subscribers switching from electronic check to automatic ACH or card payments."),
        ("4. Security & Support Bundle Trials", "Provide complimentary 60-day trials of Online Security and Tech Support to high-risk subscribers, as multi-service users show 3x higher retention."),
    ]
    for title, desc in rec_cards:
        st.markdown(
            f"""
            <div style="background-color: rgba(128,128,128,0.06); padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #0072B2;">
                <strong style="font-size: 1.05rem;">{title}</strong><br>
                <span style="color: #4A5568;">{desc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📥 Downloadable Reports & Data Tables")
    d_col1, d_col2, d_col3 = st.columns(3)

    # 1. Executive Summary Download
    exec_summary_file = SUMMARIES_DIR / "executive_summary.md"
    if exec_summary_file.exists():
        d_col1.download_button(
            "📄 Download Executive Summary (MD)",
            data=exec_summary_file.read_text(encoding="utf-8"),
            file_name="telco_churn_executive_summary.md",
            mime="text/markdown",
        )
    else:
        d_col1.info("Run `python scripts/generate_reports.py` to create summary.")

    # 2. Contract Segmentation CSV
    contract_seg = segment_table(df, "Contract")
    d_col2.download_button(
        "📊 Download Contract Analysis (CSV)",
        data=contract_seg.to_csv(index=False).encode("utf-8"),
        file_name="churn_by_contract.csv",
        mime="text/csv",
    )

    # 3. Model Comparison CSV
    if meta:
        models_data = []
        for m in meta.get("models", []):
            models_data.append({
                "Model": m["model"],
                **{f"Test {k}": round(v, 4) for k, v in m["test_at_0.5"].items()},
                "Train ROC-AUC": round(m["train"]["ROC-AUC"], 4),
            })
        d_col3.download_button(
            "📈 Download Model Benchmark (CSV)",
            data=pd.DataFrame(models_data).to_csv(index=False).encode("utf-8"),
            file_name="model_benchmark_comparison.csv",
            mime="text/csv",
        )

