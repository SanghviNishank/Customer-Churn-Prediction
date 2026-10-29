"""Page 5 — Feature Analysis."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_header
from dashboard.components.charts import horizontal_bar, CHURN_COLOR, STAY_COLOR
from dashboard.services.data_service import get_churn_model
from src.features.feature_analysis import compute_shap_for_sample


def render(df: pd.DataFrame, meta: dict | None):
    render_header(
        title="Feature Importance & Interpretability Analysis",
        subtitle="Decomposition of Model Drivers, Feature Weights, and SHAP Values",
        badge_text="Feature Analysis",
    )

    if meta is None:
        st.warning("Model metadata not found. Please run `python train_model.py` first.")
        return

    st.markdown("### 🔑 Key Drivers of Customer Churn")
    st.markdown(
        """
        Feature importance metrics reflect mathematical relationships discovered by the model in the historical training data.
        - **Tree Importance (XGBoost):** Measures the relative gain each feature split contributes to lowering log-loss.
        - **Logistic Regression Coefficients:** Standardized log-odds multipliers indicating whether a feature increases or decreases churn odds.
        - **SHAP (SHapley Additive exPlanations):** Game-theoretic attribution quantifying the marginal contribution of each feature to predictions.
        """
    )

    tabs = st.tabs([
        "XGBoost Feature Importance",
        "Logistic Regression Coefficients",
        "Global SHAP Explanations",
        "Feature Dictionary & Business Impact",
    ])

    with tabs[0]:
        imp = meta.get("feature_importance_selected")
        if imp:
            s_imp = pd.Series(imp).sort_values(ascending=False).head(20).iloc[::-1]
            fig_imp = horizontal_bar(
                s_imp,
                title=f"Top 20 Features by Model Importance ({meta['selected_model']})",
                xlabel="Relative Gain / Importance",
                single_color="#0072B2",
                height=520,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Contract type (Month-to-month), tenure, InternetService (Fiber optic), and TechSupport consistently dominate split decisions.")
        else:
            st.info("Feature importance is not supported by the selected model.")

    with tabs[1]:
        coefs = meta.get("logreg_coefficients")
        if coefs:
            s_coefs = pd.Series(coefs)
            top_coefs = s_coefs.reindex(s_coefs.abs().sort_values(ascending=False).head(20).index).iloc[::-1]
            
            direction = (top_coefs > 0).map({True: "Increases Churn Risk (+)", False: "Decreases Churn Risk (-)"})
            fig_coef = horizontal_bar(
                top_coefs,
                title="Top 20 Logistic Regression Feature Coefficients (Standardized Log-Odds)",
                xlabel="Coefficient Value (Log-Odds)",
                colors=direction,
                color_map={"Increases Churn Risk (+)": CHURN_COLOR, "Decreases Churn Risk (-)": STAY_COLOR},
                height=520,
            )
            st.plotly_chart(fig_coef, use_container_width=True)
            st.caption("Standardized continuous variables and one-hot categorical features allow direct magnitude comparisons.")
        else:
            st.info("Logistic regression coefficients not available.")

    with tabs[2]:
        model = get_churn_model()
        if model is None:
            st.warning("Model file could not be loaded for SHAP calculation.")
        else:
            st.markdown("Calculate global SHAP values across an empirical sample of customer records:")
            sample_size = st.slider("Customer Sample Size", 100, 500, 200, 50)
            if st.button("Compute Global SHAP Importances", key="btn_global_shap"):
                with st.spinner("Computing Tree SHAP values across feature space..."):
                    shap_series = compute_shap_for_sample(model, df, n_samples=sample_size)
                if shap_series is not None:
                    top_shap = shap_series.head(15).iloc[::-1]
                    fig_shap = horizontal_bar(
                        top_shap,
                        title=f"Mean |SHAP Value| Across {sample_size} Customers",
                        xlabel="Mean Absolute SHAP Value (Impact on Log-Odds)",
                        single_color="#009E73",
                        height=480,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.error("SHAP computation failed or shap library is unavailable.")

    with tabs[3]:
        feature_dict = pd.DataFrame([
            {"Feature": "Contract_Month-to-month", "Type": "Categorical (Engineered)", "Target Relationship": "Strong Positive (Increases Churn)", "Strategic Interpretation": "High flexibility with zero exit barriers leads to rapid attrition."},
            {"Feature": "tenure", "Type": "Numerical (Months)", "Target Relationship": "Strong Negative (Decreases Churn)", "Strategic Interpretation": "Subscribers past the first year build habit and brand loyalty."},
            {"Feature": "InternetService_Fiber optic", "Type": "Categorical", "Target Relationship": "Positive (Increases Churn)", "Strategic Interpretation": "High monthly expense ($80-$100+) and service expectations create pricing sensitivity."},
            {"Feature": "TotalCharges", "Type": "Numerical ($)", "Target Relationship": "Negative (Cumulative Value)", "Strategic Interpretation": "Correlates strongly with tenure; loyal customers accumulate high total spend."},
            {"Feature": "PaymentMethod_Electronic check", "Type": "Categorical", "Target Relationship": "Positive (Increases Churn)", "Strategic Interpretation": "Manual payment process creates recurring monthly friction compared to auto-debit."},
            {"Feature": "TechSupport_No", "Type": "Categorical", "Target Relationship": "Positive (Increases Churn)", "Strategic Interpretation": "Lack of technical problem resolution leads directly to customer frustration."},
            {"Feature": "OnlineSecurity_No", "Type": "Categorical", "Target Relationship": "Positive (Increases Churn)", "Strategic Interpretation": "Fewer protective digital services lead to weaker account stickiness."},
            {"Feature": "NumberOfServices", "Type": "Numerical (Count 0-6)", "Target Relationship": "Negative (Decreases Churn)", "Strategic Interpretation": "Multi-product adoption creates strong ecosystem switching barriers."},
        ])
        st.dataframe(feature_dict, use_container_width=True, hide_index=True)

