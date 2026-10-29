"""Page 8 — Prediction Interface (Single & Batch Prediction)."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.cards import render_header, render_metric_card
from dashboard.components.charts import (
    apply_theme,
    horizontal_bar,
    CHURN_COLOR,
    STAY_COLOR,
    RISK_COLORS,
)
from dashboard.services.data_service import get_churn_model, get_logreg_model
from src.data.load_data import ALLOWED, RAW_INPUT_COLUMNS, SERVICE_COLUMNS
from src.data.validate_data import validate_input
from src.features.feature_analysis import explain_single_customer_linear, compute_shap_single_customer
from src.models.predict import (
    DEFAULT_LOW_MAX,
    DEFAULT_HIGH_MIN,
    NOTEBOOK_THRESHOLD,
    predict_frame,
    generate_retention_observations,
)


def render_single_prediction_form(df: pd.DataFrame, model, lr_model, meta: dict | None):
    st.markdown("### 👤 Single Customer Real-Time Scoring")
    
    A = ALLOWED
    thr = float(meta.get("selected_threshold", NOTEBOOK_THRESHOLD)) if meta else NOTEBOOK_THRESHOLD

    with st.expander("Risk Threshold Controls", expanded=False):
        c_thr, c_bands = st.columns([1, 2])
        thr = c_thr.slider("Decision Threshold (Churn cutoff)", 0.05, 0.95, thr, 0.05, key="single_thr")
        low_max, high_min = c_bands.slider(
            "Risk Tier Boundaries: [Low < X] | [X <= Medium < Y] | [High >= Y]",
            0.05, 0.95, (DEFAULT_LOW_MAX, DEFAULT_HIGH_MIN), 0.05, key="single_bands"
        )

    # Customer Input Form
    with st.form("single_prediction_form"):
        col_profile, col_services, col_billing = st.columns(3)

        with col_profile:
            st.markdown("##### 1. Demographics")
            gender = st.selectbox("Gender", A["gender"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
            partner = st.selectbox("Partner", A["Partner"])
            dependents = st.selectbox("Dependents", A["Dependents"])
            tenure = st.slider("Tenure (Months with Provider)", 0, 72, 6)

        with col_services:
            st.markdown("##### 2. Subscribed Services")
            phone = st.selectbox("Phone Service", A["PhoneService"], index=1)
            lines = "No phone service" if phone == "No" else st.selectbox("Multiple Lines", ["No", "Yes"])
            internet = st.selectbox("Internet Service", A["InternetService"], index=1)
            
            addons = {}
            for s_col in SERVICE_COLUMNS:
                if internet == "No":
                    addons[s_col] = "No internet service"
                else:
                    addons[s_col] = st.selectbox(s_col, ["No", "Yes"], key=f"form_{s_col}")

        with col_billing:
            st.markdown("##### 3. Contract & Billing")
            contract = st.selectbox("Contract Type", A["Contract"])
            paperless = st.selectbox("Paperless Billing", A["PaperlessBilling"], index=1)
            payment = st.selectbox("Payment Method", A["PaymentMethod"], index=2)
            monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=0.5)
            calc_total = float(round(monthly * max(1, tenure), 2))
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=calc_total, step=10.0)

        submitted = st.form_submit_button("⚡ Predict Customer Churn Risk")

    if submitted:
        input_record = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            **addons,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        input_df = pd.DataFrame([input_record])[RAW_INPUT_COLUMNS]
        valid_df, invalid_df, messages = validate_input(input_df)

        for m in messages:
            st.warning(m)

        if not invalid_df.empty:
            st.error("Validation failed. Please correct input values.")
            st.dataframe(invalid_df)
            return

        result = predict_frame(model, valid_df, threshold=thr, low_max=low_max, high_min=high_min).iloc[0]
        prob = float(result["ChurnProbability"])
        predicted_class = result["PredictedChurn"]
        risk_tier = result["RiskCategory"]

        st.divider()
        st.markdown("#### 🎯 Prediction Results & Diagnostic")

        col_gauge, col_kpis = st.columns([3, 2])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Estimated Churn Probability", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1A202C"},
                    "steps": [
                        {"range": [0, low_max * 100], "color": "#D1FAE5"},
                        {"range": [low_max * 100, high_min * 100], "color": "#FEF3C7"},
                        {"range": [high_min * 100, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": "#D55E00", "width": 4}, "thickness": 0.8, "value": thr * 100},
                },
            ))
            st.plotly_chart(apply_theme(fig_gauge, height=340), use_container_width=True)

        with col_kpis:
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            render_metric_card(k1, "Predicted Outcome", "Likely to Churn" if predicted_class == "Yes" else "Likely to Stay")
            render_metric_card(k2, "Assigned Risk Tier", risk_tier)
            st.caption(f"Cutoff threshold: **{thr:.2f}**. Risk tiers: Low (<{low_max:.0%}), Medium, High (>={high_min:.0%}).")

        # Explainability Breakdown
        st.markdown("#### 🔍 Explainability: Why was this score assigned?")
        exp_tabs = st.tabs(["Logistic Regression Contributions", "Tree SHAP Local Explanation", "Retention Actions"])

        with exp_tabs[0]:
            linear_contrib = explain_single_customer_linear(lr_model, valid_df)
            if linear_contrib is not None:
                top_contrib = linear_contrib.reindex(linear_contrib.abs().sort_values(ascending=False).head(12).index).iloc[::-1]
                fig_lc = horizontal_bar(
                    top_contrib,
                    title="Top Feature Contributions to Customer's Score (Linear Log-Odds)",
                    xlabel="Contribution to Score",
                    colors=(top_contrib > 0).map({True: "Pushes Toward Churn (+)", False: "Pushes Toward Retention (-)"}),
                    color_map={"Pushes Toward Churn (+)": CHURN_COLOR, "Pushes Toward Retention (-)": STAY_COLOR},
                    height=420,
                )
                st.plotly_chart(fig_lc, use_container_width=True)
            else:
                st.info("Linear explainer not available.")

        with exp_tabs[1]:
            shap_vals = compute_shap_single_customer(model, valid_df)
            if shap_vals is not None:
                top_shap_local = shap_vals.reindex(shap_vals.abs().sort_values(ascending=False).head(12).index).iloc[::-1]
                fig_shap_local = horizontal_bar(
                    top_shap_local,
                    title="Tree SHAP Values for This Customer Record",
                    xlabel="SHAP Attribution (Log-Odds Impact)",
                    colors=(top_shap_local > 0).map({True: "Pushes Toward Churn (+)", False: "Pushes Toward Retention (-)"}),
                    color_map={"Pushes Toward Churn (+)": CHURN_COLOR, "Pushes Toward Retention (-)": STAY_COLOR},
                    height=420,
                )
                st.plotly_chart(fig_shap_local, use_container_width=True)
            else:
                st.info("Local SHAP calculation completed or tree explainer unavailable.")

        with exp_tabs[2]:
            st.markdown("##### 💡 Prescriptive Observations for Customer Success")
            notes = generate_retention_observations(valid_df.iloc[0], df, prob)
            for n in notes:
                st.markdown(f"- {n}")


def render_batch_prediction(df: pd.DataFrame, model, meta: dict | None):
    st.markdown("### 📁 Batch Customer Prediction Engine")
    st.markdown(
        "Upload a batch CSV or Excel file containing customer records. "
        "The engine validates input types, generates risk scores, and exports enriched predictions."
    )

    thr = float(meta.get("selected_threshold", NOTEBOOK_THRESHOLD)) if meta else NOTEBOOK_THRESHOLD

    c1, c2 = st.columns([1, 2])
    thr = c1.slider("Batch Threshold", 0.05, 0.95, thr, 0.05, key="batch_thr")
    low_max, high_min = c2.slider(
        "Batch Risk Tier Boundaries", 0.05, 0.95, (DEFAULT_LOW_MAX, DEFAULT_HIGH_MIN), 0.05, key="batch_bands"
    )

    # 5-Row Template Download
    template_df = df[RAW_INPUT_COLUMNS].head(5)
    st.download_button(
        "📥 Download 5-Row Template (CSV)",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Customer File (.csv or .xlsx)", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        return

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            upload_df = pd.read_csv(uploaded_file)
        else:
            upload_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    if upload_df.empty:
        st.warning("Uploaded file is empty.")
        return

    st.caption(f"Loaded {len(upload_df):,} records with {upload_df.shape[1]} columns.")
    valid_df, invalid_df, messages = validate_input(upload_df)

    for m in messages:
        st.warning(m)

    if not invalid_df.empty:
        st.error(f"{len(invalid_df):,} records failed validation and were excluded from scoring.")
        st.dataframe(invalid_df)
        st.download_button(
            "📥 Download Rejected Records with Diagnostic Reasons (CSV)",
            data=invalid_df.to_csv(index=False).encode("utf-8"),
            file_name="rejected_records.csv",
            mime="text/csv",
        )

    if valid_df.empty:
        st.error("No valid records remain to predict.")
        return

    with st.spinner(f"Scoring {len(valid_df):,} customer records..."):
        scored_df = predict_frame(model, valid_df, threshold=thr, low_max=low_max, high_min=high_min)

    # Re-attach extra columns from upload (like customerID)
    extra_cols = [c for c in upload_df.columns if c not in scored_df.columns]
    final_output = pd.concat([upload_df.loc[scored_df.index, extra_cols], scored_df], axis=1)

    st.markdown("#### 📊 Batch Prediction Summary")
    b1, b2, b3, b4 = st.columns(4)
    total_valid = len(final_output)
    churn_count = int((final_output["PredictedChurn"] == "Yes").sum())
    render_metric_card(b1, "Total Processed", f"{total_valid:,}")
    render_metric_card(b2, "Predicted Churners", f"{churn_count:,}")
    render_metric_card(b3, "Predicted Churn Rate", f"{churn_count / total_valid:.1%}")
    render_metric_card(b4, "High Risk Accounts", f"{(final_output['RiskCategory'] == 'High').sum():,}")

    col_risk_bar, col_prob_hist = st.columns(2)
    with col_risk_bar:
        risk_counts = final_output["RiskCategory"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]
        fig_rb = px.bar(
            risk_counts,
            x="Risk Tier",
            y="Count",
            color="Risk Tier",
            color_discrete_map=RISK_COLORS,
            text="Count",
            title="Distribution by Risk Tier",
        )
        st.plotly_chart(apply_theme(fig_rb, height=350), use_container_width=True)

    with col_prob_hist:
        fig_ph = px.histogram(
            final_output,
            x="ChurnProbability",
            nbins=30,
            title="Distribution of Churn Probabilities",
            color_discrete_sequence=["#0072B2"],
        )
        st.plotly_chart(apply_theme(fig_ph, height=350), use_container_width=True)

    st.dataframe(final_output.sort_values("ChurnProbability", ascending=False).head(500), use_container_width=True)

    st.download_button(
        "📥 Download Enriched Predictions (CSV)",
        data=final_output.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions_scored.csv",
        mime="text/csv",
    )


def render(df: pd.DataFrame, meta: dict | None):
    render_header(
        title="Interactive Inference & Prediction Interface",
        subtitle="Real-Time Scoring, Probability Quantification, and Batch Processing Engine",
        badge_text="Inference Engine",
    )

    model = get_churn_model()
    lr_model = get_logreg_model()

    if model is None:
        st.error("No trained machine learning pipeline found at `models/churn_pipeline.joblib`. Please run `python train_model.py` first.")
        return

    pred_mode = st.radio("Choose Mode:", ["Single Customer Scoring", "Batch Prediction File Upload"], horizontal=True)

    if pred_mode == "Single Customer Scoring":
        render_single_prediction_form(df, model, lr_model, meta)
    else:
        render_batch_prediction(df, model, meta)

