"""Page 6 — Model Training and Evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from dashboard.components.cards import render_header, render_metric_card
from dashboard.components.charts import (
    apply_theme,
    confusion_matrix_plot,
    roc_pr_curves_plot,
)
from dashboard.services.data_service import get_test_predictions


def render(meta: dict | None):
    render_header(
        title="Model Training, Benchmarking & Evaluation",
        subtitle="Hold-Out Test Validation, Threshold Optimization, and Performance Diagnostics",
        badge_text="Model Performance",
    )

    if meta is None:
        st.error("No model metadata found. Please run `python train_model.py`.")
        return

    tp_df = get_test_predictions()
    if tp_df is None:
        st.error("Hold-out predictions file not found (`models/test_predictions.csv`).")
        return

    sp = meta.get("split", {})
    st.info(
        f"**Rigorous Evaluation Design:** All models evaluated on an untouched, stratified 20% hold-out "
        f"test set ({sp.get('n_test', 1407):,} customers, {sp.get('test_churn_rate', 0.266):.1%} churn rate, "
        f"random_state={sp.get('random_state', 42)}). Training set: {sp.get('n_train', 5625):,} customers."
    )

    tabs = st.tabs([
        "Model Comparison Benchmark",
        "Threshold Tuning & Dynamic KPI Impact",
        "ROC & Precision-Recall Curves",
        "Confusion Matrix Diagnostic",
        "Threshold Sensitivity Sweep",
        "Full Classification Report",
    ])

    # 1. Comparison Benchmark
    with tabs[0]:
        rows = []
        for r in meta["models"]:
            rows.append({
                "Model": r["model"],
                **{f"Test {k}": v for k, v in r["test_at_0.5"].items()},
                "Train ROC-AUC": r["train"]["ROC-AUC"],
                "Params Origin": r["params_source"],
            })
        comp_df = pd.DataFrame(rows)

        # Plot comparison
        metrics_cols = [c for c in comp_df.columns if c.startswith("Test ") and "Accuracy" not in c]
        long_df = comp_df.melt(id_vars="Model", value_vars=metrics_cols, var_name="Metric", value_name="Value")
        long_df["Metric"] = long_df["Metric"].str.replace("Test ", "", regex=False)

        fig_comp = px.bar(
            long_df,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            range_y=[0, 1.0],
            title="Comparison of Model Metrics on 20% Hold-Out Test Set (Threshold 0.50)",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(apply_theme(fig_comp, height=440), use_container_width=True)

        st.subheader("Hold-Out Test Set Benchmark Results")
        format_dict = {c: "{:.3f}" for c in comp_df.columns if c not in ("Model", "Params Origin")}
        st.dataframe(comp_df.style.format(format_dict), use_container_width=True, hide_index=True)
        st.caption("Train ROC-AUC is displayed to monitor potential overfitting. Test ROC-AUC and Test Recall govern final selection.")

    # 2. Threshold Tuning
    selected_model_name = meta["selected_model"]
    y_true = tp_df["y_true"].to_numpy()
    y_prob = tp_df[selected_model_name].to_numpy()

    with tabs[1]:
        st.markdown(f"### Fine-Tuning Decision Threshold for `{selected_model_name}`")
        st.markdown(
            """
            In customer churn prevention, **False Negatives (missing a churner)** are far more costly than **False Positives (contacting a customer who might stay)**.
            Lowering the decision threshold from 0.50 to **0.40** significantly increases Recall with an acceptable trade-off in Precision.
            """
        )

        cv_info = meta.get("cv_roc_auc_5fold_on_train", {})
        if cv_info:
            st.write(f"5-Fold Stratified Cross-Validation ROC-AUC: **{cv_info['mean']:.4f} ± {cv_info['std']:.4f}**")

        threshold = st.slider(
            "Adjust Operational Decision Threshold:",
            min_value=0.10,
            max_value=0.90,
            value=float(meta.get("selected_threshold", 0.40)),
            step=0.05,
        )

        y_pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, y_prob)

        k1, k2, k3, k4, k5 = st.columns(5)
        render_metric_card(k1, "Accuracy", f"{acc:.3f}")
        render_metric_card(k2, "Precision", f"{prec:.3f}", help_text="Of flagged churners, how many actually left.")
        render_metric_card(k3, "Recall", f"{rec:.3f}", help_text="Proportion of actual churners successfully caught.")
        render_metric_card(k4, "F1-Score", f"{f1:.3f}")
        render_metric_card(k5, "ROC-AUC", f"{roc:.3f}", help_text="Threshold-invariant ranking power.")

        st.caption(f"At threshold {threshold:.2f}, the model intercepts {rec:.1%} of all churners.")

    # 3. ROC / PR Curves
    with tabs[2]:
        all_models = [c for c in tp_df.columns if c != "y_true"]
        models_to_plot = st.multiselect(
            "Select models to display on curves:",
            all_models,
            default=[selected_model_name, "LogisticRegression (notebook-tuned)"],
        )
        if models_to_plot:
            fig_roc, fig_pr = roc_pr_curves_plot(tp_df, models_to_plot)
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(fig_roc, use_container_width=True)
            col_b.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.warning("Please select at least one model to display curves.")

    # 4. Confusion Matrix
    with tabs[3]:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        st.plotly_chart(
            confusion_matrix_plot(cm.tolist(), title=f"Confusion Matrix ({selected_model_name} @ Threshold {threshold:.2f})"),
            use_container_width=True,
        )
        tn, fp, fn, tp = cm.ravel()
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("True Negatives (Correct Stays)", f"{tn:,}")
        m_c2.metric("False Positives (False Alarms)", f"{fp:,}")
        m_c3.metric("False Negatives (Missed Churners)", f"{fn:,}")
        m_c4.metric("True Positives (Caught Churners)", f"{tp:,}")

    # 5. Threshold Sensitivity Sweep
    with tabs[4]:
        sweep_data = meta.get("threshold_table", [])
        if sweep_data:
            sweep_df = pd.DataFrame(sweep_data)
            st.dataframe(sweep_df.style.format("{:.3f}"), use_container_width=True, hide_index=True)

            fig_sweep = px.line(
                sweep_df,
                x="threshold",
                y=["Precision", "Recall", "F1", "Accuracy"],
                markers=True,
                title="Trade-off Curves Across Decision Thresholds",
            )
            st.plotly_chart(apply_theme(fig_sweep, height=420), use_container_width=True)

    # 6. Full Classification Report
    with tabs[5]:
        report = classification_report(y_true, y_pred, target_names=["Retained (No)", "Churned (Yes)"], output_dict=True)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)

