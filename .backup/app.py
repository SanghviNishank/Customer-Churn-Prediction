"""Customer Churn dashboard (Streamlit).

Run:  python -m streamlit run app.py
Everything here is read-only with respect to the dataset and the original notebooks.
The model is loaded from models/churn_pipeline.joblib (created by train_model.py); it is
never trained at dashboard start-up and no user-supplied model files are ever loaded.
"""
from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src import churn_core as cc

st.set_page_config(page_title="Customer Churn Dashboard", page_icon=":bar_chart:", layout="wide")

CHURN_COLOR, STAY_COLOR = "#D55E00", "#0072B2"  # colour-blind-safe pair
COLOR_MAP = {"Yes": CHURN_COLOR, "No": STAY_COLOR}
RISK_COLORS = {"Low": "#009E73", "Medium": "#E69F00", "High": CHURN_COLOR}
TEMPLATE = "plotly_white"
MAX_UPLOAD_ROWS = 50_000

st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem; max-width: 1400px;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 12px 16px;
                                 border-radius: 10px;}
    </style>""", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def hbar(s: pd.Series, title: str, xlabel: str, colors: pd.Series | None = None,
         cmap: dict | None = None, single_color: str = "#0072B2") -> go.Figure:
    """Horizontal bar chart from a Series (index = feature names, order = bottom to top)."""
    d = pd.DataFrame({"Feature": s.index.astype(str), "Value": s.to_numpy()})
    if colors is not None:
        d["Direction"] = colors.to_numpy()
        fig = px.bar(d, x="Value", y="Feature", orientation="h", color="Direction",
                     color_discrete_map=cmap, title=title, labels={"Value": xlabel})
    else:
        fig = px.bar(d, x="Value", y="Feature", orientation="h", title=title,
                     labels={"Value": xlabel}, color_discrete_sequence=[single_color])
    fig.update_yaxes(categoryorder="array", categoryarray=d["Feature"].tolist())
    return fig


# --------------------------------------------------------------------------- loaders
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = cc.load_raw()
    return raw, cc.prepare_dashboard_frame(raw)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Fixed project path only. Uploaded files are never passed to joblib.
    if not cc.MODEL_PATH.exists():
        return None
    return joblib.load(cc.MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_logreg():
    p = cc.MODEL_DIR / "logreg_pipeline.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame | None:
    return pd.read_csv(cc.TEST_PRED_PATH) if cc.TEST_PRED_PATH.exists() else None


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(Customers=("ChurnFlag", "size"),
                                           Churned=("ChurnFlag", "sum"),
                                           AvgMonthly=("MonthlyCharges", "mean")).reset_index()
    g["ChurnRate"] = g["Churned"] / g["Customers"]
    g["ShareOfCustomers"] = g["Customers"] / g["Customers"].sum()
    return g


# ----------------------------------------------------------------------------- pages
def page_overview(df: pd.DataFrame) -> None:
    st.title("Executive overview")
    n, churned = len(df), int(df["ChurnFlag"].sum())
    c = st.columns(5)
    c[0].metric("Customers", f"{n:,}")
    c[1].metric("Churned", f"{churned:,}")
    c[2].metric("Retained", f"{n - churned:,}")
    c[3].metric("Churn rate", f"{churned / n:.1%}")
    c[4].metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():,.2f}")
    c = st.columns(3)
    c[0].metric("Avg tenure (months)", f"{df['tenure'].mean():.1f}")
    lost = df.loc[df["ChurnFlag"] == 1, "MonthlyCharges"].sum()
    c[1].metric("Monthly charges of churned customers", f"${lost:,.0f}",
                help="Sum of MonthlyCharges for customers labelled Churn = Yes. This is "
                     "recurring revenue already associated with churners, not a forecast.")
    c[2].metric("Share of monthly billing from churned customers",
                f"{lost / df['MonthlyCharges'].sum():.1%}")
    st.caption("Figures use the cleaned data (11 rows with blank TotalCharges removed, as in the "
               "notebook). Revenue figures are historical sums, not predictions.")

    left, right = st.columns(2)
    with left:
        counts = df[cc.TARGET].value_counts().rename_axis("Churn").reset_index(name="Customers")
        fig = px.pie(counts, names="Churn", values="Customers", hole=0.55, color="Churn",
                     color_discrete_map=COLOR_MAP, title="Churned vs retained")
        st.plotly_chart(style(fig, 380))
    with right:
        seg = st.selectbox("Segment by", ["Contract", "InternetService", "PaymentMethod",
                                           "TenureGroup", "PaperlessBilling", "SeniorCitizen"])
        t = segment_table(df, seg)
        fig = px.bar(t, x=seg, y="ChurnRate", text=t["ChurnRate"].map("{:.1%}".format),
                     hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1%"},
                     title=f"Churn rate by {seg}", color_discrete_sequence=[CHURN_COLOR])
        fig.update_yaxes(tickformat=".0%", title="Churn rate (churned / customers in segment)")
        st.plotly_chart(style(fig, 380))
    st.subheader("Segment summary")
    t = segment_table(df, seg)
    st.dataframe(t.style.format({"AvgMonthly": "${:,.2f}", "ChurnRate": "{:.1%}",
                                 "ShareOfCustomers": "{:.1%}"}), hide_index=True)


def page_eda(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    st.title("Exploratory data analysis")
    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        contracts = f1.multiselect("Contract", cc.ALLOWED["Contract"], default=cc.ALLOWED["Contract"])
        internet = f2.multiselect("Internet service", cc.ALLOWED["InternetService"],
                                  default=cc.ALLOWED["InternetService"])
        tmin, tmax = f3.slider("Tenure (months)", 0, 72, (0, 72))
    d = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)
           & df["tenure"].between(tmin, tmax)]
    st.caption(f"{len(d):,} of {len(df):,} customers match the filters.")
    if d.empty:
        st.warning("No customers match the current filters.")
        return

    tabs = st.tabs(["Target", "Numeric", "Categorical", "Relationships", "Correlation", "Missing values"])
    with tabs[0]:
        c = d[cc.TARGET].value_counts().rename_axis("Churn").reset_index(name="Customers")
        fig = px.bar(c, x="Churn", y="Customers", color="Churn", color_discrete_map=COLOR_MAP,
                     text="Customers", title="Target distribution")
        st.plotly_chart(style(fig, 380))
        if len(c) == 2:
            st.write(f"Imbalance ratio (majority : minority) = "
                     f"{c['Customers'].max() / c['Customers'].min():.2f} : 1")
    with tabs[1]:
        col = st.selectbox("Numeric feature", ["tenure", "MonthlyCharges", "TotalCharges", "NumberOfServices"])
        fig = px.histogram(d, x=col, color=cc.TARGET, nbins=40, barmode="overlay", opacity=0.65,
                           color_discrete_map=COLOR_MAP, marginal="box",
                           title=f"Distribution of {col} by churn")
        st.plotly_chart(style(fig, 460))
        st.dataframe(d.groupby(cc.TARGET)[col].describe().round(2))
    with tabs[2]:
        col = st.selectbox("Categorical feature", cc.CATEGORICAL_FEATURES, index=13)
        t = segment_table(d, col)
        fig = px.bar(t, x=col, y="ChurnRate", color_discrete_sequence=[CHURN_COLOR],
                     text=t["ChurnRate"].map("{:.1%}".format),
                     hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1%"},
                     title=f"Churn rate by {col}")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 420))
    with tabs[3]:
        kind = st.radio("Plot", ["Scatter", "Box", "Violin"], horizontal=True)
        if kind == "Scatter":
            sample = d.sample(min(len(d), 2500), random_state=cc.RANDOM_STATE)
            fig = px.scatter(sample, x="tenure", y="MonthlyCharges", color=cc.TARGET, opacity=0.6,
                             color_discrete_map=COLOR_MAP, hover_data=["Contract", "InternetService"],
                             title="Tenure vs monthly charges (random sample of up to 2,500)")
        else:
            y = st.selectbox("Numeric variable", ["tenure", "MonthlyCharges", "TotalCharges"])
            x = st.selectbox("Group by", ["Contract", "InternetService", "PaymentMethod", "TenureGroup"])
            fn = px.box if kind == "Box" else px.violin
            kwargs = {"box": True} if kind == "Violin" else {}
            fig = fn(d, x=x, y=y, color=cc.TARGET, color_discrete_map=COLOR_MAP,
                     title=f"{y} by {x} and churn", **kwargs)
        st.plotly_chart(style(fig, 480))
    with tabs[4]:
        num = d[["tenure", "MonthlyCharges", "TotalCharges", "NumberOfServices", "ChurnFlag"]]
        corr = num.corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation (Pearson) of numeric features and churn flag")
        st.plotly_chart(style(fig, 460))
        st.caption("Correlation describes association only, not causation.")
    with tabs[5]:
        miss = pd.DataFrame({
            "Missing (NaN)": raw.isna().sum(),
            "Blank strings": raw.apply(lambda s: s.astype(str).str.strip().eq("").sum()),
        })
        miss["Total problems"] = miss.sum(axis=1)
        miss["Percent"] = (miss["Total problems"] / len(raw) * 100).round(2)
        st.write("Computed on the raw file, before cleaning.")
        st.dataframe(miss[miss["Total problems"] > 0] if (miss["Total problems"] > 0).any() else miss)
        if miss["Total problems"].sum() == 0:
            st.success("No missing values or blank strings found.")


def page_segments(df: pd.DataFrame) -> None:
    st.title("Customer segmentation")
    st.caption("Churn rate = churned customers / customers in that segment. Segment size is shown "
               "alongside so small groups are not over-read.")
    df = df.copy()
    df["ChargeGroup"] = pd.qcut(df["MonthlyCharges"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    dim = st.selectbox("Segment", ["Contract", "TenureGroup", "PaymentMethod", "InternetService",
                                   "ChargeGroup", "NumberOfServices"])
    t = segment_table(df, dim)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(t, x=dim, y="ChurnRate", color="ChurnRate", color_continuous_scale="OrRd",
                     text=t["ChurnRate"].map("{:.1%}".format),
                     hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1%"},
                     title=f"Churn rate by {dim}")
        fig.update_yaxes(tickformat=".0%")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig, 400))
    with c2:
        fig = px.bar(t, x=dim, y="Customers", color_discrete_sequence=["#7f7f7f"],
                     text="Customers", title=f"Segment sizes ({dim})")
        st.plotly_chart(style(fig, 400))

    st.subheader("Animated comparison across tenure groups")
    other = st.selectbox("Compare this segment within each tenure group", ["Contract", "InternetService",
                                                                           "PaymentMethod"])
    g = df.groupby(["TenureGroup", other], observed=True).agg(
        Customers=("ChurnFlag", "size"), Churned=("ChurnFlag", "sum")).reset_index()
    g["ChurnRate"] = g["Churned"] / g["Customers"]
    g["TenureGroup"] = pd.Categorical(g["TenureGroup"], cc.TENURE_LABELS, ordered=True)
    g = g.sort_values("TenureGroup")
    fig = px.bar(g, x=other, y="ChurnRate", animation_frame="TenureGroup", range_y=[0, 1],
                 hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1%"},
                 color_discrete_sequence=[CHURN_COLOR],
                 title=f"Churn rate by {other}, stepping through tenure groups")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 440))
    st.caption("The animation steps through tenure groups (New 0-12, Regular 13-24, Established "
               "25-48, Loyal 49-72 months) with a fixed 0-100% axis so frames are comparable. "
               "Hover shows the segment size; very small cells are noisy.")

    st.subheader("Two-way view")
    a, b = st.columns(2)
    r = a.selectbox("Rows", ["Contract", "InternetService", "PaymentMethod", "TenureGroup"], key="r")
    c = b.selectbox("Columns", ["TenureGroup", "Contract", "InternetService", "PaymentMethod"], key="c")
    if r == c:
        st.info("Choose two different dimensions.")
    else:
        pv = df.pivot_table(index=r, columns=c, values="ChurnFlag", aggfunc="mean", observed=True)
        fig = px.imshow(pv, text_auto=".0%", color_continuous_scale="OrRd", aspect="auto",
                        title=f"Churn rate: {r} x {c}")
        st.plotly_chart(style(fig, 420))


def confusion_heatmap(cm: np.ndarray, title: str) -> go.Figure:
    labels = ["No churn", "Churn"]
    fig = px.imshow(cm, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
                    text_auto=True, color_continuous_scale="Blues", title=title)
    fig.update_coloraxes(showscale=False)
    return fig


def page_model() -> None:
    st.title("Model performance")
    meta, tp = cc.load_metadata(), load_test_predictions()
    if meta is None or tp is None:
        st.error("No model results found. Run `python train_model.py` first (see README).")
        return
    if not meta.get("xgboost_available_at_training", True):
        st.warning("This model was trained without XGBoost installed, so the notebook's selected "
                   "model (XGBoost) was skipped. Install xgboost and re-run train_model.py to "
                   "reproduce the notebook's final model.")
    sp = meta["split"]
    st.info(f"**Evaluation data:** a stratified 20% hold-out test set ({sp['n_test']:,} customers, "
            f"{sp['test_churn_rate']:.1%} churn; random_state={sp['random_state']}) that the models "
            f"never saw during fitting. Training set: {sp['n_train']:,} customers.")

    rows = []
    for r in meta["models"]:
        rows.append({"Model": r["model"], **{f"Test {k}": v for k, v in r["test_at_0.5"].items()},
                     "Train ROC-AUC": r["train"]["ROC-AUC"]})
    comp = pd.DataFrame(rows)
    st.subheader("Model comparison (decision threshold 0.5)")
    metrics_cols = [c for c in comp.columns if c.startswith("Test ")]
    long = comp.melt(id_vars="Model", value_vars=metrics_cols, var_name="Metric", value_name="Value")
    long["Metric"] = long["Metric"].str.replace("Test ", "", regex=False)
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group", range_y=[0, 1],
                 title="Test-set metrics by model", color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(style(fig, 440))
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if c != "Model"}), hide_index=True)
    st.caption("Train ROC-AUC is shown only to help spot overfitting; judge models by the test columns.")

    sel = meta["selected_model"]
    st.subheader(f"Selected model: {sel}")
    cv = meta["cv_roc_auc_5fold_on_train"]
    st.write(f"5-fold cross-validated ROC-AUC on the training set: **{cv['mean']:.4f} +/- {cv['std']:.4f}**.")
    thr = st.slider("Decision threshold (probability at or above which a customer is flagged as churn)",
                    0.05, 0.95, float(meta["selected_threshold"]), 0.05)
    y, p = tp["y_true"].to_numpy(), tp[sel].to_numpy()
    pred = (p >= thr).astype(int)
    c = st.columns(5)
    c[0].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
    c[1].metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}")
    c[2].metric("Recall", f"{recall_score(y, pred, zero_division=0):.3f}")
    c[3].metric("F1", f"{f1_score(y, pred, zero_division=0):.3f}")
    c[4].metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy**: share of all customers classified correctly (can look good on imbalanced data).\n"
            "- **Precision**: of customers flagged as churn, how many really churned (false alarms lower it).\n"
            "- **Recall**: of customers who really churned, how many were flagged (missed churners lower it).\n"
            "- **F1**: harmonic mean of precision and recall.\n"
            "- **ROC-AUC**: how well the probabilities rank churners above non-churners, independent of threshold.\n"
            "- Lowering the threshold usually raises recall and lowers precision. The notebook "
            "chose 0.4 for its final model.")

    t1, t2, t3, t4, t5 = st.tabs(["Confusion matrix", "ROC / PR curves", "Threshold sweep (animated)",
                                  "Classification report", "Feature importance"])
    with t1:
        cm = confusion_matrix(y, pred)
        st.plotly_chart(style(confusion_heatmap(cm, f"Confusion matrix at threshold {thr:.2f}"), 420))
    with t2:
        models = st.multiselect("Models", [m for m in tp.columns if m != "y_true"], default=[sel])
        fig1, fig2 = go.Figure(), go.Figure()
        for m in models:
            fpr, tpr, _ = roc_curve(y, tp[m])
            fig1.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC {roc_auc_score(y, tp[m]):.3f})")
            pr, rc, _ = precision_recall_curve(y, tp[m])
            fig2.add_scatter(x=rc, y=pr, mode="lines", name=m)
        fig1.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey"))
        fig1.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        fig2.add_hline(y=float(y.mean()), line_dash="dash", line_color="grey",
                       annotation_text="Churn base rate")
        fig2.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
        a, b = st.columns(2)
        a.plotly_chart(style(fig1, 430))
        b.plotly_chart(style(fig2, 430))
    with t3:
        frames = []
        for t in np.round(np.arange(0.1, 0.91, 0.1), 2):
            pr_ = (p >= t).astype(int)
            tn, fp, fn, tp_ = confusion_matrix(y, pr_, labels=[0, 1]).ravel()
            for name, val in [("True negative", tn), ("False positive", fp),
                              ("False negative", fn), ("True positive", tp_)]:
                frames.append({"Threshold": f"{t:.1f}", "Outcome": name, "Customers": int(val)})
        fig = px.bar(pd.DataFrame(frames), x="Outcome", y="Customers", color="Outcome",
                     animation_frame="Threshold", range_y=[0, len(y)],
                     color_discrete_sequence=["#0072B2", "#E69F00", "#D55E00", "#009E73"],
                     title="How the confusion-matrix counts change as the threshold moves")
        st.plotly_chart(style(fig, 440))
        st.caption("Press play or drag the slider. Raising the threshold converts true positives into "
                   "false negatives, and false positives into true negatives.")
        st.dataframe(pd.DataFrame(meta["threshold_table"]).style.format("{:.3f}"), hide_index=True)
    with t4:
        rep = classification_report(y, pred, target_names=["No Churn", "Churn"], output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3))
    with t5:
        imp = meta.get("feature_importance_selected")
        if imp:
            s = pd.Series(imp).sort_values(ascending=False).head(15).iloc[::-1]
            fig = hbar(s, "Top 15 model feature importances (selected model)", "Importance")
            st.plotly_chart(style(fig, 520))
            st.caption("Tree-based importances describe how much the model used a feature. They "
                       "do not say that changing the feature would change churn.")
        else:
            st.info("The selected model does not expose feature importances.")


def input_form(df: pd.DataFrame) -> pd.DataFrame | None:
    """Collect one customer record. Dependent fields are forced to valid combinations."""
    A = cc.ALLOWED
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Profile**")
        gender = st.selectbox("Gender", A["gender"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
        partner = st.selectbox("Partner", A["Partner"])
        dependents = st.selectbox("Dependents", A["Dependents"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", A["PhoneService"], index=1)
        lines = "No phone service" if phone == "No" else st.selectbox("Multiple lines", ["No", "Yes"])
        internet = st.selectbox("Internet service", A["InternetService"], index=1)
        addons = {}
        for col in cc.SERVICE_COLUMNS:
            if internet == "No":
                addons[col] = "No internet service"
            else:
                addons[col] = st.selectbox(col, ["No", "Yes"], key=f"in_{col}")
        if internet == "No":
            st.caption("Add-on services are set to 'No internet service' because there is no internet plan.")
    with c3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", A["Contract"])
        paperless = st.selectbox("Paperless billing", A["PaperlessBilling"], index=1)
        pay = st.selectbox("Payment method", A["PaymentMethod"])
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.05)
        total = st.number_input("Total charges ($)", 0.0, 20000.0, float(round(monthly * tenure, 2)), 1.0,
                                help="Defaults to monthly x tenure; edit if you know the real value.")
    rec = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
           "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
           **addons, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": pay,
           "MonthlyCharges": monthly, "TotalCharges": total}
    return pd.DataFrame([rec])[cc.RAW_INPUT_COLUMNS]


def retention_notes(rec: pd.Series, df: pd.DataFrame, proba: float) -> list[str]:
    """Cautious, data-derived observations (associations in this dataset, not causal claims)."""
    base = df["ChurnFlag"].mean()
    notes = []

    def rate(mask):
        return df.loc[mask, "ChurnFlag"].mean(), int(mask.sum())

    if rec["Contract"] == "Month-to-month":
        r, n = rate(df["Contract"] == "Month-to-month")
        r2, _ = rate(df["Contract"] != "Month-to-month")
        notes.append(f"Month-to-month customers churned at {r:.0%} in this data (n={n:,}) versus "
                     f"{r2:.0%} for longer contracts. A contract-conversion offer may be worth testing.")
    if rec["InternetService"] == "Fiber optic":
        r, n = rate(df["InternetService"] == "Fiber optic")
        notes.append(f"Fiber optic customers churned at {r:.0%} (n={n:,}) vs {base:.0%} overall. "
                     "Checking service quality and price perception for this group could help.")
    if rec["InternetService"] != "No" and rec["OnlineSecurity"] == "No":
        r, n = rate((df["OnlineSecurity"] == "No") & (df["InternetService"] != "No"))
        notes.append(f"Internet customers without Online Security churned at {r:.0%} (n={n:,}). "
                     "Offering a trial of add-on services is an option to test, not a proven fix.")
    if rec["PaymentMethod"] == "Electronic check":
        r, n = rate(df["PaymentMethod"] == "Electronic check")
        notes.append(f"Electronic-check payers churned at {r:.0%} (n={n:,}). Encouraging automatic "
                     "payment could be tested.")
    if rec["tenure"] <= 12:
        r, n = rate(df["tenure"] <= 12)
        notes.append(f"Customers in their first 12 months churned at {r:.0%} (n={n:,}); early-life "
                     "outreach may be useful.")
    if not notes:
        notes.append("No standout risk segments for this profile in the historical data.")
    notes.append("These are associations observed in historical data and the model's output. They do "
                 "not prove that any action will prevent churn; validate with a controlled test.")
    return notes


def page_predict(df: pd.DataFrame) -> None:
    st.title("Individual customer prediction")
    model, meta = load_model(), cc.load_metadata()
    if model is None or meta is None:
        st.error("Model file not found. Run `python train_model.py` first (see README).")
        return
    thr0 = float(meta["selected_threshold"])
    with st.expander("Risk settings", expanded=False):
        thr = st.slider("Decision threshold", 0.05, 0.95, thr0, 0.05)
        low_max, high_min = st.slider("Risk bands: Low below / High from", 0.05, 0.95, (cc.DEFAULT_LOW_MAX,
                                                                                        cc.DEFAULT_HIGH_MIN), 0.05)
    form_df = input_form(df)
    valid, invalid, msgs = cc.validate_input(form_df)
    for m in msgs:
        st.warning(m)
    if not invalid.empty:
        st.dataframe(invalid)
        return
    out = cc.predict_frame(model, valid, thr, low_max, high_min).iloc[0]
    p = float(out["ChurnProbability"])
    st.divider()
    a, b = st.columns([1, 1])
    with a:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Estimated churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333333"},
                   "steps": [{"range": [0, low_max * 100], "color": "#BFE6D8"},
                             {"range": [low_max * 100, high_min * 100], "color": "#F5DDA4"},
                             {"range": [high_min * 100, 100], "color": "#F0B99A"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100}}))
        st.plotly_chart(style(fig, 330))
    with b:
        st.metric("Predicted class", "Churn" if out["PredictedChurn"] == "Yes" else "Stay")
        st.metric("Risk category", out["RiskCategory"])
        st.caption(f"Flagged as churn when probability >= {thr:.2f}. The probability is the model's "
                   "score and may not be perfectly calibrated.")
    st.subheader("Observations from the data")
    for n in retention_notes(valid.iloc[0], df, p):
        st.write("- " + n)

    st.subheader("Why this score? (Logistic Regression contributions)")
    lr = load_logreg()
    if lr is None:
        st.info("logreg_pipeline.joblib not found; run train_model.py.")
    else:
        Xt = lr.named_steps["preprocessor"].transform(cc.model_frame(valid))
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        names = lr.named_steps["preprocessor"].get_feature_names_out()
        contrib = pd.Series(Xt[0] * lr.named_steps["classifier"].coef_[0], index=names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(12).index).iloc[::-1]
        fig = hbar(top, "Top contributions to this customer's score", "Contribution to log-odds",
                   colors=(top > 0).map({True: "Raises churn score", False: "Lowers churn score"}),
                   cmap={"Raises churn score": CHURN_COLOR, "Lowers churn score": STAY_COLOR})
        st.plotly_chart(style(fig, 440))
        st.caption("This explains the simpler Logistic Regression model (standardised numeric and "
                   "one-hot features), which is consistent with but not identical to the selected "
                   "model. Contributions are associations, not causes.")
    shap_note = shap_single(model, valid)
    if shap_note is not None:
        st.plotly_chart(shap_note)


def shap_single(model, valid: pd.DataFrame):
    """SHAP waterfall-style bar for the selected tree model, if shap is installed."""
    try:
        import shap
    except Exception:
        return None
    clf = model.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        return None
    try:
        pre = model.named_steps["preprocessor"]
        Xt = pre.transform(cc.model_frame(valid))
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        s = pd.Series(np.asarray(sv)[0], index=pre.get_feature_names_out())
        top = s.reindex(s.abs().sort_values(ascending=False).head(12).index).iloc[::-1]
        fig = hbar(top, "SHAP values for this customer (selected model)", "SHAP value (log-odds)",
                   colors=(top > 0).map({True: "Pushes toward churn", False: "Pushes toward stay"}),
                   cmap={"Pushes toward churn": CHURN_COLOR, "Pushes toward stay": STAY_COLOR})
        return style(fig, 440)
    except Exception:
        return None


def read_upload(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")


def page_batch(df: pd.DataFrame) -> None:
    st.title("Batch customer prediction")
    model, meta = load_model(), cc.load_metadata()
    if model is None or meta is None:
        st.error("Model file not found. Run `python train_model.py` first (see README).")
        return
    st.write("Upload a CSV or Excel file containing these columns (extra columns such as customerID "
             "are kept in the output):")
    st.code(", ".join(cc.RAW_INPUT_COLUMNS), language=None)
    tmpl = df[cc.RAW_INPUT_COLUMNS].head(5)
    st.download_button("Download a 5-row template", tmpl.to_csv(index=False).encode("utf-8"),
                       "churn_template.csv", "text/csv")
    c = st.columns(3)
    thr = c[0].slider("Decision threshold", 0.05, 0.95, float(meta["selected_threshold"]), 0.05)
    low_max, high_min = c[1].slider("Risk bands: Low below / High from", 0.05, 0.95,
                                    (cc.DEFAULT_LOW_MAX, cc.DEFAULT_HIGH_MIN), 0.05)
    file = st.file_uploader("Customer file", type=["csv", "xlsx", "xls"])
    if file is None:
        return
    try:
        up = read_upload(file)
    except Exception as e:  # unreadable file
        st.error(f"Could not read the file: {e}")
        return
    if up.empty:
        st.error("The file has no rows.")
        return
    if len(up) > MAX_UPLOAD_ROWS:
        st.error(f"File has {len(up):,} rows; the limit is {MAX_UPLOAD_ROWS:,}.")
        return
    st.caption(f"Read {len(up):,} rows and {up.shape[1]} columns. Your file is not modified or stored.")
    valid, invalid, msgs = cc.validate_input(up)
    for m in msgs:
        st.warning(m)
    if not invalid.empty:
        st.subheader(f"Rows not predicted ({len(invalid):,})")
        st.dataframe(invalid)
        st.download_button("Download rejected rows with reasons", invalid.to_csv(index=False).encode("utf-8"),
                           "rejected_rows.csv", "text/csv")
    if valid.empty:
        st.error("No valid rows to predict.")
        return
    pred = cc.predict_frame(model, valid, thr, low_max, high_min)
    extra = [c for c in up.columns if c not in pred.columns]
    result = pd.concat([up.loc[pred.index, extra], pred], axis=1)
    st.subheader(f"Predictions ({len(result):,} rows)")
    order = ["Low", "Medium", "High"]
    counts = result["RiskCategory"].value_counts().reindex(order, fill_value=0).rename_axis("Risk").reset_index(name="Customers")
    k = st.columns(4)
    k[0].metric("Predicted to churn", f"{int((result['PredictedChurn'] == 'Yes').sum()):,}")
    for i, r in enumerate(order):
        k[i + 1].metric(f"{r} risk", f"{int(counts.loc[counts['Risk'] == r, 'Customers'].iloc[0]):,}")
    l, r_ = st.columns(2)
    fig = px.bar(counts, x="Risk", y="Customers", color="Risk", color_discrete_map=RISK_COLORS,
                 text="Customers", title="Customers by risk category")
    l.plotly_chart(style(fig, 360))
    fig = px.histogram(result, x="ChurnProbability", nbins=30, color_discrete_sequence=["#555555"],
                       title="Distribution of churn probabilities")
    r_.plotly_chart(style(fig, 360))
    st.dataframe(result.sort_values("ChurnProbability", ascending=False))
    st.download_button("Download predictions (CSV)", result.to_csv(index=False).encode("utf-8"),
                       "churn_predictions.csv", "text/csv")


def page_explain(df: pd.DataFrame) -> None:
    st.title("Explainability")
    meta = cc.load_metadata()
    if meta is None:
        st.error("Run `python train_model.py` first.")
        return
    st.warning("Explanations describe how the models use features. They show association, not causation.")
    t1, t2, t3 = st.tabs(["Feature importance", "Logistic Regression coefficients", "SHAP"])
    with t1:
        imp = meta.get("feature_importance_selected")
        if imp:
            s = pd.Series(imp).sort_values(ascending=False).head(20).iloc[::-1]
            fig = hbar(s, f"Top 20 importances - {meta['selected_model']}", "Importance")
            st.plotly_chart(style(fig, 560))
        else:
            st.info("Selected model has no built-in importances.")
    with t2:
        co = pd.Series(meta["logreg_coefficients"])
        top = co.reindex(co.abs().sort_values(ascending=False).head(20).index).iloc[::-1]
        fig = hbar(top, "Logistic Regression coefficients (top 20 by size)", "Coefficient (log-odds)",
                   colors=(top > 0).map({True: "Higher churn score", False: "Lower churn score"}),
                   cmap={"Higher churn score": CHURN_COLOR, "Lower churn score": STAY_COLOR})
        st.plotly_chart(style(fig, 560))
        st.caption("Numeric features are standardised; categorical features are one-hot encoded, so "
                   "each coefficient is relative to the dropped/other categories.")
    with t3:
        try:
            import shap  # noqa: F401
        except Exception:
            st.info("The optional `shap` package is not installed. Run `python -m pip install shap` "
                    "to enable this tab.")
            return
        model = load_model()
        if model is None or not hasattr(model.named_steps["classifier"], "feature_importances_"):
            st.info("SHAP tab requires the saved tree-based model.")
            return
        n = st.slider("Customers to sample for SHAP", 100, 800, 300, 100,
                      help="Random sample of the cleaned dataset (kept small for speed).")
        if st.button("Compute SHAP"):
            with st.spinner("Computing SHAP values..."):
                fig = shap_global(model, df, n)
            if fig is None:
                st.error("SHAP could not be computed for this model.")
            else:
                st.plotly_chart(fig)


def shap_global(model, df: pd.DataFrame, n: int):
    import shap
    try:
        sample = df.sample(min(n, len(df)), random_state=cc.RANDOM_STATE)
        pre = model.named_steps["preprocessor"]
        Xt = pre.transform(sample[cc.NUMERIC_FEATURES + cc.CATEGORICAL_FEATURES])
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        sv = shap.TreeExplainer(model.named_steps["classifier"]).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        s = pd.Series(np.abs(np.asarray(sv)).mean(axis=0), index=pre.get_feature_names_out())
        s = s.sort_values(ascending=False).head(15).iloc[::-1]
        fig = hbar(s, f"Global SHAP importance (random sample of {len(sample)} customers)",
                   "Mean |SHAP value|", single_color="#009E73")
        return style(fig, 520)
    except Exception:
        return None


def page_explorer(df: pd.DataFrame) -> None:
    st.title("Data explorer")
    show_ids = st.checkbox("Show customer IDs", value=False,
                           help="Hidden by default to avoid unnecessarily exposing identifiers.")
    raw, _ = load_data()
    view = df.copy()
    if show_ids and cc.ID_COL in raw.columns:
        ids = raw.assign(TotalCharges=pd.to_numeric(raw["TotalCharges"], errors="coerce")).dropna(
            subset=["TotalCharges"]).reset_index(drop=True)[cc.ID_COL]
        view.insert(0, cc.ID_COL, ids.values)
    f1, f2, f3 = st.columns(3)
    contracts = f1.multiselect("Contract", cc.ALLOWED["Contract"])
    churn = f2.multiselect("Churn", ["Yes", "No"])
    tmin, tmax = f3.slider("Tenure", 0, 72, (0, 72))
    mask = view["tenure"].between(tmin, tmax)
    if contracts:
        mask &= view["Contract"].isin(contracts)
    if churn:
        mask &= view[cc.TARGET].isin(churn)
    view = view[mask].drop(columns=["ChurnFlag"])
    st.caption(f"{len(view):,} rows")
    t1, t2, t3 = st.tabs(["Preview", "Column types & missing", "Statistics"])
    with t1:
        st.dataframe(view.head(500))
        st.download_button("Download filtered rows (CSV)", view.to_csv(index=False).encode("utf-8"),
                           "filtered_customers.csv", "text/csv")
    with t2:
        info = pd.DataFrame({"dtype": view.dtypes.astype(str), "missing": view.isna().sum(),
                             "unique": view.nunique()})
        st.dataframe(info)
    with t3:
        st.dataframe(view.describe(include="all").T)


# ------------------------------------------------------------------------------- main
def main() -> None:
    try:
        raw, df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Could not read the dataset: {e}")
        st.stop()

    st.sidebar.title("Churn dashboard")
    page = st.sidebar.radio("Navigate", [
        "Executive overview", "Exploratory analysis", "Customer segmentation", "Model performance",
        "Individual prediction", "Batch prediction", "Explainability", "Data explorer"])
    st.sidebar.caption(f"Dataset: {cc.find_dataset().name} ({len(raw):,} raw rows / {len(df):,} used)")
    meta = cc.load_metadata()
    if meta:
        st.sidebar.caption(f"Model: {meta['selected_model']}")
    else:
        st.sidebar.warning("No trained model yet. Run train_model.py.")

    {"Executive overview": lambda: page_overview(df),
     "Exploratory analysis": lambda: page_eda(raw, df),
     "Customer segmentation": lambda: page_segments(df),
     "Model performance": page_model,
     "Individual prediction": lambda: page_predict(df),
     "Batch prediction": lambda: page_batch(df),
     "Explainability": lambda: page_explain(df),
     "Data explorer": lambda: page_explorer(df)}[page]()


main()
