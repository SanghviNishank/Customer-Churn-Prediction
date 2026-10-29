"""Feature importance, logistic regression coefficients, and SHAP explainability."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.data.load_data import RANDOM_STATE
from src.preprocessing.preprocessing import model_frame
from src.utils.helpers import logger


def get_feature_names_from_pipeline(pipeline) -> list[str]:
    """Extract transformed feature names from the pipeline's ColumnTransformer."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    return []


def extract_tree_feature_importances(pipeline) -> Optional[Dict[str, float]]:
    """Extract feature importances from a tree-based pipeline if supported."""
    clf = pipeline.named_steps.get("classifier")
    if clf is not None and hasattr(clf, "feature_importances_"):
        names = get_feature_names_from_pipeline(pipeline)
        if names:
            return dict(zip(names, [float(v) for v in clf.feature_importances_]))
    return None


def extract_logistic_regression_coefficients(pipeline) -> Optional[Dict[str, float]]:
    """Extract feature coefficients from LogisticRegression pipeline."""
    clf = pipeline.named_steps.get("classifier")
    if clf is not None and hasattr(clf, "coef_"):
        names = get_feature_names_from_pipeline(pipeline)
        if names:
            return dict(zip(names, [float(v) for v in clf.coef_[0]]))
    return None


def explain_single_customer_linear(
    lr_pipeline, customer_df: pd.DataFrame
) -> Optional[pd.Series]:
    """Calculate feature contributions to log-odds for a single customer using Logistic Regression."""
    if lr_pipeline is None:
        return None
    try:
        pre = lr_pipeline.named_steps["preprocessor"]
        clf = lr_pipeline.named_steps["classifier"]
        Xt = pre.transform(model_frame(customer_df))
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        names = pre.get_feature_names_out()
        contrib = pd.Series(Xt[0] * clf.coef_[0], index=names)
        return contrib
    except Exception as e:
        logger.warning("Linear explanation failed: %s", e)
        return None


def compute_shap_for_sample(
    tree_pipeline, df: pd.DataFrame, n_samples: int = 300
) -> Optional[pd.Series]:
    """Compute mean absolute SHAP values across a sample of customers."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP library is not installed.")
        return None

    clf = tree_pipeline.named_steps.get("classifier")
    if not hasattr(clf, "feature_importances_"):
        return None

    try:
        sample = df.sample(min(n_samples, len(df)), random_state=RANDOM_STATE)
        pre = tree_pipeline.named_steps["preprocessor"]
        Xt = pre.transform(model_frame(sample))
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv

        names = pre.get_feature_names_out()
        mean_abs_shap = pd.Series(np.abs(np.asarray(sv)).mean(axis=0), index=names)
        return mean_abs_shap.sort_values(ascending=False)
    except Exception as e:
        logger.warning("Failed computing SHAP values: %s", e)
        return None


def compute_shap_single_customer(
    tree_pipeline, customer_df: pd.DataFrame
) -> Optional[pd.Series]:
    """Compute local SHAP values for a single customer."""
    try:
        import shap
    except ImportError:
        return None

    clf = tree_pipeline.named_steps.get("classifier")
    if not hasattr(clf, "feature_importances_"):
        return None

    try:
        pre = tree_pipeline.named_steps["preprocessor"]
        Xt = pre.transform(model_frame(customer_df))
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv

        names = pre.get_feature_names_out()
        return pd.Series(np.asarray(sv)[0], index=names)
    except Exception as e:
        logger.warning("Single-customer SHAP computation failed: %s", e)
        return None

