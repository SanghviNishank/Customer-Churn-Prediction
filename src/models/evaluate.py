"""Model evaluation metrics, confusion matrices, curves, and threshold analysis."""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    """Generate confusion matrix as nested list."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm.tolist()


def compute_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate ROC curve coordinates."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val = float(roc_auc_score(y_true, y_prob))
    return {
        "fpr": [float(v) for v in fpr],
        "tpr": [float(v) for v in tpr],
        "thresholds": [float(v) for v in thresholds],
        "auc": auc_val,
    }


def compute_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate Precision-Recall curve coordinates."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "thresholds": [float(v) for v in thresholds],
        "base_rate": float(np.mean(y_true)),
    }


def compute_threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float] | None = None
) -> List[Dict[str, Any]]:
    """Calculate confusion matrix counts and metrics across decision thresholds."""
    thrs = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    records = []
    
    for t in thrs:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m = compute_metrics(y_true, y_pred, y_prob)
        records.append({
            "threshold": round(float(t), 2),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            **{k: round(v, 4) for k, v in m.items()},
        })
    return records


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Generate structured classification report dictionary."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
    )

