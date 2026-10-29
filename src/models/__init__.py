"""Models package."""
from src.models.evaluate import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve_data,
    compute_pr_curve_data,
    compute_threshold_sweep,
    generate_classification_report,
)
from src.models.predict import (
    DEFAULT_LOW_MAX,
    DEFAULT_HIGH_MIN,
    NOTEBOOK_THRESHOLD,
    risk_category,
    predict_frame,
    generate_retention_observations,
)
from src.models.train import candidate_models, train_and_evaluate

__all__ = [
    "compute_metrics",
    "compute_confusion_matrix",
    "compute_roc_curve_data",
    "compute_pr_curve_data",
    "compute_threshold_sweep",
    "generate_classification_report",
    "DEFAULT_LOW_MAX",
    "DEFAULT_HIGH_MIN",
    "NOTEBOOK_THRESHOLD",
    "risk_category",
    "predict_frame",
    "generate_retention_observations",
    "candidate_models",
    "train_and_evaluate",
]

