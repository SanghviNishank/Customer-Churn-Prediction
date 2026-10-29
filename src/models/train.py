"""Model training, tuning, and artifact generation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.load_data import (
    ALLOWED,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_INPUT_COLUMNS,
    TARGET,
    find_dataset,
    load_raw,
    save_processed,
)
from src.models.evaluate import (
    compute_confusion_matrix,
    compute_metrics,
    compute_threshold_sweep,
    generate_classification_report,
)
from src.models.predict import NOTEBOOK_THRESHOLD
from src.preprocessing.preprocessing import clean_for_model, create_pipeline
from src.utils.helpers import MODEL_DIR, compute_file_hash, logger

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


def candidate_models(tune: bool = False) -> Dict[str, Tuple[Pipeline, str, Dict | None]]:
    """Define candidate model pipelines, hyperparameter grids, and sources."""
    rs = RANDOM_STATE
    models = {
        "DummyClassifier": (
            create_pipeline(DummyClassifier(strategy="most_frequent")),
            "baseline",
            None,
        ),
        "LogisticRegression (notebook-tuned)": (
            create_pipeline(LogisticRegression(C=10, solver="liblinear", random_state=rs)),
            "notebook best params: C=10, l2, liblinear",
            {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        ),
        "DecisionTree (notebook-tuned)": (
            create_pipeline(
                DecisionTreeClassifier(
                    max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=rs
                )
            ),
            "notebook best params: max_depth=5, min_samples_split=2, min_samples_leaf=1",
            {
                "classifier__max_depth": [3, 5, 7, 10, None],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 5, 10],
            },
        ),
        "RandomForest (default params)": (
            create_pipeline(RandomForestClassifier(random_state=rs, n_jobs=1)),
            "library defaults",
            {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [5, 10, 15, None],
            },
        ),
        "AdaBoost (default params)": (
            create_pipeline(AdaBoostClassifier(random_state=rs)),
            "library defaults",
            {"classifier__n_estimators": [50, 100, 200], "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0]},
        ),
        "GradientBoosting (default params)": (
            create_pipeline(GradientBoostingClassifier(random_state=rs)),
            "library defaults",
            None,
        ),
    }

    if HAVE_XGB:
        models["XGBoost (notebook-tuned)"] = (
            create_pipeline(
                XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    eval_metric="logloss",
                    n_jobs=1,
                )
            ),
            "notebook best params: n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8",
            {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
            },
        )
    return models


def train_and_evaluate(
    tune: bool = False,
    threshold: float = NOTEBOOK_THRESHOLD,
    model_dir: Path | None = None,
) -> Dict[str, Any]:
    """Train all models on the dataset, evaluate them on the 20% hold-out test set, and persist artifacts."""
    output_dir = model_dir or MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset()
    if data_path is None:
        raise FileNotFoundError("Dataset file could not be found.")

    raw = load_raw(data_path)
    df = clean_for_model(raw)
    
    # Save processed dataframe for caching and external inspection
    save_processed(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].map({"Yes": 1, "No": 0})
    if not y.notna().all():
        raise ValueError("Target contains unmapped values.")

    # Stratified 80/20 train/test split matching the notebook
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(
        "Dataset rows: %d (dropped %d with empty TotalCharges). Train: %d, Test: %d",
        len(df),
        len(raw) - len(df),
        len(X_tr),
        len(X_te),
    )

    fitted_models = {}
    model_results = []
    test_probs = {}

    for name, (pipeline, src, grid) in candidate_models(tune).items():
        logger.info("Training %s...", name)
        if tune and grid is not None:
            gs = GridSearchCV(
                pipeline,
                grid,
                cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                scoring="roc_auc",
                n_jobs=1,
            )
            gs.fit(X_tr, y_tr)
            pipeline = gs.best_estimator_
            src = f"re-tuned with GridSearchCV: {gs.best_params_}"
        else:
            pipeline.fit(X_tr, y_tr)

        fitted_models[name] = pipeline
        p_tr = pipeline.predict_proba(X_tr)[:, 1]
        p_te = pipeline.predict_proba(X_te)[:, 1]
        test_probs[name] = p_te

        pred_tr_05 = (p_tr >= 0.5).astype(int)
        pred_te_05 = (p_te >= 0.5).astype(int)

        m_tr = compute_metrics(y_tr, pred_tr_05, p_tr)
        m_te = compute_metrics(y_te, pred_te_05, p_te)

        model_results.append({
            "model": name,
            "params_source": src,
            "train": m_tr,
            "test_at_0.5": m_te,
        })
        logger.info(
            "%s -> Test ROC-AUC: %.4f | Recall: %.4f",
            name,
            m_te["ROC-AUC"],
            m_te["Recall"],
        )

    # Select the model chosen by the notebook (XGBoost if available, else highest test ROC-AUC)
    xgb_key = next((k for k in fitted_models if k.startswith("XGBoost")), None)
    non_dummy = [r for r in model_results if r["model"] != "DummyClassifier"]
    selected_name = xgb_key or max(non_dummy, key=lambda r: r["test_at_0.5"]["ROC-AUC"])["model"]
    selected_pipe = fitted_models[selected_name]
    selected_probs = test_probs[selected_name]

    # Compute 5-fold cross validation score on train set for selected model
    cv_scores = cross_val_score(
        selected_pipe, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=1
    )

    # Threshold analysis
    sweep = compute_threshold_sweep(y_te.to_numpy(), selected_probs)

    # Transformed feature names and importance
    feat_names = list(
        selected_pipe.named_steps["preprocessor"].get_feature_names_out()
    )
    clf = selected_pipe.named_steps["classifier"]
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = dict(zip(feat_names, [float(v) for v in clf.feature_importances_]))

    # Logistic Regression coefficients
    lr_key = "LogisticRegression (notebook-tuned)"
    lr_pipe = fitted_models[lr_key]
    lr_clf = lr_pipe.named_steps["classifier"]
    lr_coefs = dict(zip(feat_names, [float(v) for v in lr_clf.coef_[0]]))

    pred_at_thr = (selected_probs >= threshold).astype(int)
    pred_at_05 = (selected_probs >= 0.5).astype(int)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost": __import__("xgboost").__version__ if HAVE_XGB else None,
        },
        "dataset": {
            "file": data_path.name,
            "sha256": compute_file_hash(data_path),
            "rows_raw": int(len(raw)),
            "rows_used": int(len(df)),
            "rows_dropped_blank_totalcharges": int(len(raw) - len(df)),
        },
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "test_churn_rate": float(y_te.mean()),
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "allowed_values": ALLOWED,
        "selected_model": selected_name,
        "selected_threshold": threshold,
        "xgboost_available_at_training": HAVE_XGB,
        "cv_roc_auc_5fold_on_train": {
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
            "folds": [float(v) for v in cv_scores],
        },
        "models": model_results,
        "selected_test_at_threshold": compute_metrics(y_te.to_numpy(), pred_at_thr, selected_probs),
        "confusion_matrix_at_0.5": compute_confusion_matrix(y_te.to_numpy(), pred_at_05),
        "confusion_matrix_at_selected_threshold": compute_confusion_matrix(y_te.to_numpy(), pred_at_thr),
        "classification_report_at_selected_threshold": generate_classification_report(
            y_te.to_numpy(), pred_at_thr
        ),
        "threshold_table": sweep,
        "feature_names_transformed": feat_names,
        "feature_importance_selected": importances,
        "logreg_coefficients": lr_coefs,
    }

    # Save artifacts
    pipeline_path = output_dir / "churn_pipeline.joblib"
    logreg_path = output_dir / "logreg_pipeline.joblib"
    meta_path = output_dir / "metadata.json"
    pred_path = output_dir / "test_predictions.csv"

    joblib.dump(selected_pipe, pipeline_path)
    joblib.dump(lr_pipe, logreg_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    preds_df = pd.DataFrame({"y_true": y_te.to_numpy()})
    for m_name, probs in test_probs.items():
        preds_df[m_name] = probs
    preds_df.to_csv(pred_path, index=False)

    logger.info("Saved models and metadata to %s", output_dir)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train customer churn prediction models.")
    parser.add_argument("--tune", action="store_true", help="Re-run grid searches.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=NOTEBOOK_THRESHOLD,
        help="Decision threshold (default: 0.40).",
    )
    args = parser.parse_args()

    meta = train_and_evaluate(tune=args.tune, threshold=args.threshold)
    print(f"\nModel Training Complete!")
    print(f"Selected Model: {meta['selected_model']} (Threshold {meta['selected_threshold']})")
    print("Test Set Metrics at Threshold:")
    for k, v in meta["selected_test_at_threshold"].items():
        print(f"  {k:12s}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

