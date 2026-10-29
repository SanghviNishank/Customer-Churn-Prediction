"""Train the churn models and save the fitted pipeline + genuine evaluation results.

Reads the dataset (never writes to it) and reproduces the notebook workflow in a separate
script: same cleaning, same engineered features, same stratified 80/20 split
(random_state=42), same ColumnTransformer.

Usage (Windows CMD):
    cd /d "D:\\ML Project"
    python train_model.py              # uses notebook's tuned params where they are recorded
    python train_model.py --tune       # re-run the notebook's grid searches (slower)

Outputs (in models/):
    churn_pipeline.joblib   fitted preprocessing + selected model (XGBoost if installed)
    logreg_pipeline.joblib  fitted preprocessing + Logistic Regression (for coefficients)
    metadata.json           schema, metrics, curves summary inputs, importances
    test_predictions.csv    held-out y_true and probabilities per model (for ROC / PR plots)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import churn_core as cc

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # xgboost not installed
    HAVE_XGB = False


def build_preprocessor() -> ColumnTransformer:
    # Identical to the notebook's preprocessor.
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), cc.NUMERIC_FEATURES),
        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]),
         cc.CATEGORICAL_FEATURES),
    ])


def pipe(clf) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])


def candidate_models(tune: bool) -> dict[str, tuple[Pipeline, str, dict | None]]:
    """name -> (pipeline, params_source, grid_for_tuning)."""
    rs = cc.RANDOM_STATE
    models = {
        "DummyClassifier": (pipe(DummyClassifier(strategy="most_frequent")), "baseline", None),
        # Best params printed in the notebook's saved outputs / interpretation cells.
        "LogisticRegression (notebook-tuned)": (
            pipe(LogisticRegression(C=10, solver="liblinear", random_state=rs)),  # l2 is the default
            "notebook best params: C=10, l2, liblinear",
            {"classifier__C": [0.01, 0.1, 1, 10, 100], "classifier__solver": ["liblinear", "lbfgs"]}),
        "DecisionTree (notebook-tuned)": (
            pipe(DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1,
                                        random_state=rs)),
            "notebook best params: max_depth=5, min_samples_split=2, min_samples_leaf=1",
            {"classifier__max_depth": [3, 5, 7, 10, None],
             "classifier__min_samples_split": [2, 5, 10, 20],
             "classifier__min_samples_leaf": [1, 5, 10]}),
        "RandomForest (default params)": (
            pipe(RandomForestClassifier(random_state=rs, n_jobs=1)), "library defaults",
            {"classifier__n_estimators": [50, 100, 200], "classifier__max_depth": [5, 10, 15, None],
             "classifier__min_samples_split": [2, 5, 10], "classifier__min_samples_leaf": [1, 2, 4]}),
        "AdaBoost (default params)": (
            pipe(AdaBoostClassifier(random_state=rs)), "library defaults",
            {"classifier__n_estimators": [50, 100, 200],
             "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0]}),
        "GradientBoosting (default params)": (
            pipe(GradientBoostingClassifier(random_state=rs)), "library defaults", None),
    }
    if HAVE_XGB:
        models["XGBoost (notebook-tuned)"] = (
            pipe(XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8,
                               colsample_bytree=0.8, random_state=rs, eval_metric="logloss",
                               n_jobs=1)),
            "notebook best params: n_estimators=50, max_depth=3, learning_rate=0.1, "
            "subsample=0.8, colsample_bytree=0.8",
            {"classifier__n_estimators": [50, 100], "classifier__max_depth": [3, 5, 7],
             "classifier__learning_rate": [0.01, 0.1, 0.2], "classifier__subsample": [0.8, 1.0],
             "classifier__colsample_bytree": [0.8, 1.0]})
    return models


def metrics(y, pred, prob) -> dict:
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, prob),
    }


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="re-run the notebook's grid searches")
    ap.add_argument("--threshold", type=float, default=cc.NOTEBOOK_THRESHOLD,
                    help="decision threshold stored with the model (notebook used 0.4)")
    args = ap.parse_args()

    data_path = cc.find_dataset()
    if data_path is None:
        print(f"ERROR: dataset not found in {cc.DATA_DIR}", file=sys.stderr)
        return 1
    raw = cc.load_raw(data_path)
    df = cc.clean_for_model(raw)
    X = df[cc.NUMERIC_FEATURES + cc.CATEGORICAL_FEATURES]
    y = df[cc.TARGET].map({"Yes": 1, "No": 0})
    assert y.notna().all(), "Unexpected target labels"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=cc.RANDOM_STATE, stratify=y)
    print(f"Rows used: {len(df)} (dropped {len(raw) - len(df)} with blank TotalCharges) | "
          f"train {len(X_tr)} / test {len(X_te)}")
    if not HAVE_XGB:
        print("WARNING: xgboost is not installed - the notebook's selected model (XGBoost) will "
              "be skipped and the best remaining model by test ROC-AUC will be saved instead.")

    fitted, results, test_probs = {}, [], {}
    for name, (pl, src, grid) in candidate_models(args.tune).items():
        if args.tune and grid is not None:
            gs = GridSearchCV(pl, grid, cv=StratifiedKFold(5, shuffle=True,
                              random_state=cc.RANDOM_STATE), scoring="roc_auc", n_jobs=1)
            gs.fit(X_tr, y_tr)
            pl, src = gs.best_estimator_, f"re-tuned here: {gs.best_params_}"
        else:
            pl.fit(X_tr, y_tr)
        fitted[name] = pl
        p_tr, p_te = pl.predict_proba(X_tr)[:, 1], pl.predict_proba(X_te)[:, 1]
        test_probs[name] = p_te
        results.append({
            "model": name, "params_source": src,
            "train": metrics(y_tr, pl.predict(X_tr), p_tr),
            "test_at_0.5": metrics(y_te, pl.predict(X_te), p_te),
        })
        print(f"{name:42s} test ROC-AUC {results[-1]['test_at_0.5']['ROC-AUC']:.4f}  "
              f"recall {results[-1]['test_at_0.5']['Recall']:.3f}")

    # Selection: the notebook selected tuned XGBoost; otherwise highest test ROC-AUC (non-dummy).
    xgb_name = next((n for n in fitted if n.startswith("XGBoost")), None)
    non_dummy = [r for r in results if r["model"] != "DummyClassifier"]
    selected = xgb_name or max(non_dummy, key=lambda r: r["test_at_0.5"]["ROC-AUC"])["model"]
    sel_pipe = fitted[selected]
    sel_prob = test_probs[selected]
    thr = args.threshold

    cv = cross_val_score(sel_pipe, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=1)

    thresholds = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pr = (sel_prob >= t).astype(int)
        thresholds.append({"threshold": t, **metrics(y_te, pr, sel_prob)})

    feat_names = list(sel_pipe.named_steps["preprocessor"].get_feature_names_out())
    clf = sel_pipe.named_steps["classifier"]
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = dict(zip(feat_names, map(float, clf.feature_importances_)))

    lr_name = "LogisticRegression (notebook-tuned)"
    lr_clf = fitted[lr_name].named_steps["classifier"]
    lr_coefs = dict(zip(feat_names, map(float, lr_clf.coef_[0])))  # same preprocessor fit params

    pred_thr = (sel_prob >= thr).astype(int)
    pred_05 = (sel_prob >= 0.5).astype(int)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "xgboost": __import__("xgboost").__version__ if HAVE_XGB else None},
        "dataset": {"file": data_path.name, "sha256": sha256(data_path), "rows_raw": int(len(raw)),
                    "rows_used": int(len(df)), "rows_dropped_blank_totalcharges": int(len(raw) - len(df))},
        "split": {"test_size": 0.2, "random_state": cc.RANDOM_STATE, "stratified": True,
                  "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                  "test_churn_rate": float(y_te.mean())},
        "numeric_features": cc.NUMERIC_FEATURES, "categorical_features": cc.CATEGORICAL_FEATURES,
        "raw_input_columns": cc.RAW_INPUT_COLUMNS, "allowed_values": cc.ALLOWED,
        "selected_model": selected, "selected_threshold": thr,
        "xgboost_available_at_training": HAVE_XGB,
        "cv_roc_auc_5fold_on_train": {"mean": float(cv.mean()), "std": float(cv.std()),
                                      "folds": list(map(float, cv))},
        "models": results,
        "selected_test_at_threshold": metrics(y_te, pred_thr, sel_prob),
        "confusion_matrix_at_0.5": confusion_matrix(y_te, pred_05).tolist(),
        "confusion_matrix_at_selected_threshold": confusion_matrix(y_te, pred_thr).tolist(),
        "classification_report_at_selected_threshold": classification_report(
            y_te, pred_thr, target_names=["No Churn", "Churn"], output_dict=True),
        "threshold_table": thresholds,
        "feature_names_transformed": feat_names,
        "feature_importance_selected": importances,
        "logreg_coefficients": lr_coefs,
    }
    cc.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(sel_pipe, cc.MODEL_PATH)
    joblib.dump(fitted[lr_name], cc.MODEL_DIR / "logreg_pipeline.joblib")
    cc.METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out = pd.DataFrame({"y_true": y_te.to_numpy()})
    for n, p in test_probs.items():
        out[n] = p
    out.to_csv(cc.TEST_PRED_PATH, index=False)

    print(f"\nSelected model: {selected} | threshold {thr}")
    print("Test metrics at that threshold:", {k: round(v, 4) for k, v in
                                              meta["selected_test_at_threshold"].items()})
    print(f"Saved: {cc.MODEL_PATH}\n       {cc.METADATA_PATH}\n       {cc.TEST_PRED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
