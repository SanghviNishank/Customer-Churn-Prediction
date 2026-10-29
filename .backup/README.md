# Customer Churn Prediction Dashboard

A Streamlit dashboard built **around** an existing Telco customer-churn notebook project.
The original notebooks (`CustomerFinal.ipynb`, `Customer_pratice.ipynb`) and the dataset are
**preserved unchanged**: nothing in this repository opens them for writing, and all new code lives in
separate files.

## Problem statement
A telecom company wants to identify customers likely to leave so it can act before they do.
The notebook frames this as binary classification (`Churn` = Yes/No). The dashboard adds
exploration, model evaluation, individual and batch prediction, and explanations on top of it.

## Dataset (verified)
- File: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (the file the notebooks load; IBM Telco Customer Churn).
- 7,043 rows, 21 columns, target `Churn` (Yes 26.5% / No 73.5% before cleaning).
- `TotalCharges` is text with 11 blank values; the notebook (and this project) drop those 11 rows -> 7,032 rows.
- `customerID` is an identifier and is not used by the model.
- Note: the original brief mentioned `Telco_customer_churn.xlsx`. The notebooks actually use the CSV above;
  the code uses the CSV and falls back to an `.xlsx` of that name only if present.

## ML workflow (taken from the notebook)
1. Clean: `TotalCharges` to numeric, drop 11 blank rows, drop `customerID`.
2. Features: `TenureGroup` (New 0-12, Regular 13-24, Established 25-48, Loyal 49-72) and
   `NumberOfServices` (count of 6 add-on services equal to "Yes").
3. Stratified 80/20 train/test split, `random_state=42`.
4. `ColumnTransformer`: `StandardScaler` on 4 numeric features; `OneHotEncoder(handle_unknown="ignore", drop="if_binary")` on 17 categorical features, all inside one sklearn `Pipeline` (no leakage).
5. Models: Dummy baseline, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost.
6. The notebook selected **tuned XGBoost** with a 0.4 decision threshold.

## What was added (all new files)
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (8 pages) |
| `train_model.py` | Re-creates the notebook pipeline, evaluates on the hold-out set, saves model + metadata |
| `src/churn_core.py` | Shared schema, cleaning, validation, prediction helpers |
| `tests/test_core.py` | Unit tests for validation, features, prediction, metadata |
| `requirements.txt`, `.gitignore`, `README.md` | Project setup |
| `models/` | Created by `train_model.py` (`churn_pipeline.joblib`, `logreg_pipeline.joblib`, `metadata.json`, `test_predictions.csv`) |

The notebook's last cells also try to save `../models/pipeline.pkl` and create `../app/`. This project
does **not** depend on those; it saves its own artifact at `models/churn_pipeline.joblib`.

## Dashboard pages
Executive overview - Exploratory analysis - Customer segmentation (with an animation stepping through tenure groups) -
Model performance (comparison, confusion matrix, ROC/PR, threshold sweep animation, report, importances) -
Individual prediction (validated form, probability gauge, risk band, data-based observations, explanation) -
Batch prediction (CSV/Excel upload, validation, risk summary, CSV download) - Explainability (importances,
Logistic Regression coefficients, optional SHAP) - Data explorer (customer IDs hidden by default).

## Install and run (Windows CMD)
```bat
cd /d "D:\ML Project"
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
Optional: `python train_model.py --tune` re-runs the notebook's grid searches (slower).
Tests: `python -m pytest -q` (or `python -m unittest discover -s tests`).

Place these files in `D:\ML Project` next to your existing `data\` folder. The dashboard loads the saved
model; it never trains on start-up and never loads model files supplied by users.

## Evaluation results
Evaluation set: stratified 20% hold-out (1,407 customers, 26.6% churn). Threshold 0.5 unless stated.

**Genuine results from a run of `train_model.py` in a sandbox WITHOUT XGBoost installed** (so the
selected model there was Gradient Boosting, not the notebook's XGBoost). Params for Logistic Regression and
Decision Tree are the notebook's reported best values; Random Forest, AdaBoost and Gradient Boosting use library
defaults here (the notebook tuned some of them, and their tuned values were not recoverable from its saved output).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| DummyClassifier | 0.734 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.792 | 0.631 | 0.529 | 0.576 | 0.834 |
| Decision Tree | 0.790 | 0.602 | 0.615 | 0.608 | 0.830 |
| Random Forest (defaults) | 0.781 | 0.611 | 0.484 | 0.540 | 0.814 |
| AdaBoost (defaults) | 0.789 | 0.624 | 0.519 | 0.566 | 0.835 |
| Gradient Boosting (defaults) | 0.795 | 0.640 | 0.527 | 0.578 | 0.838 |

Logistic Regression, Decision Tree and Gradient Boosting ROC-AUC match the values printed in your notebook.
The XGBoost figures in the notebook (ROC-AUC about 0.841; precision 64.4%, recall 50.8% at 0.5; recall 68.2% at 0.4)
were **not re-run by me**; run `train_model.py` with XGBoost installed and the dashboard's Model performance
page will show your own verified numbers.

## Limitations and future work
- Single hold-out split of one public dataset; no external validation.
- Probabilities are not calibrated; consider calibration before using them as literal risks.
- Feature importances, coefficients and SHAP show associations, not causes.
- Retention suggestions are rule-based observations from historical rates, not tested interventions.
- Customers with `tenure = 0` (blank TotalCharges) cannot be scored; they are listed as rejected rows in batch mode.
- Ideas: probability calibration, cost-based threshold selection, cross-validated model comparison.

## Screenshots
None included. Add genuine screenshots after running the app.
