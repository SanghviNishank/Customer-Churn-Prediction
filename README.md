
# 📊 Telco Customer Churn Intelligence & ML Analytics Platform

An enterprise-grade Machine Learning analytics dashboard and modular prediction engine built around the IBM Telco Customer Churn study.

> **Strict Preservation Guarantee:** All original Jupyter notebooks (`CustomerFinal.ipynb`, `Customer_pratice.ipynb`) remain **100% bit-for-bit identical and unchanged**. Their SHA-256 hashes are actively monitored and validated via automated test scripts.

---

## 1. Project Overview & Business Objectives

Customer retention is critical in the telecommunications industry, where acquiring a new subscriber is 5x to 25x more expensive than retaining an existing customer.

### Core Objectives:
1. **Identify Churn Drivers:** Uncover contractual, tenure, and service add-on dynamics driving attrition.
2. **Leakage-Free Modeling:** Clean and preprocess data with scikit-learn `Pipeline` and `ColumnTransformer` fitted exclusively on training splits.
3. **Threshold Optimization:** Shift from arbitrary 0.5 decision boundaries to an operational threshold (**0.40**) to capture 66.3% of churners while controlling false alarms.
4. **Transparent Explainability:** Provide SHAP tree attributions and standardized Logistic Regression odds ratios for every prediction.
5. **Interactive Operations:** Empower business stakeholders with real-time scoring, batch scoring, and downloadable executive reports.

---

## 2. Verified Project Dataset

- **Source File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (also backed up in `data/raw/`).
- **Dimensions:** 7,043 customer rows, 21 columns.
- **Target Variable:** `Churn` (`Yes` 26.5% / `No` 73.5%).
- **Data Cleaning Transformation:**
  - 11 records possess empty whitespace strings (`" "`) in `TotalCharges` (associated with `tenure = 0` accounts).
  - As verified in `CustomerFinal.ipynb`, `TotalCharges` is converted to numeric and these 11 rows are dropped, yielding **7,032 model-ready records**.
  - `customerID` is dropped to avoid identifier leakage.
  - Engineered features:
    - `TenureGroup`: Categorical binning into `New` (0–12m), `Regular` (13–24m), `Established` (25–48m), and `Loyal` (49–72m).
    - `NumberOfServices`: Integer sum (0–6) of active add-on services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).

---

## 3. Final Project Architecture

```text
Customer-Churn-Prediction-main/
│
├── README.md                           # Comprehensive documentation & startup guide
├── requirements.txt                    # Exact production dependencies
├── .gitignore                          # Git exclusions (caches, logs, virtualenvs)
├── .env.example                        # Environment variable configuration template
├── app.py                              # Root dashboard launcher (delegates to dashboard/app.py)
├── churn_core.py                       # Top-level backward-compatible integration facade
├── train_model.py                      # Top-level model training script
├── test_core.py                        # Top-level test runner
│
├── CustomerFinal.ipynb                 # Original full analysis notebook (UNTOUCHED)
├── Customer_pratice.ipynb              # Original practice notebook (UNTOUCHED)
│
├── notebooks/
│   └── original/                       # Exact bit-for-bit copies preserved with identical hashes
│       ├── CustomerFinal.ipynb
│       └── Customer_pratice.ipynb
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Active dataset referenced by original code
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── telco_cleaned.csv                  # Cleaned 7,032 rows with engineered features
│
├── src/                                # Modular production-grade Python package
│   ├── __init__.py
│   ├── churn_core.py                   # Shared schemas, constants, and utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py                # Dataset discovery, loading, and persistence
│   │   └── validate_data.py            # Input validation, type enforcement, quality checks
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py            # Feature engineering, transformers, and pipelines
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py                      # Segmentation, KPI calculations, aggregations
│   │   └── statistics.py               # Outlier analysis (IQR), correlations, Chi-Square
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_analysis.py         # Tree importances, log-odds, SHAP explainers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                    # Model definitions, CV, and training routines
│   │   ├── evaluate.py                 # Classification metrics, confusion matrix, sweeps
│   │   └── predict.py                  # Single/batch inference, risk categorization
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                  # Paths, hashing, logging, and formatting
│
├── dashboard/                          # Advanced Streamlit & Plotly Dashboard
│   ├── app.py                          # Multi-page dashboard runner
│   ├── components/
│   │   ├── cards.py                    # Metric cards, alerts, headers
│   │   ├── charts.py                   # Standardized accessible Plotly figures
│   │   └── sidebar.py                  # Modern navigation and model status
│   ├── pages/
│   │   ├── p1_overview.py              # Executive overview & visual pipeline
│   │   ├── p2_explorer.py              # Raw vs Processed dataset explorer with filters
│   │   ├── p3_quality.py               # Data quality audit & outlier diagnostics
│   │   ├── p4_eda.py                   # Interactive distributions, box plots, correlations
│   │   ├── p5_features.py              # XGBoost importances, coefficients, SHAP values
│   │   ├── p6_models.py                # Model comparison, threshold sweep, ROC/PR curves
│   │   ├── p7_insights.py              # Empirical conclusions & downloadable reports
│   │   └── p8_prediction.py            # Real-time single & batch prediction engines
│   ├── services/
│   │   └── data_service.py             # Streamlit caching for models and datasets
│   └── styles/
│       └── main.css                    # Professional CSS design system
│
├── models/                             # Trained model artifacts & evaluation records
│   ├── churn_pipeline.joblib           # Fitted XGBoost preprocessor + classifier
│   ├── logreg_pipeline.joblib          # Fitted Logistic Regression pipeline
│   ├── metadata.json                   # Verified metrics, split info, CV scores
│   └── test_predictions.csv            # Hold-out test set ground truth and model probs
│
├── reports/                            # Automatically generated analysis exports
│   ├── figures/                        # Interactive HTML visualizations
│   ├── tables/                         # CSV segmentation tables & benchmarks
│   └── summaries/
│       └── executive_summary.md        # Downloadable markdown executive report
│
├── scripts/
│   ├── train.py                        # CLI training script
│   ├── generate_reports.py             # Report and figure generation utility
│   └── verify_notebooks.py             # Cryptographic hash verification utility
│
└── tests/                              # Comprehensive test suite (pytest)
    ├── test_core.py                    # Legacy backward-compatibility test suite
    ├── test_modular_pipeline.py        # Tests for src data, preprocessing, and analysis
    └── test_dashboard_components.py    # Tests for dashboard services and charts
```

---

## 4. Machine Learning Model Benchmark

All models were evaluated on the **20% stratified hold-out test set** (1,407 customers, 26.6% churn rate; `random_state=42`) that was never seen during preprocessing or training:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Parameters / Status |
|---|---|---|---|---|---|---|
| **DummyClassifier (Baseline)** | 0.734 | 0.000 | 0.000 | 0.000 | 0.500 | `strategy="most_frequent"` |
| **Decision Tree** | 0.790 | 0.602 | 0.615 | 0.608 | 0.830 | Tuned (`max_depth=5`) |
| **Random Forest** | 0.781 | 0.611 | 0.492 | 0.545 | 0.814 | Default ensemble (`n_estimators=100`) |
| **AdaBoost** | 0.789 | 0.624 | 0.519 | 0.566 | 0.835 | Default boosting (`learning_rate=1.0`) |
| **Logistic Regression** | 0.792 | 0.631 | 0.529 | 0.576 | 0.835 | Tuned (`C=10, liblinear`) |
| **Gradient Boosting** | 0.795 | 0.640 | 0.527 | 0.578 | 0.838 | Default sklearn ensemble |
| **XGBoost (Selected)** | **0.795** | **0.644** | **0.511** | **0.570** | **0.838** | **Tuned (`lr=0.1, max_depth=3`)** |

### Selected Operational Setting:
- **Model:** Tuned XGBoost (`XGBClassifier`)
- **5-Fold Cross-Validation ROC-AUC (Train Set):** `0.8483 ± 0.015`
- **Tuned Threshold:** `0.40`
  - **Recall:** `0.6631` (Captures 66.3% of all at-risk customers, up from 51.1% at 0.50)
  - **Precision:** `0.5933`
  - **Accuracy:** `0.7896`
  - **F1-Score:** `0.6263`

---

## 5. Installation & Setup Guide (Windows PowerShell)

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- Windows PowerShell

### Step 1: Open PowerShell and Navigate to Project Root
```powershell
cd D:\MlProject\Customer-Churn-Prediction-main
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 6. Execution Commands

### Launch the Analytics Dashboard
You can run the dashboard from the project root using either command:
```powershell
python -m streamlit run app.py
```
*or directly from the dashboard folder:*
```powershell
python -m streamlit run dashboard/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### Verify Notebook Integrity
Confirm that all original notebooks match their recorded SHA-256 baselines:
```powershell
python scripts\verify_notebooks.py
```

### Retrain Models and Refresh Artifacts (Optional)
```powershell
python train_model.py
```
To run hyperparameter grid searches:
```powershell
python train_model.py --tune
```

### Export Reports, Summary Tables, and Figures
```powershell
python scripts\generate_reports.py
```

### Execute Test Suite
Run all 21 automated unit and integration tests:
```powershell
python -m pytest tests\ -v
```

---

## 7. Dashboard Pages Overview

1. **Executive Overview:** High-level KPI summary (customers, churned, billing at risk), problem statement, and 6-stage ML pipeline diagram.
2. **Dataset Explorer:** Deep inspection of raw and processed datasets with interactive column filters, search, sorting, and CSV export.
3. **Data Quality & Preprocessing:** Missing-value and blank-string audits, outlier boundaries (IQR), duplicate detection, and before/after comparisons.
4. **Exploratory Data Analysis:** Interactive distributions, box/violin plots, Pearson correlation matrix, and 2-way cross-segment churn heatmaps.
5. **Feature Analysis:** Top 20 tree importances, Logistic Regression log-odds multipliers, global SHAP values, and feature business explanations.
6. **Model Training & Evaluation:** Comprehensive model benchmark comparison, interactive threshold slider with real-time KPI re-calculation, ROC/PR curves, and confusion matrix diagnostics.
7. **Business Insights & Reports:** Strategic recommendations ('Lock & Save' campaigns, onboarding care), financial exposure analysis, and downloadable markdown/CSV summaries.
8. **Prediction Interface:** 
   - *Single Customer:* Form input with probability gauge, risk tier categorization (Low/Medium/High), score contribution breakdown, and personalized retention actions.
   - *Batch Prediction:* CSV/Excel upload, strict validation with downloadable diagnostic error log for rejected rows, probability histogram, and downloadable scored predictions CSV.

---

## 8. Troubleshooting & FAQ

- **Issue: `ModuleNotFoundError: No module named 'src'`**
  - *Solution:* Run commands from the project root directory (`D:\MlProject\Customer-Churn-Prediction-main`). All entry scripts automatically configure `sys.path`.
- **Issue: Streamlit port already in use**
  - *Solution:* Launch with a custom port: `python -m streamlit run app.py --server.port 8502`.
- **Issue: XGBoost or SHAP warning**
  - *Solution:* Ensure `xgboost` and `shap` are installed via `requirements.txt`. The platform gracefully handles environments where optional libraries are absent.
