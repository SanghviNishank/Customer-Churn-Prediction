"""Comprehensive unit tests for modular src packages."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.helpers import get_project_root, compute_file_hash, format_currency, format_percentage
from src.data.load_data import (
    find_dataset,
    load_raw,
    load_processed,
    RAW_INPUT_COLUMNS,
    ALLOWED,
    TARGET,
)
from src.data.validate_data import validate_input, check_data_quality
from src.preprocessing.preprocessing import (
    add_engineered_features,
    clean_for_model,
    prepare_dashboard_frame,
    model_frame,
    build_preprocessor,
)
from src.analysis.eda import compute_overview_kpis, segment_table, two_way_segment_table
from src.analysis.statistics import numeric_correlation, outlier_analysis, categorical_association_chi2
from src.features.feature_analysis import (
    get_feature_names_from_pipeline,
    extract_tree_feature_importances,
    extract_logistic_regression_coefficients,
    explain_single_customer_linear,
)
from src.models.evaluate import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve_data,
    compute_pr_curve_data,
    compute_threshold_sweep,
    generate_classification_report,
)
from src.models.predict import risk_category, predict_frame, generate_retention_observations


class TestModularPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_row = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.0,
            "TotalCharges": 660.0,
        }

    def test_utils_helpers(self):
        root = get_project_root()
        self.assertTrue(root.exists())
        self.assertEqual(format_currency(1234.5), "$1,234.50")
        self.assertEqual(format_percentage(0.2654), "26.5%")

    def test_data_quality_report(self):
        df = pd.DataFrame([self.sample_row, self.sample_row])
        dq = check_data_quality(df)
        self.assertIn("missing", dq)
        self.assertIn("duplicates", dq)
        self.assertIn("cardinality", dq)
        self.assertEqual(len(dq["duplicates"]), 2)

    def test_eda_kpis_and_segments(self):
        raw = load_raw()
        df = prepare_dashboard_frame(raw)
        kpis = compute_overview_kpis(df)
        self.assertGreater(kpis["total_customers"], 7000)
        self.assertGreater(kpis["churn_rate"], 0.20)
        self.assertLess(kpis["churn_rate"], 0.35)

        seg = segment_table(df, "Contract")
        self.assertEqual(len(seg), 3)
        self.assertTrue("ChurnRate" in seg.columns)

        two_way = two_way_segment_table(df, "Contract", "InternetService")
        self.assertEqual(two_way.shape, (3, 3))

    def test_statistical_functions(self):
        raw = load_raw()
        df = prepare_dashboard_frame(raw)
        corr = numeric_correlation(df)
        self.assertIn("ChurnFlag", corr.columns)
        self.assertTrue((-1 <= corr.values).all() and (corr.values <= 1).all())

        outliers = outlier_analysis(df)
        self.assertGreaterEqual(len(outliers), 3)
        self.assertTrue("Outlier Count" in outliers.columns)

        assoc = categorical_association_chi2(df, "Contract")
        self.assertIn("cramers_v", assoc)
        self.assertGreater(assoc["cramers_v"], 0.1)

    def test_evaluation_functions(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3, 0.1, 0.7])
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_true, y_pred, y_prob)
        self.assertIn("ROC-AUC", m)
        self.assertIn("F1", m)
        self.assertGreaterEqual(m["Accuracy"], 0.7)

        cm = compute_confusion_matrix(y_true, y_pred)
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)

        roc = compute_roc_curve_data(y_true, y_prob)
        self.assertIn("fpr", roc)
        self.assertIn("tpr", roc)

        sweep = compute_threshold_sweep(y_true, y_prob, [0.3, 0.5, 0.7])
        self.assertEqual(len(sweep), 3)

    def test_retention_observations(self):
        raw = load_raw()
        df = prepare_dashboard_frame(raw)
        row = pd.Series(self.sample_row)
        row["Contract"] = "Month-to-month"
        row["InternetService"] = "Fiber optic"
        notes = generate_retention_observations(row, df, 0.75)
        self.assertTrue(any("Month-to-month" in n for n in notes))
        self.assertTrue(any("Fiber optic" in n for n in notes))


if __name__ == "__main__":
    unittest.main()

