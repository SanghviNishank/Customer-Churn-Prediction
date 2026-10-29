"""Practical tests. Run with:  python -m pytest -q   (or: python -m unittest discover -s tests)

Tests that need the dataset or a trained model are skipped (not passed) when those files
are missing, so run train_model.py first for full coverage.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import churn_core as cc  # noqa: E402

HAVE_DATA = cc.find_dataset() is not None
HAVE_MODEL = cc.MODEL_PATH.exists()


def sample_record(**over):
    rec = {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
           "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
           "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
           "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
           "Contract": "Month-to-month", "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 425.0}
    rec.update(over)
    return rec


class TestValidation(unittest.TestCase):
    def test_valid_row_passes(self):
        valid, invalid, msgs = cc.validate_input(pd.DataFrame([sample_record()]))
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 0)
        self.assertEqual(msgs, [])

    def test_missing_columns_reported(self):
        df = pd.DataFrame([sample_record()]).drop(columns=["Contract", "tenure"])
        valid, invalid, msgs = cc.validate_input(df)
        self.assertEqual(len(valid), 0)
        self.assertIn("Contract", msgs[0])
        self.assertIn("tenure", msgs[0])

    def test_bad_rows_are_returned_not_dropped(self):
        df = pd.DataFrame([sample_record(), sample_record(Contract="Weekly"),
                           sample_record(tenure="abc"), sample_record(MonthlyCharges=-5)])
        valid, invalid, _ = cc.validate_input(df)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 3)
        self.assertEqual(len(valid) + len(invalid), len(df))
        self.assertTrue((invalid["Problem"] != "").all())

    def test_senior_citizen_text_accepted(self):
        valid, invalid, _ = cc.validate_input(pd.DataFrame([sample_record(SeniorCitizen="Yes")]))
        self.assertEqual(len(valid), 1)
        self.assertEqual(int(valid["SeniorCitizen"].iloc[0]), 1)

    def test_input_not_mutated(self):
        df = pd.DataFrame([sample_record(SeniorCitizen="Yes", tenure="5")])
        before = df.copy(deep=True)
        cc.validate_input(df)
        pd.testing.assert_frame_equal(df, before)


class TestFeatures(unittest.TestCase):
    def test_engineered_features(self):
        df = pd.DataFrame([sample_record(tenure=5), sample_record(tenure=60, StreamingMovies="Yes")])
        out = cc.add_engineered_features(df)
        self.assertEqual(list(out["TenureGroup"]), ["New", "Loyal"])
        self.assertEqual(list(out["NumberOfServices"]), [1, 2])

    def test_risk_categories(self):
        self.assertEqual(cc.risk_category(0.1), "Low")
        self.assertEqual(cc.risk_category(0.45), "Medium")
        self.assertEqual(cc.risk_category(0.8), "High")

    def test_model_frame_column_order(self):
        valid, _, _ = cc.validate_input(pd.DataFrame([sample_record()]))
        mf = cc.model_frame(valid)
        self.assertEqual(list(mf.columns), cc.NUMERIC_FEATURES + cc.CATEGORICAL_FEATURES)


@unittest.skipUnless(HAVE_DATA, "dataset not found in data/")
class TestData(unittest.TestCase):
    def test_cleaning_matches_notebook(self):
        raw = cc.load_raw()
        df = cc.clean_for_model(raw)
        self.assertEqual(len(raw) - len(df), int(pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum()))
        self.assertNotIn(cc.ID_COL, df.columns)
        self.assertFalse(df[cc.NUMERIC_FEATURES].isna().any().any())

    def test_allowed_values_cover_dataset(self):
        df = cc.clean_for_model(cc.load_raw())
        for col, allowed in cc.ALLOWED.items():
            self.assertTrue(set(df[col].unique()) <= set(allowed), col)


@unittest.skipUnless(HAVE_DATA and HAVE_MODEL, "run train_model.py first")
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import joblib
        cls.model = joblib.load(cc.MODEL_PATH)

    def test_prediction_shape_and_range(self):
        df = cc.clean_for_model(cc.load_raw()).head(50)
        valid, invalid, _ = cc.validate_input(df)
        self.assertEqual(len(invalid), 0)
        out = cc.predict_frame(self.model, valid, 0.4)
        self.assertEqual(len(out), 50)
        self.assertTrue(out["ChurnProbability"].between(0, 1).all())
        self.assertTrue(set(out["PredictedChurn"]) <= {"Yes", "No"})
        self.assertTrue(set(out["RiskCategory"]) <= {"Low", "Medium", "High"})

    def test_threshold_monotonic(self):
        valid, _, _ = cc.validate_input(cc.clean_for_model(cc.load_raw()).head(200))
        lo = (cc.predict_frame(self.model, valid, 0.2)["PredictedChurn"] == "Yes").sum()
        hi = (cc.predict_frame(self.model, valid, 0.8)["PredictedChurn"] == "Yes").sum()
        self.assertGreaterEqual(lo, hi)

    def test_metadata_consistent(self):
        meta = cc.load_metadata()
        self.assertIsNotNone(meta)
        m = meta["selected_test_at_threshold"]
        self.assertTrue(0 <= m["ROC-AUC"] <= 1)
        cm = np.array(meta["confusion_matrix_at_selected_threshold"])
        self.assertEqual(cm.sum(), meta["split"]["n_test"])


if __name__ == "__main__":
    unittest.main()
