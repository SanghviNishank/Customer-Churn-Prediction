"""Tests for dashboard components, services, and pages."""
import unittest
from pathlib import Path
import pandas as pd

from src.data.load_data import load_raw
from src.preprocessing.preprocessing import prepare_dashboard_frame
from src.churn_core import load_metadata
from dashboard.services.data_service import (
    get_datasets,
    get_churn_model,
    get_logreg_model,
    get_model_metadata,
    get_test_predictions,
)
from dashboard.components.charts import horizontal_bar, confusion_matrix_plot


class TestDashboardComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.processed = prepare_dashboard_frame(cls.raw)
        cls.meta = load_metadata()

    def test_services_load(self):
        raw, processed = get_datasets()
        self.assertEqual(len(raw), 7043)
        self.assertEqual(len(processed), 7032)

        model = get_churn_model()
        self.assertIsNotNone(model)

        lr_model = get_logreg_model()
        self.assertIsNotNone(lr_model)

        meta = get_model_metadata()
        self.assertIsNotNone(meta)
        self.assertEqual(meta["selected_model"], "XGBoost (notebook-tuned)")

        tp = get_test_predictions()
        self.assertIsNotNone(tp)
        self.assertEqual(len(tp), 1407)

    def test_chart_generators(self):
        s = pd.Series([0.1, 0.4, 0.2], index=["A", "B", "C"])
        fig_bar = horizontal_bar(s, "Test Bar", "Val")
        self.assertIsNotNone(fig_bar)

        cm = [[100, 20], [15, 80]]
        fig_cm = confusion_matrix_plot(cm, "Test CM")
        self.assertIsNotNone(fig_cm)


if __name__ == "__main__":
    unittest.main()

