"""Analysis package."""
from src.analysis.eda import compute_overview_kpis, segment_table, two_way_segment_table
from src.analysis.statistics import (
    numeric_correlation,
    outlier_analysis,
    categorical_association_chi2,
)

__all__ = [
    "compute_overview_kpis",
    "segment_table",
    "two_way_segment_table",
    "numeric_correlation",
    "outlier_analysis",
    "categorical_association_chi2",
]

