"""Analytics package for Aviation MRO Predictive Intelligence."""

from .eda import (
    ColumnProfile,
    DatasetProfile,
    compare_sensor_trends,
    compute_column_profile,
    detect_constant_columns,
    load_raw_data,
    load_rul,
    profile_dataset,
    run_eda_analysis,
    test_distribution_shift,
    test_normality,
    test_stationarity,
    test_variance_homogeneity,
)

__all__ = [
    "load_raw_data",
    "load_rul",
    "profile_dataset",
    "compute_column_profile",
    "detect_constant_columns",
    "test_normality",
    "test_stationarity",
    "test_distribution_shift",
    "test_variance_homogeneity",
    "compare_sensor_trends",
    "run_eda_analysis",
    "ColumnProfile",
    "DatasetProfile",
]
