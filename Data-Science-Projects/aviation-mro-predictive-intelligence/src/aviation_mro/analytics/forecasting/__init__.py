"""Forecasting Analytics Package for Aviation MRO Predictive Intelligence."""

from .baselines import (
    ForecastResult,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    naive_factory,
    seasonal_naive_factory,
)
from .data_prep import (
    FORECAST_HORIZON,
    N_WEEKS,
    PART_FAMILIES,
    RANDOM_SEED,
    SHOPS,
    START_DATE,
    TEST_END_WEEK,
    TRAIN_END_WEEK,
    VALIDATION_END_WEEK,
    ChronologicalSplit,
    compute_dataset_fingerprint,
    generate_synthetic_demand,
    load_dataset,
    save_dataset,
    save_split_metadata,
    split_chronologically,
)
from .evaluation import (
    BacktestResult,
    aggregate_results,
    evaluate_forecast,
    evaluate_model_on_all_series,
    evaluate_test_set,
    mae,
    rmse,
    rolling_origin_backtest,
    wape,
)
from .uncertainty import (
    IntervalCalibration,
    apply_intervals,
    calibrate_all_series,
    calibrate_intervals,
    collect_validation_residuals,
    evaluate_coverage,
)

__all__ = [
    # Data preparation
    "generate_synthetic_demand",
    "split_chronologically",
    "load_dataset",
    "compute_dataset_fingerprint",
    "save_dataset",
    "save_split_metadata",
    "ChronologicalSplit",
    "SHOPS",
    "PART_FAMILIES",
    "N_WEEKS",
    "START_DATE",
    "FORECAST_HORIZON",
    "TRAIN_END_WEEK",
    "VALIDATION_END_WEEK",
    "TEST_END_WEEK",
    "RANDOM_SEED",
    # Baselines
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ForecastResult",
    "naive_factory",
    "seasonal_naive_factory",
    # Evaluation
    "mae",
    "rmse",
    "wape",
    "evaluate_forecast",
    "BacktestResult",
    "rolling_origin_backtest",
    "evaluate_model_on_all_series",
    "aggregate_results",
    "evaluate_test_set",
    # Uncertainty
    "IntervalCalibration",
    "collect_validation_residuals",
    "calibrate_intervals",
    "apply_intervals",
    "evaluate_coverage",
    "calibrate_all_series",
]
