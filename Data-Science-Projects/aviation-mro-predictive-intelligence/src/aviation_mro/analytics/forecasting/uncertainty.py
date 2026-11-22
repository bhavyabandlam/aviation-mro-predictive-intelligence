"""Validation-Calibrated Prediction Intervals.

Implements empirical quantile-based prediction intervals using validation residuals.
Evaluates empirical coverage without using test outcomes for calibration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baselines import ForecastResult


@dataclass
class IntervalCalibration:
    """Calibrated prediction intervals for a model at a specific horizon."""

    horizon: int
    lower_quantiles: np.ndarray  # Shape: (horizon,) - 2.5th percentile residuals
    upper_quantiles: np.ndarray  # Shape: (horizon,) - 97.5th percentile residuals
    empirical_coverage: float  # Fraction of validation actuals within intervals
    target_coverage: float = 0.95
    n_residuals_per_horizon: list[int] | None = None


def collect_validation_residuals(
    train_series: pd.Series,
    val_series: pd.Series,
    model_factory: Callable[..., Any],
    validation_origins: list[int],
    horizon: int,
    model_kwargs: dict | None = None,
) -> dict[int, list[float]]:
    """Collect residuals from rolling-origin validation for each horizon step.

    Args:
        train_series: Training data
        val_series: Validation data
        model_factory: Model factory callable
        validation_origins: Week numbers for rolling origins
        horizon: Forecast horizon
        model_kwargs: Model arguments

    Returns:
        Dict mapping horizon step (1-indexed) -> list of residuals
    """
    if model_kwargs is None:
        model_kwargs = {}

    combined = pd.concat([train_series, val_series])
    len(train_series)

    # Convert week numbers to indices
    origin_indices = []
    for week_num in validation_origins:
        idx = week_num - 1
        if idx < len(combined) - horizon:
            origin_indices.append(idx)

    residuals_by_horizon: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}

    for origin_idx in origin_indices:
        history = combined.iloc[: origin_idx + 1]
        actuals = combined.iloc[origin_idx + 1 : origin_idx + 1 + horizon].values

        if len(actuals) != horizon:
            continue

        model = model_factory(history, **model_kwargs)
        result = model.predict(horizon)

        # Collect residuals per horizon step
        for h in range(horizon):
            residuals_by_horizon[h + 1].append(actuals[h] - result.point_forecast[h])

    return residuals_by_horizon


def calibrate_intervals(
    train_series: pd.Series,
    val_series: pd.Series,
    model_factory: Callable[..., Any],
    validation_origins: list[int],
    horizon: int,
    target_coverage: float = 0.95,
    model_kwargs: dict | None = None,
) -> IntervalCalibration:
    """Calibrate prediction intervals using validation residuals.

    Uses empirical quantiles of residuals at each horizon step.

    Args:
        train_series: Training data
        val_series: Validation data
        model_factory: Model factory callable
        validation_origins: Week numbers for rolling origins
        horizon: Forecast horizon
        target_coverage: Target prediction interval coverage (default 95%)
        model_kwargs: Model arguments

    Returns:
        IntervalCalibration with quantiles and empirical coverage
    """
    if model_kwargs is None:
        model_kwargs = {}

    residuals_by_horizon = collect_validation_residuals(
        train_series, val_series, model_factory, validation_origins, horizon, model_kwargs
    )

    # Recompute origin_indices for coverage evaluation
    combined = pd.concat([train_series, val_series])
    origin_indices = []
    for week_num in validation_origins:
        idx = week_num - 1
        if idx < len(combined) - horizon:
            origin_indices.append(idx)

    alpha = (1 - target_coverage) / 2
    lower_q = alpha
    upper_q = 1 - alpha

    lower_quantiles = np.zeros(horizon)
    upper_quantiles = np.zeros(horizon)
    n_residuals = []

    for h in range(1, horizon + 1):
        res = np.array(residuals_by_horizon[h])
        n_residuals.append(len(res))
        if len(res) > 0:
            lower_quantiles[h - 1] = np.quantile(res, lower_q)
            upper_quantiles[h - 1] = np.quantile(res, upper_q)
        else:
            # No residuals for this horizon: use 0
            lower_quantiles[h - 1] = 0.0
            upper_quantiles[h - 1] = 0.0

    # Evaluate empirical coverage on validation
    all_covered = []
    for origin_idx in origin_indices:
        if origin_idx + horizon > len(combined):
            continue
        history = combined.iloc[: origin_idx + 1]
        actuals = combined.iloc[origin_idx + 1 : origin_idx + 1 + horizon].values
        if len(actuals) != horizon:
            continue

        model = model_factory(history, **model_kwargs)
        forecast_result = model.predict(horizon)

        # Build intervals for this origin
        for h in range(horizon):
            lower = forecast_result.point_forecast[h] + lower_quantiles[h]
            upper = forecast_result.point_forecast[h] + upper_quantiles[h]
            all_covered.append(lower <= actuals[h] <= upper)

    empirical_coverage = float(np.mean(all_covered)) if all_covered else 0.0

    return IntervalCalibration(
        horizon=horizon,
        lower_quantiles=lower_quantiles,
        upper_quantiles=upper_quantiles,
        empirical_coverage=empirical_coverage,
        target_coverage=target_coverage,
        n_residuals_per_horizon=n_residuals,
    )


def apply_intervals(
    forecast: ForecastResult,
    calibration: IntervalCalibration,
) -> ForecastResult:
    """Apply calibrated intervals to a forecast result."""
    if calibration.horizon != forecast.horizon:
        raise ValueError("Calibration horizon must match forecast horizon")

    # Build interval array: (horizon, 2)
    intervals = np.column_stack(
        [
            forecast.point_forecast + calibration.lower_quantiles,
            forecast.point_forecast + calibration.upper_quantiles,
        ]
    )

    return ForecastResult(
        point_forecast=forecast.point_forecast,
        prediction_intervals=intervals,
        model_name=forecast.model_name,
        origin_date=forecast.origin_date,
        horizon=forecast.horizon,
        seasonal_period=forecast.seasonal_period,
    )


def evaluate_coverage(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    model_factory: Callable[..., Any],
    validation_origins: list[int],
    horizon: int,
    model_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Evaluate prediction interval coverage on test set using validation-calibrated intervals.

    IMPORTANT: Uses only validation data for calibration. Test data is only used
    for final coverage evaluation, never for determining interval bounds.

    Returns:
        Dict with coverage metrics
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Calibrate on validation
    calibration = calibrate_intervals(
        train_series,
        val_series,
        model_factory,
        validation_origins,
        horizon,
        model_kwargs=model_kwargs,
    )

    # Apply to test forecast
    combined_train_val = pd.concat([train_series, val_series])
    model = model_factory(combined_train_val, **model_kwargs)
    test_forecast = model.predict(horizon)
    test_forecast_with_intervals = apply_intervals(test_forecast, calibration)

    # Evaluate on test actuals
    test_actuals = test_series.iloc[:horizon].values
    if len(test_actuals) != horizon:
        return {"error": "Insufficient test data"}

    covered = []
    interval_widths = []
    intervals = test_forecast_with_intervals.prediction_intervals
    if intervals is None:
        return {"error": "No prediction intervals produced"}
    for h in range(horizon):
        lower = intervals[h, 0]
        upper = intervals[h, 1]
        covered.append(lower <= test_actuals[h] <= upper)
        interval_widths.append(upper - lower)

    return {
        "empirical_coverage": float(np.mean(covered)),
        "target_coverage": calibration.target_coverage,
        "avg_interval_width": float(np.mean(interval_widths)),
        "calibration_empirical_coverage": calibration.empirical_coverage,
        "n_residuals_per_horizon": calibration.n_residuals_per_horizon,
    }


def calibrate_all_series(
    train_series_dict: dict[str, pd.Series],
    val_series_dict: dict[str, pd.Series],
    model_factory: Callable[..., Any],
    validation_origins: list[int],
    horizon: int,
    model_kwargs: dict | None = None,
) -> dict[str, IntervalCalibration]:
    """Calibrate intervals for all series."""
    calibrations = {}
    for series_id in train_series_dict:
        calibrations[series_id] = calibrate_intervals(
            train_series_dict[series_id],
            val_series_dict[series_id],
            model_factory,
            validation_origins,
            horizon,
            model_kwargs=model_kwargs,
        )
    return calibrations
