"""Forecast Evaluation Metrics and Rolling-Origin Backtesting.

Implements MAE, RMSE, WAPE (with NaN for zero denominator),
and rolling-origin evaluation for time series forecasting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    Returns NaN when sum of absolute actual values is zero
    (i.e., all actuals are zero), since WAPE is undefined.
    Does not substitute arbitrary values like 1.0.
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics for a single forecast."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


@dataclass
class BacktestResult:
    """Results from rolling-origin backtest for one model on one series."""

    model_name: str
    series_id: str  # e.g., "SHOP_01_ENGINE"
    origin_dates: list[pd.Timestamp]
    horizon: int
    metrics_per_origin: list[dict[str, float]]  # One dict per origin
    avg_metrics: dict[str, float]  # Average across origins
    all_actuals: np.ndarray  # Flattened actuals across all origins
    all_predictions: np.ndarray  # Flattened predictions across all origins

    @property
    def n_origins(self) -> int:
        """Number of valid origins used in backtest."""
        return len(self.metrics_per_origin)

    @property
    def per_origin_metrics(self) -> list[dict[str, float]]:
        """Alias for metrics_per_origin for test compatibility."""
        return self.metrics_per_origin


def rolling_origin_backtest(
    series: pd.Series,
    model_factory: Callable[..., Any],
    origins: list[int],  # Indices in series where forecasts originate
    horizon: int,
    model_kwargs: dict | None = None,
) -> BacktestResult:
    """Perform rolling-origin backtest on a single time series.

    Args:
        series: Full time series (train + validation or train + validation + test)
        model_factory: Callable that returns a fitted model instance (e.g., NaiveForecaster)
        origins: List of indices (0-based) in series where forecasts originate
        horizon: Forecast horizon
        model_kwargs: Additional kwargs for model_factory

    Returns:
        BacktestResult with metrics per origin and averages
    """
    if model_kwargs is None:
        model_kwargs = {}

    origin_dates = []
    metrics_per_origin = []
    all_actuals: list[float] = []
    all_predictions = []

    for origin_idx in origins:
        if origin_idx + horizon > len(series):
            # Not enough future data for this origin
            continue

        # History up to and including origin
        history = series.iloc[: origin_idx + 1]
        # Actuals for evaluation
        actuals = np.asarray(
            series.iloc[origin_idx + 1 : origin_idx + 1 + horizon].values, dtype=float
        )

        if len(actuals) != horizon:
            continue

        # Fit model and predict
        model = model_factory(history, **model_kwargs)
        result = model.predict(horizon)

        # Evaluate
        metrics = evaluate_forecast(actuals, result.point_forecast)

        origin_dates.append(series.index[origin_idx])
        metrics_per_origin.append(metrics)
        all_actuals.extend(actuals)
        all_predictions.extend(result.point_forecast)

    # Average metrics across origins
    if metrics_per_origin:
        avg_metrics = {
            "mae": float(np.nanmean([m["mae"] for m in metrics_per_origin])),
            "rmse": float(np.nanmean([m["rmse"] for m in metrics_per_origin])),
            "wape": float(np.nanmean([m["wape"] for m in metrics_per_origin])),
        }
    else:
        avg_metrics = {"mae": np.nan, "rmse": np.nan, "wape": np.nan}

    return BacktestResult(
        model_name=model_kwargs.get("model_name", "unknown"),
        series_id=f"{series.name}" if series.name else "series",
        origin_dates=origin_dates,
        horizon=horizon,
        metrics_per_origin=metrics_per_origin,
        avg_metrics=avg_metrics,
        all_actuals=np.array(all_actuals),
        all_predictions=np.array(all_predictions),
    )


def evaluate_model_on_all_series(
    train_series_dict: dict[str, pd.Series],
    val_series_dict: dict[str, pd.Series],
    model_factory: Callable[..., Any],
    validation_origins: list[int],  # Week numbers (1-indexed)
    horizon: int,
    model_kwargs: dict | None = None,
) -> dict[str, BacktestResult]:
    """Evaluate a model on all series using rolling-origin backtest.

    Args:
        train_series_dict: {series_id: train_series}
        val_series_dict: {series_id: validation_series}
        model_factory: Callable returning fitted model
        validation_origins: Week numbers for rolling origins (1-indexed)
        horizon: Forecast horizon
        model_kwargs: Additional model arguments

    Returns:
        Dict mapping series_id -> BacktestResult
    """
    results = {}

    for series_id in train_series_dict:
        train_series = train_series_dict[series_id]
        val_series = val_series_dict[series_id]

        # Combine for rolling origins
        combined = pd.concat([train_series, val_series])

        # Convert week numbers to indices in combined series
        len(train_series)
        origin_indices = []
        for week_num in validation_origins:
            # week_num is 1-indexed from start of dataset
            # train ends at TRAIN_END_WEEK (104)
            # validation starts at week 105 (index 104 in combined)
            idx = week_num - 1  # Convert to 0-indexed
            if idx < len(combined) - horizon:
                origin_indices.append(idx)

        result = rolling_origin_backtest(
            combined,
            model_factory,
            origin_indices,
            horizon,
            model_kwargs,
        )
        result.series_id = series_id
        results[series_id] = result

    return results


def aggregate_results(results: dict[str, BacktestResult]) -> dict[str, float]:
    """Aggregate backtest results across all series."""
    all_mae = []
    all_rmse = []
    all_wape = []

    for res in results.values():
        all_mae.append(res.avg_metrics["mae"])
        all_rmse.append(res.avg_metrics["rmse"])
        # WAPE can be NaN, use nanmean
        all_wape.append(res.avg_metrics["wape"])

    return {
        "mae": float(np.nanmean(all_mae)),
        "rmse": float(np.nanmean(all_rmse)),
        "wape": float(np.nanmean(all_wape)),
        "n_series": len(results),
    }


def evaluate_test_set(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    model_factory: Callable[..., Any],
    horizon: int,
    model_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Evaluate model on final test set using single origin at end of validation.

    Args:
        train_series: Training data
        val_series: Validation data
        test_series: Test data (first horizon weeks used for evaluation)
        model_factory: Model factory callable
        horizon: Forecast horizon
        model_kwargs: Model arguments

    Returns:
        Test metrics
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Train on train + validation
    combined = pd.concat([train_series, val_series])
    model = model_factory(combined, **model_kwargs)
    result = model.predict(horizon)

    # Actuals from test set (first horizon weeks)
    actuals = np.asarray(test_series.iloc[:horizon].values, dtype=float)

    if len(actuals) != horizon:
        return {"mae": np.nan, "rmse": np.nan, "wape": np.nan, "error": "Insufficient test data"}

    return evaluate_forecast(actuals, result.point_forecast)
