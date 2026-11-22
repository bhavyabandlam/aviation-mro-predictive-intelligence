"""Naive and Seasonal-Naive Forecasting Baselines.

Implements two standard forecasting benchmarks:
1. Naive (Last Value Carried Forward)
2. Seasonal-Naive (Same period from previous seasonal cycle)

Both use only information available at forecast origin — no future leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Container for forecast output."""

    point_forecast: np.ndarray  # Shape: (horizon,)
    prediction_intervals: np.ndarray | None  # Shape: (horizon, 2) or None
    model_name: str
    origin_date: pd.Timestamp
    horizon: int
    seasonal_period: int | None = None


class NaiveForecaster:
    """Naive forecaster: predicts the last observed value for all horizons."""

    def __init__(self) -> None:
        self.model_name = "naive"
        self.seasonal_period = None
        self.last_value: float | None = None
        self.origin_date: pd.Timestamp | None = None
        self._is_fitted = False

    def fit(self, y: pd.Series) -> NaiveForecaster:
        """Fit the model on historical data.

        Args:
            y: Time series indexed by date (must be sorted)

        Returns:
            Self for chaining
        """
        if len(y) == 0:
            raise ValueError("Cannot fit on empty series")
        self.last_value = float(y.iloc[-1])
        self.origin_date = pd.Timestamp(y.index[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastResult:
        """Generate naive forecasts.

        Args:
            horizon: Number of steps ahead to forecast

        Returns:
            ForecastResult with point forecasts
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        point_forecast = np.full(horizon, self.last_value, dtype=float)
        origin = self.origin_date
        assert origin is not None

        return ForecastResult(
            point_forecast=point_forecast,
            prediction_intervals=None,  # Intervals added by uncertainty module
            model_name=self.model_name,
            origin_date=origin,
            horizon=horizon,
        )

    def predict_with_intervals(self, horizon: int, residual_std: float) -> ForecastResult:
        """Generate forecasts with approximate prediction intervals.

        Args:
            horizon: Number of steps ahead
            residual_std: Standard deviation of residuals from validation

        Returns:
            ForecastResult with point forecasts and intervals
        """
        result = self.predict(horizon)
        if residual_std > 0:
            # Approximate 95% interval: ±1.96 * residual_std
            margin = 1.96 * residual_std
            intervals = np.column_stack(
                [
                    result.point_forecast - margin,
                    result.point_forecast + margin,
                ]
            )
            result.prediction_intervals = intervals
        return result


class SeasonalNaiveForecaster:
    """Seasonal-Naive forecaster: predicts value from same seasonal period last cycle.

    Uses only historical data available at forecast origin.
    Falls back to naive if insufficient seasonal history.
    """

    def __init__(self, seasonal_period: int) -> None:
        if seasonal_period <= 0:
            raise ValueError("Seasonal period must be positive")
        self.seasonal_period = seasonal_period
        self.model_name = "seasonal_naive"
        self.history: np.ndarray | None = None
        self.origin_date: pd.Timestamp | None = None
        self._is_fitted = False

    def fit(self, y: pd.Series) -> SeasonalNaiveForecaster:
        """Fit the model on historical data.

        Args:
            y: Time series indexed by date (must be sorted)

        Returns:
            Self for chaining

        Raises:
            ValueError: If series has fewer observations than seasonal period
        """
        if len(y) == 0:
            raise ValueError("Cannot fit on empty series")
        if len(y) < self.seasonal_period:
            raise ValueError("at least seasonal_period observations")
        self.history = np.asarray(y.values, dtype=float)
        self.origin_date = pd.Timestamp(y.index[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastResult:
        """Generate seasonal-naive forecasts.

        For each horizon h, uses observation from (t - m + h) where m is seasonal period.
        If that index is out of bounds (insufficient history), falls back to last value.
        Tiles the seasonal pattern for horizons longer than the period.

        Args:
            horizon: Number of steps ahead to forecast

        Returns:
            ForecastResult with point forecasts
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        history = self.history
        if history is None:
            raise ValueError("Model not fitted")
        n_history = len(history)
        point_forecast = np.zeros(horizon, dtype=float)

        for h in range(1, horizon + 1):
            # Index in history for same seasonal position
            # At origin t (last index = n_history - 1), we want t - m + h
            idx = n_history - self.seasonal_period + h - 1

            # Tile the seasonal pattern if needed
            while idx < 0:
                idx += self.seasonal_period
            while idx >= n_history:
                idx -= self.seasonal_period

            if 0 <= idx < n_history:
                point_forecast[h - 1] = history[idx]
            else:
                # Should not happen with tiling, but fallback
                point_forecast[h - 1] = history[-1]

        origin = self.origin_date
        assert origin is not None
        return ForecastResult(
            point_forecast=point_forecast,
            prediction_intervals=None,
            model_name=self.model_name,
            origin_date=origin,
            horizon=horizon,
            seasonal_period=self.seasonal_period,
        )

    def predict_with_intervals(self, horizon: int, residual_std: float) -> ForecastResult:
        """Generate forecasts with approximate prediction intervals."""
        result = self.predict(horizon)
        if residual_std > 0:
            margin = 1.96 * residual_std
            intervals = np.column_stack(
                [
                    result.point_forecast - margin,
                    result.point_forecast + margin,
                ]
            )
            result.prediction_intervals = intervals
        return result


def naive_factory(history: pd.Series, **kwargs: Any) -> NaiveForecaster:
    """Factory function for NaiveForecaster."""
    model_name = kwargs.pop("model_name", "naive")
    model = NaiveForecaster()
    model.model_name = model_name
    return model.fit(history)


def seasonal_naive_factory(
    history: pd.Series, seasonal_period: int, **kwargs: Any
) -> SeasonalNaiveForecaster:
    """Factory function for SeasonalNaiveForecaster."""
    model_name = kwargs.pop("model_name", "seasonal_naive")
    model = SeasonalNaiveForecaster(seasonal_period=seasonal_period)
    model.model_name = model_name
    return model.fit(history)


def evaluate_seasonal_candidates(
    train_series: pd.Series,
    val_series: pd.Series,
    candidate_periods: list[int],
    horizon: int,
) -> dict[int, dict[str, Any]]:
    """Evaluate multiple seasonal periods on validation data.

    Uses rolling-origin evaluation on validation set to compare
    seasonal-naive against naive for each candidate period.

    Args:
        train_series: Training data
        val_series: Validation data
        candidate_periods: List of seasonal periods to evaluate (e.g., [52, 13])
        horizon: Forecast horizon

    Returns:
        Dict mapping period -> {mae, rmse, wape, better_than_naive}
    """
    from .evaluation import mae, rmse, wape

    # Combine train + validation for rolling origins
    # We'll evaluate at the end of train and within validation
    combined = pd.concat([train_series, val_series])

    results = {}

    for period in candidate_periods:
        # Skip if not enough history for at least one full cycle at first origin
        if len(train_series) < period:
            results[period] = {
                "mae": np.nan,
                "rmse": np.nan,
                "wape": np.nan,
                "better_than_naive": False,
                "error": (
                    f"Insufficient history: train length {len(train_series)} < period {period}"
                ),
            }
            continue

        # Use the same validation origins as defined in data_prep
        # Origin weeks relative to combined series
        train_len = len(train_series)
        val_len = len(val_series)
        # Origins: end of train (train_len-1), then every 6 weeks in validation
        # Convert to indices in combined series
        origin_indices = [train_len - 1]
        for offset in range(6, val_len, 6):
            if origin_indices[-1] + offset < train_len + val_len - horizon:
                origin_indices.append(train_len - 1 + offset)

        # Evaluate at each origin
        period_maes = []
        period_rmses = []
        period_wapes = []
        naive_maes = []

        for origin_idx in origin_indices:
            history = combined.iloc[: origin_idx + 1]

            # Seasonal-naive forecast
            snaive = SeasonalNaiveForecaster(seasonal_period=period).fit(history)
            snaive_result = snaive.predict(horizon)

            # Naive forecast for comparison
            naive = NaiveForecaster().fit(history)
            naive_result = naive.predict(horizon)

            # Actuals
            actuals = combined.iloc[origin_idx + 1 : origin_idx + 1 + horizon].values

            if len(actuals) == horizon:
                period_maes.append(mae(actuals, snaive_result.point_forecast))
                period_rmses.append(rmse(actuals, snaive_result.point_forecast))
                period_wapes.append(wape(actuals, snaive_result.point_forecast))
                naive_maes.append(mae(actuals, naive_result.point_forecast))

        if period_maes:
            avg_mae = float(np.nanmean(period_maes))
            avg_rmse = float(np.nanmean(period_rmses))
            avg_wape = float(np.nanmean(period_wapes))
            avg_naive_mae = float(np.nanmean(naive_maes))

            results[period] = {
                "mae": avg_mae,
                "rmse": avg_rmse,
                "wape": avg_wape,
                "better_than_naive": avg_mae < avg_naive_mae,
                "naive_mae": avg_naive_mae,
            }
        else:
            results[period] = {
                "mae": np.nan,
                "rmse": np.nan,
                "wape": np.nan,
                "better_than_naive": False,
                "error": "No valid evaluation windows",
            }

    return results
