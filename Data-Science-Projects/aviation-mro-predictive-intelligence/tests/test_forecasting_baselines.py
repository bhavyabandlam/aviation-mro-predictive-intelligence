"""Tests for forecasting baselines module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.forecasting.baselines import (
    ForecastResult,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    evaluate_seasonal_candidates,
)
from src.aviation_mro.analytics.forecasting.data_prep import (
    generate_synthetic_demand,
    split_chronologically,
)
from src.aviation_mro.analytics.forecasting.train import (
    naive_factory,
    seasonal_naive_factory,
)


class TestNaiveForecaster:
    """Test NaiveForecaster."""

    def setup_method(self):
        # Create a simple test series
        dates = pd.date_range("2022-01-03", periods=20, freq="W-MON")
        self.series = pd.Series(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            index=dates,
            name="demand_count",
        )

    def test_naive_fit_predict(self):
        model = NaiveForecaster()
        model.fit(self.series)
        result = model.predict(5)

        assert isinstance(result, ForecastResult)
        assert result.model_name == "naive"
        assert result.horizon == 5
        assert len(result.point_forecast) == 5
        assert result.prediction_intervals is None

        # Naive forecast should be last value repeated
        expected = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_naive_requires_fit(self):
        model = NaiveForecaster()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(5)

    def test_naive_origin_date(self):
        model = NaiveForecaster()
        model.fit(self.series)
        result = model.predict(5)
        assert result.origin_date == self.series.index[-1]

    def test_naive_seasonal_period_none(self):
        model = NaiveForecaster()
        model.fit(self.series)
        assert model.seasonal_period is None


class TestSeasonalNaiveForecaster:
    """Test SeasonalNaiveForecaster."""

    def setup_method(self):
        # Create a series with clear seasonal pattern (period=4)
        dates = pd.date_range("2022-01-03", periods=20, freq="W-MON")
        # Pattern: [10, 20, 30, 40] repeated
        values = [10, 20, 30, 40] * 5
        self.series = pd.Series(values, index=dates, name="demand_count")

    def test_seasonal_naive_fit_predict(self):
        model = SeasonalNaiveForecaster(seasonal_period=4)
        model.fit(self.series)
        result = model.predict(4)

        assert isinstance(result, ForecastResult)
        assert result.model_name == "seasonal_naive"
        assert result.seasonal_period == 4
        assert result.horizon == 4

        # Last cycle was [10, 20, 30, 40], so forecast should repeat
        expected = np.array([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_seasonal_naive_requires_fit(self):
        model = SeasonalNaiveForecaster(seasonal_period=4)
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(5)

    def test_seasonal_naive_seasonal_period_attribute(self):
        model = SeasonalNaiveForecaster(seasonal_period=13)
        model.fit(self.series)
        assert model.seasonal_period == 13

    def test_seasonal_naive_horizon_longer_than_period(self):
        model = SeasonalNaiveForecaster(seasonal_period=4)
        model.fit(self.series)
        result = model.predict(10)  # Longer than period

        assert len(result.point_forecast) == 10
        # Should tile the pattern: [10,20,30,40,10,20,30,40,10,20]
        expected = np.array([10, 20, 30, 40, 10, 20, 30, 40, 10, 20], dtype=float)
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_seasonal_naive_requires_enough_history(self):
        # Series shorter than seasonal period
        dates = pd.date_range("2022-01-03", periods=3, freq="W-MON")
        short_series = pd.Series([1, 2, 3], index=dates, name="demand_count")

        model = SeasonalNaiveForecaster(seasonal_period=4)
        with pytest.raises(ValueError, match="at least seasonal_period observations"):
            model.fit(short_series)


class TestFactories:
    """Test factory functions."""

    def test_naive_factory(self):
        series = pd.Series([1, 2, 3], index=pd.date_range("2022-01-03", periods=3, freq="W-MON"))
        model = naive_factory(series, model_name="test")
        assert isinstance(model, NaiveForecaster)
        assert model.model_name == "test"

    def test_seasonal_naive_factory(self):
        series = pd.Series([1, 2, 3, 4], index=pd.date_range("2022-01-03", periods=4, freq="W-MON"))
        model = seasonal_naive_factory(series, seasonal_period=4, model_name="test")
        assert isinstance(model, SeasonalNaiveForecaster)
        assert model.seasonal_period == 4
        assert model.model_name == "test"

    def test_seasonal_naive_factory_requires_period(self):
        series = pd.Series([1, 2, 3], index=pd.date_range("2022-01-03", periods=3, freq="W-MON"))
        with pytest.raises(TypeError):
            seasonal_naive_factory(series)


class TestEvaluateSeasonalCandidates:
    """Test seasonal candidate evaluation."""

    def test_evaluate_returns_dict(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        # Extract a single series for testing
        train_series = split.train[
            (split.train["shop_id"] == "SHOP_01") & (split.train["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]
        val_series = split.validation[
            (split.validation["shop_id"] == "SHOP_01")
            & (split.validation["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]

        results = evaluate_seasonal_candidates(train_series, val_series, [52, 13], 12)

        assert isinstance(results, dict)
        assert 52 in results
        assert 13 in results

        for _period, metrics in results.items():
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "wape" in metrics
            assert "better_than_naive" in metrics
            assert "naive_mae" in metrics


class TestForecastResult:
    """Test ForecastResult dataclass."""

    def test_forecast_result_creation(self):
        result = ForecastResult(
            point_forecast=np.array([1.0, 2.0, 3.0]),
            prediction_intervals=None,
            model_name="test",
            origin_date=pd.Timestamp("2022-01-03"),
            horizon=3,
            seasonal_period=None,
        )
        assert result.horizon == 3
        assert result.model_name == "test"

    def test_forecast_result_with_intervals(self):
        intervals = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
        result = ForecastResult(
            point_forecast=np.array([1.0, 2.0, 3.0]),
            prediction_intervals=intervals,
            model_name="test",
            origin_date=pd.Timestamp("2022-01-03"),
            horizon=3,
            seasonal_period=52,
        )
        assert result.prediction_intervals is not None
        assert result.prediction_intervals.shape == (3, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
