"""Tests for forecasting evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.forecasting.baselines import (
    naive_factory,
)
from src.aviation_mro.analytics.forecasting.data_prep import (
    generate_synthetic_demand,
    split_chronologically,
)
from src.aviation_mro.analytics.forecasting.evaluation import (
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


class TestMetrics:
    """Test metric functions."""

    def test_mae_basic(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.5, 2.5, 2.5, 3.5])
        assert mae(actual, pred) == 0.5

    def test_mae_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert mae(actual, pred) == 0.0

    def test_rmse_basic(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 2.0, 2.0])
        # Errors: [1, 0, 1], squared: [1, 0, 1], mean: 2/3, sqrt: ~0.816
        expected = np.sqrt(2 / 3)
        assert abs(rmse(actual, pred) - expected) < 1e-6

    def test_wape_basic(self):
        actual = np.array([10.0, 20.0, 30.0])
        pred = np.array([12.0, 18.0, 33.0])
        # |10-12| + |20-18| + |30-33| = 2+2+3 = 7
        # sum(actual) = 60
        # WAPE = 7/60 = 0.1166...
        expected = 7 / 60
        assert abs(wape(actual, pred) - expected) < 1e-6

    def test_wape_zero_denominator_returns_nan(self):
        actual = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 2.0, 3.0])
        result = wape(actual, pred)
        assert np.isnan(result)

    def test_wape_some_zeros(self):
        # Only sum of actual matters for denominator
        actual = np.array([0.0, 10.0, 0.0])
        pred = np.array([5.0, 15.0, 5.0])
        # |0-5| + |10-15| + |0-5| = 5+5+5 = 15
        # sum(actual) = 10
        # WAPE = 15/10 = 1.5
        assert wape(actual, pred) == 1.5

    def test_evaluate_forecast(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 2.5, 2.5])
        metrics = evaluate_forecast(actual, pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "wape" in metrics


class TestRollingOriginBacktest:
    """Test rolling-origin backtesting."""

    def setup_method(self):
        # Create a series with enough data for multiple origins
        dates = pd.date_range("2022-01-03", periods=50, freq="W-MON")
        values = np.random.default_rng(42).integers(0, 20, size=50)
        self.series = pd.Series(values, index=dates, name="demand_count")
        self.origins = [20, 26, 32, 38]  # Week numbers (1-indexed)
        self.horizon = 6

    def test_rolling_origin_returns_backtest_result(self):
        result = rolling_origin_backtest(
            self.series,
            naive_factory,
            self.origins,
            self.horizon,
            model_kwargs={"model_name": "naive"},
        )

        assert isinstance(result, BacktestResult)
        assert result.model_name == "naive"
        assert result.horizon == self.horizon
        assert result.n_origins == len(self.origins)

    def test_rolling_origin_metrics_exist(self):
        result = rolling_origin_backtest(
            self.series,
            naive_factory,
            self.origins,
            self.horizon,
            model_kwargs={"model_name": "naive"},
        )

        assert "mae" in result.avg_metrics
        assert "rmse" in result.avg_metrics
        assert "wape" in result.avg_metrics
        assert all(isinstance(v, float) for v in result.avg_metrics.values())

    def test_rolling_origin_per_origin_metrics(self):
        result = rolling_origin_backtest(
            self.series,
            naive_factory,
            self.origins,
            self.horizon,
            model_kwargs={"model_name": "naive"},
        )

        assert len(result.per_origin_metrics) == len(self.origins)
        for m in result.per_origin_metrics:
            assert "mae" in m
            assert "rmse" in m
            assert "wape" in m

    def test_rolling_origin_all_predictions_collected(self):
        result = rolling_origin_backtest(
            self.series,
            naive_factory,
            self.origins,
            self.horizon,
            model_kwargs={"model_name": "naive"},
        )

        # Should have n_origins * horizon predictions
        expected_n = len(self.origins) * self.horizon
        assert len(result.all_predictions) == expected_n
        assert len(result.all_actuals) == expected_n


class TestEvaluateModelOnAllSeries:
    """Test evaluating model on all series."""

    def test_evaluate_all_series(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        # Convert to dict of series
        train_dict = {}
        val_dict = {}
        for (shop, part), group in split.train.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            train_dict[series_id] = group.set_index("date")["demand_count"]

        for (shop, part), group in split.validation.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            val_dict[series_id] = group.set_index("date")["demand_count"]

        results = evaluate_model_on_all_series(
            train_dict,
            val_dict,
            naive_factory,
            split.validation_origins,
            horizon=12,
            model_kwargs={"model_name": "naive"},
        )

        assert len(results) == 15
        for _series_id, result in results.items():
            assert isinstance(result, BacktestResult)
            assert result.model_name == "naive"


class TestAggregateResults:
    """Test aggregate_results function."""

    def test_aggregate_returns_mean_metrics(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_dict = {}
        val_dict = {}
        for (shop, part), group in split.train.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            train_dict[series_id] = group.set_index("date")["demand_count"]

        for (shop, part), group in split.validation.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            val_dict[series_id] = group.set_index("date")["demand_count"]

        results = evaluate_model_on_all_series(
            train_dict,
            val_dict,
            naive_factory,
            split.validation_origins,
            horizon=12,
            model_kwargs={"model_name": "naive"},
        )

        agg = aggregate_results(results)

        assert "mae" in agg
        assert "rmse" in agg
        assert "wape" in agg
        assert "n_series" in agg
        assert agg["n_series"] == 15


class TestEvaluateTestSet:
    """Test final test set evaluation."""

    def test_evaluate_test_set(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_dict = {}
        val_dict = {}
        test_dict = {}
        for (shop, part), group in split.train.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            train_dict[series_id] = group.set_index("date")["demand_count"]

        for (shop, part), group in split.validation.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            val_dict[series_id] = group.set_index("date")["demand_count"]

        for (shop, part), group in split.test.groupby(["shop_id", "part_family"]):
            series_id = f"{shop}_{part}"
            test_dict[series_id] = group.set_index("date")["demand_count"]

        result = evaluate_test_set(
            train_dict["SHOP_01_ENGINE"],
            val_dict["SHOP_01_ENGINE"],
            test_dict["SHOP_01_ENGINE"],
            naive_factory,
            horizon=12,
            model_kwargs={"model_name": "naive"},
        )

        assert "mae" in result
        assert "rmse" in result
        assert "wape" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
