"""Tests for forecasting uncertainty module."""

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
from src.aviation_mro.analytics.forecasting.uncertainty import (
    IntervalCalibration,
    apply_intervals,
    calibrate_all_series,
    calibrate_intervals,
    collect_validation_residuals,
    evaluate_coverage,
)


class TestCollectValidationResiduals:
    """Test collecting residuals from validation."""

    def test_collect_residuals(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_series = split.train.loc[
            (split.train["shop_id"] == "SHOP_01") & (split.train["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]

        val_series = split.validation.loc[
            (
                (split.validation["shop_id"] == "SHOP_01")
                & (split.validation["part_family"] == "ENGINE")
            )
        ].set_index("date")["demand_count"]

        residuals = collect_validation_residuals(
            train_series,
            val_series,
            naive_factory,
            split.validation_origins,
            horizon=12,
            model_kwargs={"model_name": "naive"},
        )

        assert isinstance(residuals, dict)
        assert all(h in residuals for h in range(1, 13))
        # Each horizon should have 3 origins * 1 residual = 3 residuals
        # (origin 122 doesn't have enough future data: 122+12=134 > 130)
        for h in range(1, 13):
            assert len(residuals[h]) == 3


class TestCalibrateIntervals:
    """Test interval calibration."""

    def test_calibrate_intervals(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_series = split.train.loc[
            (split.train["shop_id"] == "SHOP_01") & (split.train["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]

        val_series = split.validation.loc[
            (
                (split.validation["shop_id"] == "SHOP_01")
                & (split.validation["part_family"] == "ENGINE")
            )
        ].set_index("date")["demand_count"]

        calibration = calibrate_intervals(
            train_series,
            val_series,
            naive_factory,
            split.validation_origins,
            horizon=12,
            target_coverage=0.95,
            model_kwargs={"model_name": "naive"},
        )

        assert isinstance(calibration, IntervalCalibration)
        assert calibration.horizon == 12
        assert calibration.target_coverage == 0.95
        assert len(calibration.lower_quantiles) == 12
        assert len(calibration.upper_quantiles) == 12
        assert len(calibration.n_residuals_per_horizon) == 12
        assert 0 <= calibration.empirical_coverage <= 1

    def test_calibrate_intervals_lower_less_than_upper(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_series = split.train.loc[
            (split.train["shop_id"] == "SHOP_01") & (split.train["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]

        val_series = split.validation.loc[
            (
                (split.validation["shop_id"] == "SHOP_01")
                & (split.validation["part_family"] == "ENGINE")
            )
        ].set_index("date")["demand_count"]

        calibration = calibrate_intervals(
            train_series,
            val_series,
            naive_factory,
            split.validation_origins,
            horizon=12,
            target_coverage=0.95,
            model_kwargs={"model_name": "naive"},
        )

        assert np.all(calibration.lower_quantiles <= calibration.upper_quantiles)


class TestApplyIntervals:
    """Test applying calibrated intervals to forecasts."""

    def test_apply_intervals(self):
        from src.aviation_mro.analytics.forecasting.baselines import ForecastResult

        forecast = ForecastResult(
            point_forecast=np.array([10.0, 20.0, 30.0]),
            prediction_intervals=None,
            model_name="test",
            origin_date=pd.Timestamp("2022-01-03"),
            horizon=3,
        )

        calibration = IntervalCalibration(
            horizon=3,
            lower_quantiles=np.array([-2.0, -3.0, -4.0]),
            upper_quantiles=np.array([2.0, 3.0, 4.0]),
            empirical_coverage=0.9,
            target_coverage=0.95,
            n_residuals_per_horizon=[10, 10, 10],
        )

        result = apply_intervals(forecast, calibration)

        assert result.prediction_intervals is not None
        assert result.prediction_intervals.shape == (3, 2)

        # Check intervals are point_forecast + quantiles
        expected_lower = np.array([8.0, 17.0, 26.0])
        expected_upper = np.array([12.0, 23.0, 34.0])

        np.testing.assert_array_equal(result.prediction_intervals[:, 0], expected_lower)
        np.testing.assert_array_equal(result.prediction_intervals[:, 1], expected_upper)

    def test_apply_intervals_horizon_mismatch(self):
        from src.aviation_mro.analytics.forecasting.baselines import ForecastResult

        forecast = ForecastResult(
            point_forecast=np.array([10.0, 20.0]),
            prediction_intervals=None,
            model_name="test",
            origin_date=pd.Timestamp("2022-01-03"),
            horizon=2,
        )

        calibration = IntervalCalibration(
            horizon=3,
            lower_quantiles=np.array([-2.0, -3.0, -4.0]),
            upper_quantiles=np.array([2.0, 3.0, 4.0]),
            empirical_coverage=0.9,
            target_coverage=0.95,
            n_residuals_per_horizon=[10, 10, 10],
        )

        with pytest.raises(ValueError, match="horizon must match"):
            apply_intervals(forecast, calibration)


class TestEvaluateCoverage:
    """Test coverage evaluation on test set."""

    def test_evaluate_coverage(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_series = split.train.loc[
            (split.train["shop_id"] == "SHOP_01") & (split.train["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]

        val_series = split.validation.loc[
            (
                (split.validation["shop_id"] == "SHOP_01")
                & (split.validation["part_family"] == "ENGINE")
            )
        ].set_index("date")["demand_count"]

        test_series = split.test.loc[
            (split.test["shop_id"] == "SHOP_01") & (split.test["part_family"] == "ENGINE")
        ].set_index("date")["demand_count"]

        result = evaluate_coverage(
            train_series,
            val_series,
            test_series,
            naive_factory,
            split.validation_origins,
            horizon=12,
            model_kwargs={"model_name": "naive"},
        )

        assert "empirical_coverage" in result
        assert "target_coverage" in result
        assert "avg_interval_width" in result
        assert "calibration_empirical_coverage" in result
        assert result["target_coverage"] == 0.95
        assert 0 <= result["empirical_coverage"] <= 1


class TestCalibrateAllSeries:
    """Test calibrating all series."""

    def test_calibrate_all_series(self):
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

        calibrations = calibrate_all_series(
            train_dict,
            val_dict,
            naive_factory,
            split.validation_origins,
            horizon=12,
            model_kwargs={"model_name": "naive"},
        )

        assert len(calibrations) == 15
        for _series_id, cal in calibrations.items():
            assert isinstance(cal, IntervalCalibration)
            assert cal.horizon == 12


class TestIntervalCalibration:
    """Test IntervalCalibration dataclass."""

    def test_interval_calibration_creation(self):
        cal = IntervalCalibration(
            horizon=12,
            lower_quantiles=np.zeros(12),
            upper_quantiles=np.ones(12),
            empirical_coverage=0.9,
            target_coverage=0.95,
            n_residuals_per_horizon=[4] * 12,
        )
        assert cal.horizon == 12
        assert cal.empirical_coverage == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
