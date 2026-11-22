"""Edge case tests for Phase B4 business analytics (FR-08, FR-09, FR-10).

Tests boundary conditions, zero-value inputs, invalid parameters, and
degenerate scenarios across forecasting, inventory, and financial domains.
All data is synthetic — not real airline records.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.kpis import FinancialKPICalculator
from src.aviation_mro.analytics.financial.scenarios import ScenarioAnalyzer
from src.aviation_mro.analytics.forecasting.baselines import (
    ForecastResult,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    naive_factory,
)
from src.aviation_mro.analytics.forecasting.evaluation import (
    mae,
    rmse,
    rolling_origin_backtest,
    wape,
)
from src.aviation_mro.analytics.inventory.demand_analysis import PartsDemandAnalyzer
from src.aviation_mro.analytics.inventory.reorder_point import ReorderPointCalculator
from src.aviation_mro.analytics.inventory.safety_stock import SafetyStockCalculator
from src.aviation_mro.analytics.inventory.stockout_risk import StockoutRiskAnalyzer

# ---------------------------------------------------------------------------
# Forecasting edge cases (FR-08)
# ---------------------------------------------------------------------------


class TestForecastingEdgeCases:
    """Edge cases for forecasting models and metrics."""

    def test_wape_zero_demand_returns_nan(self):
        """WAPE must return NaN when all actuals are zero (undefined)."""
        y_true = np.zeros(5)
        y_pred = np.ones(5)
        result = wape(y_true, y_pred)
        assert math.isnan(result)

    def test_wape_zero_actuals_nonzero_pred(self):
        """WAPE with zero actuals but nonzero predictions → NaN."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert math.isnan(wape(y_true, y_pred))

    def test_mae_perfect_prediction(self):
        """MAE = 0 when predictions exactly match actuals."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert mae(y_true, y_pred) == 0.0

    def test_rmse_perfect_prediction(self):
        """RMSE = 0 when predictions exactly match actuals."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_naive_constant_series(self):
        """Naive forecast on constant series predicts the constant value."""
        dates = pd.date_range("2022-01-03", periods=10, freq="W-MON")
        series = pd.Series([50.0] * 10, index=dates, name="demand_count")
        model = NaiveForecaster()
        model.fit(series)
        result = model.predict(5)
        expected = np.array([50.0] * 5)
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_naive_single_point_series(self):
        """Naive forecast on single-point series predicts that value."""
        dates = pd.date_range("2022-01-03", periods=1, freq="W-MON")
        series = pd.Series([42.0], index=dates, name="demand_count")
        model = NaiveForecaster()
        model.fit(series)
        result = model.predict(3)
        expected = np.array([42.0, 42.0, 42.0])
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_naive_negative_values(self):
        """Naive forecast handles negative values (e.g., net demand adjustments)."""
        dates = pd.date_range("2022-01-03", periods=5, freq="W-MON")
        series = pd.Series([-5.0, -3.0, -1.0, -2.0, -4.0], index=dates, name="demand_count")
        model = NaiveForecaster()
        model.fit(series)
        result = model.predict(2)
        expected = np.array([-4.0, -4.0])
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_seasonal_naive_horizon_longer_than_series(self):
        """Seasonal-naive tiles pattern when horizon exceeds series length."""
        dates = pd.date_range("2022-01-03", periods=4, freq="W-MON")
        series = pd.Series([10.0, 20.0, 30.0, 40.0], index=dates, name="demand_count")
        model = SeasonalNaiveForecaster(seasonal_period=2)
        model.fit(series)
        result = model.predict(10)
        # With history [10,20,30,40] and period 2:
        # h=1 -> idx=2 (30), h=2 -> idx=3 (40), h=3 -> idx=2 (30), etc.
        # Pattern [30, 40] tiled
        expected = np.array([30.0, 40.0] * 5)
        np.testing.assert_array_equal(result.point_forecast, expected)

    def test_rolling_origin_origin_at_series_end(self):
        """Rolling-origin skips origins that lack sufficient future data."""
        dates = pd.date_range("2022-01-03", periods=5, freq="W-MON")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, name="demand_count")
        # Origin at index 3 (4th value) with horizon 2 → needs index 4,5 (out of bounds)
        result = rolling_origin_backtest(
            series,
            naive_factory,
            origins=[0, 1, 2, 3, 4],
            horizon=2,
            model_kwargs={"model_name": "test"},
        )
        # Only origins 0, 1, 2 have enough future data (origin_idx + horizon <= len)
        assert result.n_origins == 3

    def test_rolling_origin_empty_origins(self):
        """Rolling-origin with no valid origins returns NaN metrics."""
        dates = pd.date_range("2022-01-03", periods=3, freq="W-MON")
        series = pd.Series([1.0, 2.0, 3.0], index=dates, name="demand_count")
        result = rolling_origin_backtest(
            series,
            naive_factory,
            origins=[0],
            horizon=10,
            model_kwargs={"model_name": "test"},
        )
        assert math.isnan(result.avg_metrics["mae"])
        assert math.isnan(result.avg_metrics["rmse"])
        assert math.isnan(result.avg_metrics["wape"])

    def test_forecast_result_horizon_zero(self):
        """ForecastResult with horizon 0 has empty arrays."""
        result = ForecastResult(
            point_forecast=np.array([], dtype=float),
            prediction_intervals=None,
            model_name="test",
            origin_date=pd.Timestamp("2022-01-03"),
            horizon=0,
        )
        assert result.horizon == 0
        assert len(result.point_forecast) == 0

    def test_seasonal_naive_seasonal_period_1(self):
        """Seasonal period 1 predicts the last value repeatedly."""
        dates = pd.date_range("2022-01-03", periods=5, freq="W-MON")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, name="demand_count")
        model = SeasonalNaiveForecaster(seasonal_period=1)
        model.fit(series)
        result = model.predict(3)
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(result.point_forecast, expected)


# ---------------------------------------------------------------------------
# Inventory edge cases (FR-09)
# ---------------------------------------------------------------------------


class TestInventoryEdgeCases:
    """Edge cases for inventory calculations."""

    def test_reorder_point_zero_mean_demand(self):
        """ROP with zero mean demand = safety stock only."""
        calc = ReorderPointCalculator()
        mean_ltd, ss, rop, z = calc.calculate_reorder_point(
            mean_weekly_demand=0.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
            service_level=0.95,
        )
        assert mean_ltd == 0.0
        assert ss > 0  # Safety stock from demand variability
        assert rop == ss  # ROP = 0 + safety stock

    def test_reorder_point_zero_std_demand(self):
        """ROP with zero demand std = mean lead time demand (no safety stock)."""
        calc = ReorderPointCalculator()
        mean_ltd, ss, rop, z = calc.calculate_reorder_point(
            mean_weekly_demand=10.0,
            std_weekly_demand=0.0,
            lead_time_days=14,
            service_level=0.95,
        )
        assert mean_ltd == 20.0
        assert ss == 0.0
        assert rop == 20.0

    def test_reorder_point_zero_lead_time(self):
        """ROP with zero lead time = 0 lead time demand."""
        calc = ReorderPointCalculator()
        mean_ltd, ss, rop, z = calc.calculate_reorder_point(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=0,
            service_level=0.95,
        )
        assert mean_ltd == 0.0
        assert ss == 0.0
        assert rop == 0.0

    def test_z_score_boundary_service_levels(self):
        """Z-score at service level boundaries."""
        calc = ReorderPointCalculator()
        # Very low service level → very negative z
        z_low = calc.z_score(0.01)
        assert z_low < -2.0
        # Very high service level → very positive z
        z_high = calc.z_score(0.99)
        assert z_high > 2.0

    def test_z_score_rejects_invalid(self):
        """Z-score rejects service levels outside (0, 1)."""
        calc = ReorderPointCalculator()
        with pytest.raises(ValueError):
            calc.z_score(0.0)
        with pytest.raises(ValueError):
            calc.z_score(1.0)
        with pytest.raises(ValueError):
            calc.z_score(-0.1)
        with pytest.raises(ValueError):
            calc.z_score(1.5)

    def test_safety_stock_normal_zero_std(self):
        """Safety stock with zero demand std = 0."""
        calc = SafetyStockCalculator()
        ss = calc.method_normal_constant_lt(
            mean_weekly_demand=10.0,
            std_weekly_demand=0.0,
            lead_time_days=14,
            service_level=0.95,
        )
        assert ss == 0.0

    def test_safety_stock_log_normal_zero_mean(self):
        """Log-normal safety stock falls back to normal when mean <= 0."""
        calc = SafetyStockCalculator()
        ss = calc.method_lognormal(
            mean_weekly_demand=0.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
            service_level=0.95,
        )
        # Should equal normal method result
        expected = calc.method_normal_constant_lt(0.0, 2.0, 14, 0.95)
        assert abs(ss - expected) < 0.01

    def test_stockout_probability_zero_std_at_rop(self):
        """Zero std: stockout prob 0 if mean <= ROP, 1 if mean > ROP."""
        analyzer = StockoutRiskAnalyzer()
        assert analyzer.calculate_stockout_probability(100.0, 0.0, 100.0) == 0.0
        assert analyzer.calculate_stockout_probability(100.0, 0.0, 101.0) == 0.0
        assert analyzer.calculate_stockout_probability(100.0, 0.0, 99.0) == 1.0

    def test_stockout_risk_negative_current_stock(self):
        """Negative current stock analyzed without error."""
        analyzer = StockoutRiskAnalyzer()
        df = pd.DataFrame(
            {
                "part_number": ["PART-NEG"],
                "part_family": ["ENGINE"],
                "shop_id": ["SHOP_01"],
                "current_stock": [-50.0],
                "demand_mean_weekly": [10.0],
                "demand_std_weekly": [2.0],
                "lead_time_days": [14],
                "lead_time_std_days": [0],
                "service_level_target": [0.95],
                "unit_cost": [500.0],
            }
        )
        results = analyzer.analyze_for_dataframe(df)
        assert len(results) == 1
        assert results[0].current_stock == -50.0

    def test_demand_analyzer_zero_mean(self):
        """Demand analyzer handles zero mean demand (CV = 0, intermittent pattern)."""
        analyzer = PartsDemandAnalyzer()
        df = pd.DataFrame(
            {
                "part_number": ["PART-ZERO"],
                "part_family": ["ENGINE"],
                "shop_id": ["SHOP_01"],
                "demand_mean_weekly": [0.0],
                "demand_std_weekly": [0.0],
            }
        )
        stats_list = analyzer.compute_statistics(df)
        assert len(stats_list) == 1
        assert stats_list[0].cv == 0.0
        # Zero mean → intermittency = 1.0 → intermittent pattern
        assert stats_list[0].demand_pattern == "intermittent"

    def test_demand_analyzer_high_cv_lumpy(self):
        """High CV + high intermittency = lumpy demand pattern."""
        analyzer = PartsDemandAnalyzer()
        df = pd.DataFrame(
            {
                "part_number": ["PART-LUMPY"],
                "part_family": ["AVIONICS"],
                "shop_id": ["SHOP_02"],
                "demand_mean_weekly": [1.0],
                "demand_std_weekly": [10.0],  # CV = 10, very high
            }
        )
        stats_list = analyzer.compute_statistics(df)
        assert len(stats_list) == 1
        assert stats_list[0].cv == 10.0
        assert stats_list[0].demand_pattern == "lumpy"


# ---------------------------------------------------------------------------
# Financial edge cases (FR-10)
# ---------------------------------------------------------------------------


class TestFinancialEdgeCases:
    """Edge cases for financial calculations."""

    def test_cost_per_flight_hour_zero_hours(self):
        """Cost per flight hour = 0 when flight hours = 0."""
        calc = FinancialKPICalculator()
        result = calc.calculate_cost_per_flight_hour(
            total_annual_maintenance_cost=10000000.0,
            total_annual_flight_hours=0.0,
        )
        assert result.value == 0.0
        assert result.unit == "USD/flight_hour"

    def test_cost_per_flight_hour_zero_cost(self):
        """Cost per flight hour = 0 when maintenance cost = 0."""
        calc = FinancialKPICalculator()
        result = calc.calculate_cost_per_flight_hour(
            total_annual_maintenance_cost=0.0,
            total_annual_flight_hours=200000.0,
        )
        assert result.value == 0.0

    def test_maintenance_cost_pct_revenue_zero_revenue(self):
        """Maintenance cost % revenue = 0 when revenue = 0."""
        calc = FinancialKPICalculator()
        result = calc.calculate_maintenance_cost_pct_revenue(
            total_annual_maintenance_cost=10000000.0,
            total_annual_revenue=0.0,
        )
        assert result.value == 0.0

    def test_inventory_turnover_zero_inventory_value(self):
        """Inventory turnover = 0 when average inventory value = 0."""
        calc = FinancialKPICalculator()
        result = calc.calculate_inventory_turnover(
            annual_parts_cost=5000000.0,
            average_inventory_value=0.0,
        )
        assert result.value == 0.0

    def test_stockout_rate_zero_demand(self):
        """Stockout rate = 0 when annual demand = 0."""
        calc = FinancialKPICalculator()
        result = calc.calculate_stockout_rate(
            total_expected_stockouts=100.0,
            total_annual_demand=0.0,
        )
        assert result.value == 0.0

    def test_labor_utilization_zero_capacity(self):
        """Labor utilization = 0 when capacity = 0."""
        calc = FinancialKPICalculator()
        result = calc.calculate_labor_utilization(
            total_labor_hours_required=80000.0,
            total_labor_capacity_hours=0.0,
        )
        assert result.value == 0.0

    def test_roi_zero_investment_cost(self):
        """ROI with zero investment cost → division guard."""
        calc = FinancialKPICalculator()
        # Zero investment cost with positive benefit → infinite ROI, guarded
        result = calc.calculate_roi_predictive_maintenance(
            cost_avoidance=2000000.0,
            implementation_cost=0.0,
            annual_operating_cost=100000.0,
        )
        # (2000000 - 100000) / 0 → should handle gracefully
        assert result.value == float("inf") or result.value == 0.0

    def test_roi_negative_cost_avoidance(self):
        """ROI with negative cost avoidance (predictive maintenance costs more)."""
        calc = FinancialKPICalculator()
        result = calc.calculate_roi_predictive_maintenance(
            cost_avoidance=-500000.0,
            implementation_cost=1000000.0,
            annual_operating_cost=100000.0,
        )
        # (-500000 - 100000) / 1000000 * 100 = -60%
        assert abs(result.value - (-60.0)) < 0.1

    def test_roi_positive_returns(self):
        """Standard positive ROI calculation matches SRS formula."""
        calc = FinancialKPICalculator()
        # Benefit = 2000000, Investment = 500000, Operating = 100000
        # ROI = ((2000000 - 100000) / 500000) * 100 = 380%
        result = calc.calculate_roi_predictive_maintenance(
            cost_avoidance=2000000.0,
            implementation_cost=500000.0,
            annual_operating_cost=100000.0,
        )
        assert abs(result.value - 380.0) < 0.1
        assert result.unit == "%"

    def test_scenario_zero_costs(self):
        """Scenario analysis with all-zero baseline costs."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(0, 0, 0, 0, 0)
        assert baseline["total"] == 0.0

        result = analyzer.scenario_predictive_maintenance(baseline)
        assert result.total_annual_cost == 0.0
        assert result.cost_vs_baseline == 0.0

    def test_scenario_fleet_growth_zero_rate(self):
        """Fleet growth with 0% rate returns baseline cost unchanged."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_fleet_growth(baseline, growth_rate=0.0)
        assert abs(result.total_annual_cost - baseline["total"]) < 1.0
        assert result.cost_vs_baseline == 0.0

    def test_scenario_labor_efficiency_100_percent(self):
        """Labor efficiency 100% → labor cost = 0."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_labor_efficiency(baseline, efficiency_gain=1.0)
        assert result.cost_breakdown["labor_cost"] == 0.0

    def test_calculate_all_kpis_empty_structures(self):
        """calculate_all_kpis with minimal/zero input structures."""
        calc = FinancialKPICalculator()
        labor_results = {
            "totals": {
                "total_annual_labor_cost": 0.0,
                "total_annual_labor_hours": 0.0,
                "total_overtime_cost": 0.0,
            },
            "shop_summaries": [],
        }
        parts_results = {
            "totals": {"total_annual_parts_cost": 0.0, "total_annual_demand": 0.0},
            "part_results": [],
        }
        downtime_results = {"totals": {"total_annual_downtime_cost": 0.0}}
        holding_results = {
            "totals": {"total_holding_cost": 0.0, "total_average_inventory_value": 0.0}
        }
        stockout_results = {
            "totals": {
                "total_annual_stockout_cost": 0.0,
                "total_expected_stockouts_per_year": 0.0,
            }
        }
        fleet_df = pd.DataFrame(
            [{"total_annual_flight_hours": 0.0, "total_annual_revenue_potential": 0.0}]
        )

        kpis = calc.calculate_all_kpis(
            labor_results,
            parts_results,
            downtime_results,
            holding_results,
            stockout_results,
            fleet_df,
        )
        assert "cost_per_flight_hour" in kpis
        assert "maintenance_cost_pct_revenue" in kpis
        assert "inventory_turnover" in kpis
        assert "stockout_rate" in kpis
        assert "labor_utilization" in kpis
        assert "cost_breakdown" in kpis
        assert "cost_breakdown_pct" in kpis
