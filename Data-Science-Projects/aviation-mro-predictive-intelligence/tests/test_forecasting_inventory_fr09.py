"""Tests for FR-09 Inventory Analytics: slow-moving, valuation, scenarios."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.inventory_scenarios import (
    InventoryScenarioAnalyzer,
    run_scenario_comparison,
    summarize_scenarios,
)
from src.aviation_mro.analytics.inventory.inventory_valuation import (
    InventoryValuationCalculator,
    calculate_inventory_valuation,
    summarize_valuation,
)
from src.aviation_mro.analytics.inventory.slow_moving import (
    SlowMovingAnalyzer,
    analyze_slow_moving_inventory,
    summarize_slow_moving,
)


def _make_inventory_df() -> pd.DataFrame:
    """Create a small synthetic inventory DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "part_number": "PART-01-ENG-0001",
                "part_family": "ENGINE",
                "shop_id": "SHOP_01",
                "current_stock": 100.0,
                "unit_cost": 5000.0,
                "lead_time_days": 21.0,
                "lead_time_std_days": 5.0,
                "demand_mean_weekly": 8.0,
                "demand_std_weekly": 3.2,
                "service_level_target": 0.95,
                "criticality": "A",
                "source": "synthetic_inventory_v1",
            },
            {
                "part_number": "PART-01-AV-0002",
                "part_family": "AVIONICS",
                "shop_id": "SHOP_02",
                "current_stock": 50.0,
                "unit_cost": 2000.0,
                "lead_time_days": 14.0,
                "lead_time_std_days": 3.0,
                "demand_mean_weekly": 4.0,
                "demand_std_weekly": 2.4,
                "service_level_target": 0.95,
                "criticality": "B",
                "source": "synthetic_inventory_v1",
            },
            {
                "part_number": "PART-01-AF-0003",
                "part_family": "AIRFRAME",
                "shop_id": "SHOP_03",
                "current_stock": 200.0,
                "unit_cost": 500.0,
                "lead_time_days": 10.0,
                "lead_time_std_days": 2.0,
                "demand_mean_weekly": 2.0,
                "demand_std_weekly": 1.6,
                "service_level_target": 0.90,
                "criticality": "C",
                "source": "synthetic_inventory_v1",
            },
            {
                "part_number": "PART-01-ENG-0004",
                "part_family": "ENGINE",
                "shop_id": "SHOP_04",
                "current_stock": 0.0,
                "unit_cost": 5000.0,
                "lead_time_days": 21.0,
                "lead_time_std_days": 5.0,
                "demand_mean_weekly": 0.0,
                "demand_std_weekly": 0.0,
                "service_level_target": 0.99,
                "criticality": "A",
                "source": "synthetic_inventory_v1",
            },
        ]
    )


class TestSlowMovingAnalyzer:
    """Test slow-moving inventory identification."""

    def test_classify_non_moving(self):
        analyzer = SlowMovingAnalyzer()
        assert analyzer.classify(0.0, "A") == "non_moving"

    def test_classify_slow_moving(self):
        analyzer = SlowMovingAnalyzer(slow_moving_threshold=1.0)
        assert analyzer.classify(0.5, "B") == "slow_moving"

    def test_classify_normal(self):
        analyzer = SlowMovingAnalyzer()
        assert analyzer.classify(5.0, "C") == "normal"

    def test_classify_boundary(self):
        analyzer = SlowMovingAnalyzer(slow_moving_threshold=1.0)
        # Exactly at threshold = normal (not < threshold)
        assert analyzer.classify(1.0, "A") == "normal"

    def test_weeks_of_supply_zero_demand(self):
        analyzer = SlowMovingAnalyzer()
        assert analyzer.weeks_of_supply(100.0, 0.0) == float("inf")

    def test_weeks_of_supply_normal(self):
        analyzer = SlowMovingAnalyzer()
        assert analyzer.weeks_of_supply(100.0, 10.0) == 10.0

    def test_analyze_dataframe(self):
        df = _make_inventory_df()
        results = analyze_slow_moving_inventory(df)
        assert len(results) == 4
        # PART-01-ENG-0004 has zero demand = non_moving
        non_moving = [r for r in results if r.classification == "non_moving"]
        assert len(non_moving) == 1
        assert non_moving[0].part_number == "PART-01-ENG-0004"

    def test_analyze_critical_slow_moving(self):
        df = _make_inventory_df()
        results = analyze_slow_moving_inventory(df)
        critical_slow = [r for r in results if r.is_critical_slow_moving]
        # Non-moving critical part counts as critical slow-moving
        assert any(r.part_number == "PART-01-ENG-0004" for r in critical_slow)

    def test_summarize(self):
        df = _make_inventory_df()
        results = analyze_slow_moving_inventory(df)
        summary = summarize_slow_moving(results)
        assert summary["total_parts"] == 4
        assert summary["non_moving_count"] == 1
        assert summary["non_moving_inventory_value"] == 0.0  # 0 stock × cost

    def test_inventory_value_calculation(self):
        df = _make_inventory_df()
        results = analyze_slow_moving_inventory(df)
        part1 = results[0]  # PART-01-ENG-0001: 100 stock × 5000 cost
        assert part1.inventory_value == 100.0 * 5000.0


class TestInventoryValuation:
    """Test inventory valuation and turnover."""

    def test_valuation_for_part(self):
        df = _make_inventory_df()
        row = df.iloc[0]
        calculator = InventoryValuationCalculator()
        result = calculator.calculate_for_part(row, reorder_point=60.0, safety_stock=20.0)
        assert result.part_number == "PART-01-ENG-0001"
        assert result.inventory_value == 100.0 * 5000.0  # 500,000
        assert result.rop_value == 60.0 * 5000.0  # 300,000
        assert result.safety_stock_value == 20.0 * 5000.0  # 100,000
        # Weeks of supply: 100 / 8 = 12.5
        assert result.weeks_of_supply == 12.5
        # Annual demand cost: 8 × 52 × 5000 = 2,080,000
        assert result.annual_demand_cost == 8.0 * 52.0 * 5000.0

    def test_valuation_zero_demand(self):
        df = _make_inventory_df()
        row = df.iloc[3]  # Zero demand part
        calculator = InventoryValuationCalculator()
        result = calculator.calculate_for_part(row, reorder_point=0.0, safety_stock=0.0)
        assert result.weeks_of_supply == float("inf")
        assert result.inventory_turnover == 0.0
        assert result.days_of_inventory == float("inf")

    def test_valuation_dataframe(self):
        df = _make_inventory_df()
        rop_lookup = {r["part_number"]: 50.0 for _, r in df.iterrows()}
        ss_lookup = {r["part_number"]: 10.0 for _, r in df.iterrows()}
        results = calculate_inventory_valuation(df, rop_lookup, ss_lookup)
        assert len(results) == 4
        for r in results:
            assert r.inventory_value >= 0
            assert r.rop_value >= 0
            assert r.safety_stock_value >= 0

    def test_summarize_valuation(self):
        df = _make_inventory_df()
        rop_lookup = {r["part_number"]: 50.0 for _, r in df.iterrows()}
        ss_lookup = {r["part_number"]: 10.0 for _, r in df.iterrows()}
        results = calculate_inventory_valuation(df, rop_lookup, ss_lookup)
        summary = summarize_valuation(results)
        assert summary["total_parts"] == 4
        assert summary["total_inventory_value"] > 0
        assert summary["total_annual_demand_cost"] > 0
        assert "by_family" in summary
        assert "by_shop" in summary
        assert "by_criticality" in summary

    def test_turnover_calculation(self):
        df = _make_inventory_df()
        rop_lookup = {r["part_number"]: 50.0 for _, r in df.iterrows()}
        ss_lookup = {r["part_number"]: 10.0 for _, r in df.iterrows()}
        results = calculate_inventory_valuation(df, rop_lookup, ss_lookup)
        summary = summarize_valuation(results)
        # Turnover = total annual demand cost / total inventory value
        expected = summary["total_annual_demand_cost"] / summary["total_inventory_value"]
        assert abs(summary["overall_turnover"] - expected) < 1e-6


class TestInventoryScenarios:
    """Test inventory scenario comparison."""

    def test_scenario_calculator(self):
        df = _make_inventory_df()
        analyzer = InventoryScenarioAnalyzer()
        result = analyzer.calculate_scenario(df, 0.95, "Test Scenario")
        assert result.scenario_name == "Test Scenario"
        assert result.service_level == 0.95
        assert result.z_score > 0
        assert result.total_safety_stock >= 0
        assert result.annual_holding_cost >= 0
        assert result.annual_stockout_cost >= 0
        assert result.total_annual_cost == result.annual_holding_cost + result.annual_stockout_cost
        assert result.is_estimate is True

    def test_compare_scenarios(self):
        df = _make_inventory_df()
        analyzer = InventoryScenarioAnalyzer()
        results = analyzer.compare_scenarios(df)
        assert len(results) == 3
        names = [r.scenario_name for r in results]
        assert "Baseline (current service levels)" in names
        assert "High Service (99%)" in names
        assert "Low Service (90%)" in names

    def test_high_service_higher_holding_cost(self):
        df = _make_inventory_df()
        analyzer = InventoryScenarioAnalyzer()
        results = analyzer.compare_scenarios(df)
        baseline = next(r for r in results if "Baseline" in r.scenario_name)
        high = next(r for r in results if "High Service" in r.scenario_name)
        # Higher service level = higher safety stock = higher holding cost
        assert high.annual_holding_cost > baseline.annual_holding_cost

    def test_high_service_lower_stockout(self):
        df = _make_inventory_df()
        analyzer = InventoryScenarioAnalyzer()
        results = analyzer.compare_scenarios(df)
        baseline = next(r for r in results if "Baseline" in r.scenario_name)
        high = next(r for r in results if "High Service" in r.scenario_name)
        # Higher service level = lower stockout probability
        assert high.avg_stockout_probability <= baseline.avg_stockout_probability

    def test_run_scenario_comparison(self):
        df = _make_inventory_df()
        results = run_scenario_comparison(df)
        assert len(results) == 3

    def test_summarize_scenarios(self):
        df = _make_inventory_df()
        results = run_scenario_comparison(df)
        summary = summarize_scenarios(results)
        assert summary["n_scenarios"] == 3
        assert summary["baseline"] is not None
        assert summary["best_total_cost_scenario"] is not None
        assert summary["lowest_stockout_scenario"] is not None
        assert summary["lowest_holding_cost_scenario"] is not None
        assert len(summary["scenarios"]) == 3
        # All scenarios should be marked as estimates
        for s in summary["scenarios"]:
            assert s["is_estimate"] is True

    def test_z_score_bounds(self):
        analyzer = InventoryScenarioAnalyzer()
        with pytest.raises(ValueError):
            analyzer.z_score(0.0)
        with pytest.raises(ValueError):
            analyzer.z_score(1.0)
