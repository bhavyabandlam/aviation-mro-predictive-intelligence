"""Tests for scenario analysis module."""

from __future__ import annotations

import pytest

from src.aviation_mro.analytics.financial.scenarios import (
    ScenarioAnalyzer,
    run_scenario_analysis,
)


class TestScenarioAnalyzer:
    """Test ScenarioAnalyzer class."""

    def test_calculate_baseline_costs(self):
        """Test baseline cost calculation."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(
            labor_cost=5000000.0,
            parts_cost=3000000.0,
            downtime_cost=2000000.0,
            holding_cost=500000.0,
            stockout_cost=200000.0,
        )
        assert baseline["labor_cost"] == 5000000.0
        assert baseline["parts_cost"] == 3000000.0
        assert baseline["downtime_cost"] == 2000000.0
        assert baseline["holding_cost"] == 500000.0
        assert baseline["stockout_cost"] == 200000.0
        assert baseline["total"] == 10700000.0

    def test_scenario_predictive_maintenance(self):
        """Test predictive maintenance scenario."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_predictive_maintenance(baseline, downtime_reduction=0.20)
        assert result.scenario_name == "Predictive Maintenance"
        assert result.total_annual_cost < baseline["total"]
        # Downtime reduced 20%, parts +2%, stockout -15%
        expected_downtime = 2000000 * 0.80
        expected_parts = 3000000 * 1.02
        expected_stockout = 200000 * 0.85
        expected_total = 5000000 + expected_parts + expected_downtime + 500000 + expected_stockout
        assert abs(result.total_annual_cost - expected_total) < 1.0

    def test_scenario_inventory_optimization(self):
        """Test inventory optimization scenario."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_inventory_optimization(
            baseline, holding_cost_reduction=0.15, stockout_cost_reduction=0.25
        )
        assert result.scenario_name == "Inventory Optimization"
        assert result.total_annual_cost < baseline["total"]

    def test_scenario_fleet_growth(self):
        """Test fleet growth scenario."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_fleet_growth(baseline, growth_rate=0.15)
        assert result.scenario_name == "Fleet Growth"
        assert result.total_annual_cost > baseline["total"]
        expected = baseline["total"] * 1.15
        assert abs(result.total_annual_cost - expected) < 1.0

    def test_scenario_labor_efficiency(self):
        """Test labor efficiency scenario."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_labor_efficiency(baseline, efficiency_gain=0.10)
        assert result.scenario_name == "Labor Efficiency"
        assert result.total_annual_cost < baseline["total"]

    def test_scenario_supplier_cost_change(self):
        """Test supplier cost change scenario."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        # Cost increase
        result_inc = analyzer.scenario_supplier_cost_change(baseline, parts_cost_change=0.10)
        assert result_inc.total_annual_cost > baseline["total"]
        # Cost decrease
        result_dec = analyzer.scenario_supplier_cost_change(baseline, parts_cost_change=-0.05)
        assert result_dec.total_annual_cost < baseline["total"]

    def test_scenario_combined_optimization(self):
        """Test combined optimization scenario."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        result = analyzer.scenario_combined_optimization(baseline)
        assert result.scenario_name == "Combined Optimization"
        assert result.total_annual_cost < baseline["total"]

    def test_run_all_scenarios(self):
        """Test running all scenarios."""
        analyzer = ScenarioAnalyzer()
        baseline = analyzer.calculate_baseline_costs(5000000, 3000000, 2000000, 500000, 200000)
        results = analyzer.run_all_scenarios(baseline)
        assert "scenarios" in results
        assert "baseline_total" in results
        assert "best_scenario" in results
        assert "best_savings" in results
        assert "worst_scenario" in results
        assert "worst_increase" in results
        assert len(results["scenarios"]) >= 7  # baseline + 6 scenarios


class TestRunScenarioAnalysis:
    """Test convenience function."""

    def test_function(self):
        """Test run_scenario_analysis function."""
        results = run_scenario_analysis(5000000, 3000000, 2000000, 500000, 200000)
        assert "scenarios" in results
        assert "baseline_total" in results
        assert results["baseline_total"] == 10700000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
