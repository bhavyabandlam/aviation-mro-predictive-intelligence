"""Tests for financial sensitivity analysis."""

from __future__ import annotations

import pytest

from src.aviation_mro.analytics.financial.sensitivity import (
    SENSITIVITY_PARAMETERS,
    VARIATION_LEVELS,
    SensitivityAnalyzer,
    run_sensitivity_analysis,
)


class TestSensitivityAnalyzer:
    """Test SensitivityAnalyzer class."""

    def test_variation_levels(self):
        """Test variation levels are as documented."""
        assert VARIATION_LEVELS == [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]

    def test_sensitivity_parameters(self):
        """Test sensitivity parameters are defined."""
        assert "downtime_reduction" in SENSITIVITY_PARAMETERS
        assert "holding_cost_rate" in SENSITIVITY_PARAMETERS
        assert "stockout_cost_multiplier" in SENSITIVITY_PARAMETERS
        assert "base_labor_rate" in SENSITIVITY_PARAMETERS
        assert "parts_unit_cost" in SENSITIVITY_PARAMETERS

    def test_analyze_parameter_downtime_reduction(self):
        """Test sensitivity analysis for downtime reduction parameter."""
        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze_parameter(
            parameter="downtime_reduction",
            baseline_value=0.20,
            baseline_total_cost=1000000.0,
            cost_impact_factor=0.25,  # 25% of cost is downtime
        )
        assert len(results) == 6  # 6 variation levels
        for r in results:
            assert r.parameter == "downtime_reduction"
            assert r.parameter_label == SENSITIVITY_PARAMETERS["downtime_reduction"]["label"]
            assert r.baseline_value == 0.20
            expected_varied = round(0.20 * (1 + r.variation_pct / 100), 4)
            assert abs(r.varied_value - expected_varied) < 0.0001
            assert "SYNTHETIC" in r.assumption or "estimate" in r.assumption.lower()

    def test_analyze_parameter_positive_variation_increases_cost(self):
        """Test that positive variation increases cost when impact factor > 0."""
        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze_parameter(
            parameter="holding_cost_rate",
            baseline_value=0.20,
            baseline_total_cost=1000000.0,
            cost_impact_factor=0.15,
        )
        # Positive variations should increase cost
        for r in results:
            if r.variation_pct > 0:
                assert r.cost_change_usd > 0
                assert r.varied_total_cost > r.baseline_total_cost
            if r.variation_pct < 0:
                assert r.cost_change_usd < 0
                assert r.varied_total_cost < r.baseline_total_cost

    def test_analyze_parameter_zero_impact(self):
        """Test parameter with zero impact factor produces no cost change."""
        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze_parameter(
            parameter="stockout_cost_multiplier",
            baseline_value=10.0,
            baseline_total_cost=1000000.0,
            cost_impact_factor=0.0,
        )
        for r in results:
            assert r.cost_change_usd == 0.0
            assert r.varied_total_cost == r.baseline_total_cost
            assert r.cost_change_pct == 0.0

    def test_analyze_parameter_zero_baseline_cost(self):
        """Test parameter with zero baseline cost."""
        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze_parameter(
            parameter="downtime_reduction",
            baseline_value=0.20,
            baseline_total_cost=0.0,
            cost_impact_factor=0.25,
        )
        for r in results:
            assert r.cost_change_pct == 0.0
            assert r.varied_total_cost == 0.0

    def test_analyze_all(self):
        """Test full sensitivity analysis."""
        analyzer = SensitivityAnalyzer()
        baseline_results = {
            "financial_kpis": {
                "cost_breakdown": {
                    "labor_cost": 5000000.0,
                    "parts_cost": 3000000.0,
                    "downtime_cost": 2000000.0,
                    "holding_cost": 500000.0,
                    "stockout_cost": 200000.0,
                    "total_maintenance_cost": 10700000.0,
                }
            }
        }
        result = analyzer.analyze_all(baseline_results)
        assert "baseline_total_cost" in result
        assert "parameter_impacts" in result
        assert "results" in result
        assert "most_sensitive_parameter" in result
        assert "max_impact_usd" in result
        assert "assumptions" in result
        assert result["baseline_total_cost"] == 10700000.0
        assert len(result["results"]) == 5  # 5 parameters
        for param_results in result["results"].values():
            assert len(param_results) == 6  # 6 variation levels each

    def test_most_sensitive_parameter(self):
        """Test most sensitive parameter identification."""
        analyzer = SensitivityAnalyzer()
        baseline_results = {
            "financial_kpis": {
                "cost_breakdown": {
                    "labor_cost": 8000000.0,  # Largest share
                    "parts_cost": 1000000.0,
                    "downtime_cost": 1000000.0,
                    "holding_cost": 500000.0,
                    "stockout_cost": 200000.0,
                    "total_maintenance_cost": 10700000.0,
                }
            }
        }
        result = analyzer.analyze_all(baseline_results)
        # Labor has the largest cost share (8M/10.7M ~ 75%), so it should be most sensitive
        assert result["most_sensitive_parameter"] == "base_labor_rate"

    def test_assumptions_present(self):
        """Test that assumptions are documented."""
        analyzer = SensitivityAnalyzer()
        baseline_results = {
            "financial_kpis": {
                "cost_breakdown": {
                    "labor_cost": 5000000.0,
                    "parts_cost": 3000000.0,
                    "downtime_cost": 2000000.0,
                    "holding_cost": 500000.0,
                    "stockout_cost": 200000.0,
                    "total_maintenance_cost": 10700000.0,
                }
            }
        }
        result = analyzer.analyze_all(baseline_results)
        assert isinstance(result["assumptions"], list)
        assert len(result["assumptions"]) > 0
        assert any("ESTIMATES" in a for a in result["assumptions"])
        assert any("synthetic" in a.lower() for a in result["assumptions"])


class TestRunSensitivityAnalysis:
    """Test convenience function."""

    def test_function(self):
        """Test run_sensitivity_analysis function."""
        baseline_results = {
            "financial_kpis": {
                "cost_breakdown": {
                    "labor_cost": 5000000.0,
                    "parts_cost": 3000000.0,
                    "downtime_cost": 2000000.0,
                    "holding_cost": 500000.0,
                    "stockout_cost": 200000.0,
                    "total_maintenance_cost": 10700000.0,
                }
            }
        }
        result = run_sensitivity_analysis(baseline_results)
        assert "baseline_total_cost" in result
        assert "results" in result
        assert result["baseline_total_cost"] == 10700000.0

    def test_empty_cost_breakdown(self):
        """Test with empty cost breakdown."""
        baseline_results = {"financial_kpis": {"cost_breakdown": {}}}
        result = run_sensitivity_analysis(baseline_results)
        assert result["baseline_total_cost"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
