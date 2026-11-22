"""Tests for maintenance cost per work order and per engine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.cost_per_unit import (
    ENGINES_PER_AIRCRAFT,
    CostPerUnitCalculator,
    calculate_cost_per_unit,
)


class TestCostPerUnitCalculator:
    """Test CostPerUnitCalculator class."""

    def test_calculate_cost_per_work_order(self):
        """Test cost per work order calculation."""
        calc = CostPerUnitCalculator()
        result = calc.calculate_cost_per_work_order(
            total_annual_maintenance_cost=1000000.0,
            total_annual_job_count=1000,
        )
        assert result.metric_name == "Cost per Work Order"
        assert result.value == 1000.0
        assert result.unit == "USD/work_order"
        assert "$1,000,000" in result.formula

    def test_calculate_cost_per_work_order_zero_jobs(self):
        """Test cost per work order with zero job count."""
        calc = CostPerUnitCalculator()
        result = calc.calculate_cost_per_work_order(
            total_annual_maintenance_cost=1000000.0,
            total_annual_job_count=0,
        )
        assert result.value == 0.0

    def test_calculate_total_engines(self):
        """Test total engines calculation."""
        calc = CostPerUnitCalculator()
        fleet_df = pd.DataFrame(
            [
                {"fleet_type": "NARROW_BODY", "aircraft_count": 45},
                {"fleet_type": "WIDE_BODY", "aircraft_count": 18},
                {"fleet_type": "REGIONAL", "aircraft_count": 22},
            ]
        )
        total = calc.calculate_total_engines(fleet_df)
        assert total == 45 * 2 + 18 * 2 + 22 * 2  # 170

    def test_calculate_cost_per_engine(self):
        """Test cost per engine calculation."""
        calc = CostPerUnitCalculator()
        fleet_df = pd.DataFrame(
            [
                {"fleet_type": "NARROW_BODY", "aircraft_count": 45},
                {"fleet_type": "WIDE_BODY", "aircraft_count": 18},
                {"fleet_type": "REGIONAL", "aircraft_count": 22},
            ]
        )
        result = calc.calculate_cost_per_engine(
            total_annual_maintenance_cost=1700000.0,
            fleet_df=fleet_df,
        )
        assert result.metric_name == "Cost per Engine"
        assert result.unit == "USD/engine/year"
        # 1700000 / 170 = 10000
        assert result.value == 10000.0

    def test_calculate_cost_per_engine_zero_engines(self):
        """Test cost per engine with zero engines."""
        calc = CostPerUnitCalculator()
        fleet_df = pd.DataFrame(
            [
                {"fleet_type": "NARROW_BODY", "aircraft_count": 0},
            ]
        )
        result = calc.calculate_cost_per_engine(
            total_annual_maintenance_cost=1000000.0,
            fleet_df=fleet_df,
        )
        assert result.value == 0.0

    def test_calculate_all(self):
        """Test calculate_all returns all expected keys."""
        calc = CostPerUnitCalculator()
        fleet_df = pd.DataFrame(
            [
                {"fleet_type": "NARROW_BODY", "aircraft_count": 10},
            ]
        )
        result = calc.calculate_all(
            total_annual_maintenance_cost=500000.0,
            total_annual_job_count=500,
            fleet_df=fleet_df,
        )
        assert "cost_per_work_order" in result
        assert "cost_per_engine" in result
        assert "total_annual_maintenance_cost" in result
        assert "total_annual_job_count" in result
        assert "total_engines" in result
        assert result["total_engines"] == 20  # 10 aircraft * 2 engines

    def test_engines_per_aircraft_constant(self):
        """Test that engines per aircraft defaults are as documented."""
        assert ENGINES_PER_AIRCRAFT["NARROW_BODY"] == 2
        assert ENGINES_PER_AIRCRAFT["WIDE_BODY"] == 2
        assert ENGINES_PER_AIRCRAFT["REGIONAL"] == 2


class TestCalculateCostPerUnit:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_cost_per_unit function."""
        fleet_df = pd.DataFrame(
            [
                {"fleet_type": "NARROW_BODY", "aircraft_count": 10},
            ]
        )
        result = calculate_cost_per_unit(
            total_annual_maintenance_cost=200000.0,
            total_annual_job_count=200,
            fleet_df=fleet_df,
        )
        assert result["cost_per_work_order"].value == 1000.0
        assert result["cost_per_engine"].value == 10000.0  # 200000 / 20 engines

    def test_custom_engines_per_aircraft(self):
        """Test with custom engines per aircraft configuration."""
        fleet_df = pd.DataFrame(
            [
                {"fleet_type": "NARROW_BODY", "aircraft_count": 10},
            ]
        )
        result = calculate_cost_per_unit(
            total_annual_maintenance_cost=200000.0,
            total_annual_job_count=200,
            fleet_df=fleet_df,
            engines_per_aircraft={"NARROW_BODY": 4, "WIDE_BODY": 4, "REGIONAL": 2},
        )
        assert result["total_engines"] == 40  # 10 aircraft * 4 engines
        assert result["cost_per_engine"].value == 5000.0  # 200000 / 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
