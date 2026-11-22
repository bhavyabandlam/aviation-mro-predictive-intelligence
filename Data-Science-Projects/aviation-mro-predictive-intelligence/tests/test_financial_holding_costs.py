"""Tests for holding costs module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.holding_costs import (
    HoldingCostCalculator,
    calculate_holding_costs,
)


class TestHoldingCostCalculator:
    """Test HoldingCostCalculator class."""

    def test_calculate_eoq(self):
        """Test EOQ calculation."""
        calc = HoldingCostCalculator(holding_cost_rate=0.20, order_cost=75.0)
        eoq = calc.calculate_eoq(annual_demand=520.0, unit_cost=1000.0)
        assert abs(eoq - 19.75) < 0.1

    def test_calculate_holding_cost(self):
        """Test holding cost calculation."""
        calc = HoldingCostCalculator(holding_cost_rate=0.20, order_cost=75.0)
        result = calc.calculate_holding_cost(
            part_number="PART-0001",
            part_family="ENGINE",
            shop_id="SHOP_01",
            unit_cost=1000.0,
            annual_demand=520.0,
            safety_stock=10.0,
        )
        assert result.part_number == "PART-0001"
        assert result.safety_stock == 10.0
        assert result.eoq > 0
        assert result.cycle_stock == result.eoq / 2
        assert result.average_inventory == result.cycle_stock + result.safety_stock
        assert result.safety_stock_value == 10.0 * 1000.0
        assert result.total_holding_cost == result.average_inventory * 1000.0 * 0.20

    def test_calculate_all_holding_costs(self):
        """Test all holding costs calculation."""
        calc = HoldingCostCalculator(holding_cost_rate=0.20, order_cost=75.0)
        inventory_df = pd.DataFrame(
            [
                {
                    "part_number": "PART-0001",
                    "part_family": "ENGINE",
                    "shop_id": "SHOP_01",
                    "unit_cost": 1000.0,
                    "demand_mean_weekly": 10.0,
                },
                {
                    "part_number": "PART-0002",
                    "part_family": "AVIONICS",
                    "shop_id": "SHOP_01",
                    "unit_cost": 500.0,
                    "demand_mean_weekly": 5.0,
                },
            ]
        )
        safety_stock_df = pd.DataFrame(
            [
                {"part_number": "PART-0001", "safety_stock": 10.0},
                {"part_number": "PART-0002", "safety_stock": 5.0},
            ]
        )
        results = calc.calculate_all_holding_costs(inventory_df, safety_stock_df)
        assert "part_results" in results
        assert "family_summaries" in results
        assert "shop_summaries" in results
        assert "totals" in results
        assert len(results["part_results"]) == 2
        assert results["totals"]["total_parts"] == 2


class TestCalculateHoldingCosts:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_holding_costs function."""
        inventory_df = pd.DataFrame(
            [
                {
                    "part_number": "PART-0001",
                    "part_family": "ENGINE",
                    "shop_id": "SHOP_01",
                    "unit_cost": 1000.0,
                    "demand_mean_weekly": 10.0,
                },
            ]
        )
        safety_stock_df = pd.DataFrame(
            [
                {"part_number": "PART-0001", "safety_stock": 10.0},
            ]
        )
        results = calculate_holding_costs(inventory_df, safety_stock_df)
        assert len(results["part_results"]) == 1
        assert results["totals"]["total_parts"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
