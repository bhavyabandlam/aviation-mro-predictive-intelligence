"""Tests for stockout costs module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.stockout_costs import (
    StockoutCostCalculator,
    calculate_stockout_costs,
)


class TestStockoutCostCalculator:
    """Test StockoutCostCalculator class."""

    def test_calculate_stockout_cost(self):
        """Test stockout cost calculation."""
        calc = StockoutCostCalculator(stockout_cost_multiplier=10.0)
        result = calc.calculate_stockout_cost(
            part_number="PART-0001",
            part_family="ENGINE",
            shop_id="SHOP_01",
            unit_cost=1000.0,
            stockout_probability=0.05,
            cycles_per_year=13.0,
        )
        assert result.part_number == "PART-0001"
        assert result.stockout_probability == 0.05
        assert result.cycles_per_year == 13.0
        assert result.expected_stockouts_per_year == 0.65
        assert result.stockout_cost_multiplier == 10.0
        assert result.stockout_cost_per_event == 10000.0
        assert result.annual_stockout_cost == 6500.0

    def test_calculate_all_stockout_costs(self):
        """Test all stockout costs calculation."""
        calc = StockoutCostCalculator(stockout_cost_multiplier=10.0)
        inventory_df = pd.DataFrame(
            [
                {
                    "part_number": "PART-0001",
                    "part_family": "ENGINE",
                    "shop_id": "SHOP_01",
                    "unit_cost": 1000.0,
                },
                {
                    "part_number": "PART-0002",
                    "part_family": "AVIONICS",
                    "shop_id": "SHOP_01",
                    "unit_cost": 500.0,
                },
            ]
        )
        stockout_risk_df = pd.DataFrame(
            [
                {
                    "part_number": "PART-0001",
                    "stockout_probability": 0.05,
                    "cycles_per_year": 13.0,
                    "risk_category": "Medium",
                },
                {
                    "part_number": "PART-0002",
                    "stockout_probability": 0.10,
                    "cycles_per_year": 13.0,
                    "risk_category": "High",
                },
            ]
        )
        results = calc.calculate_all_stockout_costs(inventory_df, stockout_risk_df)
        assert "part_results" in results
        assert "family_summaries" in results
        assert "shop_summaries" in results
        assert "totals" in results
        assert len(results["part_results"]) == 2
        assert results["totals"]["total_parts"] == 2


class TestCalculateStockoutCosts:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_stockout_costs function."""
        inventory_df = pd.DataFrame(
            [
                {
                    "part_number": "PART-0001",
                    "part_family": "ENGINE",
                    "shop_id": "SHOP_01",
                    "unit_cost": 1000.0,
                },
            ]
        )
        stockout_risk_df = pd.DataFrame(
            [
                {
                    "part_number": "PART-0001",
                    "stockout_probability": 0.05,
                    "cycles_per_year": 13.0,
                    "risk_category": "Medium",
                },
            ]
        )
        results = calculate_stockout_costs(inventory_df, stockout_risk_df)
        assert len(results["part_results"]) == 1
        assert results["totals"]["total_parts"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
