"""Tests for parts costs module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.parts_costs import (
    PartsCostCalculator,
    calculate_parts_costs,
)


class TestPartsCostCalculator:
    """Test PartsCostCalculator class."""

    def test_calculate_eoq(self):
        """Test EOQ calculation."""
        calc = PartsCostCalculator(holding_cost_rate=0.20, order_cost=75.0)
        # D=520, S=75, H=0.20*1000=200
        # EOQ = sqrt(2*520*75/200) = sqrt(390) = 19.75
        eoq = calc.calculate_eoq(annual_demand=520.0, unit_cost=1000.0)
        assert abs(eoq - 19.75) < 0.1

    def test_calculate_eoq_zero_demand(self):
        """Test EOQ with zero demand."""
        calc = PartsCostCalculator()
        eoq = calc.calculate_eoq(annual_demand=0.0, unit_cost=1000.0)
        assert eoq == 1.0

    def test_calculate_eoq_zero_cost(self):
        """Test EOQ with zero unit cost."""
        calc = PartsCostCalculator()
        eoq = calc.calculate_eoq(annual_demand=520.0, unit_cost=0.0)
        assert eoq == 1.0

    def test_calculate_part_costs(self):
        """Test part cost calculation."""
        calc = PartsCostCalculator(holding_cost_rate=0.20, order_cost=75.0)
        result = calc.calculate_part_costs(
            part_number="PART-0001",
            part_family="ENGINE",
            shop_id="SHOP_01",
            unit_cost=1000.0,
            annual_demand=520.0,
        )
        assert result.part_number == "PART-0001"
        assert result.unit_cost == 1000.0
        assert result.annual_demand == 520.0
        assert result.annual_parts_cost == 520000.0
        assert result.eoq > 0
        assert result.orders_per_year > 0
        assert result.annual_ordering_cost > 0
        assert result.annual_holding_cost > 0
        assert (
            result.total_inventory_cost == result.annual_ordering_cost + result.annual_holding_cost
        )

    def test_calculate_all_parts_costs(self):
        """Test all parts cost calculation."""
        calc = PartsCostCalculator(holding_cost_rate=0.20, order_cost=75.0)
        df = pd.DataFrame(
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
        results = calc.calculate_all_parts_costs(df)
        assert "part_results" in results
        assert "family_summaries" in results
        assert "shop_summaries" in results
        assert "totals" in results
        assert len(results["part_results"]) == 2
        assert len(results["family_summaries"]) == 2
        assert len(results["shop_summaries"]) == 1
        assert results["totals"]["total_parts"] == 2


class TestCalculatePartsCosts:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_parts_costs function."""
        df = pd.DataFrame(
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
        results = calculate_parts_costs(df, holding_cost_rate=0.20, order_cost=75.0)
        assert len(results["part_results"]) == 1
        assert results["totals"]["total_parts"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
