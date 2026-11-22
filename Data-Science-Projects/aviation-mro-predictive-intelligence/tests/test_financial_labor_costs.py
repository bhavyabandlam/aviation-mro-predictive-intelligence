"""Tests for labor costs module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.labor_costs import (
    LaborCostCalculator,
    LaborCostResult,
    calculate_labor_costs,
)


class TestLaborCostCalculator:
    """Test LaborCostCalculator class."""

    def test_calculate_labor_cost_per_job(self):
        """Test labor cost per job calculation."""
        calc = LaborCostCalculator()
        cost = calc.calculate_labor_cost_per_job(labor_hours=10.0, labor_rate=150.0)
        assert cost == 1500.0

    def test_calculate_shop_family_labor_cost(self):
        """Test shop-family labor cost calculation."""
        calc = LaborCostCalculator()
        result = calc.calculate_shop_family_labor_cost(
            shop_id="SHOP_01",
            part_family="ENGINE",
            labor_rate=150.0,
            mean_labor_hours_per_job=48.0,
            annual_job_count=520,
        )
        assert result.shop_id == "SHOP_01"
        assert result.part_family == "ENGINE"
        assert result.labor_rate == 150.0
        assert result.mean_labor_hours_per_job == 48.0
        assert result.labor_cost_per_job == 7200.0
        assert result.annual_job_count == 520
        assert result.annual_labor_hours == 48.0 * 520
        assert result.annual_labor_cost == 7200.0 * 520

    def test_calculate_shop_labor_summary(self):
        """Test shop labor summary calculation."""
        calc = LaborCostCalculator()
        # Create mock family results
        family_results = [
            LaborCostResult(
                shop_id="SHOP_01",
                part_family="ENGINE",
                labor_rate=150.0,
                mean_labor_hours_per_job=48.0,
                labor_cost_per_job=7200.0,
                annual_job_count=520,
                annual_labor_hours=24960.0,
                annual_labor_cost=3744000.0,
                assumptions=[],
            ),
            LaborCostResult(
                shop_id="SHOP_01",
                part_family="AVIONICS",
                labor_rate=150.0,
                mean_labor_hours_per_job=12.0,
                labor_cost_per_job=1800.0,
                annual_job_count=520,
                annual_labor_hours=6240.0,
                annual_labor_cost=936000.0,
                assumptions=[],
            ),
        ]
        shop_data = pd.Series(
            {
                "technician_count": 45,
                "base_labor_rate": 150.0,
                "overtime_labor_rate": 225.0,
                "annual_regular_capacity_hours": 93600,
                "annual_overtime_capacity_hours": 18000,
            }
        )
        summary = calc.calculate_shop_labor_summary("SHOP_01", shop_data, family_results)
        assert summary.shop_id == "SHOP_01"
        assert summary.total_annual_labor_cost == 4680000.0
        assert summary.total_annual_labor_hours == 31200.0
        assert summary.utilization_rate == 31200.0 / (93600 + 18000)
        # Overtime = 31200 - 93600 = negative, so 0
        assert summary.overtime_hours == 0
        assert summary.overtime_cost == 0

    def test_calculate_shop_labor_summary_with_overtime(self):
        """Test shop labor summary with overtime."""
        calc = LaborCostCalculator()
        family_results = [
            LaborCostResult(
                shop_id="SHOP_01",
                part_family="ENGINE",
                labor_rate=150.0,
                mean_labor_hours_per_job=48.0,
                labor_cost_per_job=7200.0,
                annual_job_count=520,
                annual_labor_hours=24960.0,
                annual_labor_cost=3744000.0,
                assumptions=[],
            ),
            LaborCostResult(
                shop_id="SHOP_01",
                part_family="AVIONICS",
                labor_rate=150.0,
                mean_labor_hours_per_job=12.0,
                labor_cost_per_job=1800.0,
                annual_job_count=520,
                annual_labor_hours=6240.0,
                annual_labor_cost=936000.0,
                assumptions=[],
            ),
            LaborCostResult(
                shop_id="SHOP_01",
                part_family="AIRFRAME",
                labor_rate=150.0,
                mean_labor_hours_per_job=8.0,
                labor_cost_per_job=1200.0,
                annual_job_count=520,
                annual_labor_hours=4160.0,
                annual_labor_cost=624000.0,
                assumptions=[],
            ),
        ]
        shop_data = pd.Series(
            {
                "technician_count": 10,  # Small team
                "base_labor_rate": 150.0,
                "overtime_labor_rate": 225.0,
                "annual_regular_capacity_hours": 20800,
                "annual_overtime_capacity_hours": 4000,
            }
        )
        summary = calc.calculate_shop_labor_summary("SHOP_01", shop_data, family_results)
        total_hours = 24960 + 6240 + 4160  # 35360
        assert summary.total_annual_labor_hours == total_hours
        # Overtime = 35360 - 20800 = 14560, capped at 4000
        assert summary.overtime_hours == 4000
        assert summary.overtime_cost == 4000 * (225.0 - 150.0)


class TestCalculateLaborCosts:
    """Test convenience function."""

    def test_function_with_default_demand(self):
        """Test function with default demand."""
        shop_labor = pd.DataFrame(
            [
                {
                    "shop_id": "SHOP_01",
                    "base_labor_rate": 150.0,
                    "overtime_labor_rate": 225.0,
                    "technician_count": 45,
                    "annual_regular_capacity_hours": 93600,
                    "annual_overtime_capacity_hours": 18000,
                },
            ]
        )
        maintenance_labor = pd.DataFrame(
            [
                {
                    "shop_id": "SHOP_01",
                    "part_family": "ENGINE",
                    "labor_rate_used": 150.0,
                    "mean_labor_hours_per_job": 48.0,
                },
                {
                    "shop_id": "SHOP_01",
                    "part_family": "AVIONICS",
                    "labor_rate_used": 150.0,
                    "mean_labor_hours_per_job": 12.0,
                },
            ]
        )
        results = calculate_labor_costs(shop_labor, maintenance_labor)
        assert "shop_family_results" in results
        assert "shop_summaries" in results
        assert "totals" in results
        assert len(results["shop_family_results"]) == 2
        assert len(results["shop_summaries"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
