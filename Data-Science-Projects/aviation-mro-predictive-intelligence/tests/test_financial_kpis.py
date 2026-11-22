"""Tests for financial KPIs module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.kpis import (
    FinancialKPICalculator,
)


class TestFinancialKPICalculator:
    """Test FinancialKPICalculator class."""

    def test_calculate_cost_per_flight_hour(self):
        """Test cost per flight hour."""
        calc = FinancialKPICalculator()
        result = calc.calculate_cost_per_flight_hour(
            total_annual_maintenance_cost=10000000.0,
            total_annual_flight_hours=200000.0,
        )
        assert result.kpi_name == "Cost per Flight Hour"
        assert result.value == 50.0
        assert result.unit == "USD/flight_hour"

    def test_calculate_cost_per_flight_hour_zero_hours(self):
        """Test cost per flight hour with zero hours."""
        calc = FinancialKPICalculator()
        result = calc.calculate_cost_per_flight_hour(
            total_annual_maintenance_cost=10000000.0,
            total_annual_flight_hours=0.0,
        )
        assert result.value == 0.0

    def test_calculate_maintenance_cost_pct_revenue(self):
        """Test maintenance cost as % of revenue."""
        calc = FinancialKPICalculator()
        result = calc.calculate_maintenance_cost_pct_revenue(
            total_annual_maintenance_cost=10000000.0,
            total_annual_revenue=100000000.0,
        )
        assert result.kpi_name == "Maintenance Cost % Revenue"
        assert result.value == 10.0
        assert result.unit == "%"

    def test_calculate_inventory_turnover(self):
        """Test inventory turnover."""
        calc = FinancialKPICalculator()
        result = calc.calculate_inventory_turnover(
            annual_parts_cost=5000000.0,
            average_inventory_value=1000000.0,
        )
        assert result.kpi_name == "Inventory Turnover"
        assert result.value == 5.0
        assert result.unit == "times/year"

    def test_calculate_stockout_rate(self):
        """Test stockout rate."""
        calc = FinancialKPICalculator()
        result = calc.calculate_stockout_rate(
            total_expected_stockouts=100.0,
            total_annual_demand=10000.0,
        )
        assert result.kpi_name == "Stockout Rate"
        assert result.value == 1.0
        assert result.unit == "%"

    def test_calculate_labor_utilization(self):
        """Test labor utilization."""
        calc = FinancialKPICalculator()
        result = calc.calculate_labor_utilization(
            total_labor_hours_required=80000.0,
            total_labor_capacity_hours=100000.0,
        )
        assert result.kpi_name == "Labor Utilization"
        assert result.value == 80.0
        assert result.unit == "%"

    def test_calculate_cost_avoidance_predictive_maintenance(self):
        """Test predictive maintenance cost avoidance."""
        calc = FinancialKPICalculator()
        result = calc.calculate_cost_avoidance_predictive_maintenance(
            baseline_downtime_cost=10000000.0,
            predictive_downtime_cost=8000000.0,
        )
        assert result.kpi_name == "Predictive Maintenance Cost Avoidance"
        assert result.value == 2000000.0
        assert result.unit == "USD/year"

    def test_calculate_roi_predictive_maintenance(self):
        """Test predictive maintenance ROI."""
        calc = FinancialKPICalculator()
        result = calc.calculate_roi_predictive_maintenance(
            cost_avoidance=2000000.0,
            implementation_cost=500000.0,
            annual_operating_cost=100000.0,
        )
        assert result.kpi_name == "Predictive Maintenance ROI"
        # (2000000 - 100000) / 500000 * 100 = 380%
        assert abs(result.value - 380.0) < 0.1
        assert result.unit == "%"

    def test_calculate_all_kpis(self):
        """Test all KPIs calculation."""
        calc = FinancialKPICalculator()
        labor_results = {
            "totals": {
                "total_annual_labor_cost": 5000000.0,
                "total_annual_labor_hours": 50000.0,
                "total_overtime_cost": 100000.0,
            },
            "shop_summaries": [
                type(
                    "obj",
                    (object,),
                    {
                        "annual_regular_capacity_hours": 50000,
                        "annual_overtime_capacity_hours": 10000,
                    },
                )(),
            ],
        }
        parts_results = {
            "totals": {"total_annual_parts_cost": 3000000.0, "total_annual_demand": 5000.0},
            "part_results": [
                type("obj", (object,), {"annual_demand": 2500.0})(),
                type("obj", (object,), {"annual_demand": 2500.0})(),
            ],
        }
        downtime_results = {"totals": {"total_annual_downtime_cost": 2000000.0}}
        holding_results = {
            "totals": {"total_holding_cost": 500000.0, "total_average_inventory_value": 1000000.0}
        }
        stockout_results = {
            "totals": {
                "total_annual_stockout_cost": 200000.0,
                "total_expected_stockouts_per_year": 50.0,
            }
        }
        fleet_df = pd.DataFrame(
            [
                {
                    "total_annual_flight_hours": 200000,
                    "total_annual_revenue_potential": 1000000000.0,
                },
            ]
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
        assert "predictive_maintenance_avoidance" in kpis
        assert "predictive_maintenance_roi" in kpis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
