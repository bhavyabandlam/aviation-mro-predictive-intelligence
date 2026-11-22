"""Tests for downtime costs module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.downtime_costs import (
    DowntimeCostCalculator,
    calculate_downtime_costs,
)


class TestDowntimeCostCalculator:
    """Test DowntimeCostCalculator class."""

    def test_calculate_downtime_cost_per_event(self):
        """Test downtime cost per event."""
        calc = DowntimeCostCalculator()
        cost = calc.calculate_downtime_cost_per_event(downtime_hours=24.0, cost_per_hour=12000.0)
        assert cost == 288000.0

    def test_calculate_fleet_family_downtime(self):
        """Test fleet-family downtime calculation."""
        calc = DowntimeCostCalculator()
        result = calc.calculate_fleet_family_downtime(
            fleet_type="NARROW_BODY",
            part_family="ENGINE",
            aircraft_count=45,
            downtime_cost_per_hour=12000.0,
            mean_downtime_hours=72.0,
            annual_event_count=300,
        )
        assert result.fleet_type == "NARROW_BODY"
        assert result.part_family == "ENGINE"
        assert result.aircraft_count == 45
        assert result.downtime_cost_per_hour == 12000.0
        assert result.mean_downtime_hours == 72.0
        assert result.downtime_cost_per_event == 864000.0
        assert result.annual_event_count == 300
        assert result.annual_downtime_hours == 21600.0
        assert result.annual_downtime_cost == 259200000.0

    def test_calculate_all_downtime_costs(self):
        """Test all downtime costs calculation."""
        calc = DowntimeCostCalculator()
        fleet_df = pd.DataFrame(
            [
                {
                    "fleet_type": "NARROW_BODY",
                    "aircraft_count": 45,
                    "avg_annual_flight_hours_per_aircraft": 2800,
                    "revenue_per_flight_hour": 8500,
                    "total_annual_flight_hours": 126000,
                    "total_annual_revenue_potential": 1071000000,
                },
            ]
        )
        downtime_df = pd.DataFrame(
            [
                {
                    "fleet_type": "NARROW_BODY",
                    "part_family": "ENGINE",
                    "mean_downtime_hours": 72.0,
                    "downtime_cost_per_hour": 12000.0,
                    "downtime_cost_per_event": 864000.0,
                },
                {
                    "fleet_type": "NARROW_BODY",
                    "part_family": "AVIONICS",
                    "mean_downtime_hours": 24.0,
                    "downtime_cost_per_hour": 12000.0,
                    "downtime_cost_per_event": 288000.0,
                },
            ]
        )
        results = calc.calculate_all_downtime_costs(fleet_df, downtime_df)
        assert "fleet_family_results" in results
        assert "scenarios" in results
        assert "totals" in results
        assert len(results["fleet_family_results"]) == 2
        assert len(results["scenarios"]) >= 3  # Baseline, improved, predictive, growth


class TestCalculateDowntimeCosts:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_downtime_costs function."""
        fleet_df = pd.DataFrame(
            [
                {
                    "fleet_type": "NARROW_BODY",
                    "aircraft_count": 45,
                    "total_annual_flight_hours": 126000,
                    "total_annual_revenue_potential": 1071000000,
                },
            ]
        )
        downtime_df = pd.DataFrame(
            [
                {
                    "fleet_type": "NARROW_BODY",
                    "part_family": "ENGINE",
                    "mean_downtime_hours": 72.0,
                    "downtime_cost_per_hour": 12000.0,
                },
            ]
        )
        results = calculate_downtime_costs(fleet_df, downtime_df)
        assert "fleet_family_results" in results
        assert "scenarios" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
