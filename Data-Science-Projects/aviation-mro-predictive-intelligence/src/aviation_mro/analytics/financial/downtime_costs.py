"""Downtime Cost Scenarios.

Calculates aircraft downtime costs for maintenance events using synthetic data.
All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Downtime Cost per Event = Downtime Hours × Cost per Hour
- Annual Downtime Cost = Σ(Event Count × Downtime Hours × Cost per Hour)
- Cost per Hour includes: Lost Revenue + Passenger Handling + Recovery Costs
- Fleet-type specific costs (Narrow-body, Wide-body, Regional)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"


@dataclass
class DowntimeCostResult:
    """Result of downtime cost calculation for a fleet-family combination."""

    fleet_type: str
    part_family: str
    aircraft_count: int
    downtime_cost_per_hour: float
    mean_downtime_hours: float
    downtime_cost_per_event: float
    annual_event_count: int
    annual_downtime_hours: float
    annual_downtime_cost: float
    assumptions: list[str]


@dataclass
class DowntimeScenarioResult:
    """Result of a downtime cost scenario."""

    scenario_name: str
    description: str
    total_annual_downtime_cost: float
    total_annual_downtime_hours: float
    cost_breakdown: dict[str, float]
    assumptions: list[str]


class DowntimeCostCalculator:
    """Calculates aircraft downtime costs."""

    def __init__(self) -> None:
        pass

    def calculate_downtime_cost_per_event(
        self,
        downtime_hours: float,
        cost_per_hour: float,
    ) -> float:
        """Calculate downtime cost for a single event."""
        return downtime_hours * cost_per_hour

    def calculate_fleet_family_downtime(
        self,
        fleet_type: str,
        part_family: str,
        aircraft_count: int,
        downtime_cost_per_hour: float,
        mean_downtime_hours: float,
        annual_event_count: int,
    ) -> DowntimeCostResult:
        """Calculate downtime costs for a fleet-family combination."""
        downtime_cost_per_event = self.calculate_downtime_cost_per_event(
            mean_downtime_hours, downtime_cost_per_hour
        )
        annual_downtime_hours = mean_downtime_hours * annual_event_count
        annual_downtime_cost = downtime_cost_per_event * annual_event_count

        assumptions = [
            f"Fleet type: {fleet_type} ({aircraft_count} aircraft)",
            f"Part family: {part_family}",
            f"Downtime cost rate: ${downtime_cost_per_hour:,.0f}/hour",
            f"Mean downtime per event: {mean_downtime_hours:.1f} hours",
            f"Annual event count: {annual_event_count} (from demand forecast)",
            "Downtime cost = Downtime hours × Cost per hour",
            "Assumes events distributed across fleet",
        ]

        return DowntimeCostResult(
            fleet_type=fleet_type,
            part_family=part_family,
            aircraft_count=aircraft_count,
            downtime_cost_per_hour=downtime_cost_per_hour,
            mean_downtime_hours=mean_downtime_hours,
            downtime_cost_per_event=downtime_cost_per_event,
            annual_event_count=annual_event_count,
            annual_downtime_hours=annual_downtime_hours,
            annual_downtime_cost=annual_downtime_cost,
            assumptions=assumptions,
        )

    def calculate_all_downtime_costs(
        self,
        fleet_df: pd.DataFrame,
        downtime_costs_df: pd.DataFrame,
        demand_forecast_df: pd.DataFrame | None = None,
        fleet_mix: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Calculate downtime costs for all fleet-family combinations."""
        # Default fleet mix if not provided
        if fleet_mix is None:
            fleet_mix = {
                "NARROW_BODY": 0.60,
                "WIDE_BODY": 0.25,
                "REGIONAL": 0.15,
            }

        # If no demand forecast, use default annual event counts
        if demand_forecast_df is None:
            # Default: distribute demand across fleet types by mix
            default_events = {}
            for fleet in fleet_df["fleet_type"].unique():
                for family in downtime_costs_df["part_family"].unique():
                    base_events = 520  # 10/week per shop-family
                    mix_factor = fleet_mix.get(fleet, 0.33)
                    default_events[(fleet, family)] = int(base_events * mix_factor)
        else:
            # Use actual demand forecast, distributed by fleet mix
            default_events = {}
            demand_forecast_df["demand_count"].sum()
            for fleet in fleet_df["fleet_type"].unique():
                for family in downtime_costs_df["part_family"].unique():
                    family_demand = demand_forecast_df[demand_forecast_df["part_family"] == family][
                        "demand_count"
                    ].sum()
                    mix_factor = fleet_mix.get(fleet, 0.33)
                    default_events[(fleet, family)] = int(family_demand * mix_factor)

        # Calculate per fleet-family
        results = []
        for _, fleet_row in fleet_df.iterrows():
            fleet = fleet_row["fleet_type"]
            aircraft_count = int(fleet_row["aircraft_count"])
            for _, dt_row in downtime_costs_df.iterrows():
                if dt_row["fleet_type"] != fleet:
                    continue
                family = dt_row["part_family"]
                event_count = default_events.get((fleet, family), 100)

                result = self.calculate_fleet_family_downtime(
                    fleet_type=fleet,
                    part_family=family,
                    aircraft_count=aircraft_count,
                    downtime_cost_per_hour=dt_row["downtime_cost_per_hour"],
                    mean_downtime_hours=dt_row["mean_downtime_hours"],
                    annual_event_count=event_count,
                )
                results.append(result)

        # Scenarios
        scenarios = self._generate_scenarios(results, fleet_df, downtime_costs_df, fleet_mix)

        # Overall totals
        total_annual_downtime_cost = sum(r.annual_downtime_cost for r in results)
        total_annual_downtime_hours = sum(r.annual_downtime_hours for r in results)

        return {
            "fleet_family_results": results,
            "scenarios": scenarios,
            "totals": {
                "total_annual_downtime_cost": total_annual_downtime_cost,
                "total_annual_downtime_hours": total_annual_downtime_hours,
            },
        }

    def _generate_scenarios(
        self,
        results: list[DowntimeCostResult],
        fleet_df: pd.DataFrame,
        downtime_costs_df: pd.DataFrame,
        fleet_mix: dict[str, float],
    ) -> list[DowntimeScenarioResult]:
        """Generate downtime cost scenarios."""
        scenarios: list[DowntimeScenarioResult] = []

        # Baseline scenario
        scenarios.append(
            DowntimeScenarioResult(
                scenario_name="Baseline",
                description="Current fleet mix and downtime parameters",
                total_annual_downtime_cost=sum(r.annual_downtime_cost for r in results),
                total_annual_downtime_hours=sum(r.annual_downtime_hours for r in results),
                cost_breakdown={r.part_family: r.annual_downtime_cost for r in results},
                assumptions=[
                    f"Fleet mix: {fleet_mix}",
                    "Current downtime hours and cost rates",
                ],
            )
        )

        # Scenario: 10% reduction in downtime hours (improved processes)
        improved_hours_cost = sum(r.annual_downtime_cost * 0.9 for r in results)
        scenarios.append(
            DowntimeScenarioResult(
                scenario_name="Improved Processes (-10% downtime)",
                description="10% reduction in mean downtime hours through process improvements",
                total_annual_downtime_cost=improved_hours_cost,
                total_annual_downtime_hours=sum(r.annual_downtime_hours for r in results) * 0.9,
                cost_breakdown={r.part_family: r.annual_downtime_cost * 0.9 for r in results},
                assumptions=[
                    "Mean downtime hours reduced by 10%",
                    "Same event count and cost rates",
                ],
            )
        )

        # Scenario: 20% reduction (predictive maintenance)
        predictive_cost = sum(r.annual_downtime_cost * 0.8 for r in results)
        scenarios.append(
            DowntimeScenarioResult(
                scenario_name="Predictive Maintenance (-20% downtime)",
                description="20% reduction in downtime through predictive maintenance",
                total_annual_downtime_cost=predictive_cost,
                total_annual_downtime_hours=sum(r.annual_downtime_hours for r in results) * 0.8,
                cost_breakdown={r.part_family: r.annual_downtime_cost * 0.8 for r in results},
                assumptions=[
                    "Mean downtime hours reduced by 20%",
                    "Same event count and cost rates",
                    "Predictive maintenance enables faster turnaround",
                ],
            )
        )

        # Scenario: Fleet growth (+15% aircraft)
        growth_cost = sum(r.annual_downtime_cost * 1.15 for r in results)
        scenarios.append(
            DowntimeScenarioResult(
                scenario_name="Fleet Growth (+15% aircraft)",
                description="15% increase in fleet size, proportional increase in events",
                total_annual_downtime_cost=growth_cost,
                total_annual_downtime_hours=sum(r.annual_downtime_hours for r in results) * 1.15,
                cost_breakdown={r.part_family: r.annual_downtime_cost * 1.15 for r in results},
                assumptions=[
                    "Event count increases proportionally with fleet size",
                    "Same downtime hours and cost rates per event",
                ],
            )
        )

        return scenarios


def calculate_downtime_costs(
    fleet_df: pd.DataFrame,
    downtime_costs_df: pd.DataFrame,
    demand_forecast_df: pd.DataFrame | None = None,
    fleet_mix: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convenience function to calculate downtime costs."""
    calculator = DowntimeCostCalculator()
    return calculator.calculate_all_downtime_costs(
        fleet_df, downtime_costs_df, demand_forecast_df, fleet_mix
    )
