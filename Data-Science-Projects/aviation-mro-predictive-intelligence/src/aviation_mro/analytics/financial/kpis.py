"""Financial KPIs.

Calculates key financial performance indicators for MRO operations using synthetic data.
All calculations use SYNTHETIC data — not real airline records.

KPIs include:
- Cost per Flight Hour
- Maintenance Cost as % of Revenue
- Inventory Turnover
- Stockout Rate
- Labor Utilization
- Cost Avoidance from Predictive Maintenance
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"


@dataclass
class FinancialKPIResult:
    """Result of a single financial KPI calculation."""

    kpi_name: str
    value: float
    unit: str
    description: str
    formula: str
    assumptions: list[str]


class FinancialKPICalculator:
    """Calculates financial KPIs for MRO operations."""

    def __init__(self) -> None:
        pass

    def calculate_cost_per_flight_hour(
        self,
        total_annual_maintenance_cost: float,
        total_annual_flight_hours: float,
    ) -> FinancialKPIResult:
        """Calculate maintenance cost per flight hour."""
        if total_annual_flight_hours <= 0:
            return FinancialKPIResult(
                kpi_name="Cost per Flight Hour",
                value=0.0,
                unit="USD/flight_hour",
                description="Total maintenance cost divided by total flight hours",
                formula="Total Annual Maintenance Cost / Total Annual Flight Hours",
                assumptions=["Flight hours > 0"],
            )

        value = total_annual_maintenance_cost / total_annual_flight_hours

        return FinancialKPIResult(
            kpi_name="Cost per Flight Hour",
            value=round(value, 2),
            unit="USD/flight_hour",
            description="Total maintenance cost divided by total flight hours",
            formula=f"${total_annual_maintenance_cost:,.0f} / {total_annual_flight_hours:,.0f} hrs",
            assumptions=[
                "Includes labor, parts, downtime, holding, and stockout costs",
                "Flight hours from fleet data",
            ],
        )

    def calculate_maintenance_cost_pct_revenue(
        self,
        total_annual_maintenance_cost: float,
        total_annual_revenue: float,
    ) -> FinancialKPIResult:
        """Calculate maintenance cost as percentage of revenue."""
        if total_annual_revenue <= 0:
            return FinancialKPIResult(
                kpi_name="Maintenance Cost % Revenue",
                value=0.0,
                unit="%",
                description="Maintenance cost as percentage of total revenue",
                formula="(Total Annual Maintenance Cost / Total Annual Revenue) × 100",
                assumptions=["Revenue > 0"],
            )

        value = (total_annual_maintenance_cost / total_annual_revenue) * 100

        return FinancialKPIResult(
            kpi_name="Maintenance Cost % Revenue",
            value=round(value, 2),
            unit="%",
            description="Maintenance cost as percentage of total revenue",
            formula=f"(${total_annual_maintenance_cost:,.0f} / ${total_annual_revenue:,.0f}) × 100",
            assumptions=[
                "Revenue based on fleet flight hours × revenue per hour",
                "Maintenance cost includes all cost categories",
            ],
        )

    def calculate_inventory_turnover(
        self,
        annual_parts_cost: float,
        average_inventory_value: float,
    ) -> FinancialKPIResult:
        """Calculate inventory turnover ratio."""
        if average_inventory_value <= 0:
            return FinancialKPIResult(
                kpi_name="Inventory Turnover",
                value=0.0,
                unit="times/year",
                description="Annual parts consumption divided by average inventory value",
                formula="Annual Parts Cost / Average Inventory Value",
                assumptions=["Average inventory value > 0"],
            )

        value = annual_parts_cost / average_inventory_value

        return FinancialKPIResult(
            kpi_name="Inventory Turnover",
            value=round(value, 2),
            unit="times/year",
            description="Annual parts consumption divided by average inventory value",
            formula=f"${annual_parts_cost:,.0f} / ${average_inventory_value:,.0f}",
            assumptions=[
                "Annual parts cost = Σ(Annual Demand × Unit Cost)",
                "Average inventory value from holding cost calculations",
            ],
        )

    def calculate_stockout_rate(
        self,
        total_expected_stockouts: float,
        total_annual_demand: float,
    ) -> FinancialKPIResult:
        """Calculate stockout rate as percentage of demand."""
        if total_annual_demand <= 0:
            return FinancialKPIResult(
                kpi_name="Stockout Rate",
                value=0.0,
                unit="%",
                description="Expected stockouts as percentage of annual demand",
                formula="(Expected Stockouts per Year / Annual Demand) × 100",
                assumptions=["Annual demand > 0"],
            )

        value = (total_expected_stockouts / total_annual_demand) * 100

        return FinancialKPIResult(
            kpi_name="Stockout Rate",
            value=round(value, 4),
            unit="%",
            description="Expected stockouts as percentage of annual demand",
            formula=f"({total_expected_stockouts:.1f} / {total_annual_demand:,.0f}) × 100",
            assumptions=[
                "Based on stockout probability × replenishment cycles",
            ],
        )

    def calculate_labor_utilization(
        self,
        total_labor_hours_required: float,
        total_labor_capacity_hours: float,
    ) -> FinancialKPIResult:
        """Calculate labor utilization rate."""
        if total_labor_capacity_hours <= 0:
            return FinancialKPIResult(
                kpi_name="Labor Utilization",
                value=0.0,
                unit="%",
                description="Required labor hours as percentage of available capacity",
                formula="(Required Labor Hours / Available Capacity Hours) × 100",
                assumptions=["Capacity hours > 0"],
            )

        value = (total_labor_hours_required / total_labor_capacity_hours) * 100

        return FinancialKPIResult(
            kpi_name="Labor Utilization",
            value=round(value, 2),
            unit="%",
            description="Required labor hours as percentage of available capacity",
            formula=(
                f"({total_labor_hours_required:,.0f} / {total_labor_capacity_hours:,.0f}) × 100"
            ),
            assumptions=[
                "Capacity includes regular + overtime hours",
                "Required hours from demand forecast × labor hours per job",
            ],
        )

    def calculate_cost_avoidance_predictive_maintenance(
        self,
        baseline_downtime_cost: float,
        predictive_downtime_cost: float,
    ) -> FinancialKPIResult:
        """Calculate cost avoidance from predictive maintenance."""
        cost_avoidance = baseline_downtime_cost - predictive_downtime_cost
        pct_reduction = (
            (cost_avoidance / baseline_downtime_cost * 100) if baseline_downtime_cost > 0 else 0
        )

        return FinancialKPIResult(
            kpi_name="Predictive Maintenance Cost Avoidance",
            value=round(cost_avoidance, 2),
            unit="USD/year",
            description=("Annual cost savings from reduced downtime via predictive maintenance"),
            formula=(
                f"${baseline_downtime_cost:,.0f} - "
                f"${predictive_downtime_cost:,.0f} = "
                f"${cost_avoidance:,.0f} ({pct_reduction:.1f}% reduction)"
            ),
            assumptions=[
                "Baseline = current reactive maintenance downtime costs",
                "Predictive = downtime costs with 20% reduction from predictive analytics",
                "Does not include implementation costs of predictive system",
            ],
        )

    def calculate_roi_predictive_maintenance(
        self,
        cost_avoidance: float,
        implementation_cost: float,
        annual_operating_cost: float,
    ) -> FinancialKPIResult:
        """Calculate ROI of predictive maintenance implementation."""
        if implementation_cost <= 0:
            return FinancialKPIResult(
                kpi_name="Predictive Maintenance ROI",
                value=0.0,
                unit="%",
                description="Return on investment for predictive maintenance",
                formula=(
                    "(Annual Cost Avoidance - Annual Operating Cost) / Implementation Cost × 100"
                ),
                assumptions=["Implementation cost > 0"],
            )

        net_annual_benefit = cost_avoidance - annual_operating_cost
        roi = (net_annual_benefit / implementation_cost) * 100

        return FinancialKPIResult(
            kpi_name="Predictive Maintenance ROI",
            value=round(roi, 2),
            unit="%",
            description="Return on investment for predictive maintenance (Year 1)",
            formula=(
                f"(${cost_avoidance:,.0f} - ${annual_operating_cost:,.0f}) / "
                f"${implementation_cost:,.0f} × 100 = {roi:.1f}%"
            ),
            assumptions=[
                f"Implementation cost: ${implementation_cost:,.0f}",
                f"Annual operating cost: ${annual_operating_cost:,.0f}",
                f"Annual cost avoidance: ${cost_avoidance:,.0f}",
                "Year 1 ROI only; does not include multi-year NPV",
            ],
        )

    def calculate_all_kpis(
        self,
        labor_results: dict[str, Any],
        parts_results: dict[str, Any],
        downtime_results: dict[str, Any],
        holding_results: dict[str, Any],
        stockout_results: dict[str, Any],
        fleet_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate all financial KPIs."""
        # Extract totals
        total_labor_cost = labor_results["totals"]["total_annual_labor_cost"]
        total_labor_hours = labor_results["totals"]["total_annual_labor_hours"]
        total_parts_cost = parts_results["totals"]["total_annual_parts_cost"]
        total_downtime_cost = downtime_results["totals"]["total_annual_downtime_cost"]
        total_holding_cost = holding_results["totals"]["total_holding_cost"]
        total_stockout_cost = stockout_results["totals"]["total_annual_stockout_cost"]

        total_maintenance_cost = (
            total_labor_cost
            + total_parts_cost
            + total_downtime_cost
            + total_holding_cost
            + total_stockout_cost
        )

        total_annual_flight_hours = fleet_df["total_annual_flight_hours"].sum()
        total_annual_revenue = fleet_df["total_annual_revenue_potential"].sum()

        # Inventory metrics
        total_annual_demand = parts_results["totals"].get("total_annual_demand", 0)
        if "total_annual_demand" not in parts_results["totals"]:
            total_annual_demand = sum(r.annual_demand for r in parts_results["part_results"])

        avg_inventory_value = holding_results["totals"].get("total_average_inventory_value", 0)
        total_expected_stockouts = stockout_results["totals"]["total_expected_stockouts_per_year"]

        # Capacity
        total_capacity = sum(
            s.annual_regular_capacity_hours + s.annual_overtime_capacity_hours
            for s in labor_results["shop_summaries"]
        )

        # Calculate KPIs
        kpis: dict[str, Any] = {}

        kpis["cost_per_flight_hour"] = self.calculate_cost_per_flight_hour(
            total_maintenance_cost, total_annual_flight_hours
        )

        kpis["maintenance_cost_pct_revenue"] = self.calculate_maintenance_cost_pct_revenue(
            total_maintenance_cost, total_annual_revenue
        )

        kpis["inventory_turnover"] = self.calculate_inventory_turnover(
            total_parts_cost, avg_inventory_value
        )

        kpis["stockout_rate"] = self.calculate_stockout_rate(
            total_expected_stockouts, total_annual_demand
        )

        kpis["labor_utilization"] = self.calculate_labor_utilization(
            total_labor_hours, total_capacity
        )

        # Cost breakdown KPIs
        kpis["cost_breakdown"] = {
            "labor_cost": total_labor_cost,
            "parts_cost": total_parts_cost,
            "downtime_cost": total_downtime_cost,
            "holding_cost": total_holding_cost,
            "stockout_cost": total_stockout_cost,
            "total_maintenance_cost": total_maintenance_cost,
        }

        kpis["cost_breakdown_pct"] = {
            k: round(v / total_maintenance_cost * 100, 1) if total_maintenance_cost > 0 else 0
            for k, v in kpis["cost_breakdown"].items()
            if k != "total_maintenance_cost"
        }

        # Predictive maintenance scenarios
        baseline_downtime = total_downtime_cost
        # Find predictive scenario
        predictive_cost = baseline_downtime * 0.8  # 20% reduction
        kpis["predictive_maintenance_avoidance"] = (
            self.calculate_cost_avoidance_predictive_maintenance(baseline_downtime, predictive_cost)
        )

        # ROI (assuming $500k implementation + $100k annual operating)
        kpis["predictive_maintenance_roi"] = self.calculate_roi_predictive_maintenance(
            cost_avoidance=kpis["predictive_maintenance_avoidance"].value,
            implementation_cost=500000,
            annual_operating_cost=100000,
        )

        return kpis


def calculate_financial_kpis(
    labor_results: dict[str, Any],
    parts_results: dict[str, Any],
    downtime_results: dict[str, Any],
    holding_results: dict[str, Any],
    stockout_results: dict[str, Any],
    fleet_df: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience function to calculate all financial KPIs."""
    calculator = FinancialKPICalculator()
    return calculator.calculate_all_kpis(
        labor_results, parts_results, downtime_results, holding_results, stockout_results, fleet_df
    )
