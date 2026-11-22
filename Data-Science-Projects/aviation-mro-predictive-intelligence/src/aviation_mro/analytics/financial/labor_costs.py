"""Maintenance Labor Cost Calculations.

Calculates labor costs for maintenance operations using synthetic data.
All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Labor Cost per Job = Labor Hours × Labor Rate
- Shop Annual Labor Cost = Σ(Job Count × Labor Hours × Labor Rate)
- Overtime Premium = Overtime Hours × (Overtime Rate - Base Rate)
- Labor Utilization = Actual Hours / Available Capacity
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"


@dataclass
class LaborCostResult:
    """Result of labor cost calculation for a shop-family combination."""

    shop_id: str
    part_family: str
    labor_rate: float
    mean_labor_hours_per_job: float
    labor_cost_per_job: float
    annual_job_count: int
    annual_labor_hours: float
    annual_labor_cost: float
    assumptions: list[str]


@dataclass
class ShopLaborSummary:
    """Summary of labor costs for a shop."""

    shop_id: str
    technician_count: int
    base_labor_rate: float
    overtime_labor_rate: float
    annual_regular_capacity_hours: int
    annual_overtime_capacity_hours: int
    total_annual_labor_cost: float
    total_annual_labor_hours: float
    utilization_rate: float
    overtime_hours: float
    overtime_cost: float
    assumptions: list[str]


class LaborCostCalculator:
    """Calculates maintenance labor costs."""

    def __init__(self) -> None:
        pass

    def calculate_labor_cost_per_job(
        self,
        labor_hours: float,
        labor_rate: float,
    ) -> float:
        """Calculate labor cost for a single job."""
        return labor_hours * labor_rate

    def calculate_shop_family_labor_cost(
        self,
        shop_id: str,
        part_family: str,
        labor_rate: float,
        mean_labor_hours_per_job: float,
        annual_job_count: int,
    ) -> LaborCostResult:
        """Calculate labor costs for a shop-family combination."""
        labor_cost_per_job = self.calculate_labor_cost_per_job(mean_labor_hours_per_job, labor_rate)
        annual_labor_hours = mean_labor_hours_per_job * annual_job_count
        annual_labor_cost = labor_cost_per_job * annual_job_count

        assumptions = [
            f"Labor rate: ${labor_rate:.2f}/hour (base rate for {shop_id})",
            f"Mean labor hours per {part_family} job: {mean_labor_hours_per_job:.1f}",
            f"Annual job count: {annual_job_count} (from demand forecast)",
            "Labor cost = Labor hours × Labor rate",
            "No overtime included in base calculation",
        ]

        return LaborCostResult(
            shop_id=shop_id,
            part_family=part_family,
            labor_rate=labor_rate,
            mean_labor_hours_per_job=mean_labor_hours_per_job,
            labor_cost_per_job=labor_cost_per_job,
            annual_job_count=annual_job_count,
            annual_labor_hours=annual_labor_hours,
            annual_labor_cost=annual_labor_cost,
            assumptions=assumptions,
        )

    def calculate_shop_labor_summary(
        self,
        shop_id: str,
        shop_labor_data: pd.Series,
        family_results: list[LaborCostResult],
    ) -> ShopLaborSummary:
        """Calculate summary labor costs for a shop."""
        total_annual_labor_cost = sum(r.annual_labor_cost for r in family_results)
        total_annual_labor_hours = sum(r.annual_labor_hours for r in family_results)

        technician_count = int(shop_labor_data["technician_count"])
        base_rate = shop_labor_data["base_labor_rate"]
        overtime_rate = shop_labor_data["overtime_labor_rate"]
        regular_capacity = int(shop_labor_data["annual_regular_capacity_hours"])
        overtime_capacity = int(shop_labor_data["annual_overtime_capacity_hours"])
        total_capacity = regular_capacity + overtime_capacity

        utilization_rate = total_annual_labor_hours / total_capacity if total_capacity > 0 else 0

        # Calculate overtime
        overtime_hours = max(0, total_annual_labor_hours - regular_capacity)
        overtime_hours = min(overtime_hours, overtime_capacity)
        overtime_cost = overtime_hours * (overtime_rate - base_rate)

        assumptions = [
            f"Technicians: {technician_count}",
            f"Base rate: ${base_rate:.2f}/hr, Overtime rate: ${overtime_rate:.2f}/hr",
            f"Regular capacity: {regular_capacity:,} hrs/yr, "
            f"Overtime capacity: {overtime_capacity:,} hrs/yr",
            f"Total labor hours required: {total_annual_labor_hours:,.1f} hrs/yr",
            f"Utilization: {utilization_rate:.1%}",
            "Overtime cost = Overtime hours × (Overtime rate - Base rate)",
        ]

        return ShopLaborSummary(
            shop_id=shop_id,
            technician_count=technician_count,
            base_labor_rate=base_rate,
            overtime_labor_rate=overtime_rate,
            annual_regular_capacity_hours=regular_capacity,
            annual_overtime_capacity_hours=overtime_capacity,
            total_annual_labor_cost=total_annual_labor_cost,
            total_annual_labor_hours=total_annual_labor_hours,
            utilization_rate=utilization_rate,
            overtime_hours=overtime_hours,
            overtime_cost=overtime_cost,
            assumptions=assumptions,
        )

    def calculate_all_labor_costs(
        self,
        shop_labor_df: pd.DataFrame,
        maintenance_labor_df: pd.DataFrame,
        demand_forecast_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Calculate labor costs for all shops and families."""
        # If no demand forecast, use default annual job counts
        if demand_forecast_df is None:
            # Default: 520 jobs/year per shop-family (10/week)
            default_job_counts = {}
            for shop in shop_labor_df["shop_id"].unique():
                for family in maintenance_labor_df["part_family"].unique():
                    default_job_counts[(shop, family)] = 520
        else:
            # Aggregate demand forecast by shop and family
            default_job_counts = {
                (str(k[0]), str(k[1])): int(v)
                for k, v in demand_forecast_df.groupby(["shop_id", "part_family"])["demand_count"]
                .sum()
                .items()
                if isinstance(k, tuple) and len(k) == 2
            }

        # Calculate per shop-family
        family_results = []
        for _, labor_row in maintenance_labor_df.iterrows():
            shop = labor_row["shop_id"]
            family = labor_row["part_family"]
            job_count = default_job_counts.get((shop, family), 520)

            result = self.calculate_shop_family_labor_cost(
                shop_id=shop,
                part_family=family,
                labor_rate=labor_row["labor_rate_used"],
                mean_labor_hours_per_job=labor_row["mean_labor_hours_per_job"],
                annual_job_count=job_count,
            )
            family_results.append(result)

        # Calculate shop summaries
        shop_summaries = []
        for _, shop_row in shop_labor_df.iterrows():
            shop = shop_row["shop_id"]
            shop_families = [r for r in family_results if r.shop_id == shop]
            summary = self.calculate_shop_labor_summary(shop, shop_row, shop_families)
            shop_summaries.append(summary)

        # Overall totals
        total_labor_cost = sum(s.total_annual_labor_cost for s in shop_summaries)
        total_labor_hours = sum(s.total_annual_labor_hours for s in shop_summaries)
        total_overtime_cost = sum(s.overtime_cost for s in shop_summaries)
        total_overtime_hours = sum(s.overtime_hours for s in shop_summaries)

        return {
            "shop_family_results": family_results,
            "shop_summaries": shop_summaries,
            "totals": {
                "total_annual_labor_cost": total_labor_cost,
                "total_annual_labor_hours": total_labor_hours,
                "total_overtime_cost": total_overtime_cost,
                "total_overtime_hours": total_overtime_hours,
            },
        }


def calculate_labor_costs(
    shop_labor_df: pd.DataFrame,
    maintenance_labor_df: pd.DataFrame,
    demand_forecast_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Convenience function to calculate labor costs."""
    calculator = LaborCostCalculator()
    return calculator.calculate_all_labor_costs(
        shop_labor_df, maintenance_labor_df, demand_forecast_df
    )
