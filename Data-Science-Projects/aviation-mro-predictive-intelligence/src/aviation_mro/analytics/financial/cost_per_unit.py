"""Maintenance Cost per Work Order and per Engine.

Calculates unit-level maintenance cost metrics from synthetic MRO data.

All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Cost per Work Order = Total Annual Maintenance Cost / Total Annual Job Count
- Cost per Engine = Total Annual Maintenance Cost / Total Number of Engines

Assumptions (synthetic, documented):
- Engines per aircraft: NARROW_BODY 2, WIDE_BODY 2, REGIONAL 2
  (simplified fleet engine count; not derived from real aircraft models)
- Total annual job count from labor calculations (shop-family annual_job_count)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"

# Engines per aircraft by fleet type (synthetic assumption).
ENGINES_PER_AIRCRAFT: dict[str, int] = {
    "NARROW_BODY": 2,
    "WIDE_BODY": 2,
    "REGIONAL": 2,
}


@dataclass
class CostPerUnitResult:
    """Result of cost-per-unit calculation."""

    metric_name: str
    value: float
    unit: str
    description: str
    formula: str
    assumptions: list[str]


class CostPerUnitCalculator:
    """Calculates maintenance cost per work order and per engine."""

    def __init__(
        self,
        engines_per_aircraft: dict[str, int] | None = None,
    ):
        self.engines_per_aircraft = (
            engines_per_aircraft if engines_per_aircraft is not None else ENGINES_PER_AIRCRAFT
        )

    def calculate_cost_per_work_order(
        self,
        total_annual_maintenance_cost: float,
        total_annual_job_count: int,
    ) -> CostPerUnitResult:
        """Calculate maintenance cost per work order."""
        if total_annual_job_count <= 0:
            return CostPerUnitResult(
                metric_name="Cost per Work Order",
                value=0.0,
                unit="USD/work_order",
                description="Total annual maintenance cost divided by total annual job count",
                formula="Total Annual Maintenance Cost / Total Annual Job Count",
                assumptions=["Job count > 0"],
            )

        value = total_annual_maintenance_cost / total_annual_job_count

        return CostPerUnitResult(
            metric_name="Cost per Work Order",
            value=round(value, 2),
            unit="USD/work_order",
            description="Total annual maintenance cost divided by total annual job count",
            formula=(
                f"${total_annual_maintenance_cost:,.0f} / {total_annual_job_count:,} "
                f"work orders = ${value:,.2f}/work order"
            ),
            assumptions=[
                "Total annual maintenance cost includes labor, parts, downtime, holding, stockout",
                "Total annual job count from labor calculations (shop-family job counts)",
                "Each job = one work order",
                "Synthetic data — not real airline records",
            ],
        )

    def calculate_total_engines(
        self,
        fleet_df: pd.DataFrame,
    ) -> int:
        """Calculate total number of engines across the fleet."""
        total_engines = 0
        for _, row in fleet_df.iterrows():
            fleet_type = row["fleet_type"]
            aircraft_count = int(row["aircraft_count"])
            engines_per = self.engines_per_aircraft.get(fleet_type, 0)
            total_engines += aircraft_count * engines_per
        return total_engines

    def calculate_cost_per_engine(
        self,
        total_annual_maintenance_cost: float,
        fleet_df: pd.DataFrame,
    ) -> CostPerUnitResult:
        """Calculate maintenance cost per engine."""
        total_engines = self.calculate_total_engines(fleet_df)

        if total_engines <= 0:
            return CostPerUnitResult(
                metric_name="Cost per Engine",
                value=0.0,
                unit="USD/engine/year",
                description="Total annual maintenance cost divided by total number of engines",
                formula="Total Annual Maintenance Cost / Total Number of Engines",
                assumptions=["Total engines > 0"],
            )

        value = total_annual_maintenance_cost / total_engines

        engine_detail = ", ".join(
            f"{ft}: {self.engines_per_aircraft.get(ft, 0)} eng/aircraft"
            for ft in sorted(self.engines_per_aircraft.keys())
        )

        return CostPerUnitResult(
            metric_name="Cost per Engine",
            value=round(value, 2),
            unit="USD/engine/year",
            description="Total annual maintenance cost divided by total number of engines",
            formula=(
                f"${total_annual_maintenance_cost:,.0f} / {total_engines:,} engines "
                f"= ${value:,.2f}/engine/year"
            ),
            assumptions=[
                f"Engines per aircraft: {engine_detail}",
                "Total annual maintenance cost includes all cost categories",
                "One engine cost metric per engine per year",
                "Synthetic data — not real airline records",
            ],
        )

    def calculate_all(
        self,
        total_annual_maintenance_cost: float,
        total_annual_job_count: int,
        fleet_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate all cost-per-unit metrics."""
        cost_per_work_order = self.calculate_cost_per_work_order(
            total_annual_maintenance_cost, total_annual_job_count
        )
        cost_per_engine = self.calculate_cost_per_engine(total_annual_maintenance_cost, fleet_df)

        return {
            "cost_per_work_order": cost_per_work_order,
            "cost_per_engine": cost_per_engine,
            "total_annual_maintenance_cost": total_annual_maintenance_cost,
            "total_annual_job_count": total_annual_job_count,
            "total_engines": self.calculate_total_engines(fleet_df),
        }


def calculate_cost_per_unit(
    total_annual_maintenance_cost: float,
    total_annual_job_count: int,
    fleet_df: pd.DataFrame,
    engines_per_aircraft: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Convenience function for cost-per-unit calculations."""
    calculator = CostPerUnitCalculator(engines_per_aircraft=engines_per_aircraft)
    return calculator.calculate_all(total_annual_maintenance_cost, total_annual_job_count, fleet_df)


if __name__ == "__main__":
    pass
