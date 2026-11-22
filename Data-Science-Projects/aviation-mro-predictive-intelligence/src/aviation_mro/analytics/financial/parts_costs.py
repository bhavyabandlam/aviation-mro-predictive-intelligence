"""Spare Parts Cost Calculations.

Calculates spare parts procurement and inventory costs using synthetic data.
All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Annual Parts Cost = Σ(Annual Demand × Unit Cost)
- EOQ = √(2 × D × S / H) where H = Unit Cost × Holding Cost Rate
- Total Annual Ordering Cost = (Annual Demand / EOQ) × Order Cost
- Total Annual Holding Cost = (EOQ / 2) × Unit Cost × Holding Cost Rate
- Total Inventory Cost = Ordering Cost + Holding Cost
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"


@dataclass
class PartsCostResult:
    """Result of parts cost calculation for a single part."""

    part_number: str
    part_family: str
    shop_id: str
    unit_cost: float
    annual_demand: float
    annual_parts_cost: float
    eoq: float
    orders_per_year: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_inventory_cost: float
    assumptions: list[str]


@dataclass
class PartsCostSummary:
    """Summary of parts costs by family or shop."""

    group_by: str  # "family" or "shop"
    group_value: str
    total_parts: int
    total_annual_parts_cost: float
    total_annual_demand: float
    avg_unit_cost: float
    total_inventory_cost: float
    assumptions: list[str]


class PartsCostCalculator:
    """Calculates spare parts costs."""

    def __init__(
        self,
        holding_cost_rate: float = 0.20,
        order_cost: float = 75.0,
    ):
        """
        Initialize calculator.

        Args:
            holding_cost_rate: Annual holding cost as fraction of unit cost
            order_cost: Fixed cost per purchase order
        """
        self.holding_cost_rate = holding_cost_rate
        self.order_cost = order_cost

    def calculate_eoq(
        self,
        annual_demand: float,
        unit_cost: float,
    ) -> float:
        """Calculate Economic Order Quantity."""
        if annual_demand <= 0 or unit_cost <= 0:
            return 1.0

        holding_cost_per_unit = unit_cost * self.holding_cost_rate
        eoq = np.sqrt(2 * annual_demand * self.order_cost / holding_cost_per_unit)
        return max(1.0, float(eoq))

    def calculate_part_costs(
        self,
        part_number: str,
        part_family: str,
        shop_id: str,
        unit_cost: float,
        annual_demand: float,
    ) -> PartsCostResult:
        """Calculate all costs for a single part."""
        annual_parts_cost = annual_demand * unit_cost
        eoq = self.calculate_eoq(annual_demand, unit_cost)
        orders_per_year = annual_demand / eoq if eoq > 0 else 0
        annual_ordering_cost = orders_per_year * self.order_cost
        annual_holding_cost = (eoq / 2) * unit_cost * self.holding_cost_rate
        total_inventory_cost = annual_ordering_cost + annual_holding_cost

        assumptions = [
            f"Unit cost: ${unit_cost:.2f}",
            f"Annual demand: {annual_demand:.1f} units",
            f"Holding cost rate: {self.holding_cost_rate:.0%} of unit cost",
            f"Order cost: ${self.order_cost:.2f} per order",
            f"EOQ = √(2 × {annual_demand:.1f} × "
            f"${self.order_cost:.2f} / (${unit_cost:.2f} × "
            f"{self.holding_cost_rate:.0%})) = {eoq:.1f}",
            f"Annual ordering cost = {orders_per_year:.1f} × "
            f"${self.order_cost:.2f} = ${annual_ordering_cost:.2f}",
            f"Annual holding cost = ({eoq:.1f}/2) × "
            f"${unit_cost:.2f} × {self.holding_cost_rate:.0%} = "
            f"${annual_holding_cost:.2f}",
        ]

        return PartsCostResult(
            part_number=part_number,
            part_family=part_family,
            shop_id=shop_id,
            unit_cost=unit_cost,
            annual_demand=annual_demand,
            annual_parts_cost=annual_parts_cost,
            eoq=eoq,
            orders_per_year=orders_per_year,
            annual_ordering_cost=annual_ordering_cost,
            annual_holding_cost=annual_holding_cost,
            total_inventory_cost=total_inventory_cost,
            assumptions=assumptions,
        )

    def calculate_all_parts_costs(
        self,
        inventory_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate parts costs for all parts in inventory."""
        results = []

        for _, row in inventory_df.iterrows():
            annual_demand = row["demand_mean_weekly"] * 52
            result = self.calculate_part_costs(
                part_number=row["part_number"],
                part_family=row["part_family"],
                shop_id=row["shop_id"],
                unit_cost=row["unit_cost"],
                annual_demand=annual_demand,
            )
            results.append(result)

        # Summaries by family
        family_summaries = []
        for family in inventory_df["part_family"].unique():
            family_results = [r for r in results if r.part_family == family]
            summary = PartsCostSummary(
                group_by="family",
                group_value=family,
                total_parts=len(family_results),
                total_annual_parts_cost=sum(r.annual_parts_cost for r in family_results),
                total_annual_demand=sum(r.annual_demand for r in family_results),
                avg_unit_cost=np.mean([r.unit_cost for r in family_results]),
                total_inventory_cost=sum(r.total_inventory_cost for r in family_results),
                assumptions=[
                    f"Aggregated across {len(family_results)} parts in {family} family",
                    "Uses individual part EOQ calculations",
                ],
            )
            family_summaries.append(summary)

        # Summaries by shop
        shop_summaries = []
        for shop in inventory_df["shop_id"].unique():
            shop_results = [r for r in results if r.shop_id == shop]
            summary = PartsCostSummary(
                group_by="shop",
                group_value=shop,
                total_parts=len(shop_results),
                total_annual_parts_cost=sum(r.annual_parts_cost for r in shop_results),
                total_annual_demand=sum(r.annual_demand for r in shop_results),
                avg_unit_cost=np.mean([r.unit_cost for r in shop_results]),
                total_inventory_cost=sum(r.total_inventory_cost for r in shop_results),
                assumptions=[
                    f"Aggregated across {len(shop_results)} parts in {shop}",
                    "Uses individual part EOQ calculations",
                ],
            )
            shop_summaries.append(summary)

        # Overall totals
        total_annual_parts_cost = sum(r.annual_parts_cost for r in results)
        total_inventory_cost = sum(r.total_inventory_cost for r in results)

        return {
            "part_results": results,
            "family_summaries": family_summaries,
            "shop_summaries": shop_summaries,
            "totals": {
                "total_parts": len(results),
                "total_annual_parts_cost": total_annual_parts_cost,
                "total_inventory_cost": total_inventory_cost,
            },
        }


def calculate_parts_costs(
    inventory_df: pd.DataFrame,
    holding_cost_rate: float = 0.20,
    order_cost: float = 75.0,
) -> dict[str, Any]:
    """Convenience function to calculate parts costs."""
    calculator = PartsCostCalculator(holding_cost_rate=holding_cost_rate, order_cost=order_cost)
    return calculator.calculate_all_parts_costs(inventory_df)
