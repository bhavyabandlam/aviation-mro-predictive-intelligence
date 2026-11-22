"""Inventory Holding Cost Calculations.

Calculates inventory holding costs using synthetic data.
All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Average Inventory Level = EOQ / 2 + Safety Stock
- Annual Holding Cost = Average Inventory × Unit Cost × Holding Cost Rate
- Safety Stock Holding Cost = Safety Stock × Unit Cost × Holding Cost Rate
- Cycle Stock Holding Cost = (EOQ / 2) × Unit Cost × Holding Cost Rate
- Total Holding Cost = Σ(Part Holding Costs)
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
class HoldingCostResult:
    """Result of holding cost calculation for a single part."""

    part_number: str
    part_family: str
    shop_id: str
    unit_cost: float
    safety_stock: float
    eoq: float
    average_inventory: float
    cycle_stock: float
    safety_stock_value: float
    cycle_stock_value: float
    total_holding_cost: float
    assumptions: list[str]


@dataclass
class HoldingCostSummary:
    """Summary of holding costs by group."""

    group_by: str
    group_value: str
    total_parts: int
    total_average_inventory: float
    total_safety_stock: float
    total_cycle_stock: float
    total_holding_cost: float
    assumptions: list[str]


class HoldingCostCalculator:
    """Calculates inventory holding costs."""

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

    def calculate_holding_cost(
        self,
        part_number: str,
        part_family: str,
        shop_id: str,
        unit_cost: float,
        annual_demand: float,
        safety_stock: float,
    ) -> HoldingCostResult:
        """Calculate holding cost for a single part."""
        eoq = self.calculate_eoq(annual_demand, unit_cost)
        cycle_stock = eoq / 2
        average_inventory = cycle_stock + safety_stock

        cycle_stock_value = cycle_stock * unit_cost
        safety_stock_value = safety_stock * unit_cost

        total_holding_cost = average_inventory * unit_cost * self.holding_cost_rate

        assumptions = [
            f"Unit cost: ${unit_cost:.2f}",
            f"Annual demand: {annual_demand:.1f} units",
            f"Safety stock: {safety_stock:.1f} units",
            f"EOQ: {eoq:.1f} units",
            f"Cycle stock (EOQ/2): {cycle_stock:.1f} units",
            f"Average inventory: {average_inventory:.1f} units",
            f"Holding cost rate: {self.holding_cost_rate:.0%} annually",
            f"Cycle stock value: ${cycle_stock_value:.2f}",
            f"Safety stock value: ${safety_stock_value:.2f}",
            f"Total holding cost = {average_inventory:.1f} × "
            f"${unit_cost:.2f} × {self.holding_cost_rate:.0%} = ${total_holding_cost:.2f}",
        ]

        return HoldingCostResult(
            part_number=part_number,
            part_family=part_family,
            shop_id=shop_id,
            unit_cost=unit_cost,
            safety_stock=safety_stock,
            eoq=eoq,
            average_inventory=average_inventory,
            cycle_stock=cycle_stock,
            safety_stock_value=safety_stock_value,
            cycle_stock_value=cycle_stock_value,
            total_holding_cost=total_holding_cost,
            assumptions=assumptions,
        )

    def calculate_all_holding_costs(
        self,
        inventory_df: pd.DataFrame,
        safety_stock_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate holding costs for all parts."""
        # Create safety stock lookup
        ss_lookup = {}
        for _, row in safety_stock_df.iterrows():
            ss_lookup[row["part_number"]] = row["safety_stock"]

        results = []
        for _, row in inventory_df.iterrows():
            part_num = row["part_number"]
            safety_stock = ss_lookup.get(part_num, 0.0)
            annual_demand = row["demand_mean_weekly"] * 52

            result = self.calculate_holding_cost(
                part_number=part_num,
                part_family=row["part_family"],
                shop_id=row["shop_id"],
                unit_cost=row["unit_cost"],
                annual_demand=annual_demand,
                safety_stock=safety_stock,
            )
            results.append(result)

        # Summaries by family
        family_summaries = []
        for family in inventory_df["part_family"].unique():
            family_results = [r for r in results if r.part_family == family]
            summary = HoldingCostSummary(
                group_by="family",
                group_value=family,
                total_parts=len(family_results),
                total_average_inventory=sum(r.average_inventory for r in family_results),
                total_safety_stock=sum(r.safety_stock for r in family_results),
                total_cycle_stock=sum(r.cycle_stock for r in family_results),
                total_holding_cost=sum(r.total_holding_cost for r in family_results),
                assumptions=[
                    f"Aggregated across {len(family_results)} parts in {family} family",
                ],
            )
            family_summaries.append(summary)

        # Summaries by shop
        shop_summaries = []
        for shop in inventory_df["shop_id"].unique():
            shop_results = [r for r in results if r.shop_id == shop]
            summary = HoldingCostSummary(
                group_by="shop",
                group_value=shop,
                total_parts=len(shop_results),
                total_average_inventory=sum(r.average_inventory for r in shop_results),
                total_safety_stock=sum(r.safety_stock for r in shop_results),
                total_cycle_stock=sum(r.cycle_stock for r in shop_results),
                total_holding_cost=sum(r.total_holding_cost for r in shop_results),
                assumptions=[
                    f"Aggregated across {len(shop_results)} parts in {shop}",
                ],
            )
            shop_summaries.append(summary)

        # Overall totals
        total_holding_cost = sum(r.total_holding_cost for r in results)
        total_safety_stock_value = sum(r.safety_stock_value for r in results)
        total_cycle_stock_value = sum(r.cycle_stock_value for r in results)

        return {
            "part_results": results,
            "family_summaries": family_summaries,
            "shop_summaries": shop_summaries,
            "totals": {
                "total_parts": len(results),
                "total_holding_cost": total_holding_cost,
                "total_safety_stock_value": total_safety_stock_value,
                total_cycle_stock_value: total_cycle_stock_value,
                "total_average_inventory_value": total_safety_stock_value + total_cycle_stock_value,
            },
        }


def calculate_holding_costs(
    inventory_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    holding_cost_rate: float = 0.20,
    order_cost: float = 75.0,
) -> dict[str, Any]:
    """Convenience function to calculate holding costs."""
    calculator = HoldingCostCalculator(holding_cost_rate=holding_cost_rate, order_cost=order_cost)
    return calculator.calculate_all_holding_costs(inventory_df, safety_stock_df)
