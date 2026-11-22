"""Inventory Valuation and Turnover Calculations.

Calculates inventory value and turnover ratios for inventory management.
All calculations use SYNTHETIC data — not real airline records.

Metrics:
- Inventory value = current stock × unit cost
- ROP value = reorder point × unit cost
- Safety stock value = safety stock × unit cost
- Weeks of supply = current stock / mean weekly demand
- Inventory turnover = annual demand cost / average inventory value
- Days of inventory on hand = 365 / turnover
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"

# Defaults
WEEKS_PER_YEAR = 52.0
DAYS_PER_YEAR = 365.0


@dataclass
class InventoryValuationResult:
    """Valuation result for a single part."""

    part_number: str
    part_family: str
    shop_id: str
    criticality: str
    current_stock: float
    unit_cost: float
    inventory_value: float  # current_stock × unit_cost
    reorder_point: float
    rop_value: float  # reorder_point × unit_cost
    safety_stock: float
    safety_stock_value: float  # safety_stock × unit_cost
    demand_mean_weekly: float
    weeks_of_supply: float  # inf if zero demand
    annual_demand_cost: float  # demand × 52 × unit_cost
    inventory_turnover: float  # annual_demand_cost / avg_inventory_value
    days_of_inventory: float  # 365 / turnover
    assumptions: list[str]


class InventoryValuationCalculator:
    """Calculates inventory value and turnover."""

    def __init__(
        self,
        weeks_per_year: float = WEEKS_PER_YEAR,
        days_per_year: float = DAYS_PER_YEAR,
    ):
        self.weeks_per_year = weeks_per_year
        self.days_per_year = days_per_year

    def calculate_for_part(
        self,
        row: pd.Series,
        reorder_point: float,
        safety_stock: float,
    ) -> InventoryValuationResult:
        """Calculate valuation for a single part."""
        current_stock = row["current_stock"]
        unit_cost = row["unit_cost"]
        demand_mean = row["demand_mean_weekly"]

        inventory_value = current_stock * unit_cost
        rop_value = reorder_point * unit_cost
        ss_value = safety_stock * unit_cost

        # Weeks of supply
        weeks_of_supply = current_stock / demand_mean if demand_mean > 0 else float("inf")

        # Annual demand cost
        annual_demand_cost = demand_mean * self.weeks_per_year * unit_cost

        # Average inventory value ≈ (current stock + safety stock) / 2
        # Or use current_stock as proxy for average inventory
        avg_inventory_value = max(inventory_value, ss_value)
        if avg_inventory_value > 0:
            inventory_turnover = annual_demand_cost / avg_inventory_value
        else:
            inventory_turnover = 0.0

        # Days of inventory on hand
        if inventory_turnover > 0:
            days_of_inventory = self.days_per_year / inventory_turnover
        else:
            days_of_inventory = float("inf")

        assumptions = [
            f"Weeks per year: {self.weeks_per_year}",
            f"Days per year: {self.days_per_year}",
            "Inventory value = current stock × unit cost",
            "Average inventory value = max(current value, safety stock value)",
            "Inventory turnover = annual demand cost / average inventory value",
            "Days of inventory = days per year / turnover",
            "All data is synthetic — not real airline records",
        ]

        return InventoryValuationResult(
            part_number=row["part_number"],
            part_family=row["part_family"],
            shop_id=row["shop_id"],
            criticality=row["criticality"],
            current_stock=current_stock,
            unit_cost=unit_cost,
            inventory_value=inventory_value,
            reorder_point=reorder_point,
            rop_value=rop_value,
            safety_stock=safety_stock,
            safety_stock_value=ss_value,
            demand_mean_weekly=demand_mean,
            weeks_of_supply=weeks_of_supply,
            annual_demand_cost=annual_demand_cost,
            inventory_turnover=inventory_turnover,
            days_of_inventory=days_of_inventory,
            assumptions=assumptions,
        )

    def calculate_for_dataframe(
        self,
        df: pd.DataFrame,
        reorder_points: dict[str, float],
        safety_stocks: dict[str, float],
    ) -> list[InventoryValuationResult]:
        """Calculate valuation for all parts."""
        results = []

        for _, row in df.iterrows():
            part_num = row["part_number"]
            rop = reorder_points.get(part_num, 0.0)
            ss = safety_stocks.get(part_num, 0.0)

            results.append(self.calculate_for_part(row, rop, ss))

        return results


def calculate_inventory_valuation(
    df: pd.DataFrame,
    reorder_points: dict[str, float],
    safety_stocks: dict[str, float],
) -> list[InventoryValuationResult]:
    """Convenience function to calculate inventory valuation."""
    calculator = InventoryValuationCalculator()
    return calculator.calculate_for_dataframe(df, reorder_points, safety_stocks)


def summarize_valuation(
    results: list[InventoryValuationResult],
) -> dict[str, Any]:
    """Summarize inventory valuation results."""
    if not results:
        return {
            "total_parts": 0,
            "total_inventory_value": 0.0,
            "total_rop_value": 0.0,
            "total_safety_stock_value": 0.0,
            "total_annual_demand_cost": 0.0,
            "overall_turnover": 0.0,
            "overall_days_of_inventory": 0.0,
            "by_family": [],
            "by_shop": [],
            "by_criticality": [],
        }

    df = pd.DataFrame(
        [
            {
                "part_number": r.part_number,
                "part_family": r.part_family,
                "shop_id": r.shop_id,
                "criticality": r.criticality,
                "inventory_value": r.inventory_value,
                "rop_value": r.rop_value,
                "safety_stock_value": r.safety_stock_value,
                "annual_demand_cost": r.annual_demand_cost,
                "inventory_turnover": r.inventory_turnover,
                "days_of_inventory": r.days_of_inventory,
                "weeks_of_supply": r.weeks_of_supply,
                "current_stock": r.current_stock,
                "unit_cost": r.unit_cost,
            }
            for r in results
        ]
    )

    # Replace inf with NaN for aggregation
    df_agg = df.replace([np.inf, -np.inf], np.nan)

    total_inv_value = float(df["inventory_value"].sum())
    total_rop_value = float(df["rop_value"].sum())
    total_ss_value = float(df["safety_stock_value"].sum())
    total_annual_demand_cost = float(df["annual_demand_cost"].sum())

    # Overall turnover = total annual demand cost / total average inventory value
    overall_turnover = total_annual_demand_cost / total_inv_value if total_inv_value > 0 else 0.0

    if overall_turnover > 0:
        overall_days = df_agg["days_of_inventory"].replace([np.inf], np.nan).mean()
        if pd.isna(overall_days):
            overall_days = float("inf")
    else:
        overall_days = float("inf")

    def _group_agg(group_col: str) -> list[dict[str, Any]]:
        agg = (
            df.groupby(group_col)
            .agg(
                n_parts=("part_number", "count"),
                inventory_value=("inventory_value", "sum"),
                annual_demand_cost=("annual_demand_cost", "sum"),
                avg_turnover=("inventory_turnover", "mean"),
            )
            .reset_index()
        )
        # Calculate group turnover = group annual demand / group inventory value
        agg["group_turnover"] = agg.apply(
            lambda row: (
                row["annual_demand_cost"] / row["inventory_value"]
                if row["inventory_value"] > 0
                else 0.0
            ),
            axis=1,
        )
        return agg.to_dict("records")  # type: ignore[return-value]

    by_family = _group_agg("part_family")
    by_shop = _group_agg("shop_id")
    by_criticality = _group_agg("criticality")

    return {
        "total_parts": len(df),
        "total_inventory_value": total_inv_value,
        "total_rop_value": total_rop_value,
        "total_safety_stock_value": total_ss_value,
        "total_annual_demand_cost": total_annual_demand_cost,
        "overall_turnover": overall_turnover,
        "overall_days_of_inventory": overall_days,
        "by_family": by_family,
        "by_shop": by_shop,
        "by_criticality": by_criticality,
        "high_value_parts": df.nlargest(10, "inventory_value")[
            ["part_number", "part_family", "shop_id", "inventory_value", "inventory_turnover"]
        ].to_dict("records"),
        "low_turnover_parts": df[df["inventory_turnover"] < 1.0][
            [
                "part_number",
                "part_family",
                "shop_id",
                "inventory_value",
                "inventory_turnover",
                "weeks_of_supply",
            ]
        ].to_dict("records"),
    }
