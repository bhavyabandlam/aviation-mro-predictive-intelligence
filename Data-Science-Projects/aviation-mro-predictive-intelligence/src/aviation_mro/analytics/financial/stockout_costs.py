"""Stockout Cost Calculations.

Calculates stockout-related costs using synthetic data.
All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Expected Stockouts per Year = Stockout Probability × Cycles per Year
- Stockout Cost per Event = Unit Cost × Stockout Cost Multiplier
- Annual Stockout Cost = Expected Stockouts per Year × Stockout Cost per Event
- Stockout Cost Multiplier represents: expediting, AOG, lost revenue, penalties
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"


@dataclass
class StockoutCostResult:
    """Result of stockout cost calculation for a single part."""

    part_number: str
    part_family: str
    shop_id: str
    unit_cost: float
    stockout_probability: float
    cycles_per_year: float
    expected_stockouts_per_year: float
    stockout_cost_multiplier: float
    stockout_cost_per_event: float
    annual_stockout_cost: float
    assumptions: list[str]


@dataclass
class StockoutCostSummary:
    """Summary of stockout costs by group."""

    group_by: str
    group_value: str
    total_parts: int
    total_expected_stockouts_per_year: float
    total_annual_stockout_cost: float
    high_risk_parts: int
    critical_risk_parts: int
    assumptions: list[str]


class StockoutCostCalculator:
    """Calculates stockout-related costs."""

    def __init__(
        self,
        stockout_cost_multiplier: float = 10.0,
    ):
        """
        Initialize calculator.

        Args:
            stockout_cost_multiplier: Multiplier for unit cost to estimate stockout cost per event
        """
        self.stockout_cost_multiplier = stockout_cost_multiplier

    def calculate_stockout_cost(
        self,
        part_number: str,
        part_family: str,
        shop_id: str,
        unit_cost: float,
        stockout_probability: float,
        cycles_per_year: float,
    ) -> StockoutCostResult:
        """Calculate stockout cost for a single part."""
        expected_stockouts_per_year = stockout_probability * cycles_per_year
        stockout_cost_per_event = unit_cost * self.stockout_cost_multiplier
        annual_stockout_cost = expected_stockouts_per_year * stockout_cost_per_event

        assumptions = [
            f"Unit cost: ${unit_cost:.2f}",
            f"Stockout probability: {stockout_probability:.1%}",
            f"Replenishment cycles per year: {cycles_per_year:.1f}",
            f"Expected stockouts/year = {stockout_probability:.1%} × "
            f"{cycles_per_year:.1f} = {expected_stockouts_per_year:.1f}",
            f"Stockout cost multiplier: {self.stockout_cost_multiplier:.0f}x unit cost",
            f"Stockout cost per event = ${unit_cost:.2f} × "
            f"{self.stockout_cost_multiplier:.0f} = ${stockout_cost_per_event:.2f}",
            f"Annual stockout cost = {expected_stockouts_per_year:.1f} × "
            f"${stockout_cost_per_event:.2f} = ${annual_stockout_cost:.2f}",
            "Multiplier includes: expediting, AOG costs, lost revenue, penalties",
        ]

        return StockoutCostResult(
            part_number=part_number,
            part_family=part_family,
            shop_id=shop_id,
            unit_cost=unit_cost,
            stockout_probability=stockout_probability,
            cycles_per_year=cycles_per_year,
            expected_stockouts_per_year=expected_stockouts_per_year,
            stockout_cost_multiplier=self.stockout_cost_multiplier,
            stockout_cost_per_event=stockout_cost_per_event,
            annual_stockout_cost=annual_stockout_cost,
            assumptions=assumptions,
        )

    def calculate_all_stockout_costs(
        self,
        inventory_df: pd.DataFrame,
        stockout_risk_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate stockout costs for all parts."""
        # Create risk lookup
        risk_lookup = {}
        for _, row in stockout_risk_df.iterrows():
            risk_lookup[row["part_number"]] = {
                "stockout_probability": row["stockout_probability"],
                "cycles_per_year": row["cycles_per_year"],
                "risk_category": row["risk_category"],
            }

        results = []
        for _, row in inventory_df.iterrows():
            part_num = row["part_number"]
            risk = risk_lookup.get(part_num, {})
            stockout_prob = risk.get("stockout_probability", 0.0)
            cycles = risk.get("cycles_per_year", 13.0)  # Default monthly

            result = self.calculate_stockout_cost(
                part_number=part_num,
                part_family=row["part_family"],
                shop_id=row["shop_id"],
                unit_cost=row["unit_cost"],
                stockout_probability=stockout_prob,
                cycles_per_year=cycles,
            )
            results.append(result)

        # Summaries by family
        family_summaries = []
        for family in inventory_df["part_family"].unique():
            family_results = [r for r in results if r.part_family == family]
            high_risk = sum(1 for r in family_results if r.stockout_probability > 0.1)
            critical_risk = sum(1 for r in family_results if r.stockout_probability > 0.2)

            summary = StockoutCostSummary(
                group_by="family",
                group_value=family,
                total_parts=len(family_results),
                total_expected_stockouts_per_year=sum(
                    r.expected_stockouts_per_year for r in family_results
                ),
                total_annual_stockout_cost=sum(r.annual_stockout_cost for r in family_results),
                high_risk_parts=high_risk,
                critical_risk_parts=critical_risk,
                assumptions=[
                    f"Aggregated across {len(family_results)} parts in {family} family",
                    f"High risk (P>10%): {high_risk}, Critical risk (P>20%): {critical_risk}",
                ],
            )
            family_summaries.append(summary)

        # Summaries by shop
        shop_summaries = []
        for shop in inventory_df["shop_id"].unique():
            shop_results = [r for r in results if r.shop_id == shop]
            high_risk = sum(1 for r in shop_results if r.stockout_probability > 0.1)
            critical_risk = sum(1 for r in shop_results if r.stockout_probability > 0.2)

            summary = StockoutCostSummary(
                group_by="shop",
                group_value=shop,
                total_parts=len(shop_results),
                total_expected_stockouts_per_year=sum(
                    r.expected_stockouts_per_year for r in shop_results
                ),
                total_annual_stockout_cost=sum(r.annual_stockout_cost for r in shop_results),
                high_risk_parts=high_risk,
                critical_risk_parts=critical_risk,
                assumptions=[
                    f"Aggregated across {len(shop_results)} parts in {shop}",
                    f"High risk (P>10%): {high_risk}, Critical risk (P>20%): {critical_risk}",
                ],
            )
            shop_summaries.append(summary)

        # Overall totals
        total_annual_stockout_cost = sum(r.annual_stockout_cost for r in results)
        total_expected_stockouts = sum(r.expected_stockouts_per_year for r in results)
        total_high_risk = sum(1 for r in results if r.stockout_probability > 0.1)
        total_critical_risk = sum(1 for r in results if r.stockout_probability > 0.2)

        return {
            "part_results": results,
            "family_summaries": family_summaries,
            "shop_summaries": shop_summaries,
            "totals": {
                "total_parts": len(results),
                "total_annual_stockout_cost": total_annual_stockout_cost,
                "total_expected_stockouts_per_year": total_expected_stockouts,
                "high_risk_parts": total_high_risk,
                "critical_risk_parts": total_critical_risk,
            },
        }


def calculate_stockout_costs(
    inventory_df: pd.DataFrame,
    stockout_risk_df: pd.DataFrame,
    stockout_cost_multiplier: float = 10.0,
) -> dict[str, Any]:
    """Convenience function to calculate stockout costs."""
    calculator = StockoutCostCalculator(stockout_cost_multiplier=stockout_cost_multiplier)
    return calculator.calculate_all_stockout_costs(inventory_df, stockout_risk_df)
