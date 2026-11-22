"""Inventory Recommendations Engine.

Generates actionable inventory recommendations based on analytics.
All recommendations use SYNTHETIC data — not real airline records.

Recommendations include:
- Reorder point adjustments
- Safety stock adjustments
- Order quantity suggestions (EOQ-based)
- Criticality-based prioritization
- Cost-benefit analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"


@dataclass
class InventoryRecommendation:
    """Single inventory recommendation."""

    part_number: str
    part_family: str
    shop_id: str
    recommendation_type: str  # "adjust_rop", "adjust_ss", "adjust_qty", "review"
    priority: str  # "Critical", "High", "Medium", "Low"
    current_value: float
    recommended_value: float
    change_pct: float
    rationale: str
    cost_impact: float  # Estimated annual cost impact
    assumptions: list[str]


class InventoryRecommendationEngine:
    """Generates inventory recommendations."""

    def __init__(
        self,
        holding_cost_rate: float = 0.20,  # 20% annual holding cost
        stockout_cost_multiplier: float = 10.0,  # Stockout cost = 10x unit cost
        min_order_quantity: float = 1.0,
    ):
        """
        Initialize recommendation engine.

        Args:
            holding_cost_rate: Annual holding cost as fraction of unit cost
            stockout_cost_multiplier: Multiplier for unit cost to estimate stockout cost
            min_order_quantity: Minimum order quantity
        """
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.min_order_quantity = min_order_quantity

    def calculate_eoq(
        self,
        annual_demand: float,
        unit_cost: float,
        order_cost: float = 50.0,  # Fixed cost per order
    ) -> float:
        """Calculate Economic Order Quantity (EOQ)."""
        if annual_demand <= 0 or unit_cost <= 0:
            return self.min_order_quantity

        holding_cost_per_unit = unit_cost * self.holding_cost_rate
        eoq = np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)
        return max(self.min_order_quantity, float(eoq))

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        reorder_points: list[Any],
        safety_stocks: list[Any],
        stockout_risks: list[Any],
    ) -> list[InventoryRecommendation]:
        """Generate comprehensive recommendations."""
        recommendations = []

        # Create lookups
        rop_lookup = {r.part_number: r for r in reorder_points}
        ss_lookup = {r.part_number: r for r in safety_stocks}
        risk_lookup = {r.part_number: r for r in stockout_risks}

        for _, row in df.iterrows():
            part_num = row["part_number"]
            unit_cost = row["unit_cost"]
            annual_demand = row["demand_mean_weekly"] * 52

            # Current values
            current_stock = row["current_stock"]

            # Reorder point recommendation
            if part_num in rop_lookup:
                rop_result = rop_lookup[part_num]
                recommended_rop = rop_result.reorder_point

                # Check if current stock is below ROP
                if current_stock < recommended_rop:
                    change_pct = (
                        (recommended_rop - current_stock) / current_stock * 100
                        if current_stock > 0
                        else 100
                    )
                    priority = "Critical" if current_stock < recommended_rop * 0.5 else "High"

                    recommendations.append(
                        InventoryRecommendation(
                            part_number=part_num,
                            part_family=row["part_family"],
                            shop_id=row["shop_id"],
                            recommendation_type="adjust_rop",
                            priority=priority,
                            current_value=current_stock,
                            recommended_value=recommended_rop,
                            change_pct=change_pct,
                            rationale=(
                                f"Current stock ({current_stock:.1f}) below reorder point "
                                f"({recommended_rop:.1f}). Place replenishment order."
                            ),
                            cost_impact=0.0,  # Cost impact depends on order
                            assumptions=[
                                "ROP calculated at target service level",
                                "Current stock is on-hand quantity",
                                "No outstanding orders considered",
                            ],
                        )
                    )

            # Safety stock recommendation
            if part_num in ss_lookup:
                ss_result = ss_lookup[part_num]
                recommended_ss = ss_result.safety_stock

                # Estimate current safety stock (current stock - mean LTD)
                mean_ltd = row["demand_mean_weekly"] * (row["lead_time_days"] / 7.0)
                current_ss = max(0, current_stock - mean_ltd)

                if current_ss < recommended_ss * 0.8:  # More than 20% below
                    change_pct = (
                        (recommended_ss - current_ss) / current_ss * 100 if current_ss > 0 else 100
                    )
                    recommendations.append(
                        InventoryRecommendation(
                            part_number=part_num,
                            part_family=row["part_family"],
                            shop_id=row["shop_id"],
                            recommendation_type="adjust_ss",
                            priority="Medium",
                            current_value=current_ss,
                            recommended_value=recommended_ss,
                            change_pct=change_pct,
                            rationale=(
                                f"Estimated current safety stock ({current_ss:.1f}) below "
                                f"target ({recommended_ss:.1f}). Consider increasing buffer."
                            ),
                            cost_impact=(
                                (recommended_ss - current_ss) * unit_cost * self.holding_cost_rate
                            ),
                            assumptions=[
                                "Safety stock = current stock - mean lead time demand",
                                "Holding cost rate applied to incremental safety stock",
                            ],
                        )
                    )

            # Order quantity (EOQ) recommendation
            order_cost = 50.0  # Assumed fixed order cost
            eoq = self.calculate_eoq(annual_demand, unit_cost, order_cost)

            # Current order quantity estimate (if we can infer from stock patterns)
            # For synthetic data, we'll recommend EOQ as review item
            if annual_demand > 0:
                recommendations.append(
                    InventoryRecommendation(
                        part_number=part_num,
                        part_family=row["part_family"],
                        shop_id=row["shop_id"],
                        recommendation_type="adjust_qty",
                        priority="Low",
                        current_value=row.get("order_quantity", eoq),  # Not available, use EOQ
                        recommended_value=eoq,
                        change_pct=0.0,
                        rationale=(
                            f"Economic Order Quantity: {eoq:.1f} units "
                            f"(annual demand: {annual_demand:.1f}, unit cost: ${unit_cost:.2f})"
                        ),
                        cost_impact=0.0,
                        assumptions=[
                            "EOQ = √(2 × D × S / H)",
                            f"Annual demand D = {annual_demand:.1f}",
                            f"Order cost S = ${order_cost:.2f}",
                            f"Holding cost H = {self.holding_cost_rate:.0%} × unit cost",
                        ],
                    )
                )

            # Stockout risk review
            if part_num in risk_lookup:
                risk_result = risk_lookup[part_num]
                if risk_result.risk_category in ["High", "Critical"]:
                    recommendations.append(
                        InventoryRecommendation(
                            part_number=part_num,
                            part_family=row["part_family"],
                            shop_id=row["shop_id"],
                            recommendation_type="review",
                            priority=risk_result.risk_category,
                            current_value=risk_result.stockout_probability,
                            recommended_value=risk_result.service_level_target,
                            change_pct=0.0,
                            rationale=(
                                f"Stockout risk: {risk_result.risk_category} "
                                f"(probability: {risk_result.stockout_probability:.1%}, "
                                f"expected stockouts/year: "
                                f"{risk_result.expected_stockouts_per_year:.1f})"
                            ),
                            cost_impact=(
                                risk_result.expected_stockouts_per_year
                                * unit_cost
                                * self.stockout_cost_multiplier
                            ),
                            assumptions=[
                                "Stockout cost estimated as multiplier of unit cost",
                                "Risk based on normal lead time demand distribution",
                            ],
                        )
                    )

        # Sort by priority
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))

        return recommendations

    def generate_summary_report(
        self,
        recommendations: list[InventoryRecommendation],
    ) -> dict[str, Any]:
        """Generate summary statistics for recommendations."""
        if not recommendations:
            return {
                "total": 0,
                "by_priority": {},
                "by_type": {},
                "total_cost_impact": 0.0,
            }

        df = pd.DataFrame(
            [
                {
                    "part_number": r.part_number,
                    "priority": r.priority,
                    "type": r.recommendation_type,
                    "cost_impact": r.cost_impact,
                }
                for r in recommendations
            ]
        )

        return {
            "total": len(recommendations),
            "by_priority": df["priority"].value_counts().to_dict(),
            "by_type": df["type"].value_counts().to_dict(),
            "total_cost_impact": float(df["cost_impact"].sum()),
            "critical_parts": df[df["priority"] == "Critical"]["part_number"].tolist(),
        }


def generate_inventory_recommendations(
    df: pd.DataFrame,
    reorder_points: list[Any],
    safety_stocks: list[Any],
    stockout_risks: list[Any],
    holding_cost_rate: float = 0.20,
) -> list[InventoryRecommendation]:
    """Convenience function to generate recommendations."""
    engine = InventoryRecommendationEngine(holding_cost_rate=holding_cost_rate)
    return engine.generate_recommendations(df, reorder_points, safety_stocks, stockout_risks)
