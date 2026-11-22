"""Slow-Moving and Non-Moving Inventory Identification.

Identifies inventory items with little or no demand for inventory rationalization.
All calculations use SYNTHETIC data — not real airline records.

Classifications:
- Non-moving: zero demand (demand_mean_weekly == 0)
- Slow-moving: demand below threshold (configurable, default < 1 unit/week)
- Critical slow-moving: slow-moving parts with criticality A or B
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"

# Default thresholds
SLOW_MOVING_DEMAND_THRESHOLD = 1.0  # units per week
NON_MOVING_DEMAND_THRESHOLD = 0.0  # units per week
LOW_TURNOVER_THRESHOLD = 1.0  # turns per year


@dataclass
class SlowMovingResult:
    """Result of slow-moving analysis for a single part."""

    part_number: str
    part_family: str
    shop_id: str
    criticality: str
    current_stock: float
    unit_cost: float
    inventory_value: float
    demand_mean_weekly: float
    demand_std_weekly: float
    lead_time_days: float
    service_level_target: float
    classification: str  # "non_moving", "slow_moving", "normal"
    weeks_of_supply: float  # current_stock / demand_mean_weekly (inf if zero demand)
    annual_demand_cost: float  # demand_mean_weekly * 52 * unit_cost
    holding_cost_annual: float  # inventory_value * holding_cost_rate
    is_critical_slow_moving: bool
    recommendation: str
    assumptions: list[str]


class SlowMovingAnalyzer:
    """Identifies slow-moving and non-moving inventory items."""

    def __init__(
        self,
        slow_moving_threshold: float = SLOW_MOVING_DEMAND_THRESHOLD,
        holding_cost_rate: float = 0.20,
        weeks_per_year: float = 52.0,
    ):
        """
        Initialize analyzer.

        Args:
            slow_moving_threshold: Demand (units/week) below this = slow-moving
            holding_cost_rate: Annual holding cost as fraction of unit cost
            weeks_per_year: Number of weeks in a year
        """
        self.slow_moving_threshold = slow_moving_threshold
        self.holding_cost_rate = holding_cost_rate
        self.weeks_per_year = weeks_per_year

    def classify(
        self,
        demand_mean_weekly: float,
        criticality: str,
    ) -> str:
        """Classify a part as non_moving, slow_moving, or normal."""
        if demand_mean_weekly <= NON_MOVING_DEMAND_THRESHOLD:
            return "non_moving"
        if demand_mean_weekly < self.slow_moving_threshold:
            return "slow_moving"
        return "normal"

    def weeks_of_supply(
        self,
        current_stock: float,
        demand_mean_weekly: float,
    ) -> float:
        """Calculate weeks of supply (infinite if zero demand)."""
        if demand_mean_weekly <= 0:
            return float("inf")
        return current_stock / demand_mean_weekly

    def analyze_for_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[SlowMovingResult]:
        """Analyze all parts for slow-moving status."""
        results = []

        for _, row in df.iterrows():
            part_num = row["part_number"]
            demand_mean = row["demand_mean_weekly"]
            current_stock = row["current_stock"]
            unit_cost = row["unit_cost"]
            criticality = row["criticality"]

            classification = self.classify(demand_mean, criticality)
            wos = self.weeks_of_supply(current_stock, demand_mean)
            inventory_value = current_stock * unit_cost
            annual_demand_cost = demand_mean * self.weeks_per_year * unit_cost
            holding_cost_annual = inventory_value * self.holding_cost_rate
            is_critical_slow = classification in ("slow_moving", "non_moving") and criticality in (
                "A",
                "B",
            )

            if classification == "non_moving":
                recommendation = "Consider discontinuing or returning stock — zero demand recorded"
            elif classification == "slow_moving":
                recommendation = "Review stock levels — reduce reorder quantity or return excess"
            else:
                recommendation = "No action — normal demand pattern"

            assumptions = [
                f"Slow-moving threshold: < {self.slow_moving_threshold} units/week",
                f"Non-moving threshold: <= {NON_MOVING_DEMAND_THRESHOLD} units/week",
                f"Holding cost rate: {self.holding_cost_rate:.0%} per year",
                f"Weeks per year: {self.weeks_per_year}",
                "All data is synthetic — not real airline records",
            ]

            results.append(
                SlowMovingResult(
                    part_number=part_num,
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    criticality=criticality,
                    current_stock=current_stock,
                    unit_cost=unit_cost,
                    inventory_value=inventory_value,
                    demand_mean_weekly=demand_mean,
                    demand_std_weekly=row["demand_std_weekly"],
                    lead_time_days=row["lead_time_days"],
                    service_level_target=row.get("service_level_target", 0.95),
                    classification=classification,
                    weeks_of_supply=wos,
                    annual_demand_cost=annual_demand_cost,
                    holding_cost_annual=holding_cost_annual,
                    is_critical_slow_moving=is_critical_slow,
                    recommendation=recommendation,
                    assumptions=assumptions,
                )
            )

        return results


def analyze_slow_moving_inventory(
    df: pd.DataFrame,
    slow_moving_threshold: float = SLOW_MOVING_DEMAND_THRESHOLD,
    holding_cost_rate: float = 0.20,
) -> list[SlowMovingResult]:
    """Convenience function to analyze slow-moving inventory."""
    analyzer = SlowMovingAnalyzer(
        slow_moving_threshold=slow_moving_threshold,
        holding_cost_rate=holding_cost_rate,
    )
    return analyzer.analyze_for_dataframe(df)


def summarize_slow_moving(
    results: list[SlowMovingResult],
) -> dict[str, Any]:
    """Summarize slow-moving analysis results."""
    if not results:
        return {
            "total_parts": 0,
            "non_moving_count": 0,
            "slow_moving_count": 0,
            "normal_count": 0,
            "critical_slow_moving_count": 0,
            "total_inventory_value": 0.0,
            "slow_moving_inventory_value": 0.0,
            "non_moving_inventory_value": 0.0,
            "total_annual_holding_cost": 0.0,
            "by_classification": {},
            "by_family": {},
            "by_shop": {},
        }

    df = pd.DataFrame(
        [
            {
                "part_number": r.part_number,
                "part_family": r.part_family,
                "shop_id": r.shop_id,
                "criticality": r.criticality,
                "classification": r.classification,
                "inventory_value": r.inventory_value,
                "current_stock": r.current_stock,
                "demand_mean_weekly": r.demand_mean_weekly,
                "weeks_of_supply": r.weeks_of_supply,
                "annual_demand_cost": r.annual_demand_cost,
                "holding_cost_annual": r.holding_cost_annual,
                "is_critical_slow_moving": r.is_critical_slow_moving,
            }
            for r in results
        ]
    )

    by_classification = (
        df.groupby("classification")
        .agg(
            count=("part_number", "count"),
            inventory_value=("inventory_value", "sum"),
            holding_cost=("holding_cost_annual", "sum"),
        )
        .reset_index()
        .to_dict("records")
    )

    by_family = (
        df.groupby("part_family")
        .agg(
            count=("part_number", "count"),
            inventory_value=("inventory_value", "sum"),
            slow_count=("classification", lambda x: (x == "slow_moving").sum()),
            non_moving_count=("classification", lambda x: (x == "non_moving").sum()),
        )
        .reset_index()
        .to_dict("records")
    )

    by_shop = (
        df.groupby("shop_id")
        .agg(
            count=("part_number", "count"),
            inventory_value=("inventory_value", "sum"),
            slow_count=("classification", lambda x: (x == "slow_moving").sum()),
            non_moving_count=("classification", lambda x: (x == "non_moving").sum()),
        )
        .reset_index()
        .to_dict("records")
    )

    non_moving = df[df["classification"] == "non_moving"]
    slow_moving = df[df["classification"] == "slow_moving"]

    return {
        "total_parts": len(df),
        "non_moving_count": len(non_moving),
        "slow_moving_count": len(slow_moving),
        "normal_count": len(df[df["classification"] == "normal"]),
        "critical_slow_moving_count": int(df["is_critical_slow_moving"].sum()),
        "total_inventory_value": float(df["inventory_value"].sum()),
        "slow_moving_inventory_value": float(slow_moving["inventory_value"].sum()),
        "non_moving_inventory_value": float(non_moving["inventory_value"].sum()),
        "total_annual_holding_cost": float(df["holding_cost_annual"].sum()),
        "by_classification": by_classification,
        "by_family": by_family,
        "by_shop": by_shop,
        "non_moving_parts": non_moving[
            ["part_number", "part_family", "shop_id", "inventory_value"]
        ].to_dict("records"),
        "critical_slow_moving_parts": df[df["is_critical_slow_moving"]][
            ["part_number", "part_family", "shop_id", "criticality", "inventory_value"]
        ].to_dict("records"),
    }
