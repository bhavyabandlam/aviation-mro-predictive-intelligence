"""Stockout Risk Analysis.

Analyzes probability of stockout for inventory items.
All calculations use SYNTHETIC data — not real airline records.

Key metrics:
- Stockout probability = P(Demand during lead time > Reorder point)
- Expected stockouts per year
- Fill rate = 1 - (Expected shortage per cycle / Mean demand per cycle)
- Time-weighted fill rate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"


@dataclass
class StockoutRiskResult:
    """Result of stockout risk analysis."""

    part_number: str
    part_family: str
    shop_id: str
    current_stock: float
    reorder_point: float
    mean_ltd: float
    std_ltd: float
    service_level_target: float
    stockout_probability: float
    expected_shortage_per_cycle: float
    fill_rate: float
    cycles_per_year: float
    expected_stockouts_per_year: float
    risk_category: str  # "Low", "Medium", "High", "Critical"
    assumptions: list[str]


class StockoutRiskAnalyzer:
    """Analyzes stockout risk for inventory items."""

    def __init__(
        self,
        risk_thresholds: dict[str, float] | None = None,
        weeks_per_year: float = 52.0,
    ):
        """
        Initialize analyzer.

        Args:
            risk_thresholds: Dict mapping risk category to max stockout probability
            weeks_per_year: Number of weeks in a year for annualization
        """
        self.risk_thresholds = risk_thresholds or {
            "Low": 0.01,
            "Medium": 0.05,
            "High": 0.10,
            "Critical": 1.0,
        }
        self.weeks_per_year = weeks_per_year

    def categorize_risk(self, stockout_prob: float) -> str:
        """Categorize risk based on stockout probability."""
        for category, threshold in self.risk_thresholds.items():
            if stockout_prob <= threshold:
                return category
        return "Critical"

    def calculate_stockout_probability(
        self,
        mean_ltd: float,
        std_ltd: float,
        reorder_point: float,
        distribution: str = "normal",
    ) -> float:
        """Calculate probability of stockout (demand > reorder point)."""
        if distribution == "normal":
            if std_ltd <= 0:
                return 0.0 if mean_ltd <= reorder_point else 1.0
            z = (reorder_point - mean_ltd) / std_ltd
            return float(1 - stats.norm.cdf(z))
        elif distribution == "lognormal":
            # For lognormal, use scipy's lognorm
            # This requires mu, sigma of the lognormal
            return float("nan")
        else:
            return float("nan")

    def calculate_expected_shortage(
        self,
        mean_ltd: float,
        std_ltd: float,
        reorder_point: float,
        distribution: str = "normal",
    ) -> float:
        """Calculate expected shortage per cycle (E[max(0, LTD - ROP)])."""
        if distribution == "normal":
            if std_ltd <= 0:
                return max(0.0, mean_ltd - reorder_point)

            z = (reorder_point - mean_ltd) / std_ltd
            # Expected shortage = σ × [φ(z) - z × (1 - Φ(z))]
            phi = stats.norm.pdf(z)
            phi_cdf = 1 - stats.norm.cdf(z)
            expected_shortage = std_ltd * (phi - z * phi_cdf)
            return max(0.0, float(expected_shortage))
        return float("nan")

    def calculate_fill_rate(
        self,
        mean_weekly_demand: float,
        expected_shortage_per_cycle: float,
        order_quantity: float | None = None,
    ) -> float:
        """Calculate fill rate (type 2 service level)."""
        if order_quantity is None:
            # Approximate: assume order quantity = 4 weeks demand
            order_quantity = mean_weekly_demand * 4

        if order_quantity <= 0:
            return 0.0

        fill_rate = 1 - (expected_shortage_per_cycle / order_quantity)
        return max(0.0, min(1.0, float(fill_rate)))

    def calculate_cycles_per_year(
        self,
        mean_weekly_demand: float,
        order_quantity: float | None = None,
    ) -> float:
        """Calculate expected order cycles per year."""
        if order_quantity is None:
            order_quantity = mean_weekly_demand * 4

        if order_quantity <= 0 or mean_weekly_demand <= 0:
            return 0.0

        annual_demand = mean_weekly_demand * self.weeks_per_year
        return annual_demand / order_quantity

    def analyze_for_dataframe(
        self,
        df: pd.DataFrame,
        reorder_points: list[Any] | None = None,
    ) -> list[StockoutRiskResult]:
        """Analyze stockout risk for all parts."""
        results = []

        # Create lookup for reorder points if provided
        rop_lookup = {}
        if reorder_points:
            for r in reorder_points:
                rop_lookup[r.part_number] = r.reorder_point

        for _, row in df.iterrows():
            part_num = row["part_number"]

            # Get reorder point (from calculation or use current stock as proxy)
            if part_num in rop_lookup:
                rop = rop_lookup[part_num]
            else:
                # Estimate ROP from current data
                sl = row.get("service_level_target", 0.95)
                z = stats.norm.ppf(sl)
                mean_ltd = row["demand_mean_weekly"] * (row["lead_time_days"] / 7.0)
                std_ltd = row["demand_std_weekly"] * np.sqrt(row["lead_time_days"] / 7.0)
                rop = mean_ltd + z * std_ltd

            current_stock = row["current_stock"]
            mean_ltd = row["demand_mean_weekly"] * (row["lead_time_days"] / 7.0)
            std_ltd = row["demand_std_weekly"] * np.sqrt(row["lead_time_days"] / 7.0)

            # Account for lead time variability if present
            lt_std = row.get("lead_time_std_days", 0)
            if lt_std > 0:
                lt_weeks = row["lead_time_days"] / 7.0
                lt_std_weeks = lt_std / 7.0
                var_ltd = (
                    lt_weeks * row["demand_std_weekly"] ** 2
                    + row["demand_mean_weekly"] ** 2 * lt_std_weeks**2
                )
                std_ltd = np.sqrt(var_ltd)

            stockout_prob = self.calculate_stockout_probability(mean_ltd, std_ltd, rop)

            expected_shortage = self.calculate_expected_shortage(mean_ltd, std_ltd, rop)

            fill_rate = self.calculate_fill_rate(row["demand_mean_weekly"], expected_shortage)

            cycles_per_year = self.calculate_cycles_per_year(row["demand_mean_weekly"])

            expected_stockouts_per_year = stockout_prob * cycles_per_year

            risk_category = self.categorize_risk(stockout_prob)

            assumptions = [
                "Lead time demand ~ Normal(mean_ltd, std_ltd)",
                f"Reorder point: {rop:.1f}",
                f"Current stock: {current_stock:.1f}",
                f"Service level target: {row.get('service_level_target', 0.95):.0%}",
                "Stockout = P(LTD > ROP)",
                "Fill rate = 1 - E[shortage] / order quantity",
                "Order quantity approximated as 4 weeks demand",
            ]

            results.append(
                StockoutRiskResult(
                    part_number=part_num,
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    current_stock=current_stock,
                    reorder_point=rop,
                    mean_ltd=mean_ltd,
                    std_ltd=std_ltd,
                    service_level_target=row.get("service_level_target", 0.95),
                    stockout_probability=stockout_prob,
                    expected_shortage_per_cycle=expected_shortage,
                    fill_rate=fill_rate,
                    cycles_per_year=cycles_per_year,
                    expected_stockouts_per_year=expected_stockouts_per_year,
                    risk_category=risk_category,
                    assumptions=assumptions,
                )
            )

        return results


def analyze_stockout_risk(
    df: pd.DataFrame,
    reorder_points: list[Any] | None = None,
) -> list[StockoutRiskResult]:
    """Convenience function to analyze stockout risk."""
    analyzer = StockoutRiskAnalyzer()
    return analyzer.analyze_for_dataframe(df, reorder_points)
