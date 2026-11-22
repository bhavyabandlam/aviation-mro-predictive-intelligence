"""Reorder Point Calculator.

Calculates reorder points for inventory management using standard formulas.
All calculations use SYNTHETIC data — not real airline records.

Formulas:
- Reorder Point (ROP) = Lead Time Demand + Safety Stock
- Lead Time Demand = Mean Weekly Demand × Lead Time (weeks)
- Safety Stock = Z × σ_LTD (where Z is service level z-score)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"


@dataclass
class ReorderPointResult:
    """Result of reorder point calculation."""

    part_number: str
    part_family: str
    shop_id: str
    lead_time_weeks: float
    mean_weekly_demand: float
    std_weekly_demand: float
    service_level: float
    z_score: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    assumptions: list[str]


class ReorderPointCalculator:
    """Calculates reorder points for inventory items."""

    def __init__(self, default_service_level: float = 0.95):
        """
        Initialize calculator.

        Args:
            default_service_level: Default service level (0-1) if not specified per part
        """
        self.default_service_level = default_service_level

    def z_score(self, service_level: float) -> float:
        """Get z-score for given service level (normal distribution)."""
        if service_level <= 0 or service_level >= 1:
            raise ValueError("Service level must be between 0 and 1")
        return float(stats.norm.ppf(service_level))

    def calculate_lead_time_demand(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
    ) -> tuple[float, float]:
        """
        Calculate lead time demand mean and standard deviation.

        Assumes weekly demand is i.i.d. normal.
        Lead time in weeks = lead_time_days / 7.

        Returns:
            (mean_ltd, std_ltd)
        """
        lead_time_weeks = lead_time_days / 7.0
        mean_ltd = mean_weekly_demand * lead_time_weeks

        # Variance of sum of i.i.d. demands over lead time
        # Var(LTD) = E[L] * Var(D) + Var(L) * E[D]^2
        # For constant lead time: Var(LTD) = L * Var(D)
        std_ltd = std_weekly_demand * np.sqrt(lead_time_weeks)

        return mean_ltd, std_ltd

    def calculate_safety_stock(
        self,
        std_ltd: float,
        service_level: float,
    ) -> float:
        """Calculate safety stock using normal distribution."""
        z = self.z_score(service_level)
        return z * std_ltd

    def calculate_reorder_point(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
        service_level: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Calculate reorder point components.

        Returns:
            (lead_time_demand, safety_stock, reorder_point, z_score)
        """
        sl = service_level or self.default_service_level
        z = self.z_score(sl)

        mean_ltd, std_ltd = self.calculate_lead_time_demand(
            mean_weekly_demand, std_weekly_demand, lead_time_days
        )

        safety_stock = self.calculate_safety_stock(std_ltd, sl)
        reorder_point = mean_ltd + safety_stock

        return mean_ltd, safety_stock, reorder_point, z

    def calculate_for_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[ReorderPointResult]:
        """Calculate reorder points for all parts in DataFrame."""
        results = []

        for _, row in df.iterrows():
            sl = row.get("service_level_target", self.default_service_level)
            # Handle NaN from pandas Series.get()
            if pd.isna(sl):
                sl = self.default_service_level

            mean_ltd, safety_stock, rop, z = self.calculate_reorder_point(
                mean_weekly_demand=row["demand_mean_weekly"],
                std_weekly_demand=row["demand_std_weekly"],
                lead_time_days=row["lead_time_days"],
                service_level=sl,
            )

            assumptions = [
                "Weekly demand is i.i.d. normal (assumption for synthetic data)",
                "Lead time is constant (using mean lead time only)",
                f"Service level target: {sl:.0%}",
                f"Z-score (normal): {z:.3f}",
                "Safety stock = Z × σ_LTD",
                "ROP = Lead Time Demand + Safety Stock",
            ]

            results.append(
                ReorderPointResult(
                    part_number=row["part_number"],
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    lead_time_weeks=row["lead_time_days"] / 7.0,
                    mean_weekly_demand=row["demand_mean_weekly"],
                    std_weekly_demand=row["demand_std_weekly"],
                    service_level=sl,
                    z_score=z,
                    lead_time_demand=mean_ltd,
                    safety_stock=safety_stock,
                    reorder_point=rop,
                    assumptions=assumptions,
                )
            )

        return results


def calculate_reorder_points(
    df: pd.DataFrame,
    default_service_level: float = 0.95,
) -> list[ReorderPointResult]:
    """Convenience function to calculate reorder points."""
    calculator = ReorderPointCalculator(default_service_level)
    return calculator.calculate_for_dataframe(df)
