"""Parts Demand Analysis.

Analyzes synthetic parts demand patterns for inventory management.
All data is SYNTHETIC — not real airline records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"


@dataclass
class DemandStatistics:
    """Demand statistics for a part."""

    part_number: str
    part_family: str
    shop_id: str
    mean_weekly: float
    std_weekly: float
    cv: float  # Coefficient of variation
    min_weekly: float
    max_weekly: float
    zero_demand_weeks: int
    total_weeks: int
    intermittency: float  # Fraction of weeks with zero demand
    demand_pattern: str  # "smooth", "erratic", "intermittent", "lumpy"


class PartsDemandAnalyzer:
    """Analyzes parts demand patterns for inventory classification."""

    def __init__(
        self,
        cv_threshold_smooth: float = 0.49,
        cv_threshold_erratic: float = 1.0,
        intermittency_threshold: float = 0.3,
    ):
        """
        Initialize demand analyzer.

        Args:
            cv_threshold_smooth: CV below this = smooth demand
            cv_threshold_erratic: CV above this = erratic demand
            intermittency_threshold: Intermittency above this = intermittent
        """
        self.cv_threshold_smooth = cv_threshold_smooth
        self.cv_threshold_erratic = cv_threshold_erratic
        self.intermittency_threshold = intermittency_threshold

    def classify_demand_pattern(self, cv: float, intermittency: float) -> str:
        """Classify demand pattern using Syntetos-Boylan approximation."""
        if intermittency <= self.intermittency_threshold:
            if cv <= self.cv_threshold_smooth:
                return "smooth"
            else:
                return "erratic"
        else:
            if cv <= self.cv_threshold_smooth:
                return "intermittent"
            else:
                return "lumpy"

    def compute_statistics(self, df: pd.DataFrame) -> list[DemandStatistics]:
        """Compute demand statistics for all parts."""
        stats_list = []

        for _, row in df.iterrows():
            mean_w = row["demand_mean_weekly"]
            std_w = row["demand_std_weekly"]
            cv = std_w / mean_w if mean_w > 0 else 0.0

            # Estimate zero demand weeks from intermittency
            # For synthetic data, we approximate
            intermittency = min(1.0, cv * 0.5) if mean_w > 0 else 1.0
            total_weeks = 52  # Annual horizon
            zero_weeks = int(intermittency * total_weeks)

            pattern = self.classify_demand_pattern(cv, intermittency)

            stats_list.append(
                DemandStatistics(
                    part_number=row["part_number"],
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    mean_weekly=mean_w,
                    std_weekly=std_w,
                    cv=cv,
                    min_weekly=max(0, mean_w - 2 * std_w),
                    max_weekly=mean_w + 2 * std_w,
                    zero_demand_weeks=zero_weeks,
                    total_weeks=total_weeks,
                    intermittency=intermittency,
                    demand_pattern=pattern,
                )
            )

        return stats_list

    def analyze_by_family(self, stats: list[DemandStatistics]) -> pd.DataFrame:
        """Aggregate statistics by part family."""
        data = []
        for s in stats:
            data.append(
                {
                    "part_number": s.part_number,
                    "part_family": s.part_family,
                    "shop_id": s.shop_id,
                    "mean_weekly": s.mean_weekly,
                    "std_weekly": s.std_weekly,
                    "cv": s.cv,
                    "intermittency": s.intermittency,
                    "demand_pattern": s.demand_pattern,
                }
            )
        df = pd.DataFrame(data)

        agg = (
            df.groupby("part_family")
            .agg(
                n_parts=("part_number", "count"),
                avg_mean=("mean_weekly", "mean"),
                avg_std=("std_weekly", "mean"),
                avg_cv=("cv", "mean"),
                avg_intermittency=("intermittency", "mean"),
                smooth_count=("demand_pattern", lambda x: (x == "smooth").sum()),
                erratic_count=("demand_pattern", lambda x: (x == "erratic").sum()),
                intermittent_count=("demand_pattern", lambda x: (x == "intermittent").sum()),
                lumpy_count=("demand_pattern", lambda x: (x == "lumpy").sum()),
            )
            .reset_index()
        )

        return agg

    def analyze_by_shop(self, stats: list[DemandStatistics]) -> pd.DataFrame:
        """Aggregate statistics by shop."""
        data = []
        for s in stats:
            data.append(
                {
                    "part_number": s.part_number,
                    "part_family": s.part_family,
                    "shop_id": s.shop_id,
                    "mean_weekly": s.mean_weekly,
                    "std_weekly": s.std_weekly,
                    "cv": s.cv,
                    "intermittency": s.intermittency,
                    "demand_pattern": s.demand_pattern,
                }
            )
        df = pd.DataFrame(data)

        agg = (
            df.groupby("shop_id")
            .agg(
                n_parts=("part_number", "count"),
                avg_mean=("mean_weekly", "mean"),
                avg_std=("std_weekly", "mean"),
                avg_cv=("cv", "mean"),
                avg_intermittency=("intermittency", "mean"),
                smooth_count=("demand_pattern", lambda x: (x == "smooth").sum()),
                erratic_count=("demand_pattern", lambda x: (x == "erratic").sum()),
                intermittent_count=("demand_pattern", lambda x: (x == "intermittent").sum()),
                lumpy_count=("demand_pattern", lambda x: (x == "lumpy").sum()),
            )
            .reset_index()
        )

        return agg


def analyze_parts_demand(df: pd.DataFrame) -> dict[str, Any]:
    """Convenience function to analyze parts demand."""
    analyzer = PartsDemandAnalyzer()
    stats = analyzer.compute_statistics(df)
    by_family = analyzer.analyze_by_family(stats)
    by_shop = analyzer.analyze_by_shop(stats)

    return {
        "statistics": stats,
        "by_family": by_family,
        "by_shop": by_shop,
        "n_parts": len(stats),
        "pattern_distribution": by_family[
            ["part_family", "smooth_count", "erratic_count", "intermittent_count", "lumpy_count"]
        ].to_dict("records"),
    }
