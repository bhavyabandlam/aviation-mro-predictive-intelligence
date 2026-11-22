"""Planned vs Unplanned Maintenance Cost Analysis.

Categorizes synthetic MRO maintenance costs into planned (scheduled,
preventive) and unplanned (breakdown, reactive, AOG) categories.

All calculations use SYNTHETIC data — not real airline records.

Assumptions (synthetic, documented, not observed real-world ratios):
- Planned share by part family: ENGINE 70%, AVIONICS 40%, AIRFRAME 50%.
  These reflect that engine overhauls are typically scheduled while
  avionics and airframe repairs are more often reactive.
- Overall default planned share: 60% when no family-specific ratio applies.

Formulas:
- Planned Cost = Category Cost × Planned Share
- Unplanned Cost = Category Cost × (1 - Planned Share)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"

# Default planned/unplanned split by part family (synthetic assumptions).
DEFAULT_PLANNED_SHARE_BY_FAMILY: dict[str, float] = {
    "ENGINE": 0.70,
    "AVIONICS": 0.40,
    "AIRFRAME": 0.50,
}

DEFAULT_PLANNED_SHARE = 0.60


@dataclass
class PlannedUnplannedResult:
    """Result of planned vs unplanned cost split for one category."""

    category: str
    total_cost: float
    planned_cost: float
    unplanned_cost: float
    planned_share: float
    assumptions: list[str]


class PlannedUnplannedAnalyzer:
    """Analyzes planned vs unplanned maintenance cost split."""

    def __init__(
        self,
        planned_share_by_family: dict[str, float] | None = None,
        default_planned_share: float = DEFAULT_PLANNED_SHARE,
    ):
        self.planned_share_by_family = (
            planned_share_by_family
            if planned_share_by_family is not None
            else DEFAULT_PLANNED_SHARE_BY_FAMILY
        )
        self.default_planned_share = default_planned_share

    def _planned_share(self, part_family: str) -> float:
        """Return planned share for a part family."""
        return self.planned_share_by_family.get(part_family, self.default_planned_share)

    def split_category(
        self,
        category: str,
        total_cost: float,
        part_family: str | None = None,
    ) -> PlannedUnplannedResult:
        """Split one cost category into planned and unplanned portions."""
        planned_share = (
            self._planned_share(part_family) if part_family else self.default_planned_share
        )
        planned_cost = total_cost * planned_share
        unplanned_cost = total_cost - planned_cost

        family_share = self.planned_share_by_family.get(
            part_family or "", self.default_planned_share
        )

        return PlannedUnplannedResult(
            category=category,
            total_cost=round(total_cost, 2),
            planned_cost=round(planned_cost, 2),
            unplanned_cost=round(unplanned_cost, 2),
            planned_share=planned_share,
            assumptions=[
                f"Category: {category}",
                f"Part family: {part_family or 'n/a'}",
                f"Planned share: {planned_share:.0%} "
                f"({family_share:.0%}"
                f" by family"
                f"{'+ default' if part_family not in self.planned_share_by_family else ''})",
                "Planned cost = Total cost × Planned share",
                "Unplanned cost = Total cost − Planned cost",
                "Synthetic assumption — not observed real-world maintenance records",
            ],
        )

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        """Get a value from either a dict or an object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def split_all_categories(
        self,
        labor_results: dict[str, Any],
        parts_results: dict[str, Any],
        downtime_results: dict[str, Any],
        holding_results: dict[str, Any],
        stockout_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Split all cost categories into planned and unplanned."""
        results: list[PlannedUnplannedResult] = []

        # Labor costs by shop-family
        for r in labor_results.get("shop_family_results", []):
            results.append(
                self.split_category(
                    category=f"Labor ({self._get(r, 'shop_id')} / {self._get(r, 'part_family')})",
                    total_cost=self._get(r, "annual_labor_cost", 0.0),
                    part_family=self._get(r, "part_family"),
                )
            )

        # Parts costs by part
        for r in parts_results.get("part_results", []):
            category = f"Parts ({self._get(r, 'part_number')} / {self._get(r, 'part_family')})"
            results.append(
                self.split_category(
                    category=category,
                    total_cost=self._get(r, "annual_parts_cost", 0.0),
                    part_family=self._get(r, "part_family"),
                )
            )

        # Downtime costs by fleet-family
        for r in downtime_results.get("fleet_family_results", []):
            category = f"Downtime ({self._get(r, 'fleet_type')} / {self._get(r, 'part_family')})"
            results.append(
                self.split_category(
                    category=category,
                    total_cost=self._get(r, "annual_downtime_cost", 0.0),
                    part_family=self._get(r, "part_family"),
                )
            )

        # Holding costs by part
        for r in holding_results.get("part_results", []):
            category = f"Holding ({self._get(r, 'part_number')} / {self._get(r, 'part_family')})"
            results.append(
                self.split_category(
                    category=category,
                    total_cost=self._get(r, "total_holding_cost", 0.0),
                    part_family=self._get(r, "part_family"),
                )
            )

        # Stockout costs by part
        for r in stockout_results.get("part_results", []):
            category = f"Stockout ({self._get(r, 'part_number')} / {self._get(r, 'part_family')})"
            results.append(
                self.split_category(
                    category=category,
                    total_cost=self._get(r, "annual_stockout_cost", 0.0),
                    part_family=self._get(r, "part_family"),
                )
            )

        # Totals
        total_cost = sum(r.total_cost for r in results)
        total_planned = sum(r.planned_cost for r in results)
        total_unplanned = sum(r.unplanned_cost for r in results)

        return {
            "category_results": results,
            "totals": {
                "total_cost": round(total_cost, 2),
                "total_planned_cost": round(total_planned, 2),
                "total_unplanned_cost": round(total_unplanned, 2),
                "planned_share_of_total": (
                    round(total_planned / total_cost, 4) if total_cost > 0 else 0.0
                ),
                "unplanned_share_of_total": (
                    round(total_unplanned / total_cost, 4) if total_cost > 0 else 0.0
                ),
            },
            "assumptions": [
                "Planned maintenance = scheduled, preventive work (known in advance)",
                "Unplanned maintenance = breakdown, reactive, AOG events",
                "Planned share by family (synthetic assumption): "
                + ", ".join(
                    f"{k}={v:.0%}" for k, v in sorted(self.planned_share_by_family.items())
                ),
                f"Default planned share: {self.default_planned_share:.0%}",
                "All data is SYNTHETIC — not real airline records.",
            ],
        }


def analyze_planned_unplanned(
    labor_results: dict[str, Any],
    parts_results: dict[str, Any],
    downtime_results: dict[str, Any],
    holding_results: dict[str, Any],
    stockout_results: dict[str, Any],
    planned_share_by_family: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convenience function for planned vs unplanned analysis."""
    analyzer = PlannedUnplannedAnalyzer(planned_share_by_family=planned_share_by_family)
    return analyzer.split_all_categories(
        labor_results, parts_results, downtime_results, holding_results, stockout_results
    )


if __name__ == "__main__":
    pass
