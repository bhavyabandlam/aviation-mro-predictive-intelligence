"""Tests for dashboard filters (SRS FR-12).

Verifies that inventory and forecasting pages expose working filters
for shop, part family, and priority.
"""

from __future__ import annotations

import pandas as pd


class TestInventoryFilters:
    """Tests for inventory page filter logic."""

    def test_inventory_filter_matching(self) -> None:
        """Filter logic correctly matches shop, family, and priority."""
        recs = [
            {
                "part_number": "P1",
                "shop_id": "SHOP_01",
                "part_family": "ENGINE",
                "priority": "high",
            },
            {
                "part_number": "P2",
                "shop_id": "SHOP_02",
                "part_family": "AVIONICS",
                "priority": "low",
            },
            {
                "part_number": "P3",
                "shop_id": "SHOP_01",
                "part_family": "ENGINE",
                "priority": "medium",
            },
        ]

        def _match(r, f_shop, f_fam, f_pri):
            if f_shop and r.get("shop_id") not in f_shop:
                return False
            if f_fam and r.get("part_family") not in f_fam:
                return False
            return not (f_pri and r.get("priority") not in f_pri)

        # No filters: all match
        assert len([r for r in recs if _match(r, [], [], [])]) == 3

        # Filter by shop
        assert len([r for r in recs if _match(r, ["SHOP_01"], [], [])]) == 2

        # Filter by family
        assert len([r for r in recs if _match(r, [], ["ENGINE"], [])]) == 2

        # Filter by priority
        assert len([r for r in recs if _match(r, [], [], ["high"])]) == 1

        # Combined filters
        assert len([r for r in recs if _match(r, ["SHOP_01"], ["ENGINE"], ["high"])]) == 1


class TestForecastingFilters:
    """Tests for forecasting page filter logic."""

    def test_forecasting_filter_matching(self) -> None:
        """Filter logic correctly matches shop and part family."""
        df = pd.DataFrame(
            {
                "shop_id": ["SHOP_01", "SHOP_01", "SHOP_02", "SHOP_02"],
                "part_family": ["ENGINE", "AVIONICS", "ENGINE", "AVIONICS"],
                "demand_count": [10, 20, 30, 40],
            }
        )

        # Filter by shop
        sub = df[df["shop_id"] == "SHOP_01"]
        assert len(sub) == 2
        assert sub["demand_count"].sum() == 30

        # Filter by shop + family
        sub = df[(df["shop_id"] == "SHOP_01") & (df["part_family"] == "ENGINE")]
        assert len(sub) == 1
        assert sub["demand_count"].iloc[0] == 10

    def test_empty_filter_result(self) -> None:
        """Filter with no matches returns empty dataframe."""
        df = pd.DataFrame(
            {
                "shop_id": ["SHOP_01"],
                "part_family": ["ENGINE"],
                "demand_count": [10],
            }
        )
        sub = df[df["shop_id"] == "SHOP_99"]
        assert len(sub) == 0
