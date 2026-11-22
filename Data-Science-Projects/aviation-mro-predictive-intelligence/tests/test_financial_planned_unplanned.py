"""Tests for planned vs unplanned maintenance cost analysis."""

from __future__ import annotations

import pytest

from src.aviation_mro.analytics.financial.planned_unplanned import (
    DEFAULT_PLANNED_SHARE,
    PlannedUnplannedAnalyzer,
    analyze_planned_unplanned,
)


class TestPlannedUnplannedAnalyzer:
    """Test PlannedUnplannedAnalyzer class."""

    def test_split_category_engine(self):
        """Test splitting ENGINE category (70% planned)."""
        analyzer = PlannedUnplannedAnalyzer()
        result = analyzer.split_category(
            category="Labor (SHOP_01 / ENGINE)",
            total_cost=100000.0,
            part_family="ENGINE",
        )
        assert result.category == "Labor (SHOP_01 / ENGINE)"
        assert result.total_cost == 100000.0
        assert result.planned_share == 0.70
        assert result.planned_cost == 70000.0
        assert result.unplanned_cost == 30000.0

    def test_split_category_avionics(self):
        """Test splitting AVIONICS category (40% planned)."""
        analyzer = PlannedUnplannedAnalyzer()
        result = analyzer.split_category(
            category="Parts (PART-001 / AVIONICS)",
            total_cost=50000.0,
            part_family="AVIONICS",
        )
        assert result.planned_share == 0.40
        assert result.planned_cost == 20000.0
        assert result.unplanned_cost == 30000.0

    def test_split_category_airframe(self):
        """Test splitting AIRFRAME category (50% planned)."""
        analyzer = PlannedUnplannedAnalyzer()
        result = analyzer.split_category(
            category="Downtime (NARROW_BODY / AIRFRAME)",
            total_cost=200000.0,
            part_family="AIRFRAME",
        )
        assert result.planned_share == 0.50
        assert result.planned_cost == 100000.0
        assert result.unplanned_cost == 100000.0

    def test_split_category_default_share(self):
        """Test splitting unknown family uses default planned share."""
        analyzer = PlannedUnplannedAnalyzer()
        result = analyzer.split_category(
            category="Holding (PART-999 / UNKNOWN)",
            total_cost=10000.0,
            part_family="UNKNOWN",
        )
        assert result.planned_share == DEFAULT_PLANNED_SHARE
        assert result.planned_cost == 6000.0
        assert result.unplanned_cost == 4000.0

    def test_split_category_no_family(self):
        """Test splitting with no part family uses default planned share."""
        analyzer = PlannedUnplannedAnalyzer()
        result = analyzer.split_category(
            category="General overhead",
            total_cost=50000.0,
        )
        assert result.planned_share == DEFAULT_PLANNED_SHARE
        assert result.planned_cost == 30000.0
        assert result.unplanned_cost == 20000.0

    def test_planned_plus_unplanned_equals_total(self):
        """Test that planned + unplanned equals total cost."""
        analyzer = PlannedUnplannedAnalyzer()
        for family in ["ENGINE", "AVIONICS", "AIRFRAME", "UNKNOWN"]:
            result = analyzer.split_category(
                category=f"Test ({family})",
                total_cost=123456.78,
                part_family=family,
            )
            assert abs(result.planned_cost + result.unplanned_cost - result.total_cost) < 0.01

    def test_zero_cost(self):
        """Test splitting zero cost."""
        analyzer = PlannedUnplannedAnalyzer()
        result = analyzer.split_category(
            category="Zero cost",
            total_cost=0.0,
            part_family="ENGINE",
        )
        assert result.total_cost == 0.0
        assert result.planned_cost == 0.0
        assert result.unplanned_cost == 0.0

    def test_custom_planned_share(self):
        """Test with custom planned share configuration."""
        custom_shares = {"ENGINE": 0.90, "AVIONICS": 0.10}
        analyzer = PlannedUnplannedAnalyzer(planned_share_by_family=custom_shares)
        result = analyzer.split_category(
            category="Test",
            total_cost=10000.0,
            part_family="ENGINE",
        )
        assert result.planned_share == 0.90
        assert result.planned_cost == 9000.0


class TestAnalyzePlannedUnplanned:
    """Test analyze_planned_unplanned convenience function."""

    def _make_results(self):
        """Create mock pipeline results for testing."""
        labor_results = {
            "shop_family_results": [
                {
                    "shop_id": "SHOP_01",
                    "part_family": "ENGINE",
                    "annual_labor_cost": 100000.0,
                    "annual_job_count": 520,
                },
                {
                    "shop_id": "SHOP_01",
                    "part_family": "AVIONICS",
                    "annual_labor_cost": 50000.0,
                    "annual_job_count": 520,
                },
            ],
            "totals": {"total_annual_labor_cost": 150000.0},
        }
        parts_results = {
            "part_results": [
                {
                    "part_number": "PART-001",
                    "part_family": "ENGINE",
                    "annual_parts_cost": 200000.0,
                },
                {
                    "part_number": "PART-002",
                    "part_family": "AVIONICS",
                    "annual_parts_cost": 100000.0,
                },
            ],
            "totals": {"total_annual_parts_cost": 300000.0},
        }
        downtime_results = {
            "fleet_family_results": [
                {
                    "fleet_type": "NARROW_BODY",
                    "part_family": "ENGINE",
                    "annual_downtime_cost": 300000.0,
                },
                {
                    "fleet_type": "WIDE_BODY",
                    "part_family": "AIRFRAME",
                    "annual_downtime_cost": 150000.0,
                },
            ],
            "totals": {"total_annual_downtime_cost": 450000.0},
        }
        holding_results = {
            "part_results": [
                {
                    "part_number": "PART-001",
                    "part_family": "ENGINE",
                    "total_holding_cost": 40000.0,
                },
            ],
            "totals": {"total_holding_cost": 40000.0},
        }
        stockout_results = {
            "part_results": [
                {
                    "part_number": "PART-001",
                    "part_family": "ENGINE",
                    "annual_stockout_cost": 20000.0,
                },
            ],
            "totals": {"total_annual_stockout_cost": 20000.0},
        }
        return labor_results, parts_results, downtime_results, holding_results, stockout_results

    def test_full_analysis(self):
        """Test full planned vs unplanned analysis."""
        (
            labor_results,
            parts_results,
            downtime_results,
            holding_results,
            stockout_results,
        ) = self._make_results()
        result = analyze_planned_unplanned(
            labor_results, parts_results, downtime_results, holding_results, stockout_results
        )
        assert "category_results" in result
        assert "totals" in result
        assert "assumptions" in result
        totals = result["totals"]
        # Total = 100000 + 50000 + 200000 + 100000 + 300000 + 150000 + 40000 + 20000 = 960000
        assert totals["total_cost"] == 960000.0
        # Planned: 70000 + 20000 + 140000 + 40000 + 210000 + 75000 + 28000 + 14000 = 597000
        assert totals["total_planned_cost"] == 597000.0
        # Unplanned: 30000 + 30000 + 60000 + 60000 + 90000 + 75000 + 12000 + 6000 = 363000
        assert totals["total_unplanned_cost"] == 363000.0
        assert (
            abs(
                totals["total_planned_cost"] + totals["total_unplanned_cost"] - totals["total_cost"]
            )
            < 0.01
        )

    def test_planned_share_of_total(self):
        """Test planned share of total calculation."""
        (
            labor_results,
            parts_results,
            downtime_results,
            holding_results,
            stockout_results,
        ) = self._make_results()
        result = analyze_planned_unplanned(
            labor_results, parts_results, downtime_results, holding_results, stockout_results
        )
        totals = result["totals"]
        expected_share = totals["total_planned_cost"] / totals["total_cost"]
        assert abs(totals["planned_share_of_total"] - expected_share) < 0.001

    def test_assumptions_present(self):
        """Test that assumptions are documented."""
        (
            labor_results,
            parts_results,
            downtime_results,
            holding_results,
            stockout_results,
        ) = self._make_results()
        result = analyze_planned_unplanned(
            labor_results, parts_results, downtime_results, holding_results, stockout_results
        )
        assert isinstance(result["assumptions"], list)
        assert len(result["assumptions"]) > 0
        assert any("SYNTHETIC" in a for a in result["assumptions"])

    def test_category_results_count(self):
        """Test that all categories are included."""
        (
            labor_results,
            parts_results,
            downtime_results,
            holding_results,
            stockout_results,
        ) = self._make_results()
        result = analyze_planned_unplanned(
            labor_results, parts_results, downtime_results, holding_results, stockout_results
        )
        # 2 labor + 2 parts + 2 downtime + 1 holding + 1 stockout = 8 categories
        assert len(result["category_results"]) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
