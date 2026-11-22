"""Tests for inventory demand analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.demand_analysis import (
    DemandStatistics,
    PartsDemandAnalyzer,
    analyze_parts_demand,
)


class TestDemandStatistics:
    """Test DemandStatistics dataclass."""

    def test_creation(self):
        stat = DemandStatistics(
            part_number="PART-0001",
            part_family="ENGINE",
            shop_id="SHOP_01",
            mean_weekly=10.0,
            std_weekly=2.0,
            cv=0.2,
            min_weekly=6.0,
            max_weekly=14.0,
            zero_demand_weeks=0,
            total_weeks=52,
            intermittency=0.0,
            demand_pattern="smooth",
        )
        assert stat.part_number == "PART-0001"
        assert stat.demand_pattern == "smooth"


class TestPartsDemandAnalyzer:
    """Test PartsDemandAnalyzer class."""

    def test_classify_smooth(self):
        """Test smooth demand classification."""
        analyzer = PartsDemandAnalyzer()
        # Low CV, low intermittency
        assert analyzer.classify_demand_pattern(0.2, 0.1) == "smooth"

    def test_classify_erratic(self):
        """Test erratic demand classification."""
        analyzer = PartsDemandAnalyzer()
        # High CV, low intermittency
        assert analyzer.classify_demand_pattern(0.8, 0.1) == "erratic"

    def test_classify_intermittent(self):
        """Test intermittent demand classification."""
        analyzer = PartsDemandAnalyzer()
        # Low CV, high intermittency
        assert analyzer.classify_demand_pattern(0.2, 0.5) == "intermittent"

    def test_classify_lumpy(self):
        """Test lumpy demand classification."""
        analyzer = PartsDemandAnalyzer()
        # High CV, high intermittency
        assert analyzer.classify_demand_pattern(0.8, 0.5) == "lumpy"

    def test_compute_statistics(self):
        """Test statistics computation."""
        analyzer = PartsDemandAnalyzer()
        df = self._create_test_df(5)
        stats = analyzer.compute_statistics(df)

        assert len(stats) == 5
        assert all(isinstance(s, DemandStatistics) for s in stats)
        assert all(s.cv >= 0 for s in stats)
        assert all(
            s.demand_pattern in ["smooth", "erratic", "intermittent", "lumpy"] for s in stats
        )

    def test_analyze_by_family(self):
        """Test aggregation by family."""
        analyzer = PartsDemandAnalyzer()
        df = self._create_test_df(10)
        stats = analyzer.compute_statistics(df)
        by_family = analyzer.analyze_by_family(stats)

        assert len(by_family) == 3  # 3 families
        assert "n_parts" in by_family.columns
        assert "avg_cv" in by_family.columns
        assert "smooth_count" in by_family.columns

    def test_analyze_by_shop(self):
        """Test aggregation by shop."""
        analyzer = PartsDemandAnalyzer()
        df = self._create_test_df(20)  # More parts to ensure all shops
        stats = analyzer.compute_statistics(df)
        by_shop = analyzer.analyze_by_shop(stats)

        assert len(by_shop) <= 5  # At most 5 shops
        assert len(by_shop) >= 3  # At least some shops
        assert "n_parts" in by_shop.columns

    def _create_test_df(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(["ENGINE", "AVIONICS", "AIRFRAME"], n),
                "shop_id": np.random.choice(
                    ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"], n
                ),
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
            }
        )


class TestAnalyzePartsDemand:
    """Test convenience function."""

    def test_function_returns_dict(self):
        """Test analyze_parts_demand returns expected structure."""
        df = self._create_test_df(10)
        result = analyze_parts_demand(df)

        assert "statistics" in result
        assert "by_family" in result
        assert "by_shop" in result
        assert "n_parts" in result
        assert "pattern_distribution" in result
        assert result["n_parts"] == 10

    def _create_test_df(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(["ENGINE", "AVIONICS", "AIRFRAME"], n),
                "shop_id": np.random.choice(
                    ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"], n
                ),
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
