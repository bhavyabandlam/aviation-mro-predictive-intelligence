"""Tests for inventory stockout risk module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.aviation_mro.analytics.inventory.stockout_risk import (
    StockoutRiskAnalyzer,
    StockoutRiskResult,
    analyze_stockout_risk,
)


class TestStockoutRiskAnalyzer:
    """Test StockoutRiskAnalyzer class."""

    def test_categorize_risk(self):
        """Test risk categorization."""
        analyzer = StockoutRiskAnalyzer()
        assert analyzer.categorize_risk(0.005) == "Low"
        assert analyzer.categorize_risk(0.02) == "Medium"
        assert analyzer.categorize_risk(0.07) == "High"
        assert analyzer.categorize_risk(0.15) == "Critical"

    def test_calculate_stockout_probability_normal(self):
        """Test stockout probability with normal distribution."""
        analyzer = StockoutRiskAnalyzer()
        # ROP = mean + 1.645 * std (95% service level)
        mean_ltd = 100.0
        std_ltd = 15.0
        rop = mean_ltd + stats.norm.ppf(0.95) * std_ltd

        prob = analyzer.calculate_stockout_probability(mean_ltd, std_ltd, rop)
        # Should be ~0.05 for 95% service level
        assert abs(prob - 0.05) < 0.01

    def test_calculate_stockout_probability_zero_std(self):
        """Test stockout probability with zero std."""
        analyzer = StockoutRiskAnalyzer()
        # If std = 0, stockout if mean > ROP
        prob1 = analyzer.calculate_stockout_probability(100.0, 0.0, 150.0)
        assert prob1 == 0.0

        prob2 = analyzer.calculate_stockout_probability(100.0, 0.0, 50.0)
        assert prob2 == 1.0

    def test_calculate_expected_shortage(self):
        """Test expected shortage calculation."""
        analyzer = StockoutRiskAnalyzer()
        mean_ltd = 100.0
        std_ltd = 15.0
        rop = mean_ltd + stats.norm.ppf(0.95) * std_ltd

        shortage = analyzer.calculate_expected_shortage(mean_ltd, std_ltd, rop)
        # For 95% service level, expected shortage = σ * [φ(z) - z*(1-Φ(z))]
        z = stats.norm.ppf(0.95)
        expected = std_ltd * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
        assert abs(shortage - expected) < 0.01

    def test_calculate_fill_rate(self):
        """Test fill rate calculation."""
        analyzer = StockoutRiskAnalyzer()
        # Expected shortage = 2.0, order quantity = 40
        fill = analyzer.calculate_fill_rate(
            mean_weekly_demand=10.0,
            expected_shortage_per_cycle=2.0,
            order_quantity=40.0,
        )
        assert abs(fill - 0.95) < 0.01

    def test_calculate_fill_rate_zero_order(self):
        """Test fill rate with zero order quantity."""
        analyzer = StockoutRiskAnalyzer()
        fill = analyzer.calculate_fill_rate(
            mean_weekly_demand=10.0,
            expected_shortage_per_cycle=2.0,
            order_quantity=0.0,
        )
        assert fill == 0.0

    def test_calculate_cycles_per_year(self):
        """Test cycles per year calculation."""
        analyzer = StockoutRiskAnalyzer()
        cycles = analyzer.calculate_cycles_per_year(
            mean_weekly_demand=10.0,
            order_quantity=40.0,
        )
        # Annual demand = 520, Q = 40, cycles = 13
        assert abs(cycles - 13.0) < 0.1

    def test_analyze_for_dataframe(self):
        """Test batch analysis."""
        analyzer = StockoutRiskAnalyzer()
        df = self._create_test_df(5)
        results = analyzer.analyze_for_dataframe(df)

        assert len(results) == 5
        assert all(isinstance(r, StockoutRiskResult) for r in results)
        assert all(r.risk_category in ["Low", "Medium", "High", "Critical"] for r in results)
        assert all(0 <= r.stockout_probability <= 1 for r in results)
        assert all(0 <= r.fill_rate <= 1 for r in results)
        assert all(r.cycles_per_year > 0 for r in results)
        assert all(len(r.assumptions) > 0 for r in results)

    def _create_test_df(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(["ENGINE", "AVIONICS", "AIRFRAME"], n),
                "shop_id": np.random.choice(
                    ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"], n
                ),
                "current_stock": np.random.randint(10, 100, n),
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
                "lead_time_days": np.random.randint(5, 30, n),
                "lead_time_std_days": np.random.randint(0, 5, n),
                "service_level_target": np.random.choice([0.90, 0.95, 0.99], n),
                "unit_cost": np.random.uniform(100, 10000, n),
            }
        )


class TestAnalyzeStockoutRisk:
    """Test convenience function."""

    def test_function(self):
        """Test analyze_stockout_risk function."""
        df = self._create_test_df(5)
        results = analyze_stockout_risk(df)

        assert len(results) == 5
        assert all(isinstance(r, StockoutRiskResult) for r in results)

    def test_function_with_reorder_points(self):
        """Test with pre-calculated reorder points."""
        from src.aviation_mro.analytics.inventory.reorder_point import ReorderPointResult

        df = self._create_test_df(5)
        # Create mock reorder points
        rop_results = [
            ReorderPointResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                lead_time_weeks=2.0,
                mean_weekly_demand=10.0,
                std_weekly_demand=2.0,
                service_level=0.95,
                z_score=1.645,
                lead_time_demand=20.0,
                safety_stock=5.0,
                reorder_point=25.0,
                assumptions=[],
            )
            for i in range(5)
        ]

        results = analyze_stockout_risk(df, rop_results)

        assert len(results) == 5
        # ROP should match
        for _i, r in enumerate(results):
            assert abs(r.reorder_point - 25.0) < 0.01

    def _create_test_df(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(["ENGINE", "AVIONICS", "AIRFRAME"], n),
                "shop_id": np.random.choice(
                    ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"], n
                ),
                "current_stock": np.random.randint(10, 100, n),
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
                "lead_time_days": np.random.randint(5, 30, n),
                "lead_time_std_days": np.random.randint(0, 5, n),
                "service_level_target": np.random.choice([0.90, 0.95, 0.99], n),
                "unit_cost": np.random.uniform(100, 10000, n),
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
