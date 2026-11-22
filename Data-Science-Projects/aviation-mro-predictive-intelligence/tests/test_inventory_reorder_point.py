"""Tests for inventory reorder point module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.reorder_point import (
    ReorderPointCalculator,
    ReorderPointResult,
    calculate_reorder_points,
)


class TestReorderPointCalculator:
    """Test ReorderPointCalculator class."""

    def test_z_score(self):
        """Test z-score calculation."""
        calc = ReorderPointCalculator()
        # Common service levels
        assert abs(calc.z_score(0.95) - 1.645) < 0.01
        assert abs(calc.z_score(0.99) - 2.326) < 0.01
        assert abs(calc.z_score(0.90) - 1.282) < 0.01

    def test_z_score_invalid(self):
        """Test z-score with invalid service level."""
        calc = ReorderPointCalculator()
        with pytest.raises(ValueError):
            calc.z_score(0.0)
        with pytest.raises(ValueError):
            calc.z_score(1.0)
        with pytest.raises(ValueError):
            calc.z_score(-0.1)
        with pytest.raises(ValueError):
            calc.z_score(1.5)

    def test_calculate_lead_time_demand(self):
        """Test lead time demand calculation."""
        calc = ReorderPointCalculator()
        mean_ltd, std_ltd = calc.calculate_lead_time_demand(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,  # 2 weeks
        )

        assert abs(mean_ltd - 20.0) < 0.01  # 10 * 2
        assert abs(std_ltd - 2.0 * np.sqrt(2)) < 0.01

    def test_calculate_safety_stock(self):
        """Test safety stock calculation."""
        calc = ReorderPointCalculator()
        ss = calc.calculate_safety_stock(std_ltd=5.0, service_level=0.95)
        expected = 1.645 * 5.0
        assert abs(ss - expected) < 0.01

    def test_calculate_reorder_point(self):
        """Test full reorder point calculation."""
        calc = ReorderPointCalculator()
        mean_ltd, ss, rop, z = calc.calculate_reorder_point(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
            service_level=0.95,
        )

        # Mean LTD = 10 * 2 = 20
        assert abs(mean_ltd - 20.0) < 0.01
        # Safety stock = 1.645 * 2 * sqrt(2) = 1.645 * 2.828 = 4.65
        assert abs(ss - 4.65) < 0.1
        # ROP = 20 + 4.65 = 24.65
        assert abs(rop - 24.65) < 0.1
        assert abs(z - 1.645) < 0.01

    def test_calculate_for_dataframe(self):
        """Test batch calculation."""
        calc = ReorderPointCalculator()
        df = self._create_test_df(5)
        results = calc.calculate_for_dataframe(df)

        assert len(results) == 5
        assert all(isinstance(r, ReorderPointResult) for r in results)
        assert all(r.reorder_point > 0 for r in results)
        assert all(r.safety_stock >= 0 for r in results)
        assert all(r.lead_time_demand > 0 for r in results)
        assert all(len(r.assumptions) > 0 for r in results)

    def test_service_level_per_part(self):
        """Test per-part service level override."""
        calc = ReorderPointCalculator(default_service_level=0.90)
        df = self._create_test_df(3)
        # Override service level for one part
        df.loc[0, "service_level_target"] = 0.99

        results = calc.calculate_for_dataframe(df)

        # Part 0 should use 0.99, others 0.90
        assert abs(results[0].service_level - 0.99) < 0.01
        assert abs(results[1].service_level - 0.90) < 0.01
        assert abs(results[2].service_level - 0.90) < 0.01

        # Higher service level = higher z = higher ROP
        assert results[0].z_score > results[1].z_score
        assert results[0].reorder_point > results[1].reorder_point

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
                "lead_time_days": np.random.randint(5, 30, n),
            }
        )


class TestCalculateReorderPoints:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_reorder_points function."""
        df = self._create_test_df(5)
        results = calculate_reorder_points(df)

        assert len(results) == 5
        assert all(isinstance(r, ReorderPointResult) for r in results)

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
                "lead_time_days": np.random.randint(5, 30, n),
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
