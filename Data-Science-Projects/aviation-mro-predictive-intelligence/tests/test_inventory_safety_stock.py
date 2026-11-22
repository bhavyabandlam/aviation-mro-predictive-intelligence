"""Tests for inventory safety stock module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.aviation_mro.analytics.inventory.safety_stock import (
    SafetyStockCalculator,
    SafetyStockResult,
    calculate_safety_stock,
)


class TestSafetyStockCalculator:
    """Test SafetyStockCalculator class."""

    def test_method_normal_constant_lt(self):
        """Test normal constant lead time method."""
        calc = SafetyStockCalculator()
        ss = calc.method_normal_constant_lt(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,  # 2 weeks
            service_level=0.95,
        )
        # SS = 1.645 * 2 * sqrt(2) = 4.65
        expected = stats.norm.ppf(0.95) * 2.0 * np.sqrt(2)
        assert abs(ss - expected) < 0.01

    def test_method_normal_variable_lt(self):
        """Test normal variable lead time method."""
        calc = SafetyStockCalculator()
        ss = calc.method_normal_variable_lt(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,  # 2 weeks
            lead_time_std_days=2,  # 2 days std
            service_level=0.95,
        )
        # Var = L * σ_D² + D² * σ_L²
        # L = 2, σ_D = 2, D = 10, σ_L = 2/7
        lt_weeks = 2.0
        lt_std_weeks = 2.0 / 7.0
        var_ltd = lt_weeks * 2.0**2 + 10.0**2 * lt_std_weeks**2
        std_ltd = np.sqrt(var_ltd)
        expected = stats.norm.ppf(0.95) * std_ltd
        assert abs(ss - expected) < 0.01

    def test_method_lognormal(self):
        """Test lognormal method."""
        calc = SafetyStockCalculator()
        ss = calc.method_lognormal(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
            service_level=0.95,
        )
        # Should be non-negative
        assert ss >= 0

    def test_method_lognormal_zero_mean(self):
        """Test lognormal with zero mean falls back to normal."""
        calc = SafetyStockCalculator()
        ss = calc.method_lognormal(
            mean_weekly_demand=0.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
            service_level=0.95,
        )
        # Should fall back to normal method
        expected = stats.norm.ppf(0.95) * 2.0 * np.sqrt(2)
        assert abs(ss - expected) < 0.01

    def test_calculate_all_methods(self):
        """Test all methods calculation."""
        calc = SafetyStockCalculator()
        df = self._create_test_df(3)
        results = calc.calculate_all_methods(df)

        assert "normal_constant_lt" in results
        assert "normal_variable_lt" in results
        assert "lognormal" in results

        for method_results in results.values():
            assert len(method_results) == 3
            assert all(isinstance(r, SafetyStockResult) for r in method_results)
            assert all(r.safety_stock >= 0 for r in method_results)

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
                "lead_time_std_days": np.random.randint(0, 5, n),
                "service_level_target": 0.95,
            }
        )


class TestCalculateSafetyStock:
    """Test convenience function."""

    def test_function(self):
        """Test calculate_safety_stock function."""
        df = self._create_test_df(5)
        results = calculate_safety_stock(df, method="normal_constant_lt")

        assert len(results) == 5
        assert all(isinstance(r, SafetyStockResult) for r in results)

    def test_invalid_method(self):
        """Test invalid method returns empty list."""
        df = self._create_test_df(5)
        results = calculate_safety_stock(df, method="invalid_method")
        assert results == []

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
                "lead_time_std_days": np.random.randint(0, 5, n),
                "service_level_target": 0.95,
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
