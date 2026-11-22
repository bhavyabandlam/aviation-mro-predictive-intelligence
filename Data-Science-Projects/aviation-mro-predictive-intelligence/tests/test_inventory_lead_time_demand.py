"""Tests for inventory lead-time demand module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.lead_time_demand import (
    LeadTimeDemandEstimator,
    LeadTimeDemandResult,
    estimate_lead_time_demand,
)


class TestLeadTimeDemandEstimator:
    """Test LeadTimeDemandEstimator class."""

    def test_estimate_normal_constant_lt(self):
        """Test normal constant lead time estimation."""
        estimator = LeadTimeDemandEstimator()
        mean_ltd, std_ltd = estimator.estimate_normal_constant_lt(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
        )
        assert abs(mean_ltd - 20.0) < 0.01  # 10 * 2
        assert abs(std_ltd - 2.0 * np.sqrt(2)) < 0.01

    def test_estimate_normal_variable_lt(self):
        """Test normal variable lead time estimation."""
        estimator = LeadTimeDemandEstimator()
        mean_ltd, std_ltd = estimator.estimate_normal_variable_lt(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
            lead_time_std_days=2,
        )
        lt_weeks = 2.0
        lt_std_weeks = 2.0 / 7.0
        expected_mean = 10.0 * 2.0
        expected_var = lt_weeks * 2.0**2 + 10.0**2 * lt_std_weeks**2
        assert abs(mean_ltd - expected_mean) < 0.01
        assert abs(std_ltd - np.sqrt(expected_var)) < 0.01

    def test_estimate_lognormal(self):
        """Test lognormal estimation."""
        estimator = LeadTimeDemandEstimator()
        mu_ln, sigma_ln, mean_ltd, std_ltd = estimator.estimate_lognormal(
            mean_weekly_demand=10.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
        )
        # Should return valid parameters
        assert mean_ltd > 0
        assert std_ltd >= 0
        assert sigma_ln >= 0

    def test_estimate_lognormal_zero_mean(self):
        """Test lognormal with zero mean falls back."""
        estimator = LeadTimeDemandEstimator()
        mu_ln, sigma_ln, mean_ltd, std_ltd = estimator.estimate_lognormal(
            mean_weekly_demand=0.0,
            std_weekly_demand=2.0,
            lead_time_days=14,
        )
        # Falls back to normal: mean_ltd = 0 * 2 = 0, std_ltd = 2 * sqrt(2)
        assert mean_ltd == 0.0
        assert abs(std_ltd - 2.0 * np.sqrt(2)) < 0.01
        assert mu_ln == 0.0
        assert sigma_ln == 0.0

    def test_compute_percentiles_normal(self):
        """Test percentile computation for normal."""
        estimator = LeadTimeDemandEstimator()
        params = {"mean": 100.0, "std": 15.0}
        pctls = estimator.compute_percentiles("normal", params, [0.5, 0.95])

        assert "p50" in pctls
        assert "p95" in pctls
        assert abs(pctls["p50"] - 100.0) < 0.1
        assert pctls["p95"] > 100.0

    def test_compute_percentiles_lognormal(self):
        """Test percentile computation for lognormal."""
        estimator = LeadTimeDemandEstimator()
        params = {"mu": 4.5, "sigma": 0.3}
        pctls = estimator.compute_percentiles("lognormal", params, [0.5, 0.95])

        assert "p50" in pctls
        assert "p95" in pctls
        assert pctls["p95"] > pctls["p50"]

    def test_estimate_for_dataframe(self):
        """Test batch estimation."""
        estimator = LeadTimeDemandEstimator()
        df = self._create_test_df(5)
        results = estimator.estimate_for_dataframe(df)

        assert len(results) == 5
        assert all(isinstance(r, LeadTimeDemandResult) for r in results)
        assert all(r.mean_ltd > 0 for r in results)
        assert all(r.std_ltd >= 0 for r in results)
        assert all("p50" in r.percentiles for r in results)
        assert all("p95" in r.percentiles for r in results)
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
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
                "lead_time_days": np.random.randint(5, 30, n),
                "lead_time_std_days": np.random.randint(0, 5, n),
            }
        )


class TestEstimateLeadTimeDemand:
    """Test convenience function."""

    def test_function(self):
        """Test estimate_lead_time_demand function."""
        df = self._create_test_df(5)
        results = estimate_lead_time_demand(df)

        assert len(results) == 5
        assert all(isinstance(r, LeadTimeDemandResult) for r in results)

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
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
