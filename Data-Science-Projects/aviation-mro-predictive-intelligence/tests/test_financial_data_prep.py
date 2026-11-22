"""Tests for financial data preparation module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.data_prep import (
    generate_all_financial_data,
    generate_downtime_costs,
    generate_fleet_data,
    generate_inventory_cost_params,
    generate_maintenance_labor_costs,
    generate_shop_labor_data,
    load_all_financial_data,
    save_financial_data,
)


class TestGenerateShopLaborData:
    """Test shop labor data generation."""

    def test_generates_all_shops(self):
        """Test all 5 shops are generated."""
        df = generate_shop_labor_data(seed=42)
        assert len(df) == 5
        assert set(df["shop_id"].unique()) == {
            "SHOP_01",
            "SHOP_02",
            "SHOP_03",
            "SHOP_04",
            "SHOP_05",
        }

    def test_reproducible_with_seed(self):
        """Test reproducibility with fixed seed."""
        df1 = generate_shop_labor_data(seed=42)
        df2 = generate_shop_labor_data(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_labor_rates_positive(self):
        """Test labor rates are positive."""
        df = generate_shop_labor_data(seed=42)
        assert (df["base_labor_rate"] > 0).all()
        assert (df["overtime_labor_rate"] > 0).all()

    def test_overtime_rate_higher(self):
        """Test overtime rate is higher than base rate."""
        df = generate_shop_labor_data(seed=42)
        assert (df["overtime_labor_rate"] > df["base_labor_rate"]).all()

    def test_capacity_hours_positive(self):
        """Test capacity hours are positive."""
        df = generate_shop_labor_data(seed=42)
        assert (df["annual_regular_capacity_hours"] > 0).all()
        assert (df["annual_overtime_capacity_hours"] >= 0).all()


class TestGenerateFleetData:
    """Test fleet data generation."""

    def test_generates_all_fleet_types(self):
        """Test all 3 fleet types are generated."""
        df = generate_fleet_data(seed=42)
        assert len(df) == 3
        assert set(df["fleet_type"].unique()) == {"NARROW_BODY", "WIDE_BODY", "REGIONAL"}

    def test_aircraft_count_positive(self):
        """Test aircraft counts are positive."""
        df = generate_fleet_data(seed=42)
        assert (df["aircraft_count"] > 0).all()

    def test_revenue_per_hour_positive(self):
        """Test revenue per hour is positive."""
        df = generate_fleet_data(seed=42)
        assert (df["revenue_per_flight_hour"] > 0).all()

    def test_total_revenue_calculated(self):
        """Test total revenue potential is calculated."""
        df = generate_fleet_data(seed=42)
        for _, row in df.iterrows():
            expected = (
                row["aircraft_count"]
                * row["avg_annual_flight_hours_per_aircraft"]
                * row["revenue_per_flight_hour"]
            )
            assert (
                abs(row["total_annual_revenue_potential"] - expected) < 1000.0
            )  # Allow floating point variance


class TestGenerateMaintenanceLaborCosts:
    """Test maintenance labor cost generation."""

    def test_generates_all_combinations(self):
        """Test all shop-family combinations."""
        df = generate_maintenance_labor_costs(seed=42)
        assert len(df) == 15  # 5 shops × 3 families

    def test_labor_cost_equals_hours_times_rate(self):
        """Test labor cost = hours × rate."""
        df = generate_maintenance_labor_costs(seed=42)
        for _, row in df.iterrows():
            # The generator rounds labor_cost_per_job but uses rounded hours and rate
            # Allow tolerance for rounding
            expected = row["mean_labor_hours_per_job"] * row["labor_rate_used"]
            assert abs(row["labor_cost_per_job"] - expected) < 10.0  # Allow rounding variance


class TestGenerateDowntimeCosts:
    """Test downtime cost generation."""

    def test_generates_all_combinations(self):
        """Test all fleet-family combinations."""
        df = generate_downtime_costs(seed=42)
        assert len(df) == 9  # 3 fleet types × 3 families

    def test_downtime_cost_calculated(self):
        """Test downtime cost = hours × rate."""
        df = generate_downtime_costs(seed=42)
        for _, row in df.iterrows():
            # The generator rounds downtime_cost_per_event but uses rounded hours
            expected = row["mean_downtime_hours"] * row["downtime_cost_per_hour"]
            assert (
                abs(row["downtime_cost_per_event"] - expected) < 2000.0
            )  # Allow rounding variance


class TestGenerateInventoryCostParams:
    """Test inventory cost parameters."""

    def test_returns_single_row(self):
        """Test single row returned."""
        df = generate_inventory_cost_params(seed=42)
        assert len(df) == 1

    def test_values_correct(self):
        """Test parameter values."""
        df = generate_inventory_cost_params(seed=42)
        row = df.iloc[0]
        assert row["holding_cost_rate"] == 0.20
        assert row["stockout_cost_multiplier"] == 10.0
        assert row["order_cost"] == 75.0


class TestGenerateAllFinancialData:
    """Test generating all financial data."""

    def test_returns_all_datasets(self):
        """Test all 5 datasets returned."""
        data = generate_all_financial_data(seed=42)
        assert set(data.keys()) == {
            "shop_labor",
            "fleet",
            "maintenance_labor",
            "downtime_costs",
            "inventory_cost_params",
        }

    def test_all_dataframes_non_empty(self):
        """Test all dataframes have data."""
        data = generate_all_financial_data(seed=42)
        for name, df in data.items():
            assert len(df) > 0, f"{name} is empty"


class TestSaveLoadFinancialData:
    """Test save and load functions."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test round-trip save and load."""
        data = generate_all_financial_data(seed=42)
        # Temporarily change DATA_DIR
        import src.aviation_mro.analytics.financial.data_prep as dp

        original_dir = dp.DATA_DIR
        dp.DATA_DIR = tmp_path
        try:
            paths = save_financial_data(data)
            assert len(paths) == 5
            for _name, path in paths.items():
                assert path.exists()
            loaded = load_all_financial_data()
            for name in data:
                pd.testing.assert_frame_equal(data[name], loaded[name])
        finally:
            dp.DATA_DIR = original_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
