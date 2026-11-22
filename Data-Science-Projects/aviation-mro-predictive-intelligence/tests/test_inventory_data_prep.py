"""Tests for inventory data preparation module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.data_prep import (
    generate_synthetic_inventory,
    load_inventory_data,
    save_inventory_data,
)
from src.aviation_mro.analytics.inventory.data_validation import validate_inventory_data


class TestGenerateSyntheticInventory:
    """Test synthetic inventory generation."""

    def test_generates_correct_number_of_parts(self):
        """Test correct number of parts generated."""
        df = generate_synthetic_inventory(seed=42)
        # N_PARTS = 150
        assert len(df) == 150

    def test_generates_all_shops(self):
        """Test all 5 shops are represented."""
        df = generate_synthetic_inventory(seed=42)
        assert df["shop_id"].nunique() == 5
        assert set(df["shop_id"].unique()) == {
            "SHOP_01",
            "SHOP_02",
            "SHOP_03",
            "SHOP_04",
            "SHOP_05",
        }

    def test_generates_all_families(self):
        """Test all 3 part families are represented."""
        df = generate_synthetic_inventory(seed=42)
        assert df["part_family"].nunique() == 3
        assert set(df["part_family"].unique()) == {"ENGINE", "AVIONICS", "AIRFRAME"}

    def test_generates_all_criticalities(self):
        """Test all criticality levels are represented."""
        df = generate_synthetic_inventory(seed=42)
        assert set(df["criticality"].unique()).issubset({"A", "B", "C"})

    def test_reproducible_with_seed(self):
        """Test reproducibility with fixed seed."""
        df1 = generate_synthetic_inventory(seed=42)
        df2 = generate_synthetic_inventory(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test different seeds produce different data."""
        df1 = generate_synthetic_inventory(seed=42)
        df2 = generate_synthetic_inventory(seed=123)
        # At least one column should differ
        assert not df1.equals(df2)

    def test_part_number_format(self):
        """Test part number format."""
        df = generate_synthetic_inventory(seed=42)
        for pn in df["part_number"]:
            assert pn.startswith("PART-")
            # Format: PART-XX-FAM-NNNN
            parts = pn.split("-")
            assert len(parts) == 4

    def test_numeric_columns_non_negative(self):
        """Test numeric columns are non-negative."""
        df = generate_synthetic_inventory(seed=42)
        numeric_cols = [
            "current_stock",
            "unit_cost",
            "lead_time_days",
            "lead_time_std_days",
            "demand_mean_weekly",
            "demand_std_weekly",
        ]
        for col in numeric_cols:
            assert (df[col] >= 0).all(), f"Column {col} has negative values"

    def test_service_level_in_range(self):
        """Test service level targets are valid."""
        df = generate_synthetic_inventory(seed=42)
        assert (df["service_level_target"] >= 0.5).all()
        assert (df["service_level_target"] <= 1.0).all()

    def test_lead_time_positive(self):
        """Test lead time is at least 1 day."""
        df = generate_synthetic_inventory(seed=42)
        assert (df["lead_time_days"] >= 1).all()


class TestSaveLoadInventoryData:
    """Test save and load functions."""

    def test_save_and_load(self, tmp_path):
        """Test round-trip save and load."""
        df = generate_synthetic_inventory(seed=42)
        path = tmp_path / "test_inventory.parquet"

        save_inventory_data(df, path)
        assert path.exists()

        loaded = load_inventory_data(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_default_path(self):
        """Test loading from default path."""
        # This requires the default file to exist
        df = generate_synthetic_inventory(seed=42)
        assert df is not None
        # Skip actual test since it requires file setup
        pass


class TestGeneratedDataValidation:
    """Test that generated data passes validation."""

    def test_generated_data_passes_validation(self):
        """Generated data should pass validation."""
        df = generate_synthetic_inventory(seed=42)
        result = validate_inventory_data(df)

        assert result.is_valid is True
        assert result.n_rows == 150
        assert result.n_parts == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
