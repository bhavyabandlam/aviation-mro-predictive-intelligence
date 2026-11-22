"""Tests for inventory data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.data_validation import (
    CRITICALITY_LEVELS,
    PART_FAMILIES,
    SHOPS,
    InventoryDataValidator,
    ValidationResult,
    validate_inventory_data,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_creation(self):
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            n_rows=100,
            n_parts=50,
        )
        assert result.is_valid is True
        assert result.n_rows == 100
        assert result.n_parts == 50


class TestInventoryDataValidator:
    """Test InventoryDataValidator class."""

    def test_valid_data_passes(self):
        """Valid synthetic data should pass validation."""
        df = self._create_valid_df()
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is True
        assert result.n_rows == 10
        assert result.n_parts == 10

    def test_missing_columns_fails(self):
        """Missing required columns should fail validation."""
        df = self._create_valid_df()
        df = df.drop(columns=["unit_cost"])
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("unit_cost" in e for e in result.errors)

    def test_duplicate_part_numbers_fails(self):
        """Duplicate part numbers should fail validation."""
        df = self._create_valid_df()
        # Create duplicate
        df = pd.concat([df, df.iloc[:1]], ignore_index=True)
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("Duplicate part_numbers" in e for e in result.errors)

    def test_invalid_part_family_fails(self):
        """Invalid part family should fail validation."""
        df = self._create_valid_df()
        df.loc[0, "part_family"] = "INVALID_FAMILY"
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("INVALID_FAMILY" in e for e in result.errors)

    def test_invalid_shop_id_fails(self):
        """Invalid shop ID should fail validation."""
        df = self._create_valid_df()
        df.loc[0, "shop_id"] = "SHOP_99"
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("SHOP_99" in e for e in result.errors)

    def test_invalid_criticality_fails(self):
        """Invalid criticality should fail validation."""
        df = self._create_valid_df()
        df.loc[0, "criticality"] = "Z"
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("Z" in e for e in result.errors)

    def test_negative_stock_fails(self):
        """Negative current stock should fail validation."""
        df = self._create_valid_df()
        df.loc[0, "current_stock"] = -5
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("current_stock" in e and "minimum" in e for e in result.errors)

    def test_negative_unit_cost_fails(self):
        """Negative unit cost should fail validation."""
        df = self._create_valid_df()
        df.loc[0, "unit_cost"] = -100
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("unit_cost" in e for e in result.errors)

    def test_zero_lead_time_fails(self):
        """Zero lead time should fail validation."""
        df = self._create_valid_df()
        df.loc[0, "lead_time_days"] = 0
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("lead_time_days" in e for e in result.errors)

    def test_service_level_out_of_bounds_fails(self):
        """Service level outside [0.5, 1.0] should fail."""
        df = self._create_valid_df()
        df.loc[0, "service_level_target"] = 1.5
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("service_level_target" in e for e in result.errors)

    def test_high_demand_cv_warning(self):
        """High coefficient of variation should generate warning."""
        df = self._create_valid_df()
        # Set very high std relative to mean
        df.loc[0, "demand_std_weekly"] = df.loc[0, "demand_mean_weekly"] * 3.0
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is True  # Warning, not error
        assert any("CV > 2.0" in w for w in result.warnings)

    def test_high_lead_time_cv_warning(self):
        """High lead time CV should generate warning."""
        df = self._create_valid_df()
        # Use int to avoid pandas dtype issues
        df.loc[0, "lead_time_std_days"] = int(df.loc[0, "lead_time_days"] * 1.5)
        validator = InventoryDataValidator()
        result = validator.validate(df)

        assert result.is_valid is True
        assert any("lead time CV > 1.0" in w for w in result.warnings)

    def _create_valid_df(self) -> pd.DataFrame:
        """Create a valid test DataFrame."""
        np.random.seed(42)
        n = 10
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(PART_FAMILIES, n),
                "shop_id": np.random.choice(SHOPS, n),
                "current_stock": np.random.randint(0, 100, n),
                "unit_cost": np.random.uniform(100, 10000, n),
                "lead_time_days": np.random.randint(5, 30, n),
                "lead_time_std_days": np.random.randint(0, 5, n),
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
                "service_level_target": np.random.choice([0.90, 0.95, 0.99], n),
                "criticality": np.random.choice(CRITICALITY_LEVELS, n, p=[0.2, 0.5, 0.3]),
                "source": "synthetic_inventory_v1",
                "generated_at": pd.Timestamp.now(),
            }
        )


class TestValidateInventoryData:
    """Test convenience function."""

    def test_convenience_function(self):
        """Test validate_inventory_data function."""
        df = self._create_valid_df()
        result = validate_inventory_data(df)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def _create_valid_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 5
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(PART_FAMILIES, n),
                "shop_id": np.random.choice(SHOPS, n),
                "current_stock": np.random.randint(0, 100, n),
                "unit_cost": np.random.uniform(100, 10000, n),
                "lead_time_days": np.random.randint(5, 30, n),
                "lead_time_std_days": np.random.randint(0, 5, n),
                "demand_mean_weekly": np.random.uniform(1, 20, n),
                "demand_std_weekly": np.random.uniform(0.5, 5, n),
                "service_level_target": np.random.choice([0.90, 0.95, 0.99], n),
                "criticality": np.random.choice(CRITICALITY_LEVELS, n),
                "source": "synthetic_inventory_v1",
                "generated_at": pd.Timestamp.now(),
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
