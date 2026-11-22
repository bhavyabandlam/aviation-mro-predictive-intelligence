"""Inventory Data Validation.

Validates synthetic inventory data for quality and consistency.
All data is SYNTHETIC — not real airline records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"

# Expected columns in inventory data
REQUIRED_COLUMNS = [
    "part_number",
    "part_family",
    "shop_id",
    "current_stock",
    "unit_cost",
    "lead_time_days",
    "lead_time_std_days",
    "demand_mean_weekly",
    "demand_std_weekly",
    "service_level_target",
    "criticality",
    "source",
]

PART_FAMILIES = ["ENGINE", "AVIONICS", "AIRFRAME"]
SHOPS = ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"]
CRITICALITY_LEVELS = ["A", "B", "C"]  # A=Critical, B=Essential, C=Standard


@dataclass
class ValidationResult:
    """Result of inventory data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    n_rows: int
    n_parts: int
    date_range: tuple[str, str] | None = None


class InventoryDataValidator:
    """Validates inventory data for quality and consistency."""

    def __init__(
        self,
        required_columns: list[str] | None = None,
        part_families: list[str] | None = None,
        shops: list[str] | None = None,
        criticality_levels: list[str] | None = None,
    ):
        self.required_columns = required_columns or REQUIRED_COLUMNS
        self.part_families = part_families or PART_FAMILIES
        self.shops = shops or SHOPS
        self.criticality_levels = criticality_levels or CRITICALITY_LEVELS

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run all validation checks on inventory data."""
        errors: list[str] = []
        warnings: list[str] = []

        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")

        # Check row count
        n_rows = len(df)
        if n_rows == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                n_rows=0,
                n_parts=0,
            )

        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows")

        # Check part_number uniqueness
        if "part_number" in df.columns:
            dup_parts = df["part_number"].duplicated().sum()
            if dup_parts > 0:
                errors.append(f"Duplicate part_numbers found: {dup_parts}")

        # Validate part_family values
        if "part_family" in df.columns:
            invalid_families = set(df["part_family"].unique()) - set(self.part_families)
            if invalid_families:
                errors.append(f"Invalid part_family values: {sorted(invalid_families)}")

        # Validate shop_id values
        if "shop_id" in df.columns:
            invalid_shops = set(df["shop_id"].unique()) - set(self.shops)
            if invalid_shops:
                errors.append(f"Invalid shop_id values: {sorted(invalid_shops)}")

        # Validate criticality values
        if "criticality" in df.columns:
            invalid_crit = set(df["criticality"].unique()) - set(self.criticality_levels)
            if invalid_crit:
                errors.append(f"Invalid criticality values: {sorted(invalid_crit)}")

        # Validate numeric columns are non-negative
        numeric_checks = [
            ("current_stock", 0, None),
            ("unit_cost", 0.01, None),
            ("lead_time_days", 1, None),
            ("lead_time_std_days", 0, None),
            ("demand_mean_weekly", 0, None),
            ("demand_std_weekly", 0, None),
            ("service_level_target", 0.5, 1.0),
        ]

        for col, min_val, max_val in numeric_checks:
            if col in df.columns:
                if (df[col] < min_val).any():
                    errors.append(f"Column {col} has values below minimum {min_val}")
                if max_val is not None and (df[col] > max_val).any():
                    errors.append(f"Column {col} has values above maximum {max_val}")

        # Check demand_std_weekly <= demand_mean_weekly (coefficient of variation warning)
        if "demand_mean_weekly" in df.columns and "demand_std_weekly" in df.columns:
            high_cv = df[df["demand_mean_weekly"] > 0]
            high_cv = high_cv[high_cv["demand_std_weekly"] / high_cv["demand_mean_weekly"] > 2.0]
            if len(high_cv) > 0:
                warnings.append(
                    f"{len(high_cv)} parts have demand CV > 2.0 (highly variable demand)"
                )

        # Check lead_time_std_days relative to lead_time_days
        if "lead_time_days" in df.columns and "lead_time_std_days" in df.columns:
            high_lt_var = df[
                (df["lead_time_days"] > 0) & (df["lead_time_std_days"] / df["lead_time_days"] > 1.0)
            ]
            if len(high_lt_var) > 0:
                warnings.append(
                    f"{len(high_lt_var)} parts have lead time CV > 1.0 (highly variable lead time)"
                )

        # Check source column
        if "source" in df.columns:
            unique_sources = df["source"].unique()
            if len(unique_sources) != 1:
                warnings.append(f"Multiple sources found: {unique_sources}")

        n_parts = df["part_number"].nunique() if "part_number" in df.columns else n_rows

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            n_rows=n_rows,
            n_parts=n_parts,
        )

    def validate_file(self, path: Path) -> ValidationResult:
        """Validate inventory data from a Parquet file."""
        df = pd.read_parquet(path)
        return self.validate(df)


def validate_inventory_data(df: pd.DataFrame) -> ValidationResult:
    """Convenience function to validate inventory data."""
    validator = InventoryDataValidator()
    return validator.validate(df)
