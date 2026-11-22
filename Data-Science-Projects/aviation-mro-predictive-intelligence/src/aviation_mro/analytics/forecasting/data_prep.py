"""Synthetic MRO Maintenance Demand Data Generation and Chronological Splitting.

This module generates clearly labeled synthetic maintenance demand data and provides
chronological train/validation/test splits with no future-data leakage.
Generates three forecast targets per series:
  - demand_count: work-order counts by week
  - labour_hours: labour-hour demand
  - maintenance_expenditure: maintenance expenditure
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Dataset configuration
SHOPS = [f"SHOP_{i:02d}" for i in range(1, 6)]
PART_FAMILIES = ["ENGINE", "AVIONICS", "AIRFRAME"]
N_WEEKS = 170  # 3 years + 14 weeks future buffer
START_DATE = pd.Timestamp("2022-01-03")  # Monday, ISO week 1 2022

# Forecast target columns
TARGET_COLUMNS = ["demand_count", "labour_hours", "maintenance_expenditure"]
PRIMARY_TARGET = "demand_count"

# Base demand rates (jobs per week) per shop-part_family combination
BASE_RATES: dict[tuple[str, str], float] = {
    ("SHOP_01", "ENGINE"): 8.0,
    ("SHOP_01", "AVIONICS"): 5.0,
    ("SHOP_01", "AIRFRAME"): 6.0,
    ("SHOP_02", "ENGINE"): 10.0,
    ("SHOP_02", "AVIONICS"): 4.0,
    ("SHOP_02", "AIRFRAME"): 7.0,
    ("SHOP_03", "ENGINE"): 6.0,
    ("SHOP_03", "AVIONICS"): 6.0,
    ("SHOP_03", "AIRFRAME"): 5.0,
    ("SHOP_04", "ENGINE"): 9.0,
    ("SHOP_04", "AVIONICS"): 3.0,
    ("SHOP_04", "AIRFRAME"): 8.0,
    ("SHOP_05", "ENGINE"): 7.0,
    ("SHOP_05", "AVIONICS"): 5.0,
    ("SHOP_05", "AIRFRAME"): 9.0,
}

# Trend rates (weekly fractional change) per series
TREND_RATES: dict[tuple[str, str], float] = {
    ("SHOP_01", "ENGINE"): 0.002,
    ("SHOP_01", "AVIONICS"): -0.001,
    ("SHOP_01", "AIRFRAME"): 0.000,
    ("SHOP_02", "ENGINE"): -0.003,
    ("SHOP_02", "AVIONICS"): 0.002,
    ("SHOP_02", "AIRFRAME"): -0.001,
    ("SHOP_03", "ENGINE"): 0.001,
    ("SHOP_03", "AVIONICS"): -0.002,
    ("SHOP_03", "AIRFRAME"): 0.003,
    ("SHOP_04", "ENGINE"): -0.002,
    ("SHOP_04", "AVIONICS"): 0.001,
    ("SHOP_04", "AIRFRAME"): -0.001,
    ("SHOP_05", "ENGINE"): 0.000,
    ("SHOP_05", "AVIONICS"): -0.001,
    ("SHOP_05", "AIRFRAME"): 0.002,
}

# Seasonal amplitudes
ANNUAL_AMPLITUDE = 0.30  # ±30% annual cycle
QUARTERLY_AMPLITUDE = 0.15  # ±15% quarterly cycle

# Holiday weeks (1-indexed): Christmas/New Year (52,1), Summer (26), Fall (39)
HOLIDAY_WEEKS = {1, 2, 26, 39, 52}

# Zero-inflation probability
ZERO_INFLATION_PROB = 0.05

# Chronological split boundaries (1-indexed week numbers)
TRAIN_END_WEEK = 104  # End of week 104 (2023-12-25)
VALIDATION_END_WEEK = 130  # End of week 130 (2024-06-24)
TEST_END_WEEK = 156  # End of week 156 (2024-12-23)

# Forecast horizon
FORECAST_HORIZON = 12

# Rolling origin validation spacing
VALIDATION_ORIGIN_SPACING = 6

# Labour hours per work order by shop-part_family
LABOUR_HOURS_PER_WO: dict[tuple[str, str], float] = {
    ("SHOP_01", "ENGINE"): 4.5,
    ("SHOP_01", "AVIONICS"): 3.0,
    ("SHOP_01", "AIRFRAME"): 3.5,
    ("SHOP_02", "ENGINE"): 5.0,
    ("SHOP_02", "AVIONICS"): 2.5,
    ("SHOP_02", "AIRFRAME"): 4.0,
    ("SHOP_03", "ENGINE"): 4.0,
    ("SHOP_03", "AVIONICS"): 3.5,
    ("SHOP_03", "AIRFRAME"): 3.0,
    ("SHOP_04", "ENGINE"): 4.8,
    ("SHOP_04", "AVIONICS"): 2.0,
    ("SHOP_04", "AIRFRAME"): 4.2,
    ("SHOP_05", "ENGINE"): 4.2,
    ("SHOP_05", "AVIONICS"): 3.2,
    ("SHOP_05", "AIRFRAME"): 4.5,
}

# Hourly labour rate by shop
LABOUR_RATE_PER_HOUR: dict[str, float] = {
    "SHOP_01": 85.0,
    "SHOP_02": 95.0,
    "SHOP_03": 80.0,
    "SHOP_04": 90.0,
    "SHOP_05": 87.5,
}

# Parts cost per work order by shop-part_family
PARTS_COST_PER_WO: dict[tuple[str, str], float] = {
    ("SHOP_01", "ENGINE"): 1200.0,
    ("SHOP_01", "AVIONICS"): 800.0,
    ("SHOP_01", "AIRFRAME"): 950.0,
    ("SHOP_02", "ENGINE"): 1500.0,
    ("SHOP_02", "AVIONICS"): 600.0,
    ("SHOP_02", "AIRFRAME"): 1100.0,
    ("SHOP_03", "ENGINE"): 1000.0,
    ("SHOP_03", "AVIONICS"): 900.0,
    ("SHOP_03", "AIRFRAME"): 750.0,
    ("SHOP_04", "ENGINE"): 1300.0,
    ("SHOP_04", "AVIONICS"): 500.0,
    ("SHOP_04", "AIRFRAME"): 1200.0,
    ("SHOP_05", "ENGINE"): 1100.0,
    ("SHOP_05", "AVIONICS"): 850.0,
    ("SHOP_05", "AIRFRAME"): 1050.0,
}

# Labour hour variation (multiplicative noise)
LABOUR_HOURS_NOISE_SD = 0.10  # ±10% noise on labour hours

# Expenditure variation (multiplicative noise)
EXPENDITURE_NOISE_SD = 0.08  # ±8% noise on expenditure


def generate_synthetic_demand(seed: int | None = None) -> pd.DataFrame:
    """Generate synthetic weekly maintenance demand data.

    Generates three forecast targets per shop-part_family series:
      - demand_count: work-order counts by week
      - labour_hours: labour-hour demand (derived from demand_count)
      - maintenance_expenditure: maintenance expenditure (derived from labour_hours)

    Args:
        seed: Random seed for reproducibility (uses RANDOM_SEED if None)

    Returns:
        DataFrame with columns: date, shop_id, part_family, demand_count,
        labour_hours, maintenance_expenditure
    """
    if seed is None:
        seed = RANDOM_SEED
    np.random.seed(seed)

    rows = []

    for week_idx in range(1, N_WEEKS + 1):
        week_date = START_DATE + pd.Timedelta(weeks=week_idx - 1)
        week_of_year = week_date.isocalendar().week

        for shop in SHOPS:
            for part_family in PART_FAMILIES:
                key = (shop, part_family)
                base_rate = BASE_RATES[key]
                trend_rate = TREND_RATES[key]

                # Trend component (multiplicative)
                trend_factor = (1 + trend_rate) ** (week_idx - 1)

                # Annual seasonality (sinusoidal, 52-week period)
                annual_phase = 2 * np.pi * (week_of_year - 1) / 52
                annual_factor = 1 + ANNUAL_AMPLITUDE * np.sin(annual_phase)

                # Quarterly seasonality (sinusoidal, 13-week period)
                quarterly_phase = 2 * np.pi * (week_of_year - 1) / 13
                quarterly_factor = 1 + QUARTERLY_AMPLITUDE * np.sin(quarterly_phase)

                # Holiday effect
                holiday_factor = 0.3 if week_of_year in HOLIDAY_WEEKS else 1.0

                # Combined expected demand (lambda for Poisson)
                lambda_week = (
                    base_rate * trend_factor * annual_factor * quarterly_factor * holiday_factor
                )
                lambda_week = max(lambda_week, 0.1)  # Floor at 0.1

                # Generate demand from Poisson
                demand = np.random.poisson(lambda_week)

                # Zero-inflation: randomly set to 0 with small probability
                if np.random.random() < ZERO_INFLATION_PROB:
                    demand = 0

                # Labour hours: demand_count * hours_per_wo * (1 + noise)
                labour_hours_per_wo = LABOUR_HOURS_PER_WO[key]
                labour_hours = (
                    demand * labour_hours_per_wo * (1 + np.random.normal(0, LABOUR_HOURS_NOISE_SD))
                )
                labour_hours = max(labour_hours, 0.0)

                # Maintenance expenditure: labour_hours * labour_rate + parts_cost * demand
                labour_rate = LABOUR_RATE_PER_HOUR[shop]
                parts_cost = PARTS_COST_PER_WO[key]
                expenditure = (labour_hours * labour_rate + demand * parts_cost) * (
                    1 + np.random.normal(0, EXPENDITURE_NOISE_SD)
                )
                expenditure = max(expenditure, 0.0)

                rows.append(
                    {
                        "date": week_date,
                        "shop_id": shop,
                        "part_family": part_family,
                        "demand_count": int(demand),
                        "labour_hours": float(labour_hours),
                        "maintenance_expenditure": float(expenditure),
                    }
                )

    df = pd.DataFrame(rows)
    # Ensure proper sorting
    df = df.sort_values(["date", "shop_id", "part_family"]).reset_index(drop=True)

    # Validate
    assert len(df) == N_WEEKS * len(SHOPS) * len(PART_FAMILIES), "Row count mismatch"
    assert df["demand_count"].min() >= 0, "Negative demand generated"
    assert df["labour_hours"].min() >= 0, "Negative labour hours generated"
    assert df["maintenance_expenditure"].min() >= 0, "Negative expenditure generated"

    return df


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute SHA-256 fingerprint of the dataset for versioning."""
    # Use relevant columns for fingerprint
    content = df[["date", "shop_id", "part_family"] + TARGET_COLUMNS].to_string()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ChronologicalSplit:
    """Container for chronological train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    future_buffer: pd.DataFrame
    train_end_week: int
    validation_end_week: int
    test_end_week: int
    forecast_horizon: int
    validation_origins: list[int]  # Week numbers for rolling origins

    @property
    def dataset_fingerprint(self) -> str:
        """Compute fingerprint from the combined data."""
        # Combine all data for fingerprinting
        combined = pd.concat([self.train, self.validation, self.test, self.future_buffer])
        return compute_dataset_fingerprint(combined)

    def get_series(
        self, shop_id: str, part_family: str, target_column: str = PRIMARY_TARGET
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Extract time series for a specific shop-part_family from each split.

        Args:
            shop_id: Shop identifier
            part_family: Part family identifier
            target_column: Target column to extract (default: PRIMARY_TARGET)
        """
        mask_train = (self.train["shop_id"] == shop_id) & (self.train["part_family"] == part_family)
        mask_val = (self.validation["shop_id"] == shop_id) & (
            self.validation["part_family"] == part_family
        )
        mask_test = (self.test["shop_id"] == shop_id) & (self.test["part_family"] == part_family)

        train_series = self.train.loc[mask_train].set_index("date")[target_column].sort_index()
        val_series = self.validation.loc[mask_val].set_index("date")[target_column].sort_index()
        test_series = self.test.loc[mask_test].set_index("date")[target_column].sort_index()

        return train_series, val_series, test_series


def split_chronologically(df: pd.DataFrame) -> ChronologicalSplit:
    """Split data chronologically into train/validation/test/future_buffer.

    Args:
        df: Full synthetic demand DataFrame (must be sorted by date)

    Returns:
        ChronologicalSplit with train, validation, test, future_buffer DataFrames
    """
    # Verify data is sorted and complete
    assert df["date"].is_monotonic_increasing, "Data must be sorted by date"

    # Split by week number (1-indexed)
    df = df.copy()
    df["week_num"] = ((df["date"] - START_DATE).dt.days // 7) + 1

    train_df = df[df["week_num"] <= TRAIN_END_WEEK]
    val_df = df[(df["week_num"] > TRAIN_END_WEEK) & (df["week_num"] <= VALIDATION_END_WEEK)]
    test_df = df[(df["week_num"] > VALIDATION_END_WEEK) & (df["week_num"] <= TEST_END_WEEK)]
    future_buffer_df = df[df["week_num"] > TEST_END_WEEK]

    # Verify no overlap
    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    test_dates = set(test_df["date"])

    assert len(train_dates & val_dates) == 0, "Train/validation date overlap"
    assert len(train_dates & test_dates) == 0, "Train/test date overlap"
    assert len(val_dates & test_dates) == 0, "Validation/test date overlap"

    # Validation rolling origins (week numbers)
    # Origin 1: week 104 (end of train)
    # Origin 2: week 110
    # Origin 3: week 116
    # Origin 4: week 122
    validation_origins = [104, 110, 116, 122]

    return ChronologicalSplit(
        train=train_df.reset_index(drop=True),
        validation=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        future_buffer=future_buffer_df.reset_index(drop=True),
        train_end_week=TRAIN_END_WEEK,
        validation_end_week=VALIDATION_END_WEEK,
        test_end_week=TEST_END_WEEK,
        forecast_horizon=FORECAST_HORIZON,
        validation_origins=validation_origins,
    )


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save dataset to Parquet with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset from Parquet."""
    return pd.read_parquet(path)


def save_split_metadata(split: ChronologicalSplit, fingerprint: str, path: Path) -> None:
    """Save split metadata as JSON."""
    meta = {
        "dataset_fingerprint": fingerprint,
        "random_seed": RANDOM_SEED,
        "n_weeks": N_WEEKS,
        "start_date": START_DATE.isoformat(),
        "shops": SHOPS,
        "part_families": PART_FAMILIES,
        "n_series": len(SHOPS) * len(PART_FAMILIES),
        "train_end_week": split.train_end_week,
        "validation_end_week": split.validation_end_week,
        "test_end_week": split.test_end_week,
        "forecast_horizon": split.forecast_horizon,
        "validation_origins": split.validation_origins,
        "train_shape": list(split.train.shape),
        "validation_shape": list(split.validation.shape),
        "test_shape": list(split.test.shape),
        "future_buffer_shape": list(split.future_buffer.shape),
        "split_dates": {
            "train_start": split.train["date"].min().isoformat(),
            "train_end": split.train["date"].max().isoformat(),
            "validation_start": split.validation["date"].min().isoformat(),
            "validation_end": split.validation["date"].max().isoformat(),
            "test_start": split.test["date"].min().isoformat(),
            "test_end": split.test["date"].max().isoformat(),
            "future_buffer_start": (
                split.future_buffer["date"].min().isoformat()
                if len(split.future_buffer) > 0
                else None
            ),
            "future_buffer_end": (
                split.future_buffer["date"].max().isoformat()
                if len(split.future_buffer) > 0
                else None
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    """Generate synthetic data, split chronologically, and save artifacts."""
    df = generate_synthetic_demand()

    # Compute fingerprint
    fingerprint = compute_dataset_fingerprint(df)

    # Save full dataset
    data_path = DATA_DIR / "maintenance_demand.parquet"
    save_dataset(df, data_path)

    # Chronological split
    split = split_chronologically(df)

    if len(split.future_buffer) > 0:
        pass

    # Save split metadata
    meta_path = REPORTS_DIR / "split_metadata.json"
    save_split_metadata(split, fingerprint, meta_path)

    # Save individual splits for downstream use
    split.train.to_parquet(DATA_DIR / "train.parquet", index=False)
    split.validation.to_parquet(DATA_DIR / "validation.parquet", index=False)
    split.test.to_parquet(DATA_DIR / "test.parquet", index=False)
    split.future_buffer.to_parquet(DATA_DIR / "future_buffer.parquet", index=False)


if __name__ == "__main__":
    main()
