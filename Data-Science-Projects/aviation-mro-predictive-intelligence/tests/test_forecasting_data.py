"""Tests for forecasting data preparation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.forecasting.data_prep import (
    FORECAST_HORIZON,
    N_WEEKS,
    PART_FAMILIES,
    PRIMARY_TARGET,
    RANDOM_SEED,
    SHOPS,
    TARGET_COLUMNS,
    generate_synthetic_demand,
    load_dataset,
    split_chronologically,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class TestDataPrepConstants:
    """Test module constants are correctly defined."""

    def test_constants_values(self):
        assert SHOPS == ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"]
        assert PART_FAMILIES == ["ENGINE", "AVIONICS", "AIRFRAME"]
        assert FORECAST_HORIZON == 12
        assert RANDOM_SEED == 42
        assert N_WEEKS == 170

    def test_target_columns(self):
        assert TARGET_COLUMNS == ["demand_count", "labour_hours", "maintenance_expenditure"]
        assert PRIMARY_TARGET == "demand_count"


class TestGenerateSyntheticData:
    """Test synthetic maintenance demand generation."""

    def test_generates_correct_shape(self):
        df = generate_synthetic_demand()
        # 5 shops * 3 part_families * 170 weeks = 2550 rows
        assert len(df) == 2550
        assert list(df.columns) == [
            "date",
            "shop_id",
            "part_family",
            "demand_count",
            "labour_hours",
            "maintenance_expenditure",
        ]

    def test_generates_all_series(self):
        df = generate_synthetic_demand()
        series_ids = df.set_index(["shop_id", "part_family"]).index.unique()
        assert len(series_ids) == 15

    def test_dates_are_weekly(self):
        df = generate_synthetic_demand()
        # Check first series
        series = df[(df["shop_id"] == "SHOP_01") & (df["part_family"] == "ENGINE")]
        dates = series["date"].sort_values()
        diffs = dates.diff().dropna()
        # All should be 7 days
        assert all(diff == pd.Timedelta(days=7) for diff in diffs)

    def test_dates_span_correct_range(self):
        df = generate_synthetic_demand()
        assert df["date"].min() == pd.Timestamp("2022-01-03")
        # 170 weeks from 2022-01-03 is 2025-03-31
        assert df["date"].max() == pd.Timestamp("2025-03-31")

    def test_demand_non_negative(self):
        df = generate_synthetic_demand()
        assert (df["demand_count"] >= 0).all()

    def test_demand_is_integer(self):
        df = generate_synthetic_demand()
        assert df["demand_count"].dtype in (np.int64, np.int32, int)

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_demand(seed=42)
        df2 = generate_synthetic_demand(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_demand(seed=42)
        df2 = generate_synthetic_demand(seed=123)
        # Should differ in at least some values
        assert not df1["demand_count"].equals(df2["demand_count"])


class TestChronologicalSplit:
    """Test chronological splitting logic."""

    def test_split_creates_correct_sizes(self):
        df = generate_synthetic_demand()
        split = split_chronologically(df)

        # Train: 104 weeks * 15 series = 1560
        assert len(split.train) == 1560
        # Validation: 26 weeks * 15 series = 390
        assert len(split.validation) == 390
        # Test: 26 weeks * 15 series = 390
        assert len(split.test) == 390

    def test_split_maintains_chronological_order(self):
        df = generate_synthetic_demand()
        split = split_chronologically(df)

        # Train dates before validation dates before test dates
        assert split.train["date"].max() < split.validation["date"].min()
        assert split.validation["date"].max() < split.test["date"].min()

    def test_split_week_boundaries(self):
        df = generate_synthetic_demand()
        split = split_chronologically(df)

        # Train: weeks 1-104
        train_weeks = sorted(split.train["week_num"].unique())
        assert train_weeks == list(range(1, 105))

        # Validation: weeks 105-130
        val_weeks = sorted(split.validation["week_num"].unique())
        assert val_weeks == list(range(105, 131))

        # Test: weeks 131-156
        test_weeks = sorted(split.test["week_num"].unique())
        assert test_weeks == list(range(131, 157))

    def test_split_preserves_all_series(self):
        df = generate_synthetic_demand()
        split = split_chronologically(df)

        for subset_name, subset in [
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ]:
            series_ids = subset.set_index(["shop_id", "part_family"]).index.unique()
            assert len(series_ids) == 15, f"{subset_name} missing series"

    def test_validation_origins_correct(self):
        df = generate_synthetic_demand()
        split = split_chronologically(df)

        # Origins are at end of each validation fold (weeks 104, 110, 116, 122)
        # These correspond to the last week of each 6-week validation fold
        assert split.validation_origins == [104, 110, 116, 122]

    def test_future_buffer_exists(self):
        df = generate_synthetic_demand()
        split = split_chronologically(df)

        # Future buffer should be weeks 157-170 (14 weeks)
        assert split.future_buffer is not None
        buffer_weeks = sorted(split.future_buffer["week_num"].unique())
        assert buffer_weeks == list(range(157, 171))

    def test_split_reproducible(self):
        df = generate_synthetic_demand(seed=42)
        split1 = split_chronologically(df)
        split2 = split_chronologically(df)

        pd.testing.assert_frame_equal(split1.train, split2.train)
        pd.testing.assert_frame_equal(split1.validation, split2.validation)
        pd.testing.assert_frame_equal(split1.test, split2.test)


class TestDatasetFingerprint:
    """Test dataset fingerprinting."""

    def test_fingerprint_deterministic(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)
        fp1 = split.dataset_fingerprint
        fp2 = split.dataset_fingerprint
        assert fp1 == fp2

    def test_fingerprint_changes_with_data(self):
        df1 = generate_synthetic_demand(seed=42)
        df2 = generate_synthetic_demand(seed=123)
        split1 = split_chronologically(df1)
        split2 = split_chronologically(df2)
        assert split1.dataset_fingerprint != split2.dataset_fingerprint

    def test_fingerprint_format(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)
        fp = split.dataset_fingerprint
        assert isinstance(fp, str)
        assert len(fp) == 16  # SHA256 truncated to 16 chars
        assert all(c in "0123456789abcdef" for c in fp)


class TestLoadDataset:
    """Test loading the main dataset."""

    def test_load_dataset_if_exists(self):
        path = DATA_DIR / "maintenance_demand.parquet"
        if path.exists():
            df = load_dataset(path)
            assert len(df) == 2550
            assert list(df.columns) == [
                "date",
                "shop_id",
                "part_family",
                "demand_count",
                "labour_hours",
                "maintenance_expenditure",
            ]


class TestNewTargetColumns:
    """Test the new forecast target columns."""

    def test_labour_hours_non_negative(self):
        df = generate_synthetic_demand()
        assert (df["labour_hours"] >= 0).all()

    def test_maintenance_expenditure_non_negative(self):
        df = generate_synthetic_demand()
        assert (df["maintenance_expenditure"] >= 0).all()

    def test_labour_hours_correlated_with_demand(self):
        df = generate_synthetic_demand()
        # labour_hours should generally increase with demand_count
        corr = df["demand_count"].corr(df["labour_hours"])
        assert corr > 0.5, f"Expected positive correlation, got {corr}"

    def test_expenditure_correlated_with_labour_hours(self):
        df = generate_synthetic_demand()
        corr = df["labour_hours"].corr(df["maintenance_expenditure"])
        assert corr > 0.5, f"Expected positive correlation, got {corr}"

    def test_reproducible_new_columns(self):
        df1 = generate_synthetic_demand(seed=42)
        df2 = generate_synthetic_demand(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_new_columns(self):
        df1 = generate_synthetic_demand(seed=42)
        df2 = generate_synthetic_demand(seed=123)
        assert not df1["labour_hours"].equals(df2["labour_hours"])
        assert not df1["maintenance_expenditure"].equals(df2["maintenance_expenditure"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
