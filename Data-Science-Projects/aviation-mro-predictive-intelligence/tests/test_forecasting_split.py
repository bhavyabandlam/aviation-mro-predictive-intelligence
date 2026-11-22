"""Tests for chronological split logic in forecasting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.forecasting.data_prep import (
    FORECAST_HORIZON,
    ChronologicalSplit,
    generate_synthetic_demand,
    split_chronologically,
)


class TestChronologicalSplitIntegration:
    """Integration tests for chronological split."""

    def test_full_split_pipeline(self):
        """Test complete split pipeline with synthetic data."""
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        # Verify split structure
        assert isinstance(split, ChronologicalSplit)
        assert len(split.train) == 1560  # 104 weeks * 15 series
        assert len(split.validation) == 390  # 26 weeks * 15 series
        assert len(split.test) == 390  # 26 weeks * 15 series

    def test_no_data_leakage(self):
        """Ensure no temporal data leakage between splits."""
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        train_max = split.train["date"].max()
        val_min = split.validation["date"].min()
        val_max = split.validation["date"].max()
        test_min = split.test["date"].min()

        # Train ends before validation starts
        assert train_max < val_min
        # Validation ends before test starts
        assert val_max < test_min

    def test_week_number_consistency(self):
        """Verify week_num column is consistent across splits."""
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        # All splits should have week_num
        assert "week_num" in split.train.columns
        assert "week_num" in split.validation.columns
        assert "week_num" in split.test.columns

        # Week numbers should be in correct ranges
        assert split.train["week_num"].min() == 1
        assert split.train["week_num"].max() == 104
        assert split.validation["week_num"].min() == 105
        assert split.validation["week_num"].max() == 130
        assert split.test["week_num"].min() == 131
        assert split.test["week_num"].max() == 156

    def test_future_buffer_properties(self):
        """Test future buffer exists with correct properties."""
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        assert split.future_buffer is not None
        assert len(split.future_buffer) == 14 * 15  # 14 weeks * 15 series
        assert split.future_buffer["week_num"].min() == 157
        assert split.future_buffer["week_num"].max() == 170

    def test_validation_origins_format(self):
        """Test validation_origins are correct week numbers."""
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        assert split.validation_origins == [104, 110, 116, 122]

        # Each origin should be a valid week in train+validation
        for origin in split.validation_origins:
            assert 1 <= origin <= 130

    def test_forecast_horizon_constant(self):
        """Test FORECAST_HORIZON constant used correctly."""
        assert FORECAST_HORIZON == 12

    def test_all_series_present_in_each_split(self):
        """Ensure all 15 series present in each split."""
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        for name, subset in [
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ]:
            series_count = subset.groupby(["shop_id", "part_family"]).ngroups
            assert series_count == 15, f"{name} has {series_count} series, expected 15"

    def test_deterministic_splits(self):
        """Splits should be deterministic given same seed."""
        df1 = generate_synthetic_demand(seed=42)
        df2 = generate_synthetic_demand(seed=42)

        split1 = split_chronologically(df1)
        split2 = split_chronologically(df2)

        pd.testing.assert_frame_equal(split1.train, split2.train)
        pd.testing.assert_frame_equal(split1.validation, split2.validation)
        pd.testing.assert_frame_equal(split1.test, split2.test)
        pd.testing.assert_frame_equal(split1.future_buffer, split2.future_buffer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
