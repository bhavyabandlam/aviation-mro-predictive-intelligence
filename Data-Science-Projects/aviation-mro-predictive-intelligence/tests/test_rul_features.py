"""Unit tests for RUL feature engineering functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aviation_mro.features.rul_features import (
    CONSTANT_SENSORS,
    INFORMATIVE_SENSORS,
    FeatureConfig,
    _rolling_slope,
    _slope,
    compute_rolling_features,
    create_rul_labels,
    engineer_features,
    get_feature_columns,
    load_and_prepare_data,
    prepare_test_last_cycle,
    split_by_engine,
)


class TestCreateRULLabels:
    """Tests for RUL target calculation."""

    def test_rul_calculation_basic(self, synthetic_train_data: pd.DataFrame) -> None:
        """RUL = max_cycle - current_cycle for each engine."""
        df = create_rul_labels(synthetic_train_data, cap=125)

        # Engine 1: max_cycle=10, so RUL at cycle 1 = 9, at cycle 10 = 0
        eng1 = df[df["unit_id"] == 1].sort_values("cycle")
        assert eng1["rul"].iloc[0] == 9
        assert eng1["rul"].iloc[-1] == 0

        # Engine 2: max_cycle=8, RUL at cycle 1 = 7, at cycle 8 = 0
        eng2 = df[df["unit_id"] == 2].sort_values("cycle")
        assert eng2["rul"].iloc[0] == 7
        assert eng2["rul"].iloc[-1] == 0

    def test_rul_nonnegative(self, synthetic_train_data: pd.DataFrame) -> None:
        """RUL values must never be negative."""
        df = create_rul_labels(synthetic_train_data, cap=125)
        assert (df["rul"] >= 0).all()

    def test_rul_capping(self, synthetic_train_data: pd.DataFrame) -> None:
        """RUL values should be capped at specified maximum."""
        df = create_rul_labels(synthetic_train_data, cap=5)
        assert (df["rul_capped"] <= 5).all()
        # Original RUL can exceed cap
        assert (df["rul"] > 5).any()

    def test_rul_calculation_multiple_engines(self, synthetic_train_data: pd.DataFrame) -> None:
        """RUL calculated independently per engine."""
        df = create_rul_labels(synthetic_train_data, cap=125)

        # Each engine has its own max cycle
        max_cycles = df.groupby("unit_id")["cycle"].transform("max")
        expected_rul = max_cycles - df["cycle"]
        pd.testing.assert_series_equal(df["rul"], expected_rul, check_names=False)

    def test_original_dataframe_unmodified(self, synthetic_train_data: pd.DataFrame) -> None:
        """Input dataframe should not be mutated."""
        original = synthetic_train_data.copy()
        create_rul_labels(synthetic_train_data, cap=125)
        pd.testing.assert_frame_equal(synthetic_train_data, original)


class TestComputeRollingFeatures:
    """Tests for past-only rolling feature computation."""

    def test_rolling_features_past_only(self) -> None:
        """Rolling features must use only past data (no future leakage)."""
        # Create simple single-engine data
        df = pd.DataFrame(
            {
                "cycle": range(1, 11),
                "unit_id": [1] * 10,
                "sensor_3": np.arange(100.0, 110.0),  # linear trend
            }
        )

        result = compute_rolling_features(df, ["sensor_3"], (3, 5))

        # At cycle 1, rolling mean with min_periods=1 should equal the value itself
        assert result["sensor_3_mean_3"].iloc[0] == pytest.approx(100.0)
        assert result["sensor_3_mean_5"].iloc[0] == pytest.approx(100.0)

        # At cycle 2, mean of first 2 values
        assert result["sensor_3_mean_3"].iloc[1] == pytest.approx(100.5)

        # No NaN at start due to min_periods=1
        assert not result["sensor_3_mean_3"].isna().any()
        assert not result["sensor_3_mean_5"].isna().any()

    def test_rolling_slope_requires_min_periods(self) -> None:
        """Rolling slope needs min_periods=2, so first value is NaN."""
        df = pd.DataFrame(
            {
                "cycle": range(1, 11),
                "unit_id": [1] * 10,
                "sensor_3": np.arange(100.0, 110.0),
            }
        )

        result = compute_rolling_features(df, ["sensor_3"], (3, 5))

        # Slope at first cycle should be NaN (need at least 2 points)
        assert np.isnan(result["sensor_3_slope_3"].iloc[0])
        assert np.isnan(result["sensor_3_slope_5"].iloc[0])

        # From cycle 2 onwards, slope should be computable
        assert not np.isnan(result["sensor_3_slope_3"].iloc[1])

    def test_rolling_features_per_engine_isolation(self) -> None:
        """Rolling features computed independently per engine via engineer_features."""
        df = pd.DataFrame(
            {
                "cycle": list(range(1, 6)) + list(range(1, 6)),
                "unit_id": [1] * 5 + [2] * 5,
                "sensor_3": [100, 101, 102, 103, 104, 200, 201, 202, 203, 204],
            }
        )

        # Use engineer_features which groups by unit_id
        from aviation_mro.features.rul_features import FeatureConfig

        config = FeatureConfig(
            window_sizes=(3,),
            include_op_settings=False,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
        )

        result = engineer_features(df, config)

        # Engine 1 rolling mean at cycle 3 = mean(100, 101, 102) = 101
        eng1_cycle3 = result[(result["unit_id"] == 1) & (result["cycle"] == 3)]
        assert eng1_cycle3["sensor_3_mean_3"].values[0] == pytest.approx(101.0)

        # Engine 2 rolling mean at cycle 3 = mean(200, 201, 202) = 201
        eng2_cycle3 = result[(result["unit_id"] == 2) & (result["cycle"] == 3)]
        assert eng2_cycle3["sensor_3_mean_3"].values[0] == pytest.approx(201.0)


class TestRollingSlopeEquivalence:
    """Numerical equivalence between vectorized _rolling_slope and reference _slope."""

    def test_slope_single_value_returns_nan(self) -> None:
        """Window with 1 value: both implementations return NaN."""
        x = np.array([5.0])
        assert np.isnan(_slope(x))
        result = _rolling_slope(pd.Series(x), w=3)
        assert np.isnan(result.iloc[0])

    def test_slope_two_points_matches_polyfit(self) -> None:
        """Two-point slope equals difference."""
        x = np.array([10.0, 20.0])
        assert _slope(x) == pytest.approx(10.0)
        result = _rolling_slope(pd.Series(x), w=3)
        assert result.iloc[1] == pytest.approx(10.0)

    def test_slope_linear_data_exact(self) -> None:
        """Perfectly linear data: slope equals trend for all window sizes."""
        for w in (2, 3, 5):
            x = np.arange(1, 21, dtype=float) * 2.5 + 10.0  # slope = 2.5
            result = _rolling_slope(pd.Series(x), w=w)
            # All full windows should have slope 2.5
            valid = result.dropna()
            assert len(valid) > 0
            assert np.allclose(valid.to_numpy(), 2.5, atol=1e-10)

    def test_slope_constant_data_zero(self) -> None:
        """Constant data: slope is 0."""
        x = np.full(20, 42.0)
        result = _rolling_slope(pd.Series(x), w=5)
        valid = result.dropna()
        assert np.allclose(valid.to_numpy(), 0.0, atol=1e-12)

    def test_slope_with_nan_values(self) -> None:
        """Windows containing NaN produce NaN in both implementations."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0])
        series = pd.Series(x)
        result = _rolling_slope(series, w=3)

        # Manually compute expected using _slope per window
        expected_vals = []
        for i in range(len(x)):
            start = max(0, i - 3 + 1)
            window = x[start : i + 1]
            if np.isnan(window).any() or len(window) < 2:
                expected_vals.append(np.nan)
            else:
                expected_vals.append(_slope(window))
        expected = pd.Series(expected_vals, index=series.index)

        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_slope_boundary_partial_windows(self) -> None:
        """At series start, partial windows match per-window _slope."""
        x = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        series = pd.Series(x)
        result = _rolling_slope(series, w=3)

        for i in range(len(x)):
            start = max(0, i - 3 + 1)
            window = x[start : i + 1]
            if len(window) < 2 or np.isnan(window).any():
                assert np.isnan(result.iloc[i]), f"index {i}: expected NaN"
            else:
                assert result.iloc[i] == pytest.approx(_slope(window), abs=1e-10), (
                    f"index {i}: mismatch"
                )

    def test_slope_random_data_matches_per_window(self) -> None:
        """Random data: vectorized matches per-window _slope for all positions."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 50)
        series = pd.Series(x)
        result = _rolling_slope(series, w=5)

        for i in range(len(x)):
            start = max(0, i - 5 + 1)
            window = x[start : i + 1]
            if np.isnan(window).any() or len(window) < 2:
                assert np.isnan(result.iloc[i]), f"index {i}: expected NaN"
            else:
                assert result.iloc[i] == pytest.approx(_slope(window), abs=1e-10), (
                    f"index {i}: mismatch"
                )

    def test_slope_feature_column_values_match(self) -> None:
        """compute_rolling_features slope column matches reference _slope per window."""
        df = pd.DataFrame(
            {
                "cycle": range(1, 21),
                "unit_id": [1] * 20,
                "sensor_3": np.arange(100.0, 120.0, dtype=float),
            }
        )
        result = compute_rolling_features(df, ["sensor_3"], (3, 5))

        for w in (3, 5):
            col = f"sensor_3_slope_{w}"
            x = df["sensor_3"].to_numpy()
            for i in range(len(x)):
                start = max(0, i - w + 1)
                window = x[start : i + 1]
                if len(window) < 2 or np.isnan(window).any():
                    assert np.isnan(result[col].iloc[i]), f"w={w}, index {i}"
                else:
                    assert result[col].iloc[i] == pytest.approx(_slope(window), abs=1e-10), (
                        f"w={w}, index {i}: {result[col].iloc[i]} != {_slope(window)}"
                    )


class TestEngineerFeatures:
    """Tests for full feature engineering pipeline."""

    def test_engineer_features_adds_rul(
        self, synthetic_train_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Training data gets RUL columns added."""
        result = engineer_features(synthetic_train_data, feature_config)
        assert "rul" in result.columns
        assert "rul_capped" in result.columns

    def test_engineer_features_adds_rolling_features(
        self, synthetic_train_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Rolling features are added for each informative sensor."""
        result = engineer_features(synthetic_train_data, feature_config)

        # Check some expected rolling columns exist
        assert "sensor_3_mean_3" in result.columns
        assert "sensor_3_std_3" in result.columns
        assert "sensor_3_slope_3" in result.columns
        assert "sensor_3_mean_5" in result.columns

    def test_engineer_features_excludes_cycle_norm_by_default(
        self, synthetic_train_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """cycle_norm should NOT be added by default (leakage-safe)."""
        result = engineer_features(synthetic_train_data, feature_config)
        assert "cycle_norm" not in result.columns

    def test_engineer_features_adds_cycle_norm_when_enabled(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Normalized cycle column added (0 to 1 per engine) when explicitly enabled."""
        config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=False,
            include_cycle_norm=True,
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_features(synthetic_train_data, config)

        assert "cycle_norm" in result.columns
        # Should be between 0 and 1
        assert (result["cycle_norm"] >= 0).all()
        assert (result["cycle_norm"] <= 1).all()
        # Last cycle of each engine should be 1.0
        last_cycles = result.groupby("unit_id")["cycle_norm"].last()
        assert (last_cycles == 1.0).all()

    def test_engineer_features_excludes_constant_sensors(
        self, synthetic_train_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Constant sensors should not generate rolling features."""
        result = engineer_features(synthetic_train_data, feature_config)

        # Constant sensors in CONSTANT_SENSORS should not have rolling features
        for sensor in CONSTANT_SENSORS:
            rolling_cols = [c for c in result.columns if c.startswith(f"{sensor}_")]
            assert len(rolling_cols) == 0, (
                f"Constant sensor {sensor} should not have rolling features"
            )

    def test_engineer_features_excludes_raw_sensors_from_feature_cols(
        self, synthetic_train_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Raw sensor columns excluded from feature column list."""
        result = engineer_features(synthetic_train_data, feature_config)
        feature_cols = get_feature_columns(result, feature_config)

        for sensor in INFORMATIVE_SENSORS:
            assert sensor not in feature_cols, (
                f"Raw sensor {sensor} should not be in feature columns"
            )


class TestGetFeatureColumns:
    """Tests for feature column extraction."""

    def test_feature_columns_exclude_targets_and_ids(
        self, synthetic_train_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Feature columns should not include targets, IDs, or raw sensors."""
        df = engineer_features(synthetic_train_data, feature_config)
        feature_cols = get_feature_columns(df, feature_config)

        excluded = {"unit_id", "cycle", "rul", "rul_capped"}
        excluded.update(INFORMATIVE_SENSORS)
        excluded.update(CONSTANT_SENSORS)

        for col in feature_cols:
            assert col not in excluded, f"Column {col} should be excluded"


class TestSplitByEngine:
    """Tests for engine-level train/validation split."""

    def test_split_by_engine_isolation(self, synthetic_train_data: pd.DataFrame) -> None:
        """Train and validation sets must have disjoint engine IDs."""
        train_ids = [1, 2]
        val_ids = [3]

        train_df, val_df = split_by_engine(synthetic_train_data, train_ids, val_ids)

        assert set(train_df["unit_id"].unique()) == {1, 2}
        assert set(val_df["unit_id"].unique()) == {3}
        assert len(set(train_ids) & set(val_ids)) == 0

    def test_split_preserves_all_rows(self, synthetic_train_data: pd.DataFrame) -> None:
        """Split should not drop any rows."""
        train_ids = [1, 2]
        val_ids = [3]

        train_df, val_df = split_by_engine(synthetic_train_data, train_ids, val_ids)

        total_split = len(train_df) + len(val_df)
        assert total_split == len(synthetic_train_data)


class TestPrepareTestLastCycle:
    """Tests for test set last-cycle extraction."""

    def test_prepare_test_last_cycle(self, synthetic_test_data: pd.DataFrame) -> None:
        """Should return only the last cycle per engine."""
        result = prepare_test_last_cycle(synthetic_test_data)

        # One row per engine
        assert len(result) == synthetic_test_data["unit_id"].nunique()

        # Each row should be the max cycle for that engine
        for unit_id in result["unit_id"].unique():
            engine_data = result[result["unit_id"] == unit_id]
            assert len(engine_data) == 1
            max_cycle = synthetic_test_data[synthetic_test_data["unit_id"] == unit_id][
                "cycle"
            ].max()
            assert engine_data["cycle"].values[0] == max_cycle


class TestLoadAndPrepareData:
    """Integration test for full data preparation pipeline."""

    def test_load_and_prepare_data_structure(self, feature_config: FeatureConfig) -> None:
        """Verify returned data structure has all required keys."""
        # This test uses the real data loading functions
        # We'll skip if real data not available, but test structure
        from aviation_mro.features.rul_features import load_raw_data

        try:
            load_raw_data("train")
            load_raw_data("test")
            load_raw_data("rul")

            # Create a mini config for faster testing
            test_config = FeatureConfig(
                window_sizes=(3, 5),
                include_op_settings=True,
                include_cycle_norm=True,
                rul_cap=125,
                random_seed=42,
            )

            data = load_and_prepare_data(test_config)

            required_keys = {
                "train_df",
                "val_df",
                "test_last",
                "rul_df",
                "feature_cols",
                "train_engine_ids",
                "val_engine_ids",
                "config",
            }
            assert set(data.keys()) == required_keys

            # Verify engine-level split
            assert len(set(data["train_engine_ids"]) & set(data["val_engine_ids"])) == 0
            assert (
                len(data["train_engine_ids"]) + len(data["val_engine_ids"]) == 100
            )  # FD001 has 100 engines

            # Verify test_last has one row per engine
            assert len(data["test_last"]) == 100

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
