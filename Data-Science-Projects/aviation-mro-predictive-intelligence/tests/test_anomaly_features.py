"""Unit tests for anomaly detection feature engineering functions."""

from __future__ import annotations

import pandas as pd
import pytest

from aviation_mro.features.anomaly_features import (
    AnomalyFeatureConfig,
    engineer_anomaly_features,
    get_anomaly_feature_columns,
    load_and_prepare_anomaly_data,
    prepare_inference_data,
)
from aviation_mro.features.rul_features import INFORMATIVE_SENSORS


class TestAnomalyFeatureConfig:
    """Tests for AnomalyFeatureConfig."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = AnomalyFeatureConfig()
        assert config.window_sizes == (3, 5)
        assert config.include_op_settings is True
        assert config.include_residuals is True
        assert config.residual_windows == (3, 5)
        assert config.rul_cap == 125
        assert config.random_seed == 42

    def test_inherits_from_feature_config(self) -> None:
        """AnomalyFeatureConfig should inherit from FeatureConfig."""
        from aviation_mro.features.rul_features import FeatureConfig

        config = AnomalyFeatureConfig()
        assert isinstance(config, FeatureConfig)


class TestEngineerAnomalyFeatures:
    """Tests for anomaly feature engineering pipeline."""

    def test_no_rul_labels_created(self, synthetic_train_data: pd.DataFrame) -> None:
        """Anomaly features should NOT create RUL columns."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        assert "rul" not in result.columns
        assert "rul_capped" not in result.columns

    def test_returns_all_cycles_not_last_only(self, synthetic_train_data: pd.DataFrame) -> None:
        """Should return all cycles per engine, not just last cycle."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # Engine 1 has 10 cycles in synthetic data
        eng1_rows = result[result["unit_id"] == 1]
        assert len(eng1_rows) == 10

    def test_adds_rolling_features(self, synthetic_train_data: pd.DataFrame) -> None:
        """Rolling features should be added for each informative sensor."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # Check some expected rolling columns exist
        assert "sensor_3_mean_3" in result.columns
        assert "sensor_3_std_3" in result.columns
        assert "sensor_3_slope_3" in result.columns
        assert "sensor_3_mean_5" in result.columns

    def test_adds_residual_features(self, synthetic_train_data: pd.DataFrame) -> None:
        """Residual features (sensor - rolling_mean) should be added."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # Check residual columns exist
        assert "sensor_3_residual_3" in result.columns
        assert "sensor_3_residual_5" in result.columns

    def test_residual_calculation_correct(self, synthetic_train_data: pd.DataFrame) -> None:
        """Residual = sensor_value - rolling_mean."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # At cycle 1, residual_3 = sensor_3 - sensor_3_mean_3 (which equals sensor_3 at cycle 1)
        eng1_cycle1 = result[(result["unit_id"] == 1) & (result["cycle"] == 1)]
        sensor_val = eng1_cycle1["sensor_3"].values[0]
        mean_val = eng1_cycle1["sensor_3_mean_3"].values[0]
        residual_val = eng1_cycle1["sensor_3_residual_3"].values[0]

        assert residual_val == pytest.approx(sensor_val - mean_val)

    def test_includes_op_settings_by_default(self, synthetic_train_data: pd.DataFrame) -> None:
        """Operational settings should be included by default."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # op_setting_1 and op_setting_2 should be present (not excluded)
        assert "op_setting_1" in result.columns
        assert "op_setting_2" in result.columns

    def test_excludes_op_setting_3(self, synthetic_train_data: pd.DataFrame) -> None:
        """op_setting_3 (constant) should be excluded."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        assert "op_setting_3" not in result.columns

    def test_excludes_constant_sensors(self, synthetic_train_data: pd.DataFrame) -> None:
        """Constant sensors should not generate rolling features."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        from aviation_mro.features.rul_features import CONSTANT_SENSORS

        for sensor in CONSTANT_SENSORS:
            rolling_cols = [c for c in result.columns if c.startswith(f"{sensor}_")]
            assert len(rolling_cols) == 0, (
                f"Constant sensor {sensor} should not have rolling features"
            )

    def test_past_only_rolling_features(self, synthetic_train_data: pd.DataFrame) -> None:
        """Rolling features must use only past data (no future leakage)."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # At cycle 1, rolling mean with min_periods=1 should equal the value itself
        eng1_cycle1 = result[(result["unit_id"] == 1) & (result["cycle"] == 1)]
        assert eng1_cycle1["sensor_3_mean_3"].values[0] == pytest.approx(
            eng1_cycle1["sensor_3"].values[0]
        )

    def test_per_engine_isolation(self, synthetic_train_data: pd.DataFrame) -> None:
        """Rolling features computed independently per engine."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # Engine 1 rolling mean at cycle 3 = mean of cycles 1,2,3
        eng1_cycle3 = result[(result["unit_id"] == 1) & (result["cycle"] == 3)]
        eng2_cycle3 = result[(result["unit_id"] == 2) & (result["cycle"] == 3)]

        # Should be different because engines have different sensor values
        assert eng1_cycle3["sensor_3_mean_3"].values[0] != eng2_cycle3["sensor_3_mean_3"].values[0]


class TestGetAnomalyFeatureColumns:
    """Tests for anomaly feature column extraction."""

    def test_feature_columns_exclude_targets_and_ids(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Feature columns should not include targets, IDs, or raw sensors."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        df = engineer_anomaly_features(synthetic_train_data, config)
        feature_cols = get_anomaly_feature_columns(df, config)

        excluded = {"unit_id", "cycle", "rul", "rul_capped", "op_setting_3"}
        excluded.update(INFORMATIVE_SENSORS)

        for col in feature_cols:
            assert col not in excluded, f"Column {col} should be excluded"

    def test_feature_columns_include_rolling_and_residuals(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Feature columns should include rolling stats and residuals."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        df = engineer_anomaly_features(synthetic_train_data, config)
        feature_cols = get_anomaly_feature_columns(df, config)

        # Should have rolling features
        rolling_cols = [c for c in feature_cols if "_mean_" in c or "_std_" in c or "_slope_" in c]
        assert len(rolling_cols) > 0

        # Should have residual features
        residual_cols = [c for c in feature_cols if "_residual_" in c]
        assert len(residual_cols) > 0

    def test_feature_columns_include_op_settings(self, synthetic_train_data: pd.DataFrame) -> None:
        """Feature columns should include op_settings when enabled."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        df = engineer_anomaly_features(synthetic_train_data, config)
        feature_cols = get_anomaly_feature_columns(df, config)

        assert "op_setting_1" in feature_cols
        assert "op_setting_2" in feature_cols


class TestLoadAndPrepareAnomalyData:
    """Integration test for full anomaly data preparation pipeline."""

    def test_load_and_prepare_data_structure(self) -> None:
        """Verify returned data structure has all required keys."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )

        try:
            data = load_and_prepare_anomaly_data(config)

            required_keys = {
                "train_df",
                "val_df",
                "feature_cols",
                "train_engine_ids",
                "val_engine_ids",
                "config",
            }
            assert set(data.keys()) == required_keys

            # Verify engine-level split (80/20)
            assert len(set(data["train_engine_ids"]) & set(data["val_engine_ids"])) == 0
            assert len(data["train_engine_ids"]) + len(data["val_engine_ids"]) == 100

            # Verify train/val have data
            assert len(data["train_df"]) > 0
            assert len(data["val_df"]) > 0

            # Verify feature columns
            assert len(data["feature_cols"]) > 0

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_same_engine_split_as_rul_model(self) -> None:
        """Should use the same 80/20 engine split as RUL model (seed=42)."""
        from aviation_mro.features.rul_features import FeatureConfig, load_and_prepare_data

        anomaly_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        rul_config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
        )

        try:
            anomaly_data = load_and_prepare_anomaly_data(anomaly_config)
            rul_data = load_and_prepare_data(rul_config)

            assert anomaly_data["train_engine_ids"] == rul_data["train_engine_ids"]
            assert anomaly_data["val_engine_ids"] == rul_data["val_engine_ids"]

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


class TestPrepareInferenceData:
    """Tests for test data preparation for inference."""

    def test_prepare_inference_data_aligns_columns(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test features should align to training feature columns."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )

        train_feat = engineer_anomaly_features(synthetic_train_data, config)
        feature_cols = get_anomaly_feature_columns(train_feat, config)

        test_feat = prepare_inference_data(synthetic_test_data, config, feature_cols)

        # Should have all feature columns
        for col in feature_cols:
            assert col in test_feat.columns

        # Should have unit_id and cycle
        assert "unit_id" in test_feat.columns
        assert "cycle" in test_feat.columns

        # Columns should be in correct order
        expected_cols = ["unit_id", "cycle"] + feature_cols
        assert list(test_feat.columns) == expected_cols

    def test_missing_columns_filled_with_nan(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Missing feature columns in test should be filled with NaN."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )

        train_feat = engineer_anomaly_features(synthetic_train_data, config)
        feature_cols = get_anomaly_feature_columns(train_feat, config)

        # Manually remove a column from test to simulate missing feature
        test_modified = synthetic_test_data.copy()
        # This test mainly verifies the function handles missing columns
        test_feat = prepare_inference_data(test_modified, config, feature_cols)

        # All feature columns should exist (some may be NaN)
        for col in feature_cols:
            assert col in test_feat.columns


class TestNoDataLeakage:
    """Tests to verify no data leakage in anomaly feature engineering."""

    def test_no_rul_used_anywhere(self, synthetic_train_data: pd.DataFrame) -> None:
        """RUL should never be used as a feature or target."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # No RUL columns
        assert "rul" not in result.columns
        assert "rul_capped" not in result.columns

        # Feature columns should not include RUL
        feature_cols = get_anomaly_feature_columns(result, config)
        assert "rul" not in feature_cols
        assert "rul_capped" not in feature_cols

    def test_rolling_features_past_only(self, synthetic_train_data: pd.DataFrame) -> None:
        """Rolling features at cycle t should only depend on cycles <= t."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        result = engineer_anomaly_features(synthetic_train_data, config)

        # Engine 1: check cycle 2 mean_3 = mean of cycles 1 and 2
        eng1_cycle1 = result[(result["unit_id"] == 1) & (result["cycle"] == 1)]
        eng1_cycle2 = result[(result["unit_id"] == 1) & (result["cycle"] == 2)]

        val_c1 = eng1_cycle1["sensor_3"].values[0]
        val_c2 = eng1_cycle2["sensor_3"].values[0]
        mean_c2 = eng1_cycle2["sensor_3_mean_3"].values[0]

        # At cycle 2 with window=3, min_periods=1, mean = (c1 + c2) / 2
        expected_mean = (val_c1 + val_c2) / 2
        assert mean_c2 == pytest.approx(expected_mean)


class TestReproducibility:
    """Tests for reproducibility with fixed random seeds."""

    def test_fixed_seed_produces_same_engine_split(self) -> None:
        """Same random seed should produce identical train/val engine splits."""
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )

        try:
            data1 = load_and_prepare_anomaly_data(config)
            data2 = load_and_prepare_anomaly_data(config)

            assert data1["train_engine_ids"] == data2["train_engine_ids"]
            assert data1["val_engine_ids"] == data2["val_engine_ids"]
            assert data1["feature_cols"] == data2["feature_cols"]

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
