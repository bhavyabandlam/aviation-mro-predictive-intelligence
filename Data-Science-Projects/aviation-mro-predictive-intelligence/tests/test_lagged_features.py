"""Tests for lagged sensor features (SRS v1.1 FR-05).

Verifies past-only, per-engine lagged sensor values with no future leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from aviation_mro.features.rul_features import (  # noqa: E402
    CONSTANT_SENSORS,
    FeatureConfig,
    compute_lagged_features,
    engineer_features,
    get_feature_columns,
)


class TestComputeLaggedFeatures:
    def test_lag_1_equals_previous_cycle_same_engine(self) -> None:
        """At cycle t, sensor_lag_1 equals the sensor value at cycle t-1."""
        df = pd.DataFrame(
            {
                "cycle": list(range(1, 6)) * 2,
                "unit_id": [1] * 5 + [2] * 5,
                "sensor_3": [100.0, 101.0, 102.0, 103.0, 104.0, 200.0, 201.0, 202.0, 203.0, 204.0],
            }
        )
        result = compute_lagged_features(df, ["sensor_3"], (1,))

        eng1 = result[result["unit_id"] == 1].sort_values("cycle")
        # cycle 2 lag_1 = cycle 1 value
        assert eng1[eng1["cycle"] == 2]["sensor_3_lag_1"].values[0] == pytest.approx(100.0)
        # cycle 3 lag_1 = cycle 2 value
        assert eng1[eng1["cycle"] == 3]["sensor_3_lag_1"].values[0] == pytest.approx(101.0)
        # cycle 1 lag_1 = NaN (no prior observation)
        assert np.isnan(eng1[eng1["cycle"] == 1]["sensor_3_lag_1"].values[0])

    def test_lag_2_equals_two_cycles_ago(self) -> None:
        """At cycle t, sensor_lag_2 equals the sensor value at cycle t-2."""
        df = pd.DataFrame(
            {
                "cycle": list(range(1, 6)),
                "unit_id": [1] * 5,
                "sensor_3": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        result = compute_lagged_features(df, ["sensor_3"], (2,))
        eng1 = result.sort_values("cycle")
        # cycle 3 lag_2 = cycle 1 value
        assert eng1[eng1["cycle"] == 3]["sensor_3_lag_2"].values[0] == pytest.approx(100.0)
        # cycle 1 and 2 lag_2 = NaN
        assert eng1[eng1["cycle"] == 1]["sensor_3_lag_2"].isna().all()
        assert eng1[eng1["cycle"] == 2]["sensor_3_lag_2"].isna().all()

    def test_per_engine_isolation(self) -> None:
        """Lagged features computed independently per engine."""
        df = pd.DataFrame(
            {
                "cycle": list(range(1, 4)) * 2,
                "unit_id": [1] * 3 + [2] * 3,
                "sensor_3": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )
        result = compute_lagged_features(df, ["sensor_3"], (1,))
        eng1_c2 = result[(result["unit_id"] == 1) & (result["cycle"] == 2)][
            "sensor_3_lag_1"
        ].values[0]
        eng2_c2 = result[(result["unit_id"] == 2) & (result["cycle"] == 2)][
            "sensor_3_lag_1"
        ].values[0]
        assert eng1_c2 == pytest.approx(10.0)
        assert eng2_c2 == pytest.approx(100.0)

    def test_no_future_leakage(self) -> None:
        """Lagged value at cycle t must never equal a future cycle's value."""
        # Strictly increasing sensor so any future reference is detectable.
        df = pd.DataFrame(
            {
                "cycle": list(range(1, 11)),
                "unit_id": [1] * 10,
                "sensor_3": [float(c) for c in range(1, 11)],
            }
        )
        result = compute_lagged_features(df, ["sensor_3"], (1, 3))
        result = result.sort_values("cycle")
        for _, row in result.iterrows():
            t = int(row["cycle"])
            # lag_1 at cycle t must equal value at t-1, never t or t+1
            lag1 = row["sensor_3_lag_1"]
            if pd.notna(lag1):
                assert lag1 == pytest.approx(float(t - 1))
            lag3 = row["sensor_3_lag_3"]
            if pd.notna(lag3):
                assert lag3 == pytest.approx(float(t - 3))


class TestLaggedFeaturesInPipeline:
    def test_engineer_features_adds_lagged_columns(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        config = FeatureConfig(
            window_sizes=(3,),
            include_op_settings=False,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
            lags=(1, 2),
        )
        result = engineer_features(synthetic_train_data, config)
        assert "sensor_3_lag_1" in result.columns
        assert "sensor_3_lag_2" in result.columns

    def test_lagged_columns_in_feature_cols(self, synthetic_train_data: pd.DataFrame) -> None:
        config = FeatureConfig(
            window_sizes=(3,),
            include_op_settings=False,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
            lags=(1,),
        )
        result = engineer_features(synthetic_train_data, config)
        feature_cols = get_feature_columns(result, config)
        lag_cols = [c for c in feature_cols if "_lag_" in c]
        assert len(lag_cols) > 0
        assert "sensor_3_lag_1" in feature_cols

    def test_default_config_has_lags(self) -> None:
        config = FeatureConfig()
        assert config.lags == (1, 2)

    def test_empty_lags_no_columns(self, synthetic_train_data: pd.DataFrame) -> None:
        config = FeatureConfig(
            window_sizes=(3,),
            include_op_settings=False,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
            lags=(),
        )
        result = engineer_features(synthetic_train_data, config)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        assert len(lag_cols) == 0

    def test_lag_excludes_constant_sensors(self, synthetic_train_data: pd.DataFrame) -> None:
        config = FeatureConfig(
            window_sizes=(3,),
            include_op_settings=False,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
            lags=(1,),
        )
        result = engineer_features(synthetic_train_data, config)
        for sensor in CONSTANT_SENSORS:
            lag_cols = [c for c in result.columns if c.startswith(f"{sensor}_lag_")]
            assert len(lag_cols) == 0, f"Constant sensor {sensor} should not have lags"

    def test_lag_past_only_at_each_cycle(self, synthetic_train_data: pd.DataFrame) -> None:
        """Verify lagged value at cycle t equals observed value at cycle t-lag."""
        config = FeatureConfig(
            window_sizes=(3,),
            include_op_settings=False,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
            lags=(1,),
        )
        result = engineer_features(synthetic_train_data, config)
        # Engine 1: max_cycle=10
        eng1 = result[result["unit_id"] == 1].sort_values("cycle")
        for _, row in eng1.iterrows():
            t = int(row["cycle"])
            lag_val = row["sensor_3_lag_1"]
            if pd.notna(lag_val):
                prior = synthetic_train_data[
                    (synthetic_train_data["unit_id"] == 1)
                    & (synthetic_train_data["cycle"] == t - 1)
                ]["sensor_3"]
                assert len(prior) == 1
                assert lag_val == pytest.approx(float(prior.values[0]))
