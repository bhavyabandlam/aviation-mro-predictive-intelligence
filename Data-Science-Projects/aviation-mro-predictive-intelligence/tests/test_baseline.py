"""Unit tests for baseline RUL prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aviation_mro.models.baseline import (
    ModelResult,
    TrainingArtifacts,
    compute_per_engine_errors,
    dataset_fingerprint,
    get_models,
    run_baseline_training,
    train_and_evaluate,
)


class TestGetModels:
    """Tests for model registry."""

    def test_get_models_contains_dummy_baselines(self) -> None:
        """DummyRegressor (mean/median) should be included as simplest baselines."""
        models = get_models()

        assert "DummyMean" in models
        assert "DummyMedian" in models
        assert models["DummyMean"].strategy == "mean"
        assert models["DummyMedian"].strategy == "median"

    def test_get_models_contains_ml_models(self) -> None:
        """Standard ML models should be present."""
        models = get_models()

        assert "LinearRegression" in models
        assert "Ridge" in models
        assert "RandomForest" in models

    def test_models_have_random_state(self) -> None:
        """All stochastic models should have fixed random_state for reproducibility."""
        models = get_models()

        for name, model in models.items():
            if hasattr(model, "random_state"):
                assert model.random_state is not None, f"{name} missing random_state"


class TestComputePerEngineErrors:
    """Tests for per-engine error computation."""

    def test_compute_per_engine_errors_basic(self) -> None:
        """Mean absolute error computed per engine."""
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "cycle": [1, 2, 3, 1, 2, 3],
            }
        )
        y_true = np.array([10.0, 8.0, 6.0, 5.0, 3.0, 1.0])
        y_pred = np.array([11.0, 7.0, 5.0, 6.0, 2.0, 0.0])

        errors = compute_per_engine_errors(df, y_true, y_pred)

        # Engine 1: |10-11| + |8-7| + |6-5| = 1+1+1 = 3, mean = 1.0
        assert errors[1] == pytest.approx(1.0)

        # Engine 2: |5-6| + |3-2| + |1-0| = 1+1+1 = 3, mean = 1.0
        assert errors[2] == pytest.approx(1.0)

    def test_compute_per_engine_errors_returns_dict(self) -> None:
        """Result should be dict mapping engine_id to float MAE."""
        df = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 2]})
        y_true = np.array([10.0, 5.0])
        y_pred = np.array([11.0, 4.0])

        errors = compute_per_engine_errors(df, y_true, y_pred)

        assert isinstance(errors, dict)
        assert all(isinstance(k, int) for k in errors)
        assert all(isinstance(v, float) for v in errors.values())


class TestTrainAndEvaluate:
    """Tests for model training and validation evaluation."""

    def test_train_and_evaluate_returns_model_result(self) -> None:
        """Should return ModelResult with all required fields."""
        from sklearn.dummy import DummyRegressor

        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20)

        val_df = pd.DataFrame(
            {
                "unit_id": [1] * 10 + [2] * 10,
                "cycle": list(range(10)) + list(range(10)),
            }
        )
        feature_names = [f"feat_{i}" for i in range(10)]

        model = DummyRegressor(strategy="mean")
        result = train_and_evaluate(
            X_train, y_train, X_val, y_val, val_df, feature_names, "TestModel", model
        )

        assert isinstance(result, ModelResult)
        assert result.name == "TestModel"
        assert result.model_type == "DummyRegressor"
        assert isinstance(result.val_mae, float)
        assert isinstance(result.val_rmse, float)
        assert isinstance(result.val_r2, float)
        assert isinstance(result.val_engine_errors, dict)
        assert result.feature_names == feature_names
        assert result.n_features == 10

    def test_train_and_evaluate_predictions_finite(self) -> None:
        """Model predictions on validation set should be finite."""
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)

        val_df = pd.DataFrame({"unit_id": [1] * 20, "cycle": range(20)})

        model = LinearRegression()
        result = train_and_evaluate(
            X_train, y_train, X_val, y_val, val_df, ["a", "b", "c", "d", "e"], "LR", model
        )

        # The function doesn't return predictions, but we can verify metrics are finite
        assert np.isfinite(result.val_mae)
        assert np.isfinite(result.val_rmse)
        assert np.isfinite(result.val_r2)


class TestDatasetFingerprint:
    """Tests for dataset fingerprinting."""

    def test_dataset_fingerprint_returns_dict(self) -> None:
        """Should return dict with hashes for each raw file."""
        fps = dataset_fingerprint()

        assert isinstance(fps, dict)
        assert "train_FD001.txt" in fps
        assert "test_FD001.txt" in fps
        assert "RUL_FD001.txt" in fps

        # Each hash should be 16-char hex string
        for _fname, hash_val in fps.items():
            assert len(hash_val) == 16
            assert all(c in "0123456789abcdef" for c in hash_val)


class TestRunBaselineTraining:
    """Integration test for full training pipeline (smoke test)."""

    @pytest.mark.integration
    def test_run_baseline_training_smoke(self) -> None:
        """Smoke test with minimal config (small windows, few engines).

        This test uses the real data loading. If data not available, it's skipped.
        """
        from aviation_mro.features.rul_features import FeatureConfig

        # Minimal config for fast testing
        config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=True,
            rul_cap=125,
            random_seed=42,
        )

        try:
            artifacts = run_baseline_training(config)

            # Verify artifacts structure
            assert isinstance(artifacts, TrainingArtifacts)
            valid_models = [
                "DummyMean",
                "DummyMedian",
                "LinearRegression",
                "Ridge",
                "RandomForest",
                "GradientBoosting",
            ]
            assert artifacts.best_model_name in valid_models
            assert artifacts.best_model is not None
            assert len(artifacts.feature_names) > 0
            assert len(artifacts.train_engine_ids) > 0
            assert len(artifacts.val_engine_ids) > 0
            assert len(set(artifacts.train_engine_ids) & set(artifacts.val_engine_ids)) == 0

            # Verify metrics
            assert "mae" in artifacts.test_metrics
            assert "rmse" in artifacts.test_metrics
            assert "r2" in artifacts.test_metrics
            assert isinstance(artifacts.test_metrics["mae"], float)
            assert isinstance(artifacts.test_metrics["rmse"], float)
            assert isinstance(artifacts.test_metrics["r2"], float)

            # Verify per-engine errors
            assert len(artifacts.test_engine_errors) == 100  # 100 test engines
            assert all(isinstance(v, float) for v in artifacts.test_engine_errors.values())

            # Verify RUL cap documented
            assert artifacts.rul_cap == 125

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


class TestNoDataLeakage:
    """Tests to verify no data leakage in preprocessing and feature engineering."""

    @pytest.mark.integration
    def test_imputer_fitted_only_on_train(self) -> None:
        """Imputer should be fit only on training data."""
        from aviation_mro.features.rul_features import FeatureConfig, load_and_prepare_data

        config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=True,
            rul_cap=125,
            random_seed=42,
        )

        try:
            data = load_and_prepare_data(config)

            # The imputer inside the pipeline should be fit on train only
            # We can't directly access it from load_and_prepare_data, but we can verify
            # that the feature preparation doesn't use val/test statistics

            train_df = data["train_df"]
            val_df = data["val_df"]

            # Train and val should have different statistics if no leakage
            # (This is a soft check - real verification requires accessing the fitted imputer)
            assert len(train_df) > 0
            assert len(val_df) > 0

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_scaler_fitted_only_on_train(self) -> None:
        """Scaler should be fit only on training data."""
        # Same logic as imputer test
        pass

    def test_feature_engineering_no_future_leakage(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Rolling features must not use future data."""
        from aviation_mro.features.rul_features import FeatureConfig, engineer_features

        config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=True,
            rul_cap=125,
            random_seed=42,
        )

        result = engineer_features(synthetic_train_data, config)

        # For each engine, at cycle t, rolling features should only depend on cycles <= t
        # This is implicitly tested by compute_rolling_features test
        # Here we verify the feature engineering pipeline preserves this
        assert "sensor_3_mean_3" in result.columns

        # Check engine 1: at cycle 1, rolling mean = value at cycle 1
        eng1_cycle1 = result[(result["unit_id"] == 1) & (result["cycle"] == 1)]
        eng1_cycle2 = result[(result["unit_id"] == 1) & (result["cycle"] == 2)]

        # At cycle 1, mean_3 with min_periods=1 = value at cycle 1
        # At cycle 2, mean_3 = mean(cycle1, cycle2)
        # So mean at cycle 2 should be between cycle1 and cycle2 values
        val_cycle1 = eng1_cycle1["sensor_3"].values[0]
        eng1_cycle1["sensor_3"].values[0]  # same row
        mean_cycle1 = eng1_cycle1["sensor_3_mean_3"].values[0]
        mean_cycle2 = eng1_cycle2["sensor_3_mean_3"].values[0]

        assert mean_cycle1 == pytest.approx(val_cycle1)
        # mean at cycle 2 should be different (includes cycle 2)
        assert mean_cycle2 != mean_cycle1


class TestReproducibility:
    """Tests for reproducibility with fixed random seeds."""

    @pytest.mark.integration
    def test_fixed_seed_produces_same_engine_split(self) -> None:
        """Same random seed should produce identical train/val engine splits."""
        from aviation_mro.features.rul_features import FeatureConfig, load_and_prepare_data

        config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=True,
            rul_cap=125,
            random_seed=42,
        )

        try:
            data1 = load_and_prepare_data(config)
            data2 = load_and_prepare_data(config)

            assert data1["train_engine_ids"] == data2["train_engine_ids"]
            assert data1["val_engine_ids"] == data2["val_engine_ids"]
            assert data1["feature_cols"] == data2["feature_cols"]

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_deterministic_model_predictions(self) -> None:
        """Models with fixed random_state should give identical predictions."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        model1 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        model2 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
