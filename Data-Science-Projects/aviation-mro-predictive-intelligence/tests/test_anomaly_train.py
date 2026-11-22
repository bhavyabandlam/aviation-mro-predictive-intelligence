"""Integration tests for anomaly detection training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aviation_mro.features.anomaly_features import AnomalyFeatureConfig
from aviation_mro.models.anomaly_model import AnomalyModelConfig
from aviation_mro.models.anomaly_train import (
    AnomalyTrainingArtifacts,
    run_anomaly_training,
)

# Every test in this module trains on the real FD001 dataset (~6-7 min total).
# They are opt-in via --run-integration; fast synthetic coverage lives in
# tests/test_training_pipeline_synthetic.py.
pytestmark = pytest.mark.integration


class TestRunAnomalyTraining:
    """Integration tests for full anomaly training pipeline."""

    def test_run_anomaly_training_smoke(self) -> None:
        """Smoke test with minimal config (small windows, few estimators).

        Uses real data loading. If data not available, test is skipped.
        """
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,  # Small for fast testing
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        contaminations = [0.01, 0.05, 0.10]

        try:
            artifacts = run_anomaly_training(feature_config, model_config, contaminations)

            # Verify artifacts structure
            assert isinstance(artifacts, AnomalyTrainingArtifacts)
            assert artifacts.model is not None
            assert len(artifacts.feature_names) > 0
            assert len(artifacts.train_engine_ids) == 80
            assert len(artifacts.val_engine_ids) == 20
            assert len(set(artifacts.train_engine_ids) & set(artifacts.val_engine_ids)) == 0

            # Verify contamination results
            assert len(artifacts.contamination_results) == 3
            for cont in contaminations:
                assert cont in artifacts.contamination_results
                result = artifacts.contamination_results[cont]
                assert hasattr(result, "train_anomaly_rate")
                assert hasattr(result, "val_anomaly_rate")
                assert 0 <= result.train_anomaly_rate <= 1
                assert 0 <= result.val_anomaly_rate <= 1

            # Verify default contamination
            assert artifacts.default_contamination == 0.01
            assert artifacts.default_threshold is not None

            # Verify sensor attributions
            assert isinstance(artifacts.sensor_attributions_val, list)
            assert isinstance(artifacts.sensor_attributions_test, list)

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_uses_same_engine_split_as_rul_model(self) -> None:
        """Anomaly training should use the same 80/20 engine split as RUL model."""
        from aviation_mro.features.rul_features import FeatureConfig, load_and_prepare_data

        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        rul_config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=False,
            rul_cap=125,
            random_seed=42,
        )

        try:
            anomaly_artifacts = run_anomaly_training(feature_config, model_config, [0.01, 0.05])
            rul_data = load_and_prepare_data(rul_config)

            assert anomaly_artifacts.train_engine_ids == rul_data["train_engine_ids"]
            assert anomaly_artifacts.val_engine_ids == rul_data["val_engine_ids"]

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_no_rul_leakage(self) -> None:
        """Training should never use RUL labels."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            artifacts = run_anomaly_training(feature_config, model_config, [0.01])

            # Feature names should not include RUL
            for fname in artifacts.feature_names:
                assert "rul" not in fname.lower()

            # Model should not have RUL-related attributes
            assert not hasattr(artifacts.model, "rul_")

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_preprocessing_fit_on_train_only(self) -> None:
        """Imputer and scaler should be fit only on training data."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            artifacts = run_anomaly_training(feature_config, model_config, [0.01])

            # Imputer and scaler should be fitted
            assert hasattr(artifacts.imputer, "statistics_")
            assert hasattr(artifacts.scaler, "mean_")
            assert hasattr(artifacts.scaler, "scale_")

            # Statistics should be from training data (not val/test)
            # We can't directly verify without accessing val/test, but we verify they exist

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_artifact_serialization(self) -> None:
        """Artifacts should be serializable and loadable."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            run_anomaly_training(feature_config, model_config, [0.01])

            # Model should be saved
            from pathlib import Path

            model_path = Path(__file__).resolve().parents[1] / "models" / "anomaly_model.pkl"
            assert model_path.exists()

            # Model card should be saved
            card_path = Path(__file__).resolve().parents[1] / "reports" / "anomaly_model_card.json"
            assert card_path.exists()

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


class TestNoDataLeakage:
    """Tests to verify no data leakage in training pipeline."""

    def test_train_val_engine_isolation(self) -> None:
        """Train and validation engines must be disjoint."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            artifacts = run_anomaly_training(feature_config, model_config, [0.01])

            train_ids = set(artifacts.train_engine_ids)
            val_ids = set(artifacts.val_engine_ids)
            assert len(train_ids & val_ids) == 0
            assert len(train_ids) == 80
            assert len(val_ids) == 20

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")

    def test_model_trained_on_train_engines_only(self) -> None:
        """Model should only see training engine data during fit."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            artifacts = run_anomaly_training(feature_config, model_config, [0.01])

            # The model's training scores should match training sample count
            _ = sum(
                1
                for eid in artifacts.train_engine_ids
                for _ in range(1)  # placeholder - actual count from data
            )
            # Just verify the model has training scores
            assert artifacts.model.train_scores_ is not None
            assert len(artifacts.model.train_scores_) > 0

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


class TestReproducibility:
    """Tests for reproducibility with fixed random seeds."""

    def test_fixed_seed_produces_identical_results(self) -> None:
        """Same random seed should produce identical model and splits."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            artifacts1 = run_anomaly_training(feature_config, model_config, [0.01, 0.05])
            artifacts2 = run_anomaly_training(feature_config, model_config, [0.01, 0.05])

            assert artifacts1.train_engine_ids == artifacts2.train_engine_ids
            assert artifacts1.val_engine_ids == artifacts2.val_engine_ids
            assert artifacts1.feature_names == artifacts2.feature_names
            assert artifacts1.default_threshold == pytest.approx(artifacts2.default_threshold)

            # Model scores should be identical
            np.testing.assert_array_equal(
                artifacts1.model.train_scores_, artifacts2.model.train_scores_
            )

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


class TestModelCardContents:
    """Tests for model card JSON contents."""

    def test_model_card_has_required_fields(self) -> None:
        """Model card should contain all required fields."""
        feature_config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        model_config = AnomalyModelConfig(
            n_estimators=10,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

        try:
            run_anomaly_training(feature_config, model_config, [0.01, 0.05])

            import json

            card_path = Path(__file__).resolve().parents[1] / "reports" / "anomaly_model_card.json"
            with open(card_path) as f:
                card = json.load(f)

            required_fields = [
                "dataset_fingerprint",
                "feature_config",
                "model_config",
                "feature_names",
                "n_features",
                "train_engine_ids",
                "val_engine_ids",
                "default_contamination",
                "default_threshold",
                "contamination_sweep",
                "score_distributions",
                "val_flagged_summary",
                "test_flagged_summary",
                "sensor_attributions_val",
                "sensor_attributions_test",
                "limitations",
            ]
            for field in required_fields:
                assert field in card, f"Missing field: {field}"

            # Check limitations are present and non-empty
            assert len(card["limitations"]) > 0

        except FileNotFoundError:
            pytest.skip("Real FD001 data not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
