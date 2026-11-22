"""Unit tests for IsolationForest anomaly model wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aviation_mro.models.anomaly_model import (
    AnomalyModelConfig,
    AnomalyResult,
    IsolationForestAnomalyDetector,
    contamination_sweep,
)


class TestAnomalyModelConfig:
    """Tests for AnomalyModelConfig."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = AnomalyModelConfig()
        assert config.n_estimators == 200
        assert config.contamination == 0.01
        assert config.random_state == 42
        assert config.n_jobs == -1

    def test_custom_config(self) -> None:
        """Custom config values should be preserved."""
        config = AnomalyModelConfig(
            n_estimators=100,
            contamination=0.05,
            random_state=123,
            n_jobs=1,
        )
        assert config.n_estimators == 100
        assert config.contamination == 0.05
        assert config.random_state == 123
        assert config.n_jobs == 1


class TestIsolationForestAnomalyDetector:
    """Tests for IsolationForestAnomalyDetector."""

    def test_fit_sets_attributes(self) -> None:
        """fit() should set all required attributes."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        assert detector.model is not None
        assert detector.feature_names == feature_names
        assert detector.train_scores_ is not None
        assert len(detector.train_scores_) == 100
        assert detector.threshold_ is not None
        assert isinstance(detector.threshold_, float)

    def test_fit_with_contamination_auto(self) -> None:
        """fit() should work with contamination='auto'."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, contamination="auto", random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        assert detector.model is not None
        assert detector.train_scores_ is not None

    def test_score_samples_returns_scores(self) -> None:
        """score_samples() should return anomaly scores."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        scores = detector.score_samples(X_test)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 20
        assert np.all(np.isfinite(scores))

    def test_predict_returns_flags(self) -> None:
        """predict() should return -1 (anomaly) or 1 (normal)."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        preds = detector.predict(X_test)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 20
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_flag_anomalies_uses_threshold(self) -> None:
        """flag_anomalies() should use threshold to flag anomalies."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        # Use fitted threshold
        flags = detector.flag_anomalies(X_test)

        assert isinstance(flags, np.ndarray)
        assert len(flags) == 20
        assert set(np.unique(flags)).issubset({0, 1})

        # Custom threshold should work
        custom_thresh = -1000.0  # Very low threshold -> no anomalies
        flags_custom = detector.flag_anomalies(X_test, threshold=custom_thresh)
        assert flags_custom.sum() == 0

    def test_reproducibility_fixed_seed(self) -> None:
        """Same random_state should produce identical results."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config1 = AnomalyModelConfig(n_estimators=10, random_state=42)
        config2 = AnomalyModelConfig(n_estimators=10, random_state=42)

        detector1 = IsolationForestAnomalyDetector(config1)
        detector1.fit(X_train, feature_names)
        scores1 = detector1.score_samples(X_test)

        detector2 = IsolationForestAnomalyDetector(config2)
        detector2.fit(X_train, feature_names)
        scores2 = detector2.score_samples(X_test)

        np.testing.assert_array_equal(scores1, scores2)

    def test_save_and_load(self, tmp_path) -> None:
        """Model should be saveable and loadable."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        model_path = tmp_path / "test_model.pkl"
        detector.save(model_path)

        # Load and verify
        loaded = IsolationForestAnomalyDetector.load(model_path)
        assert loaded.feature_names == detector.feature_names
        assert loaded.config.n_estimators == detector.config.n_estimators
        assert loaded.config.random_state == detector.config.random_state
        assert loaded.threshold_ == pytest.approx(detector.threshold_)

        # Scores should match
        X_test = np.random.randn(20, 10)
        scores_orig = detector.score_samples(X_test)
        scores_loaded = loaded.score_samples(X_test)
        np.testing.assert_array_almost_equal(scores_orig, scores_loaded)

    def test_raises_if_not_fitted(self) -> None:
        """Methods should raise if called before fit()."""
        detector = IsolationForestAnomalyDetector()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.score_samples(X)

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(X)

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.flag_anomalies(X)


class TestComputeSensorAttribution:
    """Tests for sensor attribution computation."""

    def test_compute_sensor_attribution_returns_list(self) -> None:
        """Should return list of SensorAttribution objects."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        feature_names = [
            "sensor_3_mean_3",
            "sensor_3_std_3",
            "sensor_4_mean_3",
            "sensor_4_std_3",
            "sensor_7_mean_3",
            "sensor_7_std_3",
            "sensor_11_mean_3",
            "sensor_11_std_3",
            "sensor_12_mean_3",
            "sensor_12_std_3",
        ]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        X_flagged = X_train[:5]  # Use first 5 as "flagged"
        df_meta = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2, 3],
                "cycle": [10, 11, 5, 6, 8],
            }
        )

        attributions = detector.compute_sensor_attribution(
            X_flagged, df_meta, feature_names, top_k=3
        )

        assert isinstance(attributions, list)
        assert len(attributions) == 5

        for attr in attributions:
            assert hasattr(attr, "unit_id")
            assert hasattr(attr, "cycle")
            assert hasattr(attr, "anomaly_score")
            assert hasattr(attr, "top_sensors")
            assert hasattr(attr, "all_contributions")
            assert len(attr.top_sensors) <= 3

    def test_top_sensors_sorted_by_contribution(self) -> None:
        """Top sensors should be sorted by contribution (descending)."""
        np.random.seed(42)
        X_train = np.random.randn(100, 6)
        feature_names = [
            "sensor_3_mean_3",
            "sensor_3_std_3",
            "sensor_4_mean_3",
            "sensor_4_std_3",
            "sensor_7_mean_3",
            "sensor_7_std_3",
        ]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        X_flagged = X_train[:1]
        df_meta = pd.DataFrame({"unit_id": [1], "cycle": [10]})

        attributions = detector.compute_sensor_attribution(
            X_flagged, df_meta, feature_names, top_k=3
        )

        # Contributions should be descending
        contributions = [c for _, c in attributions[0].top_sensors]
        assert contributions == sorted(contributions, reverse=True)


class TestContaminationSweep:
    """Tests for contamination threshold sweep."""

    def test_contamination_sweep_returns_results(self) -> None:
        """Should return dict mapping contamination to AnomalyResult."""
        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        X_val = np.random.randn(50, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        contaminations = [0.01, 0.05, 0.10]
        results = contamination_sweep(detector, X_train, X_val, None, contaminations)

        assert isinstance(results, dict)
        assert len(results) == 3

        for cont, result in results.items():
            assert isinstance(result, AnomalyResult)
            assert result.contamination == cont
            assert isinstance(result.threshold, float)
            assert isinstance(result.train_anomaly_rate, float)
            assert isinstance(result.val_anomaly_rate, float)
            assert 0 <= result.train_anomaly_rate <= 1
            assert 0 <= result.val_anomaly_rate <= 1

    def test_contamination_sweep_with_test(self) -> None:
        """Should include test results when X_test provided."""
        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        X_val = np.random.randn(50, 10)
        X_test = np.random.randn(100, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        contaminations = [0.01, 0.05]
        results = contamination_sweep(detector, X_train, X_val, X_test, contaminations)

        for _cont, result in results.items():
            assert result.test_anomaly_rate is not None
            assert result.test_scores is not None
            assert result.test_flags is not None
            assert result.n_test_samples == 100

    def test_higher_contamination_more_anomalies(self) -> None:
        """Higher contamination should flag more anomalies (on average)."""
        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        X_val = np.random.randn(50, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        config = AnomalyModelConfig(n_estimators=10, random_state=42)
        detector = IsolationForestAnomalyDetector(config)
        detector.fit(X_train, feature_names)

        contaminations = [0.001, 0.01, 0.10]
        results = contamination_sweep(detector, X_train, X_val, None, contaminations)

        # Higher contamination -> higher anomaly rate (generally)
        rates = [results[c].train_anomaly_rate for c in contaminations]
        # Should be non-decreasing (allowing small random variation)
        assert rates[2] >= rates[1] >= rates[0]


class TestAnomalyResult:
    """Tests for AnomalyResult dataclass."""

    def test_anomaly_result_fields(self) -> None:
        """AnomalyResult should have all required fields."""
        result = AnomalyResult(
            contamination=0.01,
            threshold=-0.5,
            train_scores=np.array([0.1, 0.2]),
            train_flags=np.array([0, 0]),
            val_scores=np.array([0.1, -1.0]),
            val_flags=np.array([0, 1]),
            test_scores=None,
            test_flags=None,
            train_anomaly_rate=0.0,
            val_anomaly_rate=0.5,
            test_anomaly_rate=None,
            n_train_samples=2,
            n_val_samples=2,
            n_test_samples=None,
        )

        assert result.contamination == 0.01
        assert result.threshold == -0.5
        assert result.train_anomaly_rate == 0.0
        assert result.val_anomaly_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
