"""IsolationForest wrapper for anomaly detection with fixed random seed and documented assumptions.

This module provides a clean interface for training and scoring with IsolationForest.
Key design decisions:
- Fixed random_state for reproducibility
- Contamination is a hyperparameter (default 0.01), not a claim about true fault rate
- No ground-truth fault labels used — purely unsupervised
- Provides anomaly scores, flags, and sensor-context attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnomalyModelConfig:
    """Configuration for IsolationForest anomaly detector."""

    n_estimators: int = 200
    max_samples: str | int = "auto"
    contamination: float | str = 0.01
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0


@dataclass
class AnomalyResult:
    """Results for a single contamination threshold."""

    contamination: float
    threshold: float
    train_scores: np.ndarray
    train_flags: np.ndarray
    val_scores: np.ndarray
    val_flags: np.ndarray
    test_scores: np.ndarray | None
    test_flags: np.ndarray | None
    train_anomaly_rate: float
    val_anomaly_rate: float
    test_anomaly_rate: float | None
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int | None


@dataclass
class SensorAttribution:
    """Sensor-level attribution for flagged observations."""

    unit_id: int
    cycle: int
    anomaly_score: float
    top_sensors: list[tuple[str, float]]  # (sensor_name, contribution)
    all_contributions: dict[str, float]


class IsolationForestAnomalyDetector:
    """IsolationForest wrapper with preprocessing and sensor attribution.

    Attributes:
        config: Model hyperparameters
        imputer: Fitted SimpleImputer (median strategy)
        scaler: Fitted StandardScaler
        model: Fitted IsolationForest
        feature_names: List of feature column names
        train_scores_: Anomaly scores on training data (for threshold setting)
        threshold_: Decision threshold (percentile of train scores)
    """

    def __init__(self, config: AnomalyModelConfig | None = None):
        self.config = config or AnomalyModelConfig()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: IsolationForest | None = None
        self.feature_names: list[str] = []
        self.train_scores_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, X_train: np.ndarray, feature_names: list[str]) -> IsolationForestAnomalyDetector:
        """Fit imputer, scaler, and IsolationForest on training data only.

        Args:
            X_train: Training feature matrix (n_samples, n_features)
            feature_names: Names of feature columns

        Returns:
            Self for chaining
        """
        self.feature_names = feature_names

        # Fit preprocessing on train only
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)

        # Fit IsolationForest
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
        self.model.fit(X_train_scaled)

        # Compute training scores for threshold calibration
        # decision_function: higher = more normal, lower = more anomalous
        self.train_scores_ = self.model.decision_function(X_train_scaled)

        # Set threshold at contamination percentile of training scores
        # e.g., contamination=0.01 -> 1st percentile
        # If contamination="auto", use 1% as default threshold percentile
        if isinstance(self.config.contamination, str) and self.config.contamination == "auto":
            threshold_percentile = 1.0
        else:
            threshold_percentile = float(self.config.contamination) * 100
        self.threshold_ = float(np.percentile(self.train_scores_, threshold_percentile))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted imputer and scaler."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled  # type: ignore[no-any-return]

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (decision_function).

        Higher = more normal, lower = more anomalous.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.transform(X)
        return self.model.decision_function(X_scaled)  # type: ignore[no-any-return]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly flags (-1 = anomaly, 1 = normal)."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.transform(X)
        return self.model.predict(X_scaled)  # type: ignore[no-any-return]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Alias for score_samples."""
        return self.score_samples(X)

    def flag_anomalies(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Flag anomalies using threshold (default: fitted threshold)."""
        scores = self.score_samples(X)
        thresh = threshold if threshold is not None else self.threshold_
        if thresh is None:
            raise RuntimeError("Threshold not set. Call fit() first or provide threshold.")
        # Anomaly if score < threshold
        return (scores < thresh).astype(int)

    def compute_sensor_attribution(
        self,
        X: np.ndarray,
        df_meta: pd.DataFrame,
        feature_names: list[str],
        top_k: int = 3,
    ) -> list[SensorAttribution]:
        """Compute sensor-level attribution for flagged observations.

        For each flagged observation, compute contribution of each sensor group
        based on deviation from training rolling mean (z-score style).

        Args:
            X: Feature matrix for flagged observations
            df_meta: DataFrame with unit_id, cycle for each row
            feature_names: Feature column names
            top_k: Number of top sensors to return per observation

        Returns:
            List of SensorAttribution objects
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get training statistics per feature for z-score computation
        # We'll use the scaler's (mean, scale)
        train_mean = self.scaler.mean_
        train_scale = self.scaler.scale_

        attributions = []

        for idx in range(len(X)):
            row = X[idx]
            unit_id = int(df_meta.iloc[idx]["unit_id"])
            cycle = int(df_meta.iloc[idx]["cycle"])
            anomaly_score = self.score_samples(row.reshape(1, -1))[0]

            # Compute z-scores for each feature
            row_imputed = self.imputer.transform(row.reshape(1, -1))
            z_scores = (row_imputed[0] - train_mean) / np.maximum(train_scale, 1e-10)

            # Group by sensor (features are like sensor_3_mean_3, sensor_3_std_3, etc.)
            sensor_contributions: dict[str, float] = {}
            for feat_name, z in zip(feature_names, z_scores, strict=False):
                # Extract sensor name from feature (e.g., "sensor_3_mean_3" -> "sensor_3")
                if feat_name.startswith("op_setting_"):
                    sensor_key: str = feat_name
                else:
                    # Find which informative sensor this belongs to
                    sensor_key = ""
                    for s in [
                        "sensor_2",
                        "sensor_3",
                        "sensor_4",
                        "sensor_6",
                        "sensor_7",
                        "sensor_8",
                        "sensor_9",
                        "sensor_11",
                        "sensor_12",
                        "sensor_13",
                        "sensor_14",
                        "sensor_15",
                        "sensor_17",
                        "sensor_20",
                        "sensor_21",
                    ]:
                        if feat_name.startswith(s + "_"):
                            sensor_key = s
                            break
                    if not sensor_key:
                        sensor_key = feat_name.split("_")[0] + "_" + feat_name.split("_")[1]

                # Contribution = absolute z-score (magnitude of deviation)
                contrib = abs(z)
                if sensor_key in sensor_contributions:
                    sensor_contributions[sensor_key] += contrib
                else:
                    sensor_contributions[sensor_key] = contrib

            # Sort by contribution
            sorted_sensors = sorted(sensor_contributions.items(), key=lambda x: x[1], reverse=True)
            top_sensors: list[tuple[str, float]] = sorted_sensors[:top_k]

            attributions.append(
                SensorAttribution(
                    unit_id=unit_id,
                    cycle=cycle,
                    anomaly_score=float(anomaly_score),
                    top_sensors=top_sensors,
                    all_contributions=sensor_contributions,
                )
            )

        return attributions

    def save(self, path: Path) -> None:
        """Save model artifact."""
        artifact = {
            "config": self.config,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "model": self.model,
            "feature_names": self.feature_names,
            "train_scores_": self.train_scores_,
            "threshold_": self.threshold_,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, path: Path) -> IsolationForestAnomalyDetector:
        """Load model artifact."""
        artifact = joblib.load(path)
        detector = cls(artifact["config"])
        detector.imputer = artifact["imputer"]
        detector.scaler = artifact["scaler"]
        detector.model = artifact["model"]
        detector.feature_names = artifact["feature_names"]
        detector.train_scores_ = artifact["train_scores_"]
        detector.threshold_ = artifact["threshold_"]
        return detector


def contamination_sweep(
    detector: IsolationForestAnomalyDetector,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray | None,
    contaminations: list[float] | None = None,
) -> dict[float, AnomalyResult]:
    """Evaluate anomaly detector across multiple contamination thresholds.

    This does NOT retrain the model — it only changes the decision threshold.
    The IsolationForest is trained once with contamination="auto" or a default,
    then we sweep the threshold percentile.

    Args:
        detector: Fitted IsolationForestAnomalyDetector
        X_train: Training features
        X_val: Validation features
        X_test: Test features (optional)
        contaminations: List of contamination percentiles to evaluate

    Returns:
        Dict mapping contamination -> AnomalyResult
    """
    if detector.model is None or detector.train_scores_ is None:
        raise RuntimeError("Detector must be fitted first.")

    if contaminations is None:
        contaminations = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    results = {}
    train_scores = detector.train_scores_

    for cont in contaminations:
        # Threshold at contamination percentile of training scores
        threshold = float(np.percentile(train_scores, cont * 100))

        # Train flags
        train_flags = (train_scores < threshold).astype(int)
        train_anomaly_rate = float(train_flags.mean())

        # Val flags
        val_scores = detector.score_samples(X_val)
        val_flags = (val_scores < threshold).astype(int)
        val_anomaly_rate = float(val_flags.mean())

        # Test flags (if provided)
        if X_test is not None:
            test_scores = detector.score_samples(X_test)
            test_flags = (test_scores < threshold).astype(int)
            test_anomaly_rate = float(test_flags.mean())
            n_test = len(X_test)
        else:
            test_scores = None
            test_flags = None
            test_anomaly_rate = None
            n_test = None

        results[cont] = AnomalyResult(
            contamination=cont,
            threshold=threshold,
            train_scores=train_scores,
            train_flags=train_flags,
            val_scores=val_scores,
            val_flags=val_flags,
            test_scores=test_scores,
            test_flags=test_flags,
            train_anomaly_rate=train_anomaly_rate,
            val_anomaly_rate=val_anomaly_rate,
            test_anomaly_rate=test_anomaly_rate,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val),
            n_test_samples=n_test,
        )

    return results


def main() -> None:
    """Demo."""
    config = AnomalyModelConfig(contamination=0.01)
    _ = f"Config: {config}"


if __name__ == "__main__":
    main()
