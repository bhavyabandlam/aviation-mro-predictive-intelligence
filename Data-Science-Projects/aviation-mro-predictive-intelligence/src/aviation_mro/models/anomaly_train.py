"""Anomaly detection training and evaluation pipeline.

Mirrors the structure of baseline.py but for unsupervised IsolationForest.
Uses the SAME engine-level split as the RUL model (seed=42, 80/20).
Evaluates on validation (20 engines) and test (100 engines) separately.
Performs contamination threshold sweep without claiming true fault rates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aviation_mro.analytics.eda import load_raw_data
from aviation_mro.features.anomaly_features import (
    AnomalyFeatureConfig,
    load_and_prepare_anomaly_data,
    prepare_inference_data,
)
from aviation_mro.models.anomaly_model import (
    AnomalyModelConfig,
    AnomalyResult,
    IsolationForestAnomalyDetector,
    contamination_sweep,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fd001"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnomalyTrainingArtifacts:
    """Complete training artifacts for reproducibility."""

    dataset_fingerprint: dict[str, str]
    feature_config: dict
    model_config: dict
    feature_names: list[str]
    train_engine_ids: list[int]
    val_engine_ids: list[int]
    imputer: Any
    scaler: Any
    model: IsolationForestAnomalyDetector
    contamination_results: dict[float, AnomalyResult]
    default_contamination: float
    default_threshold: float
    sensor_attributions_val: list[Any]  # SensorAttribution objects
    sensor_attributions_test: list[Any] | None


def dataset_fingerprint() -> dict[str, str]:
    """Compute SHA-256 hashes of raw data files."""
    hashes = {}
    for fname in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
        path = RAW_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()[:16]
        else:
            hashes[fname] = "MISSING"
    return hashes


def run_anomaly_training(
    feature_config: AnomalyFeatureConfig | None = None,
    model_config: AnomalyModelConfig | None = None,
    contaminations: list[float] | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> AnomalyTrainingArtifacts:
    """Complete anomaly detection training pipeline.

    Args:
        feature_config: Feature engineering configuration
        model_config: IsolationForest configuration
        contaminations: Contamination percentiles to evaluate (not true fault rates)
        models_dir: Directory for the saved model artifact. Defaults to the
            production ``models/`` directory.
        reports_dir: Directory for the saved model card. Defaults to the
            production ``reports/`` directory.

    Returns:
        AnomalyTrainingArtifacts with all results and fitted components

    Tests pass temporary directories so that training never overwrites the
    preserved production artifacts.
    """
    if feature_config is None:
        feature_config = AnomalyFeatureConfig()
    if model_config is None:
        model_config = AnomalyModelConfig()
    if contaminations is None:
        contaminations = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    out_models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    out_reports_dir = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_reports_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data using SAME engine split as RUL model
    data = load_and_prepare_anomaly_data(feature_config)

    train_df = data["train_df"]
    val_df = data["val_df"]
    feature_cols = data["feature_cols"]
    train_ids = [int(x) for x in data["train_engine_ids"]]
    val_ids = [int(x) for x in data["val_engine_ids"]]

    # Prepare feature matrices (all cycles, not just last)
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values

    # Fit anomaly detector on training engines only
    detector = IsolationForestAnomalyDetector(model_config)
    detector.fit(X_train, feature_cols)

    # Contamination sweep (threshold sensitivity analysis)
    contamination_results = contamination_sweep(detector, X_train, X_val, None, contaminations)

    # Default contamination for reporting
    default_contamination = 0.01
    default_result = contamination_results[default_contamination]
    default_threshold = default_result.threshold

    # Sensor attribution for validation anomalies (using default threshold)
    val_flagged_mask = default_result.val_flags == 1
    if val_flagged_mask.any():
        X_val_flagged = X_val[val_flagged_mask]
        val_meta_flagged = val_df[val_flagged_mask][["unit_id", "cycle"]].reset_index(drop=True)
        sensor_attributions_val = detector.compute_sensor_attribution(
            X_val_flagged, val_meta_flagged, feature_cols, top_k=3
        )
    else:
        sensor_attributions_val = []

    # Prepare test data for final evaluation
    test_raw = load_raw_data("test")
    test_feat = prepare_inference_data(test_raw, feature_config, feature_cols)
    X_test = test_feat[feature_cols].values

    # Evaluate on test set with same thresholds
    test_contamination_results = contamination_sweep(
        detector, X_train, X_val, X_test, contaminations
    )

    # Sensor attribution for test anomalies
    test_default_result = test_contamination_results[default_contamination]
    test_scores = (
        test_default_result.test_scores
        if test_default_result.test_scores is not None
        else np.array([])
    )
    test_flagged_mask = np.asarray(test_default_result.test_flags == 1)
    if bool(test_flagged_mask.any()):
        X_test_flagged = X_test[test_flagged_mask]
        test_meta_flagged = test_feat.loc[test_flagged_mask, ["unit_id", "cycle"]].reset_index(
            drop=True
        )
        sensor_attributions_test = detector.compute_sensor_attribution(
            X_test_flagged, test_meta_flagged, feature_cols, top_k=3
        )
    else:
        sensor_attributions_test = []

    # Compile artifacts
    artifacts = AnomalyTrainingArtifacts(
        dataset_fingerprint=dataset_fingerprint(),
        feature_config=asdict(feature_config),
        model_config=asdict(model_config),
        feature_names=feature_cols,
        train_engine_ids=train_ids,
        val_engine_ids=val_ids,
        imputer=detector.imputer,
        scaler=detector.scaler,
        model=detector,
        contamination_results=contamination_results,
        default_contamination=default_contamination,
        default_threshold=default_threshold,
        sensor_attributions_val=sensor_attributions_val,
        sensor_attributions_test=sensor_attributions_test,
    )

    # Save model artifact
    model_path = out_models_dir / "anomaly_model.pkl"
    detector.save(model_path)

    # Save human-readable model card
    card_path = out_reports_dir / "anomaly_model_card.json"
    card_data = {
        "dataset_fingerprint": artifacts.dataset_fingerprint,
        "feature_config": artifacts.feature_config,
        "model_config": artifacts.model_config,
        "feature_names": artifacts.feature_names,
        "n_features": len(artifacts.feature_names),
        "train_engine_ids": artifacts.train_engine_ids,
        "val_engine_ids": artifacts.val_engine_ids,
        "default_contamination": artifacts.default_contamination,
        "default_threshold": artifacts.default_threshold,
        "contamination_sweep": {
            str(cont): {
                "contamination": result.contamination,
                "threshold": result.threshold,
                "train_anomaly_rate": result.train_anomaly_rate,
                "val_anomaly_rate": result.val_anomaly_rate,
                "test_anomaly_rate": result.test_anomaly_rate,
                "n_train_samples": result.n_train_samples,
                "n_val_samples": result.n_val_samples,
                "n_test_samples": result.n_test_samples,
            }
            for cont, result in test_contamination_results.items()
        },
        "score_distributions": {
            "train": {
                "mean": float(np.mean(default_result.train_scores)),
                "std": float(np.std(default_result.train_scores)),
                "min": float(np.min(default_result.train_scores)),
                "max": float(np.max(default_result.train_scores)),
                "percentiles": {
                    "1%": float(np.percentile(default_result.train_scores, 1)),
                    "5%": float(np.percentile(default_result.train_scores, 5)),
                    "10%": float(np.percentile(default_result.train_scores, 10)),
                    "25%": float(np.percentile(default_result.train_scores, 25)),
                    "50%": float(np.percentile(default_result.train_scores, 50)),
                    "75%": float(np.percentile(default_result.train_scores, 75)),
                    "90%": float(np.percentile(default_result.train_scores, 90)),
                    "95%": float(np.percentile(default_result.train_scores, 95)),
                    "99%": float(np.percentile(default_result.train_scores, 99)),
                },
            },
            "val": {
                "mean": float(np.mean(default_result.val_scores)),
                "std": float(np.std(default_result.val_scores)),
                "min": float(np.min(default_result.val_scores)),
                "max": float(np.max(default_result.val_scores)),
                "percentiles": {
                    "1%": float(np.percentile(default_result.val_scores, 1)),
                    "5%": float(np.percentile(default_result.val_scores, 5)),
                    "10%": float(np.percentile(default_result.val_scores, 10)),
                    "25%": float(np.percentile(default_result.val_scores, 25)),
                    "50%": float(np.percentile(default_result.val_scores, 50)),
                    "75%": float(np.percentile(default_result.val_scores, 75)),
                    "90%": float(np.percentile(default_result.val_scores, 90)),
                    "95%": float(np.percentile(default_result.val_scores, 95)),
                    "99%": float(np.percentile(default_result.val_scores, 99)),
                },
            },
            "test": {
                "mean": float(np.mean(test_scores)),
                "std": float(np.std(test_scores)),
                "min": float(np.min(test_scores)),
                "max": float(np.max(test_scores)),
                "percentiles": {
                    "1%": float(np.percentile(test_scores, 1)),
                    "5%": float(np.percentile(test_scores, 5)),
                    "10%": float(np.percentile(test_scores, 10)),
                    "25%": float(np.percentile(test_scores, 25)),
                    "50%": float(np.percentile(test_scores, 50)),
                    "75%": float(np.percentile(test_scores, 75)),
                    "90%": float(np.percentile(test_scores, 90)),
                    "95%": float(np.percentile(test_scores, 95)),
                    "99%": float(np.percentile(test_scores, 99)),
                },
            },
        },
        "val_flagged_summary": {
            "n_flagged": int(default_result.val_flags.sum()),
            "n_total": len(default_result.val_flags),
            "anomaly_rate": default_result.val_anomaly_rate,
            "per_engine": (
                val_df.assign(flag=default_result.val_flags)
                .groupby("unit_id")["flag"]
                .mean()
                .to_dict()
            ),
        },
        "test_flagged_summary": {
            "n_flagged": (
                int(test_default_result.test_flags.sum())
                if test_default_result.test_flags is not None
                else 0
            ),
            "n_total": (
                len(test_default_result.test_flags)
                if test_default_result.test_flags is not None
                else 0
            ),
            "anomaly_rate": test_default_result.test_anomaly_rate,
            "per_engine": (
                test_feat.assign(flag=test_default_result.test_flags)
                .groupby("unit_id")["flag"]
                .mean()
                .to_dict()
            )
            if test_default_result.test_flags is not None
            else {},
        },
        "sensor_attributions_val": [
            {
                "unit_id": a.unit_id,
                "cycle": a.cycle,
                "anomaly_score": a.anomaly_score,
                "top_sensors": a.top_sensors,
            }
            for a in artifacts.sensor_attributions_val[:50]  # Limit to top 50
        ],
        "sensor_attributions_test": [
            {
                "unit_id": a.unit_id,
                "cycle": a.cycle,
                "anomaly_score": a.anomaly_score,
                "top_sensors": a.top_sensors,
            }
            for a in (artifacts.sensor_attributions_test or [])[:50]
        ],
        "limitations": [
            "No ground-truth fault labels available — contamination is a "
            "threshold percentile, not a fault rate",
            "IsolationForest flags statistical outliers; late-life degradation "
            "appears as anomalies (expected)",
            "Sensor attribution uses z-score deviation from training rolling "
            "statistics — not causal",
            "Test engines have truncated trajectories; anomaly scores may differ "
            "from full-life engines",
            "Results are dataset-specific (FD001, single operating condition); may not generalize",
        ],
    }

    with open(card_path, "w") as f:
        json.dump(card_data, f, indent=2, default=str)

    return artifacts


def main() -> None:
    """Main entry point."""
    feature_config = AnomalyFeatureConfig(
        window_sizes=(3, 5),
        include_op_settings=True,
        include_residuals=True,
        residual_windows=(3, 5),
        rul_cap=125,
        random_seed=42,
    )
    model_config = AnomalyModelConfig(
        n_estimators=200,
        contamination="auto",  # Training uses "auto", threshold sweep handles specific rates
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    contaminations = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    run_anomaly_training(feature_config, model_config, contaminations)


if __name__ == "__main__":
    main()
