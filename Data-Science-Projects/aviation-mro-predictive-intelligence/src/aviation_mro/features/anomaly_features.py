"""Feature engineering for anomaly detection — past-only rolling features with residual deviations.

All features computed using only historical data within each engine (unit_id).
No future leakage. No RUL targets. Fitted on training engines only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aviation_mro.analytics.eda import load_raw_data
from aviation_mro.features.rul_features import (
    CONSTANT_SENSORS,
    EXCLUDE_COLS,
    INFORMATIVE_SENSORS,
    OP_SETTINGS,
    FeatureConfig,
    compute_rolling_features,
    split_by_engine,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fd001"


@dataclass
class AnomalyFeatureConfig(FeatureConfig):
    """Configuration for anomaly detection feature engineering."""

    include_residuals: bool = True
    residual_windows: tuple[int, ...] = (3, 5)
    include_op_settings: bool = True


def engineer_anomaly_features(
    df: pd.DataFrame,
    config: AnomalyFeatureConfig,
    sensor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline for anomaly detection.

    Differences from RUL feature engineering:
    - No RUL labels created (unsupervised)
    - Returns all cycles (not just last cycle)
    - Adds residual features: sensor - rolling_mean
    - Includes operational settings by default
    - Drops constant columns (op_setting_3, constant sensors)
    """
    if sensor_cols is None:
        sensor_cols = INFORMATIVE_SENSORS

    df = df.copy()
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Apply rolling features per engine (reuse existing function)
    feature_frames = []
    for _unit_id, group in df.groupby("unit_id", sort=False):
        g = compute_rolling_features(group, sensor_cols, config.window_sizes)
        feature_frames.append(g)

    df = pd.concat(feature_frames, ignore_index=True)

    # Add residual features: deviation from rolling mean
    if config.include_residuals:
        residual_cols = {}
        for sensor in sensor_cols:
            if sensor not in df.columns:
                continue
            for w in config.residual_windows:
                mean_col = f"{sensor}_mean_{w}"
                if mean_col in df.columns:
                    residual_cols[f"{sensor}_residual_{w}"] = df[sensor] - df[mean_col]

        if residual_cols:
            residual_df = pd.DataFrame(residual_cols, index=df.index)
            df = pd.concat([df, residual_df], axis=1)

    # Drop constant columns that should not be used as features
    drop_cols = ["op_setting_3"] + CONSTANT_SENSORS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_anomaly_feature_columns(df: pd.DataFrame, config: AnomalyFeatureConfig) -> list[str]:
    """Extract feature column names for anomaly detection.

    Excludes: IDs, raw sensors, RUL targets, constant sensors, op_setting_3
    Includes: rolling stats, residuals, operational settings (if enabled)
    """
    exclude = set(EXCLUDE_COLS)
    exclude.update({"rul", "rul_capped"})

    if not config.include_op_settings:
        exclude.update(OP_SETTINGS)

    # Also exclude raw sensor columns (we use rolling features only)
    exclude.update(INFORMATIVE_SENSORS)

    feature_cols = [c for c in df.columns if c not in exclude]
    return sorted(feature_cols)


def load_and_prepare_anomaly_data(config: AnomalyFeatureConfig) -> dict[str, Any]:
    """Load raw data, engineer features, split train/val using SAME split as RUL model.

    Returns feature matrices for all cycles (not just last cycle).
    """
    # Load raw
    train_raw = load_raw_data("train")

    # Engineer features on full train data
    train_feat = engineer_anomaly_features(train_raw, config)

    # Use SAME engine split as RUL model (seed=42, 80/20)
    np.random.seed(config.random_seed)
    all_engine_ids = sorted(train_feat["unit_id"].unique())
    np.random.shuffle(all_engine_ids)
    n_train = int(0.8 * len(all_engine_ids))
    train_ids = sorted(all_engine_ids[:n_train])
    val_ids = sorted(all_engine_ids[n_train:])

    train_df, val_df = split_by_engine(train_feat, train_ids, val_ids)

    # Feature columns
    feature_cols = get_anomaly_feature_columns(train_df, config)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "feature_cols": feature_cols,
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
        "config": config,
    }


def prepare_inference_data(
    test_raw: pd.DataFrame,
    config: AnomalyFeatureConfig,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Prepare test data for inference using same feature engineering.

    Args:
        test_raw: Raw test data (truncated trajectories)
        config: Same config used for training
        feature_cols: Feature columns from training (must match)

    Returns:
        DataFrame with features for all test cycles, aligned to training feature_cols
    """
    test_feat = engineer_anomaly_features(test_raw, config)

    # Ensure all training feature columns exist (add missing as NaN)
    for col in feature_cols:
        if col not in test_feat.columns:
            test_feat[col] = np.nan

    # Select and order columns to match training
    test_feat = test_feat[["unit_id", "cycle"] + feature_cols]

    return test_feat


def main() -> None:
    """Demo anomaly feature engineering."""
    config = AnomalyFeatureConfig()
    data = load_and_prepare_anomaly_data(config)

    _ = f"Train engines: {len(data['train_engine_ids'])}"
    _ = f"Val engines: {len(data['val_engine_ids'])}"
    _ = f"Train rows: {len(data['train_df'])}"
    _ = f"Val rows: {len(data['val_df'])}"
    _ = f"Features: {len(data['feature_cols'])}"
    for fc in data["feature_cols"][:20]:
        _ = f"  {fc}"
    if len(data["feature_cols"]) > 20:
        _ = f"  ... and {len(data['feature_cols']) - 20} more"


if __name__ == "__main__":
    main()
