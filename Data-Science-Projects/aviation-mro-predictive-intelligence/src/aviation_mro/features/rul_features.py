"""Feature engineering for RUL prediction — past-only rolling features.

All features computed using only historical data within each engine (unit_id).
No future leakage. Fitted on training engines only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aviation_mro.analytics.eda import load_raw_data, load_rul

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fd001"

# Constant sensors identified in EDA (zero variance)
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

# Informative sensors (varying with degradation)
INFORMATIVE_SENSORS = [
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
]

# Operational settings (op_setting_3 is constant 100.0 in FD001, excluded)
OP_SETTINGS = ["op_setting_1", "op_setting_2"]

# Columns to exclude from feature generation
EXCLUDE_COLS = ["unit_id", "cycle", "rul", "rul_capped", "op_setting_3"] + CONSTANT_SENSORS


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    window_sizes: tuple[int, ...] = (3, 5)
    include_op_settings: bool = True
    include_cycle_norm: bool = False
    rul_cap: int = 125
    random_seed: int = 42
    lags: tuple[int, ...] = (1, 2)


def create_rul_labels(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Add RUL column to training data: cycles until failure per engine."""
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    df["rul_capped"] = df["rul"].clip(upper=cap)
    return df


def _slope(x: np.ndarray) -> float:
    """Least-squares slope of y=x values over window positions 0..n-1.

    Matches ``np.polyfit(np.arange(len(x)), x, 1)[0]`` using a closed-form
    formula. Kept as reference for equivalence testing.
    """
    n = len(x)
    if n < 2:
        return np.nan
    idx = np.arange(n, dtype=float)
    sum_x = float(np.sum(x))
    sum_ix = float(np.sum(idx * x))
    sum_i = n * (n - 1) / 2
    sum_i2 = n * (n - 1) * (2 * n - 1) / 6
    denominator = n * sum_i2 - sum_i * sum_i
    if denominator == 0:
        return np.nan
    return float((n * sum_ix - sum_i * sum_x) / denominator)


def _rolling_slope(vals: pd.Series, w: int) -> pd.Series:
    """Vectorized rolling linear slope.

    Matches ``vals.rolling(w, min_periods=2).apply(_slope, raw=True)`` where
    ``_slope`` fits a degree-1 polynomial to window positions 0..n-1. Uses the
    closed-form least-squares slope:

        m = (n * sum(i*x[i]) - sum(i) * sum(x)) / (n * sum(i^2) - sum(i)^2)

    NaN handling: any window containing NaN yields NaN (matches polyfit NaN
    propagation). Windows with fewer than 2 non-NaN values yield NaN.
    """
    n = len(vals)
    if n == 0:
        return pd.Series(index=vals.index, dtype=float)

    x = vals.to_numpy(dtype=float)
    is_nan = np.isnan(x)

    # Cumulative sums (NaN treated as 0; invalid windows masked later)
    x_filled = np.where(is_nan, 0.0, x)
    idx = np.arange(n, dtype=float)
    x_idx = x_filled * idx

    cumsum_x = np.concatenate([[0.0], np.cumsum(x_filled)])
    cumsum_x_idx = np.concatenate([[0.0], np.cumsum(x_idx)])
    cumsum_count = np.concatenate([[0], np.cumsum(~is_nan).astype(np.int64)])

    # Window boundaries for each end position t (0-indexed)
    ends = np.arange(1, n + 1)  # exclusive end for cumsum slicing
    starts = np.maximum(0, np.arange(n) - w + 1)

    count = cumsum_count[ends] - cumsum_count[starts]  # non-NaN values in window
    n_vals = ends - starts  # window length (may be < w at series boundary)

    # Need at least min_periods=2 non-NaN values to compute
    valid = count >= 2

    sum_x = cumsum_x[ends] - cumsum_x[starts]
    sum_jx = cumsum_x_idx[ends] - cumsum_x_idx[starts]
    # Shift from global-index weighting to window-relative positions
    sum_ix = sum_jx - starts.astype(float) * sum_x

    sum_i = n_vals * (n_vals - 1) / 2
    sum_i2 = n_vals * (n_vals - 1) * (2 * n_vals - 1) / 6
    denominator = n_vals * sum_i2 - sum_i**2

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n_vals * sum_ix - sum_i * sum_x) / denominator

    # Mask: NaN if any NaN in window (count < n_vals), or degenerate denominator
    valid = valid & (count == n_vals) & (denominator != 0)

    out = np.full(n, np.nan, dtype=float)
    out[valid] = slope[valid]
    return pd.Series(out, index=vals.index)


def compute_rolling_features(
    group: pd.DataFrame,
    sensor_cols: list[str],
    window_sizes: tuple[int, ...],
) -> pd.DataFrame:
    """Compute past-only rolling features for a single engine group."""
    g = group.copy()
    g = g.sort_values("cycle")

    new_cols = {}
    for sensor in sensor_cols:
        if sensor not in g.columns:
            continue
        vals = g[sensor]

        for w in window_sizes:
            # Rolling statistics (min_periods=1 ensures past-only, no NaN at start)
            new_cols[f"{sensor}_mean_{w}"] = vals.rolling(w, min_periods=1).mean()
            new_cols[f"{sensor}_std_{w}"] = vals.rolling(w, min_periods=1).std()
            new_cols[f"{sensor}_min_{w}"] = vals.rolling(w, min_periods=1).min()
            new_cols[f"{sensor}_max_{w}"] = vals.rolling(w, min_periods=1).max()

            # Rolling linear slope (degradation rate over window)
            new_cols[f"{sensor}_slope_{w}"] = _rolling_slope(vals, w)

    # Add all new columns at once to avoid fragmentation
    new_df = pd.DataFrame(new_cols, index=g.index)
    return pd.concat([g, new_df], axis=1)


def compute_lagged_features(
    group: pd.DataFrame,
    sensor_cols: list[str],
    lags: tuple[int, ...],
) -> pd.DataFrame:
    """Compute past-only lagged sensor values for a single engine group.

    Each lagged column ``<sensor>_lag_{k}`` holds the sensor value observed
    ``k`` cycles earlier within the same engine. Only past observations are
    used (``shift``), so there is no future leakage. Lagged columns for the
    first ``k`` cycles of an engine are NaN (no prior observation exists).

    Args:
        group: Single-engine DataFrame sorted by ``cycle``.
        sensor_cols: Sensor columns to lag.
        lags: Positive integer cycle lags to compute.

    Returns:
        The group with lagged columns appended.
    """
    g = group.copy()
    g = g.sort_values(["unit_id", "cycle"])
    new_cols: dict[str, pd.Series] = {}
    for sensor in sensor_cols:
        if sensor not in g.columns:
            continue
        for lag in lags:
            if lag < 1:
                continue
            # Shift within each engine only — no cross-engine contamination.
            new_cols[f"{sensor}_lag_{lag}"] = g.groupby("unit_id")[sensor].shift(lag)
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=g.index)
        g = pd.concat([g, new_df], axis=1)
    return g


def engineer_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    sensor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline for a dataset."""
    if sensor_cols is None:
        sensor_cols = INFORMATIVE_SENSORS

    df = df.copy()
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Add RUL labels if not present (for training data)
    if "rul" not in df.columns and "rul_capped" not in df.columns:
        df = create_rul_labels(df, config.rul_cap)

    # Apply rolling features per engine
    feature_frames = []
    for _unit_id, group in df.groupby("unit_id", sort=False):
        g = compute_rolling_features(group, sensor_cols, config.window_sizes)
        # Past-only lagged sensor values (per engine, chronological).
        if config.lags:
            g = compute_lagged_features(g, sensor_cols, config.lags)
        feature_frames.append(g)

    df = pd.concat(feature_frames, ignore_index=True)

    # Normalized cycle (life fraction) - uses max cycle per engine from CURRENT dataset
    # For training: true total life (known). For test: truncated last cycle (observed).
    # WARNING: test distribution shift expected (cycle_norm=1.0 at last observed cycle).
    if config.include_cycle_norm:
        max_cycle_per_unit = df.groupby("unit_id")["cycle"].transform("max")
        df = df.copy()
        df["cycle_norm"] = df["cycle"] / max_cycle_per_unit

    return df


def get_feature_columns(df: pd.DataFrame, config: FeatureConfig) -> list[str]:
    """Extract feature column names (exclude IDs, targets, raw sensors)."""
    exclude = set(EXCLUDE_COLS)
    if not config.include_op_settings:
        exclude.update(OP_SETTINGS)

    # Also exclude raw sensor columns (we use rolling features only)
    exclude.update(INFORMATIVE_SENSORS)

    feature_cols = [c for c in df.columns if c not in exclude]
    return sorted(feature_cols)


def split_by_engine(
    df: pd.DataFrame,
    train_engine_ids: list[int],
    val_engine_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by engine ID (no row-level random split)."""
    train_df = df[df["unit_id"].isin(train_engine_ids)].copy()
    val_df = df[df["unit_id"].isin(val_engine_ids)].copy()
    return train_df, val_df


def prepare_test_last_cycle(test_df: pd.DataFrame) -> pd.DataFrame:
    """Extract last cycle per engine from test set for prediction."""
    return test_df.groupby("unit_id", sort=False).tail(1).copy()


def load_and_prepare_data(config: FeatureConfig) -> dict[str, Any]:
    """Load raw data, engineer features, split train/val, prepare test."""
    # Load raw
    train_raw = load_raw_data("train")
    test_raw = load_raw_data("test")
    rul_df = load_rul()

    # Engineer features on full train + test separately
    train_feat = engineer_features(train_raw, config)
    test_feat = engineer_features(test_raw, config)

    # Engine-level split (80/20 for FD001's 100 engines)
    np.random.seed(config.random_seed)
    all_engine_ids = sorted(train_feat["unit_id"].unique())
    np.random.shuffle(all_engine_ids)
    n_train = int(0.8 * len(all_engine_ids))
    train_ids = sorted(all_engine_ids[:n_train])
    val_ids = sorted(all_engine_ids[n_train:])

    train_df, val_df = split_by_engine(train_feat, train_ids, val_ids)

    # Test: only last cycle per engine
    test_last = prepare_test_last_cycle(test_feat)

    # Feature columns
    feature_cols = get_feature_columns(train_df, config)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_last": test_last,
        "rul_df": rul_df,
        "feature_cols": feature_cols,
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
        "config": config,
    }


def main() -> None:
    """Demo feature engineering."""
    config = FeatureConfig()
    data = load_and_prepare_data(config)

    for _fc in data["feature_cols"][:20]:
        pass
    if len(data["feature_cols"]) > 20:
        pass


if __name__ == "__main__":
    main()
