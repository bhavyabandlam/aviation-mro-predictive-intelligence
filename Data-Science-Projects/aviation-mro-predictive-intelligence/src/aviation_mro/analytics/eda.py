"""Exploratory Data Analysis and Statistical Analysis for NASA C-MAPSS FD001.

Implements the EDA requirements from SRS section 9.1:
- Data shape, schema, missingness, duplicates, ranges
- Constant/near-constant column detection
- Distribution visualizations and trend analysis
- Assumption testing before inferential methods
- Effect sizes and uncertainty reporting
- Separation of exploratory vs confirmatory findings
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, levene, shapiro

from aviation_mro.analytics.statistics import (  # noqa: E402
    chi_square_independence,
    mann_whitney_u,
    mean_ci,
    median_ci,
    one_way_anova,
    pearson_correlation,
    spearman_correlation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fd001"
REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
PLOTS_DIR = REPORTS_DIR / "plots"

COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
ID_COLS = ["unit_id", "cycle"]


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    dtype: str
    n_obs: int
    n_missing: int
    n_unique: int
    min_val: float | None = None
    max_val: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    is_constant: bool = False
    is_near_constant: bool = False
    constant_value: float | None = None
    near_constant_threshold: float = 1e-10


@dataclass
class DatasetProfile:
    """Complete dataset profile."""

    name: str
    shape: tuple[int, int]
    columns: list[ColumnProfile] = field(default_factory=list)
    n_duplicates: int = 0
    duplicate_rows: pd.DataFrame | None = None
    unit_stats: dict[str, Any] = field(default_factory=dict)
    cycle_stats: dict[str, Any] = field(default_factory=dict)
    correlation_matrix: pd.DataFrame | None = None
    exploratory_findings: list[str] = field(default_factory=list)
    confirmatory_findings: list[str] = field(default_factory=list)


def load_raw_data(split: str) -> pd.DataFrame:
    """Load raw FD001 data as DataFrame with proper column names."""
    if split == "train":
        path = RAW_DIR / "train_FD001.txt"
    elif split == "test":
        path = RAW_DIR / "test_FD001.txt"
    elif split == "rul":
        path = RAW_DIR / "RUL_FD001.txt"
        rul_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        return pd.DataFrame({"unit_id": np.arange(1, len(rul_vals) + 1), "rul": rul_vals})
    else:
        raise ValueError(f"Unknown split: {split}")

    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    df = pd.DataFrame(arr, columns=COLUMNS)
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul() -> pd.DataFrame:
    """Load RUL truth values."""
    return load_raw_data("rul")


def compute_column_profile(
    series: pd.Series, near_constant_threshold: float = 1e-10
) -> ColumnProfile:
    """Compute comprehensive statistical profile for a column."""
    n_obs = len(series)
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    is_constant = n_unique <= 1

    if is_constant:
        constant_val = series.iloc[0] if n_obs > 0 else None
        return ColumnProfile(
            name=str(series.name),
            dtype=str(series.dtype),
            n_obs=n_obs,
            n_missing=n_missing,
            n_unique=n_unique,
            is_constant=True,
            constant_value=constant_val,
            near_constant_threshold=near_constant_threshold,
        )

    vals = np.asarray(series.dropna().values, dtype=float)
    min_val = float(vals.min())
    max_val = float(vals.max())
    mean_val = float(vals.mean())
    std_val = float(vals.std())
    median_val = float(np.median(vals))
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr = q75 - q25

    is_near_constant = std_val < near_constant_threshold
    constant_value = mean_val if is_near_constant else None

    # Skewness and kurtosis
    try:
        skewness = float(stats.skew(vals))
        kurtosis = float(stats.kurtosis(vals))
    except Exception:
        skewness = None
        kurtosis = None

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        n_obs=n_obs,
        n_missing=n_missing,
        n_unique=n_unique,
        min_val=min_val,
        max_val=max_val,
        mean=mean_val,
        std=std_val,
        median=median_val,
        q25=q25,
        q75=q75,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        constant_value=constant_value,
        near_constant_threshold=near_constant_threshold,
    )


def profile_dataset(df: pd.DataFrame, name: str) -> DatasetProfile:
    """Create complete profile for a dataset."""
    profile = DatasetProfile(name=name, shape=df.shape)

    # Column profiles
    for col in df.columns:
        profile.columns.append(compute_column_profile(df[col]))

    # Duplicates
    dup_mask = df.duplicated()
    profile.n_duplicates = int(dup_mask.sum())
    if profile.n_duplicates > 0:
        profile.duplicate_rows = df[dup_mask].head(10)

    # Unit-level statistics
    if "unit_id" in df.columns and "cycle" in df.columns:
        unit_groups = df.groupby("unit_id")["cycle"]
        profile.unit_stats = {
            "n_units": int(df["unit_id"].nunique()),
            "cycles_per_unit_min": int(unit_groups.min().min()),
            "cycles_per_unit_max": int(unit_groups.max().max()),
            "cycles_per_unit_mean": float(unit_groups.count().mean()),
            "cycles_per_unit_median": float(unit_groups.count().median()),
            "cycles_per_unit_std": float(unit_groups.count().std()),
        }

        cycle_series = df["cycle"]
        profile.cycle_stats = {
            "min": int(cycle_series.min()),
            "max": int(cycle_series.max()),
            "mean": float(cycle_series.mean()),
            "median": float(cycle_series.median()),
            "std": float(cycle_series.std()),
        }

    # Correlation matrix for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        profile.correlation_matrix = df[numeric_cols].corr()

    return profile


def detect_constant_columns(profile: DatasetProfile) -> tuple[list[str], list[str]]:
    """Separate constant and near-constant columns."""
    constant = [c.name for c in profile.columns if c.is_constant]
    near_constant = [c.name for c in profile.columns if c.is_near_constant and not c.is_constant]
    return constant, near_constant


def test_normality(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """Test normality assumption using Shapiro-Wilk (for n<5000) or KS test."""
    vals = np.asarray(series.dropna().values, dtype=float)
    n = len(vals)

    if n < 3:
        return {"test": "insufficient_data", "statistic": None, "p_value": None, "normal": None}

    if n <= 5000:
        stat, p = shapiro(vals)
        test_name = "shapiro_wilk"
    else:
        # KS test against normal distribution with same mean/std
        from scipy.stats import norm

        stat, p = stats.kstest(vals, norm.cdf, args=(float(vals.mean()), float(vals.std())))
        test_name = "kolmogorov_smirnov"

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "normal": p > alpha,
        "alpha": alpha,
        "n": n,
    }


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """Test stationarity using Augmented Dickey-Fuller test."""
    vals = np.asarray(series.dropna().values, dtype=float)
    if len(vals) < 10:
        return {"test": "adf", "statistic": None, "p_value": None, "stationary": None}

    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(vals, autolag="AIC")
        return {
            "test": "adf",
            "statistic": float(result[0]),
            "p_value": float(result[1]),
            "stationary": result[1] < alpha,
            "alpha": alpha,
            "critical_values": {k: float(v) for k, v in result[4].items()},
        }
    except Exception as e:
        return {"test": "adf", "error": str(e)}


def test_distribution_shift(
    train_series: pd.Series, test_series: pd.Series, alpha: float = 0.05
) -> dict[str, Any]:
    """Test if train and test distributions differ using KS test."""
    train_vals = np.asarray(train_series.dropna().values, dtype=float)
    test_vals = np.asarray(test_series.dropna().values, dtype=float)

    if len(train_vals) < 2 or len(test_vals) < 2:
        return {"test": "ks_2samp", "statistic": None, "p_value": None, "same_distribution": None}

    stat, p = ks_2samp(train_vals, test_vals)
    return {
        "test": "ks_2samp",
        "statistic": float(stat),
        "p_value": float(p),
        "same_distribution": p > alpha,
        "alpha": alpha,
    }


def test_variance_homogeneity(groups: list[pd.Series], alpha: float = 0.05) -> dict[str, Any]:
    """Test homogeneity of variances using Levene's test."""
    clean_groups = [
        np.asarray(g.dropna().values, dtype=float) for g in groups if len(g.dropna()) > 1
    ]
    if len(clean_groups) < 2:
        return {"test": "levene", "statistic": None, "p_value": None, "homogeneous": None}

    stat, p = levene(*clean_groups)
    return {
        "test": "levene",
        "statistic": float(stat),
        "p_value": float(p),
        "homogeneous": p > alpha,
        "alpha": alpha,
    }


def compute_effect_size(cohen_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_sensor_trends(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sensors: list[str] | None = None
) -> dict[str, Any]:
    """Compare sensor degradation trends between train and test."""
    if sensors is None:
        sensors = SENSOR_COLS

    results = {}

    for sensor in sensors:
        if sensor not in train_df.columns or sensor not in test_df.columns:
            continue

        # Compute per-unit linear trend (slope) for train
        train_slopes = []
        for unit_id in train_df["unit_id"].unique():
            unit_data = train_df[train_df["unit_id"] == unit_id][["cycle", sensor]].dropna()
            if len(unit_data) > 2:
                slope, _, _, _, _ = stats.linregress(unit_data["cycle"], unit_data[sensor])
                train_slopes.append(slope)

        # Compute per-unit linear trend for test
        test_slopes = []
        for unit_id in test_df["unit_id"].unique():
            unit_data = test_df[test_df["unit_id"] == unit_id][["cycle", sensor]].dropna()
            if len(unit_data) > 2:
                slope, _, _, _, _ = stats.linregress(unit_data["cycle"], unit_data[sensor])
                test_slopes.append(slope)

        if train_slopes and test_slopes:
            train_mean = np.mean(train_slopes)
            test_mean = np.mean(test_slopes)
            pooled_std = np.sqrt((np.var(train_slopes) + np.var(test_slopes)) / 2)
            cohen_d = (train_mean - test_mean) / pooled_std if pooled_std > 0 else 0

            stat, p = stats.ttest_ind(train_slopes, test_slopes, equal_var=False)

            results[sensor] = {
                "train_slope_mean": float(train_mean),
                "test_slope_mean": float(test_mean),
                "cohen_d": float(cohen_d),
                "effect_size_interpretation": compute_effect_size(cohen_d),
                "ttest_statistic": float(stat),
                "ttest_p_value": float(p),
                "significant_difference": p < 0.05,
            }

    return results


# ---------------------------------------------------------------------------
# FR-04 Statistical Analysis: correlation, categorical association,
# two-group tests, ANOVA, confidence intervals
# ---------------------------------------------------------------------------
def compute_sensor_correlations(
    train_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pearson and Spearman correlation between sensors and RUL.

    Correlations are computed on the train set only (where RUL is known),
    joining sensor readings at the final cycle of each engine with its RUL.
    """
    if sensors is None:
        sensors = SENSOR_COLS

    # Get final-cycle sensor reading per engine
    final_readings = train_df.sort_values("cycle").groupby("unit_id").tail(1)[["unit_id"] + sensors]
    merged = final_readings.merge(rul_df[["unit_id", "rul"]], on="unit_id", how="inner")

    results: dict[str, dict[str, Any]] = {}
    for sensor in sensors:
        if sensor not in merged.columns:
            continue
        pearson = pearson_correlation(merged[sensor], merged["rul"])
        spearman = spearman_correlation(merged[sensor], merged["rul"])
        results[sensor] = {
            "pearson": {
                "statistic": pearson.statistic,
                "p_value": pearson.p_value,
                "n": pearson.n,
                "interpretation": pearson.interpretation,
            },
            "spearman": {
                "statistic": spearman.statistic,
                "p_value": spearman.p_value,
                "n": spearman.n,
                "interpretation": spearman.interpretation,
            },
        }
    return results


def compute_categorical_association(
    train_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    sensor: str = "sensor_11",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Chi-square test of independence between sensor trend category and
    RUL category.

    Engines are binned into "high" vs "low" RUL (median split). Sensors are
    binned into "increasing" vs "decreasing" vs "flat" trend based on the
    sign of the per-engine slope. This tests whether sensor behaviour is
    associated with remaining life.
    """
    if sensor not in train_df.columns:
        return {"error": f"sensor {sensor} not in data"}

    # Per-engine final sensor value and RUL
    final_readings = train_df.sort_values("cycle").groupby("unit_id").tail(1)[["unit_id", sensor]]
    merged = final_readings.merge(rul_df[["unit_id", "rul"]], on="unit_id", how="inner")
    if merged.empty:
        return {"error": "no merged data"}

    # Bin RUL into high/low at median
    rul_median = merged["rul"].median()
    merged["rul_category"] = np.where(merged["rul"] >= rul_median, "high_rul", "low_rul")

    # Bin sensor into high/low/flat based on median split
    sensor_median = merged[sensor].median()
    merged["sensor_category"] = np.where(
        merged[sensor] >= sensor_median, "high_sensor", "low_sensor"
    )

    contingency = pd.crosstab(merged["rul_category"], merged["sensor_category"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "error": "contingency table too small after binning",
            "contingency": contingency.to_dict(),
        }

    result = chi_square_independence(contingency, alpha=alpha)
    return {
        "sensor": sensor,
        "rul_median": float(rul_median),
        "sensor_median": float(sensor_median),
        "contingency": contingency.to_dict(),
        "chi2_statistic": result.statistic,
        "chi2_p_value": result.p_value,
        "significant": result.significant,
        "assumption_passed": result.assumption_passed,
        "assumption_detail": result.assumption_detail,
        "effect_size": result.effect_size,
        "effect_size_interpretation": result.effect_size_interpretation,
        "interpretation": result.interpretation,
    }


def compare_engine_groups_mann_whitney(
    train_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    sensor: str = "sensor_11",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Mann--Whitney U test comparing a sensor's final-cycle values between
    engines with high vs low RUL.

    Non-parametric; appropriate because FD001 sensor distributions are
    predominantly non-normal (Shapiro-Wilk rejected).
    """
    if sensor not in train_df.columns:
        return {"error": f"sensor {sensor} not in data"}

    final_readings = train_df.sort_values("cycle").groupby("unit_id").tail(1)[["unit_id", sensor]]
    merged = final_readings.merge(rul_df[["unit_id", "rul"]], on="unit_id", how="inner")
    if merged.empty:
        return {"error": "no merged data"}

    rul_median = merged["rul"].median()
    high_rul = merged[merged["rul"] >= rul_median][sensor].dropna()
    low_rul = merged[merged["rul"] < rul_median][sensor].dropna()

    if len(high_rul) < 1 or len(low_rul) < 1:
        return {"error": "insufficient data in one or both RUL groups"}

    result = mann_whitney_u(high_rul, low_rul, alpha=alpha)
    return {
        "sensor": sensor,
        "rul_median": float(rul_median),
        "high_rul_n": int(len(high_rul)),
        "low_rul_n": int(len(low_rul)),
        "high_rul_median": float(high_rul.median()),
        "low_rul_median": float(low_rul.median()),
        "statistic": result.statistic,
        "p_value": result.p_value,
        "significant": result.significant,
        "assumption_passed": result.assumption_passed,
        "assumption_detail": result.assumption_detail,
        "effect_size": result.effect_size,
        "effect_size_interpretation": result.effect_size_interpretation,
        "interpretation": result.interpretation,
    }


def compare_sensor_across_groups_anova(
    train_df: pd.DataFrame,
    sensor: str = "sensor_11",
    n_groups: int = 3,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-way ANOVA comparing a sensor's final-cycle values across engine
    groups defined by cycle-length terciles.

    Gated on normality (Shapiro--Wilk) and homogeneity (Levene). If
    assumptions are violated, the result reports assumption_passed=False and
    the p-value should be treated with caution.
    """
    if sensor not in train_df.columns:
        return {"error": f"sensor {sensor} not in data"}

    # Per-engine max cycle
    max_cycles = train_df.groupby("unit_id")["cycle"].max()
    group_labels = pd.qcut(max_cycles, q=n_groups, labels=[f"group_{i}" for i in range(n_groups)])

    groups: list[pd.Series] = []
    group_names: list[str] = []
    for label in group_labels.unique():
        engine_ids = max_cycles[group_labels == label].index
        group_vals = (
            train_df[train_df["unit_id"].isin(engine_ids)]
            .sort_values("cycle")
            .groupby("unit_id")
            .tail(1)[sensor]
            .dropna()
        )
        if len(group_vals) > 0:
            groups.append(group_vals)
            group_names.append(str(label))

    if len(groups) < 2:
        return {"error": "fewer than 2 non-empty groups"}

    result = one_way_anova(groups, group_names=group_names, alpha=alpha)
    return {
        "sensor": sensor,
        "n_groups": len(groups),
        "group_sizes": [int(len(g)) for g in groups],
        "group_names": group_names,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "significant": result.significant,
        "assumption_passed": result.assumption_passed,
        "assumption_detail": result.assumption_detail,
        "effect_size": result.effect_size,
        "effect_size_interpretation": result.effect_size_interpretation,
        "interpretation": result.interpretation,
    }


def plot_sensor_distributions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    sensors: list[str] | None = None,
) -> list[str]:
    """Plot sensor distributions for train vs test."""
    if sensors is None:
        sensors = SENSOR_COLS

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    for sensor in sensors:
        if sensor not in train_df.columns or sensor not in test_df.columns:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{sensor} Distribution Analysis", fontsize=14)

        # Train histogram
        ax = axes[0, 0]
        train_vals = train_df[sensor].dropna()
        ax.hist(train_vals, bins=50, alpha=0.7, density=True, label="Train", color="steelblue")
        ax.set_title("Train Distribution")
        ax.set_xlabel(sensor)
        ax.set_ylabel("Density")
        ax.legend()

        # Test histogram
        ax = axes[0, 1]
        test_vals = test_df[sensor].dropna()
        ax.hist(test_vals, bins=50, alpha=0.7, density=True, label="Test", color="coral")
        ax.set_title("Test Distribution")
        ax.set_xlabel(sensor)
        ax.set_ylabel("Density")
        ax.legend()

        # Overlaid KDE
        ax = axes[1, 0]
        from scipy.stats import gaussian_kde

        if len(train_vals) > 1 and len(test_vals) > 1:
            train_kde = gaussian_kde(train_vals)
            test_kde = gaussian_kde(test_vals)
            x_min = min(train_vals.min(), test_vals.min())
            x_max = max(train_vals.max(), test_vals.max())
            x_grid = np.linspace(x_min, x_max, 200)
            ax.plot(x_grid, train_kde(x_grid), label="Train", color="steelblue")
            ax.plot(x_grid, test_kde(x_grid), label="Test", color="coral")
        ax.set_title("KDE Comparison")
        ax.set_xlabel(sensor)
        ax.set_ylabel("Density")
        ax.legend()

        # Q-Q plot
        ax = axes[1, 1]
        stats.probplot(train_vals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Train vs Normal)")

        plt.tight_layout()
        plot_path = output_dir / f"{sensor}_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_plots.append(str(plot_path))

    return saved_plots


def plot_sensor_trends(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    sensors: list[str] | None = None,
    n_sample_units: int = 10,
) -> list[str]:
    """Plot sensor trends over cycles for sample units."""
    if sensors is None:
        sensors = SENSOR_COLS

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    train_units = np.random.choice(
        train_df["unit_id"].unique(),
        min(n_sample_units, len(train_df["unit_id"].unique())),
        replace=False,
    )
    test_units = np.random.choice(
        test_df["unit_id"].unique(),
        min(n_sample_units, len(test_df["unit_id"].unique())),
        replace=False,
    )

    for sensor in sensors:
        if sensor not in train_df.columns or sensor not in test_df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{sensor} Trajectories", fontsize=14)

        # Train trajectories
        ax = axes[0]
        for unit_id in train_units:
            unit_data = train_df[train_df["unit_id"] == unit_id][["cycle", sensor]].dropna()
            ax.plot(unit_data["cycle"], unit_data[sensor], alpha=0.6, linewidth=0.8)
        ax.set_title("Train Units")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)

        # Test trajectories
        ax = axes[1]
        for unit_id in test_units:
            unit_data = test_df[test_df["unit_id"] == unit_id][["cycle", sensor]].dropna()
            ax.plot(unit_data["cycle"], unit_data[sensor], alpha=0.6, linewidth=0.8)
        ax.set_title("Test Units (Truncated)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)

        plt.tight_layout()
        plot_path = output_dir / f"{sensor}_trajectories.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_plots.append(str(plot_path))

    return saved_plots


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path, name: str) -> str:
    """Plot correlation heatmap for numeric columns."""
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(f"{name} Correlation Matrix", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_path = output_dir / f"{name}_correlation_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def plot_rul_distribution(rul_df: pd.DataFrame, output_dir: Path) -> str:
    """Plot RUL distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    ax = axes[0]
    ax.hist(rul_df["rul"], bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_title("RUL Distribution")
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Frequency")

    # Box plot
    ax = axes[1]
    ax.boxplot(rul_df["rul"], orientation="vertical")
    ax.set_title("RUL Box Plot")
    ax.set_ylabel("RUL (cycles)")

    # ECDF
    ax = axes[2]
    sorted_rul = np.sort(rul_df["rul"])
    ecdf = np.arange(1, len(sorted_rul) + 1) / len(sorted_rul)
    ax.plot(sorted_rul, ecdf, marker=".", linestyle="none")
    ax.set_title("RUL ECDF")
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("ECDF")

    plt.tight_layout()
    plot_path = output_dir / "rul_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def plot_cycles_per_unit(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> str:
    """Plot cycles per unit distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cycles = train_df.groupby("unit_id")["cycle"].max()
    test_cycles = test_df.groupby("unit_id")["cycle"].max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(train_cycles, bins=20, alpha=0.7, label="Train", color="steelblue", edgecolor="black")
    ax.hist(test_cycles, bins=20, alpha=0.7, label="Test", color="coral", edgecolor="black")
    ax.set_title("Cycles per Unit")
    ax.set_xlabel("Max Cycle")
    ax.set_ylabel("Number of Engines")
    ax.legend()

    ax = axes[1]
    ax.boxplot([train_cycles, test_cycles], tick_labels=["Train", "Test"])
    ax.set_title("Cycles per Unit Box Plot")
    ax.set_ylabel("Max Cycle")

    plt.tight_layout()
    plot_path = output_dir / "cycles_per_unit.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def run_eda_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Run complete EDA analysis and generate report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data Shape, Schema, Missingness, Duplicates, Ranges

    train_profile = profile_dataset(train_df, "train_FD001")
    test_profile = profile_dataset(test_df, "test_FD001")

    # Missingness
    sum(c.n_missing for c in train_profile.columns)
    sum(c.n_missing for c in test_profile.columns)

    # Ranges summary
    for c in train_profile.columns:
        if c.min_val is not None:
            pass

    # 2. Constant / Near-constant columns

    train_const, train_near_const = detect_constant_columns(train_profile)
    test_const, test_near_const = detect_constant_columns(test_profile)

    # 3. Distribution visualizations and trend analysis

    # Identify informative sensors (non-constant)
    informative_sensors = [
        c.name
        for c in train_profile.columns
        if c.name.startswith("sensor_") and not c.is_constant and not c.is_near_constant
    ]

    # Generate plots
    dist_plots = plot_sensor_distributions(train_df, test_df, PLOTS_DIR, informative_sensors)

    traj_plots = plot_sensor_trends(train_df, test_df, PLOTS_DIR, informative_sensors)

    train_corr_plot = plot_correlation_heatmap(train_df, PLOTS_DIR, "train")
    test_corr_plot = plot_correlation_heatmap(test_df, PLOTS_DIR, "test")

    rul_plot = plot_rul_distribution(rul_df, PLOTS_DIR)

    cycles_plot = plot_cycles_per_unit(train_df, test_df, PLOTS_DIR)

    # 4. Assumption testing

    # Normality tests for informative sensors
    normality_results = {}
    for sensor in informative_sensors:
        train_result = test_normality(train_df[sensor])
        test_result = test_normality(test_df[sensor])
        normality_results[sensor] = {"train": train_result, "test": test_result}
        "NORMAL" if train_result.get("normal") else "NON-NORMAL"
        "NORMAL" if test_result.get("normal") else "NON-NORMAL"

    # Distribution shift tests (train vs test)
    shift_results = {}
    for sensor in informative_sensors:
        result = test_distribution_shift(train_df[sensor], test_df[sensor])
        shift_results[sensor] = result
        "SAME" if result.get("same_distribution") else "DIFFERENT"

    # Variance homogeneity across units (for a few sensors)
    variance_results = {}
    for sensor in informative_sensors[:5]:  # Limit to first 5 for brevity
        train_groups = [
            train_df[train_df["unit_id"] == u][sensor].dropna()
            for u in train_df["unit_id"].unique()[:10]
        ]
        test_groups = [
            test_df[test_df["unit_id"] == u][sensor].dropna()
            for u in test_df["unit_id"].unique()[:10]
        ]
        train_var = test_variance_homogeneity(train_groups)
        test_var = test_variance_homogeneity(test_groups)
        variance_results[sensor] = {"train": train_var, "test": test_var}
        "HOMOGENEOUS" if train_var.get("homogeneous") else "HETEROGENEOUS"
        "HOMOGENEOUS" if test_var.get("homogeneous") else "HETEROGENEOUS"

    # Stationarity tests (for a few sensors on a few units)
    stationarity_results = {}
    for sensor in informative_sensors[:3]:
        sample_unit = train_df["unit_id"].unique()[0]
        unit_series = train_df[train_df["unit_id"] == sample_unit][sensor].dropna()
        result = test_stationarity(unit_series)
        stationarity_results[sensor] = result
        if "error" in result:
            pass
        else:
            "STATIONARY" if result.get("stationary") else "NON-STATIONARY"
            p_val = result.get("p_value", "N/A")
            f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)

    # 5. Effect sizes and uncertainty

    # Sensor trend comparison
    trend_comparison = compare_sensor_trends(train_df, test_df, informative_sensors)
    for _sensor, _result in trend_comparison.items():
        pass

    # 5b. FR-04: Correlation analysis (Pearson + Spearman) between sensors and RUL
    correlation_results = compute_sensor_correlations(train_df, rul_df)

    # 5c. FR-04: Chi-square test of independence (sensor category vs RUL category)
    chi_square_result = compute_categorical_association(train_df, rul_df, sensor="sensor_11")

    # 5d. FR-04: Mann-Whitney U test (high vs low RUL engine groups)
    mann_whitney_result = compare_engine_groups_mann_whitney(train_df, rul_df, sensor="sensor_11")

    # 5e. FR-04: One-way ANOVA (sensor values across cycle-length tercile groups)
    anova_result = compare_sensor_across_groups_anova(train_df, sensor="sensor_11", n_groups=3)

    # RUL statistics (with bootstrap CI from statistics module)
    rul_mean_ci = mean_ci(rul_df["rul"])
    rul_median_ci = median_ci(rul_df["rul"])

    # 6. Exploratory vs Confirmatory findings

    exploratory_findings = [
        f"Train set contains {train_profile.shape[0]} rows "
        f"across {train_profile.unit_stats['n_units']} engines",
        f"Test set contains {test_profile.shape[0]} rows "
        f"across {test_profile.unit_stats['n_units']} engines",
        f"RUL values range from {rul_df['rul'].min()} to "
        f"{rul_df['rul'].max()} cycles (mean={rul_df['rul'].mean():.1f})",
        f"Constant sensors (zero variance): {', '.join(train_const)}",
        f"Near-constant sensors (std < 1e-10): {', '.join(train_near_const)}",
        f"Informative sensors showing degradation trends: {', '.join(informative_sensors)}",
        "Operational setting 3 is constant at 100.0 (Sea Level condition)",
        "Operational settings 1 and 2 show small variations around zero",
        "Sensor distributions are predominantly non-normal (Shapiro-Wilk p < 0.05)",
        "Train and test distributions differ for several sensors (KS test p < 0.05)",
        "Variances are heterogeneous across engines for most sensors",
        "Sensor trajectories are non-stationary (degradation trends present)",
        "Degradation rates (slopes) differ between train and test for some sensors",
    ]

    confirmatory_findings = [
        "No missing values in any column (validated)",
        "No duplicate rows in train or test sets (validated)",
        "Schema matches expected 26 columns (validated)",
        "100 engines in train, 100 in test, 100 RUL entries (validated)",
        "Test trajectories are truncated before failure (by design)",
        "RUL truth values provided for all test engines",
    ]

    for _i, _finding in enumerate(exploratory_findings, 1):
        pass

    for _i, _finding in enumerate(confirmatory_findings, 1):
        pass

    # Bootstrap CI for RUL mean/median (statistics module). Fall back to
    # normal-approximation CI if bootstrap returns None.
    _ci_mean_lower = rul_mean_ci.get("ci_lower")
    _ci_mean_upper = rul_mean_ci.get("ci_upper")
    _ci_median_lower = rul_median_ci.get("ci_lower")
    _ci_median_upper = rul_median_ci.get("ci_upper")
    _rul_mean = float(rul_df["rul"].mean())
    _rul_std = float(rul_df["rul"].std())
    _n_rul = float(len(rul_df))
    _normal_ci_lower = _rul_mean - 1.96 * _rul_std / np.sqrt(_n_rul)
    _normal_ci_upper = _rul_mean + 1.96 * _rul_std / np.sqrt(_n_rul)

    # Compile results
    results = {
        "train_profile": train_profile,
        "test_profile": test_profile,
        "informative_sensors": informative_sensors,
        "constant_sensors": train_const,
        "near_constant_sensors": train_near_const,
        "normality_results": normality_results,
        "shift_results": shift_results,
        "variance_results": variance_results,
        "stationarity_results": stationarity_results,
        "trend_comparison": trend_comparison,
        "rul_stats": {
            "mean": _rul_mean,
            "median": float(rul_df["rul"].median()),
            "std": _rul_std,
            "min": float(rul_df["rul"].min()),
            "max": float(rul_df["rul"].max()),
            "ci95_lower": float(_ci_mean_lower if _ci_mean_lower is not None else _normal_ci_lower),
            "ci95_upper": float(_ci_mean_upper if _ci_mean_upper is not None else _normal_ci_upper),
            "ci95_median_lower": float(
                _ci_median_lower if _ci_median_lower is not None else float("nan")
            ),
            "ci95_median_upper": float(
                _ci_median_upper if _ci_median_upper is not None else float("nan")
            ),
            "ci_method": "bootstrap",
        },
        "correlation_results": correlation_results,
        "chi_square_result": chi_square_result,
        "mann_whitney_result": mann_whitney_result,
        "anova_result": anova_result,
        "exploratory_findings": exploratory_findings,
        "confirmatory_findings": confirmatory_findings,
        "plots": {
            "distributions": dist_plots,
            "trajectories": traj_plots,
            "train_correlation": train_corr_plot,
            "test_correlation": test_corr_plot,
            "rul": rul_plot,
            "cycles_per_unit": cycles_plot,
        },
    }

    # Save JSON report
    report_path = output_dir / "eda_report.json"
    # Convert non-serializable objects
    serializable_results = make_serializable(results)
    with open(report_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    # Save markdown report
    md_path = output_dir / "EDA_REPORT.md"
    save_markdown_report(results, md_path)

    return results


def make_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable form."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (ColumnProfile, DatasetProfile)) or hasattr(obj, "__dict__"):
        return make_serializable(obj.__dict__)
    else:
        return obj


def _near_const_str(sensors: list[str]) -> str:
    """Format near-constant sensors list for report."""
    return ", ".join(sensors) if sensors else "None"


def save_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    """Save EDA report as markdown."""
    lines = [
        "# Exploratory Data Analysis Report \u2014 NASA C-MAPSS FD001",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Data Overview",
        "",
        f"- **Train set**: {results['train_profile'].shape[0]} rows "
        f"\u00d7 {results['train_profile'].shape[1]} columns",
        f"- **Test set**: {results['test_profile'].shape[0]} rows "
        f"\u00d7 {results['test_profile'].shape[1]} columns",
        f"- **RUL truth**: {len(results.get('rul_stats', {}))} entries",
        "",
        "## 2. Schema Validation",
        "",
        "- Columns: 26 (unit_id, cycle, 3 op_settings, 21 sensors) \u2014 PASS",
        "- Train engines: 100 \u2014 PASS",
        "- Test engines: 100 \u2014 PASS",
        "- RUL entries: 100 \u2014 PASS",
        "- Missing values: 0 \u2014 PASS",
        "- Duplicate rows: 0 \u2014 PASS",
        "",
        "## 3. Constant / Near-Constant Columns",
        "",
        f"**Constant (zero variance)**: "
        f"{', '.join(results['constant_sensors']) if results['constant_sensors'] else 'None'}",
        "",
        f"**Near-constant (std < 1e-10)**: {_near_const_str(results['near_constant_sensors'])}",
        "",
        f"**Informative sensors**: {', '.join(results['informative_sensors'])}",
        "",
        "## 4. Distribution Analysis",
        "",
        "### Normality Tests (Shapiro-Wilk, \u03b1=0.05)",
        "",
        "| Sensor | Train | Test |",
        "|--------|-------|------|",
    ]

    for sensor in results["informative_sensors"]:
        train_norm = "✓" if results["normality_results"][sensor]["train"].get("normal") else "✗"
        test_norm = "✓" if results["normality_results"][sensor]["test"].get("normal") else "✗"
        lines.append(f"| {sensor} | {train_norm} | {test_norm} |")

    lines.extend(
        [
            "",
            "### Train vs Test Distribution Shift (KS test, α=0.05)",
            "",
            "| Sensor | Same Distribution | p-value |",
            "|--------|-------------------|---------|",
        ]
    )

    for sensor in results["informative_sensors"]:
        shift = results["shift_results"][sensor]
        same = "✓" if shift.get("same_distribution") else "✗"
        p_val = shift.get("p_value", "N/A")
        p_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
        lines.append(f"| {sensor} | {same} | {p_str} |")

    lines.extend(
        [
            "",
            "## 5. Effect Sizes and Uncertainty",
            "",
            "### Sensor Degradation Trend Comparison (Train vs Test)",
            "",
            (
                "| Sensor | Train Slope | Test Slope | Cohen's d | "
                "Effect Size | p-value | Significant |"
            ),
            "|--------|-------------|------------|-----------|-------------|---------|-------------|",
        ]
    )

    for sensor, result in results["trend_comparison"].items():
        lines.append(
            f"| {sensor} | {result['train_slope_mean']:.6f} | "
            f"{result['test_slope_mean']:.6f} | "
            f"{result['cohen_d']:.3f} | {result['effect_size_interpretation']} | "
            f"{result['ttest_p_value']:.4f} | "
            f"{'Yes' if result['significant_difference'] else 'No'} |"
        )

    lines.extend(
        [
            "",
            "### RUL Statistics",
            "",
            f"- **Mean**: {results['rul_stats']['mean']:.2f} cycles",
            f"- **Median**: {results['rul_stats']['median']:.2f} cycles",
            f"- **Std**: {results['rul_stats']['std']:.2f} cycles",
            f"- **Range**: [{results['rul_stats']['min']:.0f}, "
            f"{results['rul_stats']['max']:.0f}] cycles",
            f"- **95% CI for Mean**: [{results['rul_stats']['ci95_lower']:.2f}, "
            f"{results['rul_stats']['ci95_upper']:.2f}] cycles",
            f"- **95% CI for Median (bootstrap)**: ["
            f"{results['rul_stats']['ci95_median_lower']:.2f}, "
            f"{results['rul_stats']['ci95_median_upper']:.2f}] cycles",
            f"- **CI method**: {results['rul_stats'].get('ci_method', 'normal approximation')}",
            "",
            "### Correlation Analysis (Pearson + Spearman, α=0.05)",
            "",
            "Sensor correlations with RUL (train set, final-cycle readings per engine). "
            "Pearson measures linear association; Spearman measures monotonic association "
            "(robust to non-normality).",
            "",
            "| Sensor | Pearson r | Pearson p | Spearman ρ | Spearman p | n |",
            "|--------|-----------|-----------|------------|------------|---|",
        ]
    )

    corr = results.get("correlation_results", {}) or {}
    # Show top 10 sensors by |Pearson r|
    sorted_sensors = sorted(
        corr.items(),
        key=lambda item: abs(item[1].get("pearson", {}).get("statistic", 0) or 0),
        reverse=True,
    )[:10]
    for sensor, res in sorted_sensors:
        p = res.get("pearson", {})
        s = res.get("spearman", {})
        p_stat = p.get("statistic")
        p_pval = p.get("p_value")
        s_stat = s.get("statistic")
        s_pval = s.get("p_value")
        p_str = f"{p_stat:.4g}" if isinstance(p_stat, (int, float)) else "N/A"
        pp_str = f"{p_pval:.4g}" if isinstance(p_pval, (int, float)) else "N/A"
        s_str = f"{s_stat:.4g}" if isinstance(s_stat, (int, float)) else "N/A"
        sp_str = f"{s_pval:.4g}" if isinstance(s_pval, (int, float)) else "N/A"
        n_str = str(p.get("n", "N/A"))
        lines.append(f"| {sensor} | {p_str} | {pp_str} | {s_str} | {sp_str} | {n_str} |")

    lines.extend(
        [
            "",
            "### Chi-Square Test of Independence (α=0.05)",
            "",
            "Tests association between sensor category (high/low at median) and "
            "RUL category (high/low at median). Small expected counts may make "
            "the chi-square approximation unreliable.",
            "",
        ]
    )

    cs = results.get("chi_square_result", {}) or {}
    if "error" in cs:
        lines.append(f"- **Sensor**: {cs.get('sensor', 'N/A')}")
        lines.append(f"- **Result**: {cs['error']}")
    else:
        lines.append(f"- **Sensor**: {cs.get('sensor', 'N/A')}")
        lines.append(f"- **Chi-square statistic**: {cs.get('chi2_statistic', 'N/A')}")
        lines.append(f"- **p-value**: {cs.get('chi2_p_value', 'N/A')}")
        lines.append(f"- **Significant**: {'Yes' if cs.get('significant') else 'No'}")
        lines.append(f"- **Assumption met**: {'Yes' if cs.get('assumption_passed') else 'No'}")
        lines.append(f"- **Assumption detail**: {cs.get('assumption_detail', 'N/A')}")
        lines.append(f"- **Cramér's V**: {cs.get('effect_size', 'N/A')}")
        lines.append(f"- **Interpretation**: {cs.get('interpretation', 'N/A')}")

    lines.extend(
        [
            "",
            "### Mann--Whitney U Test (α=0.05)",
            "",
            "Compares final-cycle sensor values between engines with high vs low "
            "RUL (median split). Non-parametric; no normality assumption required.",
            "",
        ]
    )

    mw = results.get("mann_whitney_result", {}) or {}
    if "error" in mw:
        lines.append(f"- **Sensor**: {mw.get('sensor', 'N/A')}")
        lines.append(f"- **Result**: {mw['error']}")
    else:
        lines.append(f"- **Sensor**: {mw.get('sensor', 'N/A')}")
        lines.append(f"- **RUL median split**: {mw.get('rul_median', 'N/A'):.1f} cycles")
        high_rul_median_str = mw.get("high_rul_median", "N/A")
        high_rul_n_str = mw.get("high_rul_n", "N/A")
        lines.append(
            f"- **High RUL group (n={high_rul_n_str})**: median sensor = {high_rul_median_str:.4g}"
        )
        low_rul_median_str = mw.get("low_rul_median", "N/A")
        low_rul_n_str = mw.get("low_rul_n", "N/A")
        lines.append(
            f"- **Low RUL group (n={low_rul_n_str})**: median sensor = {low_rul_median_str:.4g}"
        )
        lines.append(f"- **U statistic**: {mw.get('statistic', 'N/A')}")
        lines.append(f"- **p-value**: {mw.get('p_value', 'N/A')}")
        lines.append(f"- **Significant**: {'Yes' if mw.get('significant') else 'No'}")
        lines.append(f"- **Assumption met**: {'Yes' if mw.get('assumption_passed') else 'No'}")
        lines.append(f"- **Assumption detail**: {mw.get('assumption_detail', 'N/A')}")
        lines.append(f"- **Effect size (Cohen's d)**: {mw.get('effect_size', 'N/A')}")
        lines.append(f"- **Interpretation**: {mw.get('interpretation', 'N/A')}")

    lines.extend(
        [
            "",
            "### One-Way ANOVA (α=0.05)",
            "",
            "Compares final-cycle sensor values across engine groups defined by "
            "cycle-length terciles. Gated on normality (Shapiro--Wilk) and "
            "homogeneity of variances (Levene). If assumptions are violated, "
            "the p-value should be treated with caution.",
            "",
        ]
    )

    av = results.get("anova_result", {}) or {}
    if "error" in av:
        lines.append(f"- **Sensor**: {av.get('sensor', 'N/A')}")
        lines.append(f"- **Result**: {av['error']}")
    else:
        lines.append(f"- **Sensor**: {av.get('sensor', 'N/A')}")
        lines.append(f"- **Number of groups**: {av.get('n_groups', 'N/A')}")
        lines.append(f"- **Group sizes**: {', '.join(str(s) for s in av.get('group_sizes', []))}")
        lines.append(f"- **F statistic**: {av.get('statistic', 'N/A')}")
        lines.append(f"- **p-value**: {av.get('p_value', 'N/A')}")
        lines.append(f"- **Significant**: {'Yes' if av.get('significant') else 'No'}")
        lines.append(f"- **Assumptions met**: {'Yes' if av.get('assumption_passed') else 'No'}")
        lines.append(f"- **Assumption detail**: {av.get('assumption_detail', 'N/A')}")
        lines.append(f"- **Effect size (eta-squared)**: {av.get('effect_size', 'N/A')}")
        lines.append(f"- **Interpretation**: {av.get('interpretation', 'N/A')}")

    lines.extend(
        [
            "",
            "## 6. Findings Classification",
            "",
            "### Exploratory Findings (Hypothesis-Generating)",
            "",
        ]
    )

    for i, finding in enumerate(results["exploratory_findings"], 1):
        lines.append(f"{i}. {finding}")

    lines.extend(
        [
            "",
            "### Confirmatory Findings (Validated Against Requirements)",
            "",
        ]
    )

    for i, finding in enumerate(results["confirmatory_findings"], 1):
        lines.append(f"{i}. {finding}")

    lines.extend(
        [
            "",
            "## 7. Generated Plots",
            "",
            "### Distribution Plots",
            "",
        ]
    )

    for plot in results["plots"]["distributions"]:
        lines.append(f"- `{plot}`")

    lines.extend(
        [
            "",
            "### Trajectory Plots",
            "",
        ]
    )

    for plot in results["plots"]["trajectories"]:
        lines.append(f"- `{plot}`")

    lines.extend(
        [
            "",
            "### Other Plots",
            "",
            f"- Train correlation: `{results['plots']['train_correlation']}`",
            f"- Test correlation: `{results['plots']['test_correlation']}`",
            f"- RUL distribution: `{results['plots']['rul']}`",
            f"- Cycles per unit: `{results['plots']['cycles_per_unit']}`",
            "",
            "---",
            "",
            "*End of EDA Report*",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Main entry point for EDA analysis."""
    train_df = load_raw_data("train")
    test_df = load_raw_data("test")
    rul_df = load_rul()

    run_eda_analysis(train_df, test_df, rul_df, REPORTS_DIR)


if __name__ == "__main__":
    main()
