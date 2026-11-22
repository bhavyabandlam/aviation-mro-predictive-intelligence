"""Baseline RUL prediction models with engine-level validation.

Implements SRS 9.2 requirements:
- Engine/unit split (no row-level leakage)
- Validation for model selection, test for final eval
- Imputation/scaling/feature selection fit on train only
- Past-only rolling features
- RUL capping documented
- MAE, RMSE, R² + per-engine error distributions
- Identical splits/features for fair comparison
- Save dataset fingerprint, model params, feature names, metrics
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from aviation_mro.features.rul_features import (
    FeatureConfig,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fd001"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelResult:
    """Results for a single model."""

    name: str
    model_type: str
    params: dict
    val_mae: float
    val_rmse: float
    val_r2: float
    val_engine_errors: dict[int, float]
    feature_names: list[str]
    n_features: int


@dataclass
class TrainingArtifacts:
    """Complete training artifacts for reproducibility."""

    dataset_fingerprint: dict[str, str]
    feature_config: dict
    feature_names: list[str]
    train_engine_ids: list[int]
    val_engine_ids: list[int]
    imputer: Any
    scaler: Any
    model_results: list[ModelResult]
    best_model_name: str
    best_model: Any
    test_metrics: dict[str, float]
    test_engine_errors: dict[int, float]
    rul_cap: int


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


def get_models() -> dict[str, Any]:
    """Return baseline models to compare."""
    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "DummyMedian": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def compute_per_engine_errors(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[int, float]:
    """Mean absolute error per engine."""
    df = df.copy()
    df["pred"] = y_pred
    df["true"] = y_true
    grouped = df.groupby("unit_id")
    return {int(str(k)): float(np.mean(np.abs(g["true"] - g["pred"]))) for k, g in grouped}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    model: Any,
) -> ModelResult:
    """Train model and evaluate on validation set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    engine_errors = compute_per_engine_errors(val_df, y_val, y_pred)

    return ModelResult(
        name=model_name,
        model_type=type(model).__name__,
        params=model.get_params() if hasattr(model, "get_params") else {},
        val_mae=float(mae),
        val_rmse=float(rmse),
        val_r2=float(r2),
        val_engine_errors=engine_errors,
        feature_names=feature_names,
        n_features=len(feature_names),
    )


def run_baseline_training(
    config: FeatureConfig | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> TrainingArtifacts:
    """Complete baseline training pipeline.

    Args:
        config: Feature engineering configuration.
        models_dir: Directory for the saved model artifact. Defaults to the
            production ``models/`` directory.
        reports_dir: Directory for the saved model card. Defaults to the
            production ``reports/`` directory.

    Tests pass temporary directories so that training never overwrites the
    preserved production artifacts.
    """
    if config is None:
        config = FeatureConfig()

    out_models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    out_reports_dir = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_reports_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    data = load_and_prepare_data(config)

    train_df = data["train_df"]
    val_df = data["val_df"]
    test_last = data["test_last"]
    rul_df = data["rul_df"]
    feature_cols = data["feature_cols"]
    train_ids = [int(x) for x in data["train_engine_ids"]]
    val_ids = [int(x) for x in data["val_engine_ids"]]

    # Prepare arrays
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train_df[feature_cols]))
    y_train = train_df["rul_capped"].values

    X_val = scaler.transform(imputer.transform(val_df[feature_cols]))
    y_val = val_df["rul_capped"].values

    X_test = scaler.transform(imputer.transform(test_last[feature_cols]))
    y_test_true = rul_df.set_index("unit_id").loc[test_last["unit_id"].values, "rul"].values

    # Train and evaluate models on validation
    models = get_models()
    model_results = []

    for name, model in models.items():
        result = train_and_evaluate(
            X_train, y_train, X_val, y_val, val_df, feature_cols, name, model
        )
        model_results.append(result)

    # Select best model (lowest validation RMSE)
    best_result = min(model_results, key=lambda r: r.val_rmse)
    best_model = models[best_result.name]

    # Final evaluation on test set
    y_test_pred = best_model.predict(X_test)

    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_r2 = r2_score(y_test_true, y_test_pred)

    test_engine_errors = compute_per_engine_errors(
        test_last.assign(rul_true=y_test_true), y_test_true, y_test_pred
    )

    # Compile artifacts
    artifacts = TrainingArtifacts(
        dataset_fingerprint=dataset_fingerprint(),
        feature_config=asdict(config),
        feature_names=feature_cols,
        train_engine_ids=train_ids,
        val_engine_ids=val_ids,
        imputer=imputer,
        scaler=scaler,
        model_results=model_results,
        best_model_name=best_result.name,
        best_model=best_model,
        test_metrics={
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "r2": float(test_r2),
        },
        test_engine_errors={int(k): float(v) for k, v in test_engine_errors.items()},
        rul_cap=config.rul_cap,
    )

    # Save artifacts
    model_path = out_models_dir / "baseline_rul_model.pkl"
    joblib.dump(artifacts, model_path)

    # Save human-readable model card
    card_path = out_reports_dir / "model_card_baseline.json"
    card_data = {
        "dataset_fingerprint": artifacts.dataset_fingerprint,
        "feature_config": artifacts.feature_config,
        "feature_names": artifacts.feature_names,
        "n_features": len(artifacts.feature_names),
        "train_engine_ids": artifacts.train_engine_ids,
        "val_engine_ids": artifacts.val_engine_ids,
        "rul_cap": artifacts.rul_cap,
        "model_comparison": [
            {
                "name": r.name,
                "model_type": r.model_type,
                "val_mae": r.val_mae,
                "val_rmse": r.val_rmse,
                "val_r2": r.val_r2,
                "n_features": r.n_features,
            }
            for r in artifacts.model_results
        ],
        "best_model": artifacts.best_model_name,
        "test_metrics": artifacts.test_metrics,
        "test_per_engine_errors": artifacts.test_engine_errors,
        "test_error_summary": {
            "mean": float(np.mean(list(artifacts.test_engine_errors.values()))),
            "std": float(np.std(list(artifacts.test_engine_errors.values()))),
            "min": float(np.min(list(artifacts.test_engine_errors.values()))),
            "max": float(np.max(list(artifacts.test_engine_errors.values()))),
            "median": float(np.median(list(artifacts.test_engine_errors.values()))),
        },
    }
    with open(card_path, "w") as f:
        json.dump(card_data, f, indent=2)

    return artifacts


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals: actual minus predicted.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Model predictions.

    Returns:
        Residual array (actual - predicted).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_true - y_pred  # type: ignore[no-any-return]


def residual_statistics(residuals: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for a residual array.

    Returns mean, std, RMSE, MAE, median, min, max and selected percentiles.
    All values are finite; NaN residuals are dropped before computation.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(r.size),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(r**2))),
        "mae": float(np.mean(np.abs(r))),
        "median": float(np.median(r)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "p5": float(np.percentile(r, 5)),
        "p25": float(np.percentile(r, 25)),
        "p75": float(np.percentile(r, 75)),
        "p95": float(np.percentile(r, 95)),
    }


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    output_dir: str | Path,
    prefix: str = "residual",
) -> list[str]:
    """Generate residual diagnostic plots (histogram, Q-Q, residual vs predicted).

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Model predictions.
        residuals: Residual array (actual - predicted).
        output_dir: Directory to write plots.
        prefix: Filename prefix for saved plots.

    Returns:
        List of saved plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    saved: list[str] = []

    # Residual histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    if r.size > 0:
        ax.hist(r, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual Histogram")
    ax.set_xlabel("Residual (actual - predicted, cycles)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    path = output_dir / f"{prefix}_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(path))

    # Q-Q plot of residuals vs normal
    fig, ax = plt.subplots(figsize=(6, 4))
    if r.size > 2:
        stats.probplot(r, dist="norm", plot=ax)  # type: ignore[call-overload]
    ax.set_title("Residual Q-Q Plot (vs Normal)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    plt.tight_layout()
    path = output_dir / f"{prefix}_qq.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(path))

    # Residual vs predicted
    fig, ax = plt.subplots(figsize=(6, 4))
    if r.size > 0:
        ax.scatter(y_pred, r, alpha=0.6, s=20, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual vs Predicted RUL")
    ax.set_xlabel("Predicted RUL (cycles)")
    ax.set_ylabel("Residual (cycles)")
    plt.tight_layout()
    path = output_dir / f"{prefix}_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(path))

    return saved


def run_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "residual",
) -> dict[str, Any]:
    """Run complete residual analysis: statistics + diagnostic plots.

    Read-only: does not train, refit, or modify any model or artifact.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Model predictions (e.g. from the saved model on the held-out split).
        output_dir: Directory for residual statistics and plots.
        prefix: Filename prefix.

    Returns:
        Dict with residual statistics and plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = compute_residuals(y_true, y_pred)
    stats = residual_statistics(residuals)
    plots = plot_residuals(y_true, y_pred, residuals, output_dir, prefix)
    return {
        "residual_statistics": stats,
        "plots": plots,
        "n_residuals": int(stats["count"]),
    }


def main() -> None:
    """Main entry point."""
    config = FeatureConfig(rul_cap=125)
    run_baseline_training(config)


if __name__ == "__main__":
    main()
