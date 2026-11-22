"""Forecasting Training Pipeline Orchestration.

Coordinates the complete forecasting workflow:
1. Load data and chronological splits
2. Train baseline models
3. Evaluate on validation (rolling-origin) and test
4. Calibrate prediction intervals
5. Select best seasonal period
6. Save artifacts and generate report
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    evaluate_seasonal_candidates,
    naive_factory,
    seasonal_naive_factory,
)
from .data_prep import (
    FORECAST_HORIZON,
    PART_FAMILIES,
    PRIMARY_TARGET,
    SHOPS,
    TARGET_COLUMNS,
    ChronologicalSplit,
    compute_dataset_fingerprint,
    load_dataset,
    split_chronologically,
)
from .evaluation import (
    aggregate_results,
    evaluate_model_on_all_series,
    evaluate_test_set,
)
from .uncertainty import (
    calibrate_all_series,
    evaluate_coverage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"
MODELS_DIR = REPORTS_DIR / "models"
PLOTS_DIR = REPORTS_DIR / "plots"


@dataclass
class ModelArtifact:
    """Serializable model artifact with metadata."""

    model_type: str  # "naive" or "seasonal_naive"
    seasonal_period: int | None
    model_params: dict
    target_column: str
    train_dates: dict[str, str]  # start, end
    validation_dates: dict[str, str]
    test_dates: dict[str, str]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    interval_calibration: dict | None  # Serialized IntervalCalibration
    dataset_fingerprint: str
    random_seed: int
    forecast_horizon: int


def get_series_dict(
    split: ChronologicalSplit, target_column: str = PRIMARY_TARGET
) -> dict[str, pd.Series]:
    """Extract time series for all shop-part_family combinations from a split.

    Args:
        split: ChronologicalSplit with train/validation/test/future_buffer
        target_column: Target column to extract (default: PRIMARY_TARGET)
    """
    series_dict = {}
    for shop in SHOPS:
        for part_family in PART_FAMILIES:
            series_id = f"{shop}_{part_family}"
            mask = (split.train["shop_id"] == shop) & (split.train["part_family"] == part_family)
            series = split.train.loc[mask].set_index("date")[target_column].sort_index()
            series.name = series_id
            series_dict[series_id] = series
    return series_dict


def get_all_series_dicts(
    split: ChronologicalSplit, target_column: str = PRIMARY_TARGET
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]]:
    """Get train, validation, test series dicts for all combinations.

    Args:
        split: ChronologicalSplit
        target_column: Target column to extract (default: PRIMARY_TARGET)
    """
    train_dict: dict[str, pd.Series] = {}
    val_dict: dict[str, pd.Series] = {}
    test_dict: dict[str, pd.Series] = {}

    for shop in SHOPS:
        for part_family in PART_FAMILIES:
            series_id = f"{shop}_{part_family}"

            for split_df, target_dict in [
                (split.train, train_dict),
                (split.validation, val_dict),
                (split.test, test_dict),
            ]:
                mask = (split_df["shop_id"] == shop) & (split_df["part_family"] == part_family)
                series = split_df.loc[mask].set_index("date")[target_column].sort_index()
                series.name = series_id
                target_dict[series_id] = series

    return train_dict, val_dict, test_dict


def run_forecasting_pipeline(
    target_column: str = PRIMARY_TARGET,
) -> dict[str, Any]:
    """Run the complete forecasting pipeline for a given target column.

    Args:
        target_column: Target column to forecast (default: PRIMARY_TARGET)

    Returns:
        Results summary dict
    """
    is_primary = target_column == PRIMARY_TARGET

    # 1. Load data and split
    df = load_dataset(DATA_DIR / "maintenance_demand.parquet")
    split = split_chronologically(df)
    fingerprint = compute_dataset_fingerprint(df)

    train_dict, val_dict, test_dict = get_all_series_dicts(split, target_column=target_column)

    # 2. Evaluate seasonal period candidates
    first_series_id = list(train_dict.keys())[0]
    train_series = train_dict[first_series_id]
    val_series = val_dict[first_series_id]

    candidate_results = evaluate_seasonal_candidates(
        train_series, val_series, candidate_periods=[52, 13], horizon=FORECAST_HORIZON
    )

    for _period, metrics in candidate_results.items():
        if "error" in metrics:
            pass
        else:
            "BETTER than naive" if metrics["better_than_naive"] else "worse than naive"

    # Select best seasonal period
    best_period = None
    best_mae = float("inf")
    for period, metrics in candidate_results.items():
        if "error" not in metrics and metrics["better_than_naive"] and metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_period = period

    if best_period:
        pass
    else:
        pass

    # 3. Train and evaluate baseline models on validation

    # Naive model
    naive_results = evaluate_model_on_all_series(
        train_dict,
        val_dict,
        naive_factory,
        split.validation_origins,
        FORECAST_HORIZON,
        model_kwargs={"model_name": "naive"},
    )
    naive_agg = aggregate_results(naive_results)

    # Seasonal-naive model (if selected)
    seasonal_results = {}
    seasonal_agg = {}
    if best_period:

        def snaive_factory(history: pd.Series, **kwargs: Any) -> Any:
            seasonal_period = kwargs.pop("seasonal_period", best_period)
            return seasonal_naive_factory(history, seasonal_period, **kwargs)

        seasonal_results = evaluate_model_on_all_series(
            train_dict,
            val_dict,
            snaive_factory,
            split.validation_origins,
            FORECAST_HORIZON,
            model_kwargs={
                "model_name": f"seasonal_naive_m{best_period}",
                "seasonal_period": best_period,
            },
        )
        seasonal_agg = aggregate_results(seasonal_results)

    # 4. Calibrate prediction intervals on validation

    naive_calibrations = calibrate_all_series(
        train_dict,
        val_dict,
        naive_factory,
        split.validation_origins,
        FORECAST_HORIZON,
        model_kwargs={"model_name": "naive"},
    )

    seasonal_calibrations = {}
    if best_period:
        seasonal_calibrations = calibrate_all_series(
            train_dict,
            val_dict,
            snaive_factory,
            split.validation_origins,
            FORECAST_HORIZON,
            model_kwargs={
                "model_name": f"seasonal_naive_m{best_period}",
                "seasonal_period": best_period,
            },
        )

    # Print calibration coverage
    for _series_id in train_dict:
        if best_period:
            pass

    # 5. Final test evaluation
    test_results = {}

    for series_id in train_dict:
        # Naive test evaluation
        naive_test = evaluate_test_set(
            train_dict[series_id],
            val_dict[series_id],
            test_dict[series_id],
            naive_factory,
            FORECAST_HORIZON,
            model_kwargs={"model_name": "naive"},
        )
        # Evaluate coverage on test
        naive_coverage = evaluate_coverage(
            train_dict[series_id],
            val_dict[series_id],
            test_dict[series_id],
            naive_factory,
            split.validation_origins,
            FORECAST_HORIZON,
            model_kwargs={"model_name": "naive"},
        )

        test_results[series_id] = {
            "naive": {**naive_test, **naive_coverage},
        }

        if best_period:
            seasonal_test = evaluate_test_set(
                train_dict[series_id],
                val_dict[series_id],
                test_dict[series_id],
                snaive_factory,
                FORECAST_HORIZON,
                model_kwargs={
                    "model_name": f"seasonal_naive_m{best_period}",
                    "seasonal_period": best_period,
                },
            )
            seasonal_coverage = evaluate_coverage(
                train_dict[series_id],
                val_dict[series_id],
                test_dict[series_id],
                snaive_factory,
                split.validation_origins,
                FORECAST_HORIZON,
                model_kwargs={
                    "model_name": f"seasonal_naive_m{best_period}",
                    "seasonal_period": best_period,
                },
            )
            test_results[series_id]["seasonal_naive"] = {**seasonal_test, **seasonal_coverage}

    # Aggregate test results
    naive_test_mae = np.nanmean([r["naive"]["mae"] for r in test_results.values()])
    naive_test_rmse = np.nanmean([r["naive"]["rmse"] for r in test_results.values()])
    naive_test_wape = np.nanmean([r["naive"]["wape"] for r in test_results.values()])
    naive_test_cov = np.nanmean([r["naive"]["empirical_coverage"] for r in test_results.values()])

    if best_period:
        seasonal_test_mae = np.nanmean([r["seasonal_naive"]["mae"] for r in test_results.values()])
        seasonal_test_rmse = np.nanmean(
            [r["seasonal_naive"]["rmse"] for r in test_results.values()]
        )
        seasonal_test_wape = np.nanmean(
            [r["seasonal_naive"]["wape"] for r in test_results.values()]
        )
        seasonal_test_cov = np.nanmean(
            [r["seasonal_naive"]["empirical_coverage"] for r in test_results.values()]
        )

    # 6. Save artifacts and generate report

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine output paths (backward compatible for primary target)
    if is_primary:
        naive_model_path = MODELS_DIR / "naive_baseline.pkl"
        seasonal_model_path = MODELS_DIR / f"seasonal_naive_m{best_period}_baseline.pkl"
        results_path = REPORTS_DIR / "forecast_results.json"
    else:
        naive_model_path = MODELS_DIR / f"naive_baseline_{target_column}.pkl"
        seasonal_model_path = (
            MODELS_DIR / f"seasonal_naive_m{best_period}_{target_column}_baseline.pkl"
        )
        results_path = REPORTS_DIR / f"forecast_results_{target_column}.json"

    # Save model artifacts
    # Naive artifact
    naive_artifact = ModelArtifact(
        model_type="naive",
        seasonal_period=None,
        model_params={},
        target_column=target_column,
        train_dates={
            "start": str(split.train["date"].min()),
            "end": str(split.train["date"].max()),
        },
        validation_dates={
            "start": str(split.validation["date"].min()),
            "end": str(split.validation["date"].max()),
        },
        test_dates={
            "start": str(split.test["date"].min()),
            "end": str(split.test["date"].max()),
        },
        validation_metrics=naive_agg,
        test_metrics={
            "mae": float(naive_test_mae),
            "rmse": float(naive_test_rmse),
            "wape": float(naive_test_wape),
            "coverage": float(naive_test_cov),
        },
        interval_calibration={
            "horizon": FORECAST_HORIZON,
            "empirical_coverage": {
                sid: cal.empirical_coverage for sid, cal in naive_calibrations.items()
            },
            "n_residuals_per_horizon": {
                sid: cal.n_residuals_per_horizon for sid, cal in naive_calibrations.items()
            },
        },
        dataset_fingerprint=fingerprint,
        random_seed=42,
        forecast_horizon=FORECAST_HORIZON,
    )

    with open(naive_model_path, "wb") as f:
        pickle.dump(naive_artifact, f)

    if best_period:
        seasonal_artifact = ModelArtifact(
            model_type="seasonal_naive",
            seasonal_period=best_period,
            model_params={"seasonal_period": best_period},
            target_column=target_column,
            train_dates={
                "start": str(split.train["date"].min()),
                "end": str(split.train["date"].max()),
            },
            validation_dates={
                "start": str(split.validation["date"].min()),
                "end": str(split.validation["date"].max()),
            },
            test_dates={
                "start": str(split.test["date"].min()),
                "end": str(split.test["date"].max()),
            },
            validation_metrics=seasonal_agg,
            test_metrics={
                "mae": float(seasonal_test_mae),
                "rmse": float(seasonal_test_rmse),
                "wape": float(seasonal_test_wape),
                "coverage": float(seasonal_test_cov),
            },
            interval_calibration={
                "horizon": FORECAST_HORIZON,
                "empirical_coverage": {
                    sid: cal.empirical_coverage for sid, cal in seasonal_calibrations.items()
                },
                "n_residuals_per_horizon": {
                    sid: cal.n_residuals_per_horizon for sid, cal in seasonal_calibrations.items()
                },
            },
            dataset_fingerprint=fingerprint,
            random_seed=42,
            forecast_horizon=FORECAST_HORIZON,
        )

        with open(seasonal_model_path, "wb") as f:
            pickle.dump(seasonal_artifact, f)

    # Save comprehensive results JSON
    results_summary = {
        "target_column": target_column,
        "dataset_fingerprint": fingerprint,
        "random_seed": 42,
        "forecast_horizon": FORECAST_HORIZON,
        "n_series": len(train_dict),
        "series_ids": list(train_dict.keys()),
        "split": {
            "train_weeks": f"1-{split.train_end_week}",
            "validation_weeks": f"{split.train_end_week + 1}-{split.validation_end_week}",
            "test_weeks": f"{split.validation_end_week + 1}-{split.test_end_week}",
            "validation_origins": split.validation_origins,
        },
        "seasonal_candidates": candidate_results,
        "selected_seasonal_period": best_period,
        "validation_aggregate": {
            "naive": naive_agg,
            "seasonal_naive": seasonal_agg if best_period else None,
        },
        "test_aggregate": {
            "naive": {
                "mae": float(naive_test_mae),
                "rmse": float(naive_test_rmse),
                "wape": float(naive_test_wape),
                "coverage": float(naive_test_cov),
            },
            "seasonal_naive": {
                "mae": float(seasonal_test_mae),
                "rmse": float(seasonal_test_rmse),
                "wape": float(seasonal_test_wape),
                "coverage": float(seasonal_test_cov),
            }
            if best_period
            else None,
        },
        "per_series_test_results": test_results,
        "limitations": [
            "Data is SYNTHETIC (source: synthetic_mro_v1) — not real maintenance records",
            (
                "Seasonal patterns were built into the generator; "
                "actual seasonal strength evaluated from training data"
            ),
            (
                "Prediction intervals use empirical quantile method on "
                "validation residuals — approximate coverage"
            ),
            "WAPE is undefined (NaN) for series with zero demand throughout evaluation window",
            (
                "Test evaluation uses only first 12 weeks of test set; "
                "weeks 143-156 held out as future buffer"
            ),
            "Only naive and seasonal-naive baselines implemented; advanced models deferred",
        ],
    }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results_summary, fh, indent=2, default=str)

    return results_summary


def run_all_targets() -> dict[str, dict[str, Any]]:
    """Run the forecasting pipeline for all target columns.

    Returns:
        Dict mapping target_column -> results summary
    """
    results = {}
    for target_column in TARGET_COLUMNS:
        results[target_column] = run_forecasting_pipeline(target_column=target_column)
    return results


def main() -> None:
    """Main entry point for forecasting pipeline."""
    run_all_targets()


if __name__ == "__main__":
    main()
