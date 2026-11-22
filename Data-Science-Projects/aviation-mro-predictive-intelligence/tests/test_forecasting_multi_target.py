"""Tests for multi-target forecasting pipeline."""

from __future__ import annotations

from pathlib import Path

from src.aviation_mro.analytics.forecasting.data_prep import (
    PRIMARY_TARGET,
    TARGET_COLUMNS,
    generate_synthetic_demand,
    split_chronologically,
)
from src.aviation_mro.analytics.forecasting.train import (
    get_all_series_dicts,
    run_forecasting_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"


class TestMultiTargetPipeline:
    """Test the forecasting pipeline runs for all target columns."""

    def test_get_all_series_dicts_for_each_target(self):
        df = generate_synthetic_demand(seed=42)
        split = split_chronologically(df)

        for target_column in TARGET_COLUMNS:
            train_dict, val_dict, test_dict = get_all_series_dicts(
                split, target_column=target_column
            )
            assert len(train_dict) == 15
            assert len(val_dict) == 15
            assert len(test_dict) == 15
            # Verify the correct column is extracted
            first_key = list(train_dict.keys())[0]
            series = train_dict[first_key]
            assert series.name == first_key

    def test_run_pipeline_for_each_target(self):
        for target_column in TARGET_COLUMNS:
            results = run_forecasting_pipeline(target_column=target_column)
            assert results["target_column"] == target_column
            assert results["n_series"] == 15
            assert results["selected_seasonal_period"] in (13, 52)
            assert "naive" in results["test_aggregate"]
            assert "seasonal_naive" in results["test_aggregate"]

    def test_pipeline_saves_per_target_results(self):
        for target_column in TARGET_COLUMNS:
            if target_column == PRIMARY_TARGET:
                results_path = REPORTS_DIR / "forecast_results.json"
            else:
                results_path = REPORTS_DIR / f"forecast_results_{target_column}.json"
            assert results_path.exists(), f"Missing results for {target_column}"

    def test_pipeline_saves_per_target_models(self):
        models_dir = REPORTS_DIR / "models"
        for target_column in TARGET_COLUMNS:
            if target_column == PRIMARY_TARGET:
                naive_path = models_dir / "naive_baseline.pkl"
            else:
                naive_path = models_dir / f"naive_baseline_{target_column}.pkl"
            assert naive_path.exists(), f"Missing naive model for {target_column}"

    def test_seasonal_naive_better_than_naive(self):
        for target_column in TARGET_COLUMNS:
            results = run_forecasting_pipeline(target_column=target_column)
            test_agg = results["test_aggregate"]
            naive_mae = test_agg["naive"]["mae"]
            seasonal_mae = test_agg["seasonal_naive"]["mae"]
            assert seasonal_mae < naive_mae, (
                f"Seasonal-naive should beat naive for {target_column}: "
                f"{seasonal_mae} vs {naive_mae}"
            )

    def test_results_have_required_metrics(self):
        for target_column in TARGET_COLUMNS:
            results = run_forecasting_pipeline(target_column=target_column)
            test_agg = results["test_aggregate"]
            for model in ("naive", "seasonal_naive"):
                metrics = test_agg[model]
                assert "mae" in metrics
                assert "rmse" in metrics
                assert "wape" in metrics
                assert "coverage" in metrics
