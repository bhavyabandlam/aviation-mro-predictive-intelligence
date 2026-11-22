"""Tests for financial pipeline module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.aviation_mro.analytics.financial.pipeline import (
    run_financial_pipeline,
    save_results,
)


class TestPipeline:
    """Test financial pipeline."""

    def test_run_pipeline(self, tmp_path):
        """Test running the full pipeline."""
        # Create minimal synthetic data for testing
        from src.aviation_mro.analytics.financial.data_prep import generate_all_financial_data
        from src.aviation_mro.analytics.inventory.data_prep import generate_synthetic_inventory
        from src.aviation_mro.analytics.inventory.reorder_point import calculate_reorder_points
        from src.aviation_mro.analytics.inventory.safety_stock import calculate_safety_stock
        from src.aviation_mro.analytics.inventory.stockout_risk import analyze_stockout_risk

        financial_data = generate_all_financial_data(seed=42)
        inventory_df = generate_synthetic_inventory(seed=42)
        safety_stock_results = calculate_safety_stock(inventory_df, method="normal_constant_lt")
        safety_stock_df = pd.DataFrame(
            [
                {"part_number": r.part_number, "safety_stock": r.safety_stock}
                for r in safety_stock_results
            ]
        )
        rop_results = calculate_reorder_points(inventory_df)
        stockout_risk_results = analyze_stockout_risk(inventory_df, rop_results)
        stockout_risk_df = pd.DataFrame(
            [
                {
                    "part_number": r.part_number,
                    "stockout_probability": r.stockout_probability,
                    "cycles_per_year": r.cycles_per_year,
                    "risk_category": r.risk_category,
                }
                for r in stockout_risk_results
            ]
        )

        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        # Run pipeline with temporary data directory
        # We need to patch the data loading
        import src.aviation_mro.analytics.financial.data_prep as dp

        original_dir = dp.DATA_DIR
        dp.DATA_DIR = tmp_path / "data" / "processed"
        dp.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in financial_data.items():
            df.to_parquet(dp.DATA_DIR / f"financial_{name}.parquet", index=False)

        try:
            results = run_financial_pipeline(
                input_dir=dp.DATA_DIR,
                output_dir=output_dir,
                inventory_df=inventory_df,
                safety_stock_df=safety_stock_df,
                stockout_risk_df=stockout_risk_df,
            )

            # Check results structure
            assert "labor_costs" in results
            assert "parts_costs" in results
            assert "downtime_costs" in results
            assert "holding_costs" in results
            assert "stockout_costs" in results
            assert "financial_kpis" in results
            assert "scenario_analysis" in results

            # Check output files
            assert (output_dir / "financial_analysis.json").exists()
            assert (output_dir / "FINANCIAL_REPORT.md").exists()

        finally:
            dp.DATA_DIR = original_dir


class TestSaveResults:
    """Test save_results function."""

    def test_save_results_creates_json(self, tmp_path):
        """Test save_results creates valid JSON."""
        results = {
            "test_key": "test_value",
            "number": 42,
        }
        output_path = tmp_path / "test_results.json"
        save_results(results, output_path)
        assert output_path.exists()

        import json

        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["test_key"] == "test_value"
        assert loaded["number"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
