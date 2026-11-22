"""Tests for inventory pipeline module."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.data_prep import generate_synthetic_inventory
from src.aviation_mro.analytics.inventory.pipeline import (
    run_inventory_pipeline,
    save_results,
)


class TestPipeline:
    """Test inventory pipeline."""

    def test_run_pipeline(self, tmp_path):
        """Test running the full pipeline."""
        # Generate test data
        df = generate_synthetic_inventory(seed=42)
        data_path = tmp_path / "inventory_data.parquet"
        df.to_parquet(data_path, index=False)

        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        # Run pipeline
        results = run_inventory_pipeline(input_path=data_path, output_dir=output_dir)

        # Check results structure
        assert "validation" in results
        assert "demand_analysis" in results
        assert "reorder_points" in results
        assert "safety_stock" in results
        assert "lead_time_demand" in results
        assert "stockout_risk" in results
        assert "recommendations" in results
        assert "metadata" in results

        # Check validation
        assert results["validation"].is_valid is True
        assert results["validation"].n_parts == 150

        # Check demand analysis
        assert results["demand_analysis"]["n_parts"] == 150

        # Check reorder points
        assert len(results["reorder_points"]) == 150

        # Check safety stock
        assert len(results["safety_stock"]) == 150

        # Check lead time demand
        assert len(results["lead_time_demand"]) == 150

        # Check stockout risk
        assert len(results["stockout_risk"]) == 150

        # Check recommendations
        assert len(results["recommendations"]) > 0

        # Check metadata
        assert results["metadata"]["source"] == "synthetic_inventory_v1"
        assert results["metadata"]["n_parts"] == 150

        # Check output files
        assert (output_dir / "inventory_analysis.json").exists()
        assert (output_dir / "INVENTORY_REPORT.md").exists()

    def test_pipeline_fails_on_invalid_data(self, tmp_path):
        """Test pipeline fails on invalid data."""
        # Create invalid data (missing required column)
        df = pd.DataFrame(
            {
                "part_number": ["PART-0001"],
                "part_family": ["ENGINE"],
                # Missing required columns
            }
        )
        data_path = tmp_path / "invalid_inventory.parquet"
        df.to_parquet(data_path, index=False)

        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="Data validation failed"):
            run_inventory_pipeline(input_path=data_path, output_dir=output_dir)


class TestSaveResults:
    """Test save_results function."""

    def test_save_results_creates_json(self, tmp_path):
        """Test save_results creates valid JSON."""
        from src.aviation_mro.analytics.inventory.data_validation import ValidationResult

        results = {
            "validation": ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                n_rows=10,
                n_parts=10,
            ),
            "reorder_points": [],  # Empty for simplicity
        }

        output_path = tmp_path / "test_results.json"
        save_results(results, output_path)

        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["validation"]["is_valid"] is True
        assert loaded["validation"]["n_parts"] == 10

    def test_save_results_handles_dataclasses(self, tmp_path):
        """Test save_results handles dataclass objects."""
        from src.aviation_mro.analytics.inventory.reorder_point import ReorderPointResult

        results = {
            "reorder_points": [
                ReorderPointResult(
                    part_number="PART-0001",
                    part_family="ENGINE",
                    shop_id="SHOP_01",
                    lead_time_weeks=2.0,
                    mean_weekly_demand=10.0,
                    std_weekly_demand=2.0,
                    service_level=0.95,
                    z_score=1.645,
                    lead_time_demand=20.0,
                    safety_stock=5.0,
                    reorder_point=25.0,
                    assumptions=["Test"],
                )
            ]
        }

        output_path = tmp_path / "test_results.json"
        save_results(results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert len(loaded["reorder_points"]) == 1
        assert loaded["reorder_points"][0]["part_number"] == "PART-0001"
        assert loaded["reorder_points"][0]["reorder_point"] == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
