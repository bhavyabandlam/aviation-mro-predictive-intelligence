"""Tests for RUL residual analysis (SRS v1.1 FR-06).

Tests are read-only: they use the residual-analysis helpers on synthetic
predictions and never retrain or modify any model artifact or saved metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from aviation_mro.models.baseline import (  # noqa: E402
    compute_residuals,
    plot_residuals,
    residual_statistics,
    run_residual_analysis,
)


class TestComputeResiduals:
    def test_residuals_are_actual_minus_predicted(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 18.0, 35.0])
        r = compute_residuals(y_true, y_pred)
        np.testing.assert_array_equal(r, np.array([-1.0, 2.0, -5.0]))

    def test_residuals_handle_nan(self) -> None:
        y_true = np.array([10.0, np.nan, 30.0])
        y_pred = np.array([11.0, 18.0, 35.0])
        r = compute_residuals(y_true, y_pred)
        assert np.isnan(r[1])
        assert r[0] == -1.0


class TestResidualStatistics:
    def test_statistics_finite(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0, 5, 100)
        stats = residual_statistics(r)
        assert stats["count"] == 100
        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["std"])
        assert np.isfinite(stats["rmse"])
        assert np.isfinite(stats["mae"])
        assert stats["min"] <= stats["median"] <= stats["max"]
        assert stats["p5"] <= stats["p25"] <= stats["p75"] <= stats["p95"]

    def test_statistics_known_values(self) -> None:
        r = np.array([1.0, -1.0, 2.0, -2.0])
        stats = residual_statistics(r)
        assert stats["count"] == 4
        assert stats["mean"] == pytest.approx(0.0)
        assert stats["mae"] == pytest.approx(1.5)
        assert stats["rmse"] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))

    def test_empty_residuals(self) -> None:
        stats = residual_statistics(np.array([]))
        assert stats["count"] == 0
        assert np.isnan(stats["mean"])

    def test_all_nan_residuals(self) -> None:
        stats = residual_statistics(np.array([np.nan, np.nan]))
        assert stats["count"] == 0


class TestPlotResiduals:
    def test_plot_residuals_generates_files(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        y_true = rng.normal(50, 10, 50)
        y_pred = y_true + rng.normal(0, 3, 50)
        residuals = compute_residuals(y_true, y_pred)
        plots = plot_residuals(y_true, y_pred, residuals, tmp_path, prefix="test_res")
        assert len(plots) == 3
        names = [Path(p).name for p in plots]
        assert "test_res_histogram.png" in names
        assert "test_res_qq.png" in names
        assert "test_res_vs_predicted.png" in names
        for p in plots:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0


class TestRunResidualAnalysis:
    def test_run_residual_analysis_structure(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        y_true = rng.normal(50, 10, 60)
        y_pred = y_true + rng.normal(0, 3, 60)
        result = run_residual_analysis(y_true, y_pred, tmp_path, prefix="res")
        assert "residual_statistics" in result
        assert "plots" in result
        assert "n_residuals" in result
        assert result["n_residuals"] == 60
        stats = result["residual_statistics"]
        assert np.isfinite(stats["mae"])
        assert np.isfinite(stats["rmse"])
        assert len(result["plots"]) == 3
