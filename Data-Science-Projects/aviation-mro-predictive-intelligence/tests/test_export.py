"""Tests for the reporting export helpers (SRS v1.1 FR-13).

Covers CSV, HTML and PNG export plus the bundled report exporter, including
the required timestamp, source-lineage and synthetic-data warning metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from aviation_mro.reporting import (  # noqa: E402
    EXPORTS_DIR,
    SYNTHETIC_WARNING,
    export_dataframe_csv,
    export_dataframe_html,
    export_figure_png,
    export_report,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shop_id": ["SHOP_01", "SHOP_02"],
            "part_family": ["ENGINE", "AVIONICS"],
            "demand_count": [12, 7],
        }
    )


class TestCsvExport:
    def test_csv_round_trip(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        written = export_dataframe_csv(sample_df, out)
        assert written == out
        assert out.exists()
        back = pd.read_csv(out)
        pd.testing.assert_frame_equal(back, sample_df)


class TestHtmlExport:
    def test_html_contains_metadata(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        export_dataframe_html(
            sample_df,
            out,
            title="My Report",
            source_lineage="NASA FD001 + synthetic MRO",
            synthetic=True,
        )
        text = out.read_text(encoding="utf-8")
        assert "My Report" in text
        assert "Generated:" in text
        assert "T" in text and "Z" in text  # ISO-8601 UTC timestamp
        assert "Source lineage:" in text
        assert "NASA FD001 + synthetic MRO" in text
        assert "Warning:" in text
        assert SYNTHETIC_WARNING in text
        assert "ENGINE" in text and "AVIONICS" in text

    def test_html_no_warning_when_not_synthetic(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        export_dataframe_html(sample_df, out, title="No Warn", synthetic=False)
        text = out.read_text(encoding="utf-8")
        assert SYNTHETIC_WARNING not in text


class TestPngExport:
    def test_png_from_figure(self, tmp_path: Path) -> None:
        fig = Figure()
        ax = fig.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        out = tmp_path / "chart.png"
        written = export_figure_png(fig, out)
        assert written == out
        assert out.exists()
        assert out.stat().st_size > 0


class TestBundledReport:
    def test_export_report_writes_csv_html_metadata(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        frames = {"demand": sample_df}
        written = export_report(
            "mro_demand",
            frames,
            out_dir=tmp_path,
            source_lineage="synthetic MRO (seed=42)",
            synthetic=True,
        )
        assert "demand_csv" in written
        assert "demand_html" in written
        assert "metadata" in written
        for path in written.values():
            assert path.exists()
        meta = json_load(written["metadata"])
        assert meta["report_name"] == "mro_demand"
        assert meta["synthetic"] is True
        assert meta["source_lineage"] == "synthetic MRO (seed=42)"
        assert "generated_utc" in meta
        assert "T" in meta["generated_utc"] and "Z" in meta["generated_utc"]

    def test_default_exports_dir_exists(self) -> None:
        assert EXPORTS_DIR.parent.name == "data"
        assert EXPORTS_DIR.name == "exports"


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
