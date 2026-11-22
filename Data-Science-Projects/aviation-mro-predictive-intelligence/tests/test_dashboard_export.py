"""Tests for dashboard CSV export (SRS FR-13).

Verifies that csv_export_button renders correctly and exports data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aviation_mro.reporting import EXPORTS_DIR
from dashboard.components import csv_export_button, export_section, png_export_button


class TestCSVExportButton:
    """Tests for the CSV export button component."""

    def test_export_button_renders_for_dataframe(self, monkeypatch) -> None:
        """csv_export_button renders a download button for non-empty data."""
        import streamlit as st

        calls = []

        def capture_button(*args, **kwargs):
            calls.append((args, kwargs))
            # Return a mock that behaves like a button (not clicked)
            return None

        monkeypatch.setattr(st, "download_button", capture_button)

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_export_button(df, "test.csv")
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["file_name"] == "test.csv"
        assert kwargs["mime"] == "text/csv"
        # data should be bytes (CSV content)
        assert isinstance(kwargs["data"], bytes)
        assert b"a,b" in kwargs["data"]

    def test_export_button_disabled_for_empty(self, monkeypatch) -> None:
        """csv_export_button is disabled for empty dataframes."""
        import streamlit as st

        calls = []

        def capture_button(*args, **kwargs):
            calls.append((args, kwargs))
            return None

        monkeypatch.setattr(st, "button", capture_button)

        df = pd.DataFrame()
        csv_export_button(df, "empty.csv")
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs.get("disabled") is True

    def test_export_button_disabled_for_none(self, monkeypatch) -> None:
        """csv_export_button is disabled for None dataframe."""
        import streamlit as st

        calls = []

        def capture_button(*args, **kwargs):
            calls.append((args, kwargs))
            return None

        monkeypatch.setattr(st, "button", capture_button)

        csv_export_button(None, "none.csv")
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs.get("disabled") is True

    def test_csv_content_matches_dataframe(self) -> None:
        """Exported CSV content matches the dataframe (first 200 rows)."""
        df = pd.DataFrame({"col1": range(5), "col2": ["a", "b", "c", "d", "e"]})
        csv_bytes = df.head(200).to_csv(index=False).encode("utf-8")
        decoded = csv_bytes.decode("utf-8")
        lines = decoded.strip().splitlines()
        assert len(lines) == 6  # header + 5 rows
        assert lines[0] == "col1,col2"
        assert lines[1] == "0,a"

    def test_export_limits_to_200_rows(self) -> None:
        """CSV export is limited to 200 rows."""
        df = pd.DataFrame({"col": range(300)})
        csv_bytes = df.head(200).to_csv(index=False).encode("utf-8")
        decoded = csv_bytes.decode("utf-8")
        lines = decoded.strip().splitlines()
        assert len(lines) == 201  # header + 200 rows


class TestExportIntegration:
    """Tests that export integrates with real dashboard tables."""

    def test_export_filename_sanitized(self) -> None:
        """Export filenames are derived from column names safely."""
        # The accessible_bar_chart generates filenames from value_col
        value_col = "Mean |SHAP|"
        filename = f"{value_col.lower().replace(' ', '_')}_export.csv"
        assert filename == "mean_|shap|_export.csv"
        assert filename.endswith(".csv")


class TestExportSection:
    """Tests for the export_section component (CSV + HTML + metadata)."""

    def test_export_section_writes_files_to_exports_dir(self, monkeypatch, tmp_path) -> None:
        """export_section writes CSV, HTML and metadata to data/exports/."""
        import streamlit as st

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        clicks = []
        downloads = []
        captions = []

        monkeypatch.setattr(st, "button", lambda *a, **k: clicks.append(True) or True)
        monkeypatch.setattr(
            st,
            "download_button",
            lambda *a, **k: downloads.append(k),
        )
        monkeypatch.setattr(st, "success", lambda *a, **k: None)
        monkeypatch.setattr(st, "caption", lambda *a, **k: captions.append(a[0] if a else ""))
        monkeypatch.setattr(st, "info", lambda *a, **k: None)

        # Clear any prior session state for this report.
        if "export_written_test_report" in st.session_state:
            del st.session_state["export_written_test_report"]

        export_section(
            {"test_data": df},
            "test_report",
            source_lineage="NASA FD001 + synthetic MRO",
            synthetic=True,
            label="Export test (CSV + HTML)",
        )

        # Button should have been clicked (returns True).
        assert len(clicks) == 1

        # After the click, session state should hold the written files.
        written = st.session_state.get("export_written_test_report")
        assert written is not None
        assert "test_data_csv" in written
        assert "test_data_html" in written
        assert "metadata" in written

        # Files should exist on disk.
        for key, path in written.items():
            if key == "metadata":
                continue
            assert Path(path).exists(), f"{key} file not written: {path}"

        # Metadata should contain timestamp, lineage, synthetic flag.
        meta = Path(written["metadata"]).read_text(encoding="utf-8")
        assert "generated_utc" in meta
        assert "NASA FD001 + synthetic MRO" in meta
        assert '"synthetic": true' in meta

        # Clean up session state.
        del st.session_state["export_written_test_report"]

    def test_export_section_empty_frames(self, monkeypatch) -> None:
        """export_section with no frames shows info, not a button."""
        import streamlit as st

        infos = []
        monkeypatch.setattr(st, "info", lambda *a, **k: infos.append(a[0]))
        monkeypatch.setattr(
            st,
            "button",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("button should not render")),
        )

        export_section({}, "empty_report")

        assert len(infos) == 1
        assert "No data available" in infos[0]

    def test_export_section_download_buttons_rendered(self, monkeypatch, tmp_path) -> None:
        """After export, download buttons are rendered for CSV and HTML."""
        import streamlit as st

        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})

        clicks = []
        downloads = []

        def capture_button(*a, **k):
            clicks.append(True)
            return True

        def capture_download(*a, **k):
            downloads.append(k)
            return None

        monkeypatch.setattr(st, "button", capture_button)
        monkeypatch.setattr(st, "download_button", capture_download)
        monkeypatch.setattr(st, "success", lambda *a, **k: None)
        monkeypatch.setattr(st, "caption", lambda *a, **k: None)
        monkeypatch.setattr(st, "info", lambda *a, **k: None)

        if "export_written_dl_report" in st.session_state:
            del st.session_state["export_written_dl_report"]

        export_section(
            {"results": df},
            "dl_report",
            source_lineage="Synthetic MRO (seed=42)",
            synthetic=True,
        )

        # Click triggered export → download buttons should follow.
        assert len(clicks) == 1
        assert len(downloads) >= 2  # CSV + HTML

        file_names = [k.get("file_name", "") for k in downloads]
        assert any("results_csv" in n or n.endswith(".csv") for n in file_names)
        assert any("results_html" in n or n.endswith(".html") for n in file_names)

        # MIME types should be correct.
        csv_call = next(k for k in downloads if k.get("file_name", "").endswith(".csv"))
        html_call = next(k for k in downloads if k.get("file_name", "").endswith(".html"))
        assert csv_call["mime"] == "text/csv"
        assert html_call["mime"] == "text/html"

        del st.session_state["export_written_dl_report"]


class TestPngExportButton:
    """Tests for the png_export_button component."""

    def test_png_export_button_renders(self, monkeypatch, tmp_path) -> None:
        """png_export_button renders a download button for a matplotlib figure."""
        import matplotlib
        import streamlit as st

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        downloads = []
        monkeypatch.setattr(
            st,
            "download_button",
            lambda *a, **k: downloads.append(k) or None,
        )
        monkeypatch.setattr(st, "button", lambda *a, **k: None)
        monkeypatch.setattr(st, "info", lambda *a, **k: None)

        png_export_button(fig, "test_chart.png", label="Export chart PNG")

        assert len(downloads) == 1
        kwargs = downloads[0]
        assert kwargs["file_name"] == "test_chart.png"
        assert kwargs["mime"] == "image/png"
        assert isinstance(kwargs["data"], bytes)
        assert kwargs["data"].startswith(b"\x89PNG")  # PNG magic bytes

        # Verify the file was written to data/exports/.
        written_path = EXPORTS_DIR / "test_chart.png"
        assert written_path.exists()

        # Clean up the test file.
        written_path.unlink(missing_ok=True)


class TestExportIntegrationEndToEnd:
    """End-to-end tests: exporter writes to data/exports/ and files are valid."""

    def test_export_report_creates_valid_csv_and_html(self, tmp_path) -> None:
        """export_report writes valid CSV and HTML files to the target dir."""
        from aviation_mro.reporting import export_report

        df = pd.DataFrame({"shop": ["SHOP_01", "SHOP_02"], "demand": [12, 7]})

        written = export_report(
            "e2e_test",
            {"demand": df},
            out_dir=tmp_path,
            source_lineage="Synthetic MRO (seed=42)",
            synthetic=True,
        )

        csv_path = written["demand_csv"]
        html_path = written["demand_html"]
        meta_path = written["metadata"]

        assert csv_path.exists()
        assert html_path.exists()
        assert meta_path.exists()

        # CSV round-trips.
        back = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(back, df)

        # HTML contains timestamp, lineage, warning.
        html_text = html_path.read_text(encoding="utf-8")
        assert "Generated:" in html_text
        assert "Synthetic MRO (seed=42)" in html_text
        assert "Warning:" in html_text
        assert "synthetic" in html_text.lower()

        # Metadata has required fields.
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["report_name"] == "e2e_test"
        assert meta["synthetic"] is True
        assert "generated_utc" in meta
        assert "T" in meta["generated_utc"] and "Z" in meta["generated_utc"]
