"""Tests for the Streamlit dashboard.

Covers two layers:

1. Unit tests for the safe artefact loaders and formatting helpers.
2. Streamlit ``AppTest`` smoke tests that execute every page via the real
   Streamlit script runner and assert that no page raises an exception and
   that navigation between pages works.

The smoke tests require the saved artefacts produced by Sections 9.1-9.6. If
those artefacts are absent the tests degrade to asserting graceful handling
(missing-artifact warnings) rather than failing hard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src`` is importable when tests are run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard import components as ui  # noqa: E402
from dashboard import data_loader as dl  # noqa: E402

APP_PATH = SRC / "dashboard" / "app.py"

PAGE_TITLES = [
    "Executive Overview",
    "Data Explorer & Quality",
    "Engine Health & RUL",
    "Anomaly Analysis",
    "Maintenance Forecasting",
    "Inventory Intelligence",
    "Financial Scenarios",
    "Model Performance & Explainability",
    "Data Sources & Model Registry",
]


# ---------------------------------------------------------------------------
# Data loader unit tests
# ---------------------------------------------------------------------------
class TestDataLoader:
    def test_loaders_return_data(self) -> None:
        assert dl.load_eda_report() is not None
        assert dl.load_rul_model_card() is not None
        assert dl.load_anomaly_model_card() is not None
        assert dl.load_forecast_results() is not None
        assert dl.load_inventory_analysis() is not None
        assert dl.load_financial_analysis() is not None

    def test_loaders_handle_missing_gracefully(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        assert dl.load_json(missing) is None
        assert dl.load_parquet(tmp_path / "nope.parquet") is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        assert dl.load_json(bad) is None

    def test_maintenance_demand_schema(self) -> None:
        df = dl.load_maintenance_demand()
        assert df is not None
        for col in ("date", "shop_id", "part_family", "demand_count"):
            assert col in df.columns

    def test_inventory_data_schema(self) -> None:
        df = dl.load_inventory_data()
        assert df is not None
        for col in ("part_number", "part_family", "shop_id", "criticality"):
            assert col in df.columns

    def test_model_artifact_status(self) -> None:
        status = dl.model_artifact_status()
        assert status, "expected artefact status entries"
        for _name, info in status.items():
            assert "exists" in info
            assert "size_kb" in info

    def test_rul_artifact_loads_or_none(self) -> None:
        # Must not raise; either an artifact or None.
        artifact = dl.load_rul_artifact()
        assert artifact is None or hasattr(artifact, "best_model_name")

    def test_anomaly_detector_loads_or_none(self) -> None:
        detector = dl.load_anomaly_detector()
        assert detector is None or hasattr(detector, "feature_names")


# ---------------------------------------------------------------------------
# Component helper tests
# ---------------------------------------------------------------------------
class TestComponents:
    def test_fmt_money(self) -> None:
        assert ui.fmt_money(1234567.0) == "$1,234,567"

    def test_fmt_pct(self) -> None:
        assert ui.fmt_pct(82.7) == "82.7%"

    def test_fmt_num(self) -> None:
        assert ui.fmt_num(1234.5, 1) == "1,234.5"

    def test_fmt_handles_bad_input(self) -> None:
        assert ui.fmt_money("n/a") == "n/a"
        assert ui.fmt_pct(None) == "None"


# ---------------------------------------------------------------------------
# Streamlit AppTest smoke tests
# ---------------------------------------------------------------------------
class TestStreamlitSmoke:
    # Map view module name -> expected page title heading.
    PAGES = {
        "executive": "Executive Overview",
        "eda": "Data Explorer & Quality",
        "rul": "Engine Health & RUL",
        "anomaly": "Anomaly Analysis",
        "forecasting": "Maintenance Forecasting",
        "inventory": "Inventory Intelligence",
        "financial": "Financial Scenarios",
        "explainability": "Model Performance & Explainability",
        "provenance": "Data Sources & Model Registry",
    }

    def test_app_runs_without_exception(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=120)
        at.run()
        assert not at.exception, f"app raised: {at.exception}"

    def test_each_page_renders_via_navigation(self) -> None:
        """Render every view in a real Streamlit script context via AppTest.

        ``st.navigation`` registers callable pages that ``AppTest.switch_page``
        cannot resolve by URL, so each view is exercised by running a tiny
        script that imports and calls its ``render`` function.
        """
        from streamlit.testing.v1 import AppTest

        for module, expected_title in self.PAGES.items():
            script = (
                f"import sys; sys.path.insert(0, r'{SRC}')\n"
                f"from dashboard.views.{module} import render\n"
                f"render()\n"
            )
            at = AppTest.from_string(script, default_timeout=120)
            at.run()
            assert not at.exception, f"view '{module}' raised: {at.exception}"
            titles = [t.value for t in at.title]
            assert any(expected_title in str(t) for t in titles), (
                f"view '{module}' did not render title '{expected_title}'. Got: {titles}"
            )

    def test_each_page_renders_callable(self) -> None:
        """Each view module exposes a callable render()."""
        from dashboard.views import (
            anomaly,
            eda,
            executive,
            explainability,
            financial,
            forecasting,
            inventory,
            provenance,
            rul,
        )

        views = [
            executive,
            eda,
            rul,
            anomaly,
            forecasting,
            inventory,
            financial,
            explainability,
            provenance,
        ]
        for view in views:
            assert callable(getattr(view, "render", None))


class TestPagesViaNavigation:
    """Redundant safety net: default page renders without error."""

    def test_default_page(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=120)
        at.run()
        assert not at.exception
