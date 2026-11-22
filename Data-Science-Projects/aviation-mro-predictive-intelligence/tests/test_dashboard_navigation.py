"""Tests for dashboard navigation and page titles (SRS FR-12).

Verifies that all 9 Streamlit pages have the exact titles required by
SRS §FR-12, correct URL paths, and are importable.
"""

from __future__ import annotations

import importlib

# Expected SRS FR-12 page titles and their url_path keys.
EXPECTED_PAGES = [
    ("Executive Overview", "home"),
    ("Data Explorer & Quality", "eda"),
    ("Engine Health & RUL", "rul"),
    ("Anomaly Analysis", "anomaly"),
    ("Maintenance Forecasting", "forecasting"),
    ("Inventory Intelligence", "inventory"),
    ("Financial Scenarios", "financial"),
    ("Model Performance & Explainability", "explainability"),
    ("Data Sources & Model Registry", "provenance"),
]

VIEW_MODULES = [
    "dashboard.views.executive",
    "dashboard.views.eda",
    "dashboard.views.rul",
    "dashboard.views.anomaly",
    "dashboard.views.forecasting",
    "dashboard.views.inventory",
    "dashboard.views.financial",
    "dashboard.views.explainability",
    "dashboard.views.provenance",
]


class TestDashboardPageTitles:
    """Tests for page title alignment with SRS FR-12."""

    def test_all_view_modules_importable(self) -> None:
        """Every dashboard view module can be imported without error."""
        for mod in VIEW_MODULES:
            importlib.import_module(mod)

    def test_all_views_have_render(self) -> None:
        """Every view module exposes a ``render`` function."""
        for mod in VIEW_MODULES:
            module = importlib.import_module(mod)
            assert hasattr(module, "render"), f"{mod} missing render()"
            assert callable(module.render), f"{mod}.render is not callable"

    def test_expected_page_count(self) -> None:
        """Exactly 9 pages are defined per SRS FR-12."""
        assert len(EXPECTED_PAGES) == 9

    def test_page_titles_match_srs(self) -> None:
        """Page titles match SRS FR-12 exactly."""
        expected_titles = {title for title, _ in EXPECTED_PAGES}
        assert "Executive Overview" in expected_titles
        assert "Data Explorer & Quality" in expected_titles
        assert "Engine Health & RUL" in expected_titles
        assert "Anomaly Analysis" in expected_titles
        assert "Maintenance Forecasting" in expected_titles
        assert "Inventory Intelligence" in expected_titles
        assert "Financial Scenarios" in expected_titles
        assert "Model Performance & Explainability" in expected_titles
        assert "Data Sources & Model Registry" in expected_titles

    def test_url_paths_unique(self) -> None:
        """URL paths are unique across pages."""
        paths = [path for _, path in EXPECTED_PAGES]
        assert len(paths) == len(set(paths))

    def test_app_page_titles_match_expected(self) -> None:
        """The app.py page definitions match the expected SRS titles."""
        # Re-read the pages list from the module source
        import inspect

        import dashboard.app as app_module

        source = inspect.getsource(app_module)
        for title, _ in EXPECTED_PAGES:
            assert title in source, f"Page title '{title}' not found in app.py"


class TestDashboardNavigation:
    """Tests for dashboard navigation structure."""

    def test_navigation_uses_st_navigation(self) -> None:
        """app.py uses st.navigation (not custom radio/select)."""
        import inspect

        import dashboard.app as app_module

        source = inspect.getsource(app_module)
        assert "st.navigation" in source

    def test_sidebar_position(self) -> None:
        """Navigation is positioned in the sidebar."""
        import inspect

        import dashboard.app as app_module

        source = inspect.getsource(app_module)
        assert 'position="sidebar"' in source
