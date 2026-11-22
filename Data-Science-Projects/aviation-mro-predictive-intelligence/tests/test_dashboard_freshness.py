"""Tests for dashboard data freshness (SRS FR-12).

Verifies that artefact freshness badges show correct modification times
and handle missing files gracefully.
"""

from __future__ import annotations

from pathlib import Path

from dashboard.components import freshness_badge
from dashboard.data_loader import artefact_freshness


class TestArtefactFreshness:
    """Tests for artefact freshness helper."""

    def test_freshness_returns_iso_timestamp(self, tmp_path) -> None:
        """artefact_freshness returns an ISO-8601 UTC timestamp."""
        f = tmp_path / "test.json"
        f.write_text("{}")
        result = artefact_freshness(f)
        assert result is not None
        assert "T" in result or " " in result
        assert "UTC" in result

    def test_freshness_none_for_missing_file(self) -> None:
        """artefact_freshness returns None for missing files."""
        result = artefact_freshness(Path("/nonexistent/path.json"))
        assert result is None

    def test_freshness_none_for_none(self) -> None:
        """artefact_freshness returns None for None input."""
        result = artefact_freshness(None)
        assert result is None

    def test_freshness_badge_renders_without_error(self, tmp_path, monkeypatch) -> None:
        """freshness_badge renders without error for valid and invalid paths."""

        f = tmp_path / "test.json"
        f.write_text("{}")

        # These should not raise
        freshness_badge(f, "Test freshness")
        freshness_badge(None, "Test freshness")
        freshness_badge(Path("/nonexistent"), "Test freshness")

    def test_freshness_badge_includes_timestamp(self, tmp_path, monkeypatch) -> None:
        """freshness_badge includes the timestamp in its output."""
        import streamlit as st

        f = tmp_path / "test.json"
        f.write_text("{}")

        # Capture st.caption calls
        calls = []
        original_caption = st.caption

        def capture_caption(text, *args, **kwargs):
            calls.append(text)
            original_caption(text, *args, **kwargs)

        monkeypatch.setattr(st, "caption", capture_caption)
        freshness_badge(f, "Test freshness")
        assert any("UTC" in c for c in calls)


class TestFreshnessIntegration:
    """Tests that freshness integrates with real report files."""

    def test_real_report_freshness(self) -> None:
        """Freshness works for real report files that exist."""
        from dashboard.data_loader import REPORTS_DIR

        rul_card = REPORTS_DIR / "model_card_baseline_v2.json"
        if rul_card.exists():
            result = artefact_freshness(rul_card)
            assert result is not None
            assert len(result) > 0
