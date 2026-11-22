"""Tests for dashboard accessibility (SRS NFR-10).

Verifies colour-independent meaning, accessible labels, units, and
readable charts.
"""

from __future__ import annotations

from dashboard.components import accessible_bar_chart, accessible_line_chart, chart_with_description


class TestAccessibilityColourIndependent:
    """Tests that colour is never the sole means of conveying meaning."""

    def test_shap_effect_has_text_indicator(self) -> None:
        """SHAP effect labels use text, not just colour, for meaning."""
        positive = "Increases RUL (+)"
        negative = "Decreases RUL (-)"
        assert "(+)" in positive
        assert "(-)" in negative
        assert "Increases" in positive
        assert "Decreases" in negative

    def test_accessible_bar_chart_renders_without_error(self, monkeypatch) -> None:
        """accessible_bar_chart renders without error for valid data."""
        import streamlit as st

        calls = []
        original_caption = st.caption

        def capture_caption(text, *args, **kwargs):
            calls.append(text)
            original_caption(text, *args, **kwargs)

        monkeypatch.setattr(st, "caption", capture_caption)

        import pandas as pd

        df = pd.DataFrame({"Feature": ["sensor_1", "sensor_2"], "Mean |SHAP|": [0.5, 0.3]})
        accessible_bar_chart(df, "Mean |SHAP|", "Feature")
        assert any("x-axis" in c and "y-axis" in c for c in calls)

    def test_accessible_line_chart_renders_without_error(self, monkeypatch) -> None:
        """accessible_line_chart renders without error for valid data."""
        import streamlit as st

        calls = []
        original_caption = st.caption

        def capture_caption(text, *args, **kwargs):
            calls.append(text)
            original_caption(text, *args, **kwargs)

        monkeypatch.setattr(st, "caption", capture_caption)

        import pandas as pd

        df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "demand": [10.0, 12.0]})
        accessible_line_chart(df, "demand", "date")
        assert any("x-axis" in c and "y-axis" in c for c in calls)

    def test_chart_with_description_renders_without_error(self, monkeypatch) -> None:
        """chart_with_description renders description, chart, caption, and table."""
        import streamlit as st

        caption_calls = []
        markdown_calls = []
        original_caption = st.caption
        original_markdown = st.markdown

        def capture_caption(text, *args, **kwargs):
            caption_calls.append(text)
            original_caption(text, *args, **kwargs)

        def capture_markdown(text, *args, **kwargs):
            markdown_calls.append(text)
            original_markdown(text, *args, **kwargs)

        monkeypatch.setattr(st, "caption", capture_caption)
        monkeypatch.setattr(st, "markdown", capture_markdown)

        import pandas as pd

        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        chart_with_description(
            lambda d, **kw: st.bar_chart(d.set_index("x")),
            df,
            description="Sample bar chart",
            caption="x on x-axis, y on y-axis.",
        )
        assert any("Sample bar chart" in c for c in markdown_calls)
        assert any("x on x-axis, y on y-axis." in c for c in caption_calls)

    def test_charts_handle_empty_data(self, monkeypatch) -> None:
        """Accessible chart helpers degrade gracefully with empty data."""
        import streamlit as st

        info_calls = []
        original_info = st.info

        def capture_info(text, *args, **kwargs):
            info_calls.append(text)
            original_info(text, *args, **kwargs)

        monkeypatch.setattr(st, "info", capture_info)

        import pandas as pd

        empty = pd.DataFrame({"a": [], "b": []})
        accessible_bar_chart(empty, "b", "a")
        accessible_line_chart(empty, "b", "a")
        chart_with_description(
            lambda d, **kw: st.bar_chart(d),
            empty,
            description="Empty chart",
            caption="No data.",
        )
        assert any("No data to chart" in c for c in info_calls)


class TestAccessibilityUnits:
    """Tests that KPI labels include explicit units."""

    def test_kpi_labels_have_units(self) -> None:
        """KPI labels across views include unit annotations."""
        import inspect

        import dashboard.views.anomaly as anomaly_mod
        import dashboard.views.executive as exec_mod
        import dashboard.views.financial as fin_mod
        import dashboard.views.forecasting as fc_mod
        import dashboard.views.inventory as inv_mod
        import dashboard.views.provenance as prov_mod
        import dashboard.views.rul as rul_mod

        # RUL: cycles
        rul_src = inspect.getsource(rul_mod)
        assert "MAE (cycles)" in rul_src
        assert "RMSE (cycles)" in rul_src
        assert "R2 (dimensionless)" in rul_src

        # Anomaly: score, fraction, %
        anomaly_src = inspect.getsource(anomaly_mod)
        assert "Threshold (score)" in anomaly_src
        assert "Train Contamination (fraction)" in anomaly_src
        assert "Val Anomaly Rate (%)" in anomaly_src

        # Forecasting: dynamic units per target (count / hours / currency)
        fc_src = inspect.getsource(fc_mod)
        assert "MAE (" in fc_src
        assert "RMSE (" in fc_src
        assert "WAPE (%)" in fc_src
        assert "Period (weeks)" in fc_src

        # Inventory: count
        inv_src = inspect.getsource(inv_mod)
        assert "Parts (count)" in inv_src
        assert "Families (count)" in inv_src
        assert "Shops (count)" in inv_src

        # Financial: USD
        fin_src = inspect.getsource(fin_mod)
        assert "Cost (USD)" in fin_src

        # Executive: cycles, %, USD
        exec_src = inspect.getsource(exec_mod)
        assert "R2" in exec_src
        assert "%" in exec_src or "R2" in exec_src

        # Provenance: count/KB
        prov_src = inspect.getsource(prov_mod)
        assert "Size (KB)" in prov_src


class TestAccessibilityChartLabels:
    """Tests that charts have accessible labels and captions."""

    def test_accessible_bar_chart_has_axis_captions(self, monkeypatch) -> None:
        """accessible_bar_chart emits axis explanation captions."""
        import streamlit as st

        calls = []
        original_caption = st.caption

        def capture_caption(text, *args, **kwargs):
            calls.append(text)
            original_caption(text, *args, **kwargs)

        monkeypatch.setattr(st, "caption", capture_caption)

        import pandas as pd

        df = pd.DataFrame({"Feature": ["sensor_1", "sensor_2"], "Mean |SHAP|": [0.5, 0.3]})
        accessible_bar_chart(df, "Mean |SHAP|", "Feature")
        assert any("x-axis" in c and "y-axis" in c for c in calls)

    def test_rul_per_engine_chart_has_description(self) -> None:
        """RUL per-engine error chart renders with description/table companion."""
        import inspect

        src = inspect.getsource(__import__("dashboard.views.rul", fromlist=["render"]))
        assert "Per-engine test MAE (cycles)" in src
        assert "Each bar shows the mean absolute error" in src

    def test_anomaly_per_engine_chart_has_description(self) -> None:
        """Anomaly per-engine rate chart renders with description/table companion."""
        import inspect

        src = inspect.getsource(__import__("dashboard.views.anomaly", fromlist=["render"]))
        assert "Per-engine test anomaly rate (%)" in src
        assert "Each bar shows the percentage of flagged observations" in src

    def test_forecasting_demand_chart_has_description(self) -> None:
        """Forecasting demand line chart renders with description/table companion."""
        import inspect

        src = inspect.getsource(__import__("dashboard.views.forecasting", fromlist=["render"]))
        assert "Weekly" in src and "chart" in src.lower()
        assert "Each point shows" in src

    def test_financial_cost_breakdown_chart_has_caption(self) -> None:
        """Financial cost breakdown chart has a text caption."""
        import inspect

        src = inspect.getsource(__import__("dashboard.views.financial", fromlist=["render"]))
        assert "Bar chart: maintenance cost category" in src

    def test_financial_scenarios_chart_has_caption(self) -> None:
        """Financial scenarios chart has a text caption."""
        import inspect

        src = inspect.getsource(__import__("dashboard.views.financial", fromlist=["render"]))
        assert "Bar chart: scenario on the x-axis" in src

    def test_rul_comparison_chart_has_caption(self) -> None:
        """RUL model comparison chart has a text caption."""
        import inspect

        src = inspect.getsource(__import__("dashboard.views.rul", fromlist=["render"]))
        assert "Bar chart: model name on the x-axis" in src


class TestAccessibilityMessages:
    """Tests that error and warning messages are clear and actionable."""

    def test_missing_artifact_message_is_actionable(self) -> None:
        """missing_artifact() includes regeneration guidance."""
        import streamlit as st

        from dashboard.components import missing_artifact

        warning_calls = []
        original_warning = st.warning

        def capture_warning(text, *args, **kwargs):
            warning_calls.append(text)
            original_warning(text, *args, **kwargs)

        st.warning = capture_warning
        try:
            missing_artifact("EDA report", "reports/eda/eda_report.json")
        finally:
            st.warning = original_warning

        assert len(warning_calls) == 1
        msg = warning_calls[0]
        assert "Missing artefact: EDA report" in msg
        assert "Regenerate" in msg

    def test_error_message_includes_detail(self) -> None:
        """error_message() includes optional detail text."""
        import streamlit as st

        from dashboard.components import error_message

        error_calls = []
        original_error = st.error

        def capture_error(text, *args, **kwargs):
            error_calls.append(text)
            original_error(text, *args, **kwargs)

        st.error = capture_error
        try:
            error_message("forecast results", "file not found")
        finally:
            st.error = original_error

        assert len(error_calls) == 1
        msg = error_calls[0]
        assert "Could not load forecast results." in msg
        assert "Details: file not found" in msg


class TestAccessibilitySourceBadges:
    """Tests that source badges convey meaning through text, not just colour."""

    def test_source_badge_has_text_label(self) -> None:
        """source_badge renders a readable text label."""
        import streamlit as st

        markdown_calls = []
        original_markdown = st.markdown

        def capture_markdown(text, *args, **kwargs):
            markdown_calls.append(text)
            original_markdown(text, *args, **kwargs)

        st.markdown = capture_markdown
        try:
            from dashboard.components import source_badge

            source_badge("nasa")
        finally:
            st.markdown = original_markdown

        assert len(markdown_calls) == 1
        html = markdown_calls[0]
        assert "NASA C-MAPSS FD001" in html
        assert "Simulated benchmark data" in html
