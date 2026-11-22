"""Reusable Streamlit UI components for the MRO dashboard.

All components are presentation-only helpers. They never compute analytics
and never invent values: callers pass in numbers read from saved artefacts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from aviation_mro.reporting import (
    EXPORTS_DIR,
    export_figure_png,
    export_report,
)

# Colour palette (kept simple; Streamlit themes handle most styling).
NASA_COLOUR = "#0B3D91"
SYNTH_COLOUR = "#1B7F3B"
WARN_COLOUR = "#B23B00"

__all__ = [
    "accessible_bar_chart",
    "accessible_line_chart",
    "chart_with_description",
    "source_badge",
    "freshness_badge",
    "csv_export_button",
    "png_export_button",
    "export_section",
    "page_header",
    "section",
    "kpi",
    "kpi_row",
    "missing_artifact",
    "error_message",
    "fmt_money",
    "fmt_pct",
    "fmt_num",
    "limitations_box",
    "data_table",
]


def source_badge(kind: str) -> None:
    """Render a coloured provenance badge.

    ``kind`` must be one of ``nasa``, ``synthetic_mro``, ``synthetic_fin``.
    """
    meta = {
        "nasa": ("NASA C-MAPSS FD001", "Simulated benchmark data", NASA_COLOUR),
        "synthetic_mro": ("Synthetic MRO", "Synthetic business data (seed=42)", SYNTH_COLOUR),
        "synthetic_fin": ("Synthetic Financial", "Synthetic cost data (seed=42)", SYNTH_COLOUR),
        "mixed": (
            "Mixed sources",
            "NASA FD001 simulation + Synthetic MRO data",
            "#5B2C83",
        ),
    }.get(kind)
    if meta is None:
        st.caption(f"Unknown source kind: {kind}")
        return
    label, detail, colour = meta
    st.markdown(
        f"<span style='background:{colour};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8rem;font-weight:600;'>{label}</span>"
        f"&nbsp;<span style='color:#666;font-size:0.8rem;'>{detail}</span>",
        unsafe_allow_html=True,
    )


def freshness_badge(path: Path | str | None, label: str = "Data freshness") -> None:
    """Render a data-freshness badge from a file's modification time.

    Shows the ISO-8601 UTC modification time of *path*. If the file is
    missing or unreadable, shows "unknown". Never cached — always reads
    the live mtime so stale data is visible.
    """
    if path is None:
        st.caption(f"{label}: unknown")
        return
    p = Path(path)
    try:
        ts = p.stat().st_mtime
        iso = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"{label}: {iso}")
    except OSError:
        st.caption(f"{label}: unknown (file not readable)")


def csv_export_button(df: pd.DataFrame, filename: str, label: str = "Export CSV") -> None:
    """Render a CSV download button for a dataframe.

    Only the first 200 rows are exported to keep payloads small. The button
    is disabled when the dataframe is empty.
    """
    if df is None or df.empty:
        st.button(label, disabled=True, help="No data to export")
        return
    export_df = df.head(200)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        help=f"Download up to 200 rows as {filename}",
    )


def png_export_button(fig: Any, filename: str, label: str = "Export PNG") -> None:
    """Render a PNG download button for a matplotlib figure.

    The figure is written to ``data/exports/`` and offered as a download.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)

    path = EXPORTS_DIR / filename
    try:
        export_figure_png(fig, path)
        data = path.read_bytes()
        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime="image/png",
            help=f"Download chart as {filename}",
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.button(label, disabled=True, help=f"PNG export unavailable: {exc}")


def export_section(
    frames: dict[str, pd.DataFrame],
    report_name: str,
    source_lineage: str | None = None,
    synthetic: bool = False,
    label: str = "Export this page (CSV + HTML)",
) -> None:
    """Render an export section bundling CSV, HTML and metadata sidecar.

    Writes artefacts to ``data/exports/`` (FR-13) with a UTC timestamp,
    source lineage and synthetic-data warning, then offers each file as a
    Streamlit download.
    """
    if not frames:
        st.info("No data available to export.")
        return

    key = f"export_written_{report_name}"
    if st.button(label, key=f"export_btn_{report_name}"):
        result = export_report(
            report_name,
            frames,
            source_lineage=source_lineage,
            synthetic=synthetic,
        )
        st.session_state[key] = result

    written = cast(dict[str, Path] | None, st.session_state.get(key))
    if written:
        st.success(f"Exported ``{report_name}`` to ``data/exports/``.")
        for export_key, path in written.items():
            if export_key == "metadata":
                continue
            p = Path(path)
            if not p.exists():
                continue
            data = p.read_bytes()
            mime = "text/csv" if p.suffix == ".csv" else "text/html"
            st.download_button(
                label=f"Download {export_key.upper()}: {p.name}",
                data=data,
                file_name=p.name,
                mime=mime,
                key=f"dl_{report_name}_{export_key}",
            )
        meta_path = written.get("metadata")
        if meta_path and Path(meta_path).exists():
            import json

            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            st.caption(f"Generated: {meta.get('generated_utc')}")
            if meta.get("source_lineage"):
                st.caption(f"Source lineage: {meta['source_lineage']}")
            if meta.get("synthetic"):
                st.caption(
                    "⚠️ This report contains synthetic MRO business data "
                    "(seed=42). Not real airline records."
                )


def accessible_bar_chart(
    df: pd.DataFrame, value_col: str, label_col: str, height: int = 360
) -> None:
    """Render a bar chart with explicit text labels alongside colour.

    Ensures colour is never the sole means of conveying meaning (NFR-10).
    Also renders a data table below the chart so values are readable without
    relying on colour.
    """
    if df is None or df.empty:
        st.info("No data to chart.")
        return
    chart_df = df[[label_col, value_col]].copy()
    st.bar_chart(chart_df.set_index(label_col), height=height)
    st.caption(f"Bar chart: {label_col} on the x-axis, {value_col} on the y-axis.")
    st.dataframe(df.head(50), width="stretch")
    csv_export_button(df, f"{value_col.lower().replace(' ', '_')}_export.csv")


def accessible_line_chart(
    df: pd.DataFrame, value_col: str, label_col: str, height: int = 360
) -> None:
    """Render a line chart with a data-table companion for accessibility.

    Streamlit's ``st.line_chart`` renders an SVG/canvas element without alt
    text, so a data table is rendered alongside so the same values are
    perceivable without relying on the chart alone (WCAG 1.1.1).
    """
    if df is None or df.empty:
        st.info("No data to chart.")
        return
    chart_df = df[[label_col, value_col]].copy()
    st.line_chart(chart_df.set_index(label_col), height=height)
    st.caption(f"Line chart: {label_col} on the x-axis, {value_col} on the y-axis.")
    st.dataframe(df.head(50), width="stretch")
    csv_export_button(df, f"{value_col.lower().replace(' ', '_')}_export.csv")


def chart_with_description(
    render_fn: Any,
    df: pd.DataFrame,
    description: str,
    caption: str,
    height: int = 360,
) -> None:
    """Render any chart callable with a text description and data table.

    Wraps a Streamlit chart call (e.g. ``st.bar_chart``, ``st.line_chart``)
    so that a human-readable description precedes the chart and a data table
    follows it, ensuring the chart's information is available in text form
    (WCAG 1.1.1 non-text content, WCAG 1.4.1 use of colour).
    """
    if df is None or df.empty:
        st.info("No data to chart.")
        return
    st.markdown(f"**{description}**")
    try:  # noqa: SIM105 - defensive: chart rendering must never crash the page
        render_fn(df, height=height)
    except Exception:
        pass
    st.caption(caption)
    st.dataframe(df.head(50), width="stretch")


def page_header(title: str, subtitle: str, source_kind: str) -> None:
    """Standard page header with a provenance badge."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    source_badge(source_kind)
    st.divider()


def section(title: str) -> None:
    """Render a section heading."""
    st.subheader(title)


def kpi(label: str, value: Any, helper: str | None = None) -> None:
    """Render a single KPI metric card."""
    st.metric(label=label, value=value, help=helper)


def kpi_row(items: list[dict[str, Any]]) -> None:
    """Render a row of up to four KPI cards from a list of dicts."""
    items = items[:4]
    cols = st.columns(len(items)) if items else []
    for col, item in zip(cols, items, strict=False):
        with col:
            kpi(item["label"], item["value"], item.get("helper"))


def missing_artifact(what: str, path_hint: str = "") -> None:
    """Render a clear, non-fatal missing-artifact warning."""
    msg = f"Missing artefact: {what}"
    if path_hint:
        msg += f" (`{path_hint}`)"
    st.warning(msg + ". Regenerate the relevant pipeline to populate this page.")


def error_message(what: str, detail: str = "") -> None:
    """Render a non-fatal error message."""
    st.error(f"Could not load {what}." + (f" Details: {detail}" if detail else ""))


def fmt_money(value: float) -> str:
    """Format a number as whole-dollar currency."""
    try:
        return f"${value:,.0f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: float, digits: int = 1) -> str:
    """Format a fraction/percentage value with a % suffix."""
    try:
        return f"{value:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_num(value: float, digits: int = 2) -> str:
    """Format a number with thousands separators."""
    try:
        return f"{value:,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def limitations_box(items: list[str], title: str = "Assumptions & limitations") -> None:
    """Render an expander listing model/data limitations verbatim."""
    if not items:
        return
    with st.expander(title, expanded=False):
        for item in items:
            st.markdown(f"- {item}")


def data_table(records: list[dict[str, Any]], max_rows: int = 200) -> None:
    """Render a dataframe from a list of dicts (empty-safe)."""
    if not records:
        st.info("No records to display.")
        return
    import pandas as pd

    df = pd.DataFrame(records)
    st.dataframe(df.head(max_rows), width="stretch")
