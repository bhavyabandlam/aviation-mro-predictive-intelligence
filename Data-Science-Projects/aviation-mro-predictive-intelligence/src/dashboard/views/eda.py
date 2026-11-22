"""Data Explorer & Quality page (NASA C-MAPSS FD001).

Visualises ``reports/eda/eda_report.json`` and the saved matplotlib plots in
``reports/eda/plots``. Shows data quality checks, sensor informativeness and
the train/test distribution-shift analysis.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _render_quality(report: dict[str, Any]) -> None:
    ui.section("Data Quality Summary")
    train = report.get("train_profile", {}) or {}
    test = report.get("test_profile", {}) or {}
    tu = train.get("unit_stats", {}) or {}
    te = test.get("unit_stats", {}) or {}

    ui.kpi_row(
        [
            {
                "label": "Train Rows",
                "value": ui.fmt_num(train.get("shape", [0])[0], 0),
                "helper": f"{tu.get('n_units')} engines",
            },
            {
                "label": "Test Rows",
                "value": ui.fmt_num(test.get("shape", [0])[0], 0),
                "helper": f"{te.get('n_units')} engines (truncated)",
            },
            {
                "label": "Informative Sensors",
                "value": len(report.get("informative_sensors", []) or []),
                "helper": "show degradation trends",
            },
            {
                "label": "Constant Sensors",
                "value": len(report.get("constant_sensors", []) or []),
                "helper": "zero variance (excluded)",
            },
        ]
    )

    st.markdown("**Confirmatory findings**")
    for item in report.get("confirmatory_findings", []) or []:
        st.markdown(f"- {item}")


def _render_sensors(report: dict[str, Any]) -> None:
    ui.section("Sensor Analysis")
    informative = report.get("informative_sensors", []) or []
    constant = report.get("constant_sensors", []) or []
    st.write(f"**Informative sensors ({len(informative)}):**")
    st.code(", ".join(informative))
    st.write(f"**Constant / excluded sensors ({len(constant)}):**")
    st.code(", ".join(constant))

    # Distribution-shift summary (KS test): count same vs shifted.
    shift = report.get("shift_results", {}) or {}
    shifted = sum(
        1 for v in shift.values() if isinstance(v, dict) and not v.get("same_distribution", True)
    )
    total = len(shift)
    if total:
        st.write(
            f"**Train/test distribution shift:** {shifted}/{total} sensors differ "
            f"(KS test, alpha=0.05)."
        )

    # Trend comparison (Cohen's d) summary.
    trend = report.get("trend_comparison", {}) or {}
    if trend:
        rows = [
            {
                "Sensor": name,
                "Train slope": round(v.get("train_slope_mean", 0.0), 5),
                "Test slope": round(v.get("test_slope_mean", 0.0), 5),
                "Cohen's d": round(v.get("cohen_d", 0.0), 3),
                "Effect": v.get("effect_size_interpretation", ""),
                "Significant": "Yes" if v.get("significant_difference") else "No",
            }
            for name, v in trend.items()
            if isinstance(v, dict)
        ]
        st.write("**Degradation-rate comparison (train vs test):**")
        ui.data_table(rows)
        if rows:
            import pandas as pd

            df = pd.DataFrame(rows)
            ui.csv_export_button(df, "sensor_trend_comparison.csv")


def _render_rul(report: dict[str, Any]) -> None:
    rul = report.get("rul_stats", {}) or {}
    if not rul:
        return
    ui.section("Remaining Useful Life (RUL) Distribution")
    ui.kpi_row(
        [
            {"label": "Mean RUL", "value": ui.fmt_num(rul.get("mean", 0.0), 1)},
            {"label": "Median RUL", "value": ui.fmt_num(rul.get("median", 0.0), 1)},
            {"label": "Std Dev", "value": ui.fmt_num(rul.get("std", 0.0), 1)},
            {
                "label": "Range",
                "value": f"{rul.get('min')} - {rul.get('max')}",
            },
            {
                "label": "95% CI Mean",
                "value": f"{rul.get('ci95_lower', 0):.1f} - {rul.get('ci95_upper', 0):.1f}",
            },
            {
                "label": "95% CI Median",
                "value": (
                    f"{rul.get('ci95_median_lower', 0):.1f} - {rul.get('ci95_median_upper', 0):.1f}"
                ),
            },
        ]
    )


def _render_correlations(report: dict[str, Any]) -> None:
    """Render FR-04 Pearson/Spearman correlation results."""
    corr = report.get("correlation_results", {}) or {}
    if not corr:
        return
    ui.section("Correlation Analysis (Pearson + Spearman)")
    st.write(
        "Sensor correlations with RUL (train set, final-cycle readings per engine). "
        "Pearson measures linear association; Spearman measures monotonic association "
        "(robust to non-normality)."
    )
    rows = []
    for name, res in corr.items():
        p = res.get("pearson", {})
        s = res.get("spearman", {})
        rows.append(
            {
                "Sensor": name,
                "Pearson r": round(p.get("statistic", float("nan")), 4)
                if p.get("statistic") is not None
                else "N/A",
                "Pearson p": round(p.get("p_value", float("nan")), 4)
                if p.get("p_value") is not None
                else "N/A",
                "Spearman ρ": round(s.get("statistic", float("nan")), 4)
                if s.get("statistic") is not None
                else "N/A",
                "Spearman p": round(s.get("p_value", float("nan")), 4)
                if s.get("p_value") is not None
                else "N/A",
                "n": p.get("n", "N/A"),
            }
        )
    if rows:
        ui.data_table(rows)
        import pandas as pd

        df = pd.DataFrame(rows)
        ui.csv_export_button(df, "sensor_rul_correlation.csv")


def _render_chi_square(report: dict[str, Any]) -> None:
    """Render FR-04 Chi-square test result."""
    cs = report.get("chi_square_result", {}) or {}
    if not cs or "error" in cs:
        return
    ui.section("Chi-Square Test of Independence")
    st.write(
        f"Tests association between {cs.get('sensor', 'sensor')} category (high/low at median) "
        "and RUL category (high/low at median)."
    )
    ui.kpi_row(
        [
            {"label": "Chi-square", "value": ui.fmt_num(cs.get("chi2_statistic", 0.0), 4)},
            {"label": "p-value", "value": ui.fmt_num(cs.get("chi2_p_value", 0.0), 4)},
            {"label": "Significant", "value": "Yes" if cs.get("significant") else "No"},
            {"label": "Assumption met", "value": "Yes" if cs.get("assumption_passed") else "No"},
            {"label": "Cramér's V", "value": ui.fmt_num(cs.get("effect_size", 0.0), 4)},
        ]
    )
    st.markdown(f"**Interpretation:** {cs.get('interpretation', '')}")
    st.markdown(f"**Assumption detail:** {cs.get('assumption_detail', '')}")


def _render_mann_whitney(report: dict[str, Any]) -> None:
    """Render FR-04 Mann-Whitney U test result."""
    mw = report.get("mann_whitney_result", {}) or {}
    if not mw or "error" in mw:
        return
    ui.section("Mann--Whitney U Test")
    st.write(
        f"Compares final-cycle {mw.get('sensor', 'sensor')} values between engines "
        "with high vs low RUL (median split). Non-parametric."
    )
    ui.kpi_row(
        [
            {"label": "U statistic", "value": ui.fmt_num(mw.get("statistic", 0.0), 2)},
            {"label": "p-value", "value": ui.fmt_num(mw.get("p_value", 0.0), 4)},
            {"label": "Significant", "value": "Yes" if mw.get("significant") else "No"},
            {"label": "High RUL median", "value": ui.fmt_num(mw.get("high_rul_median", 0.0), 2)},
            {"label": "Low RUL median", "value": ui.fmt_num(mw.get("low_rul_median", 0.0), 2)},
        ]
    )
    st.markdown(f"**Interpretation:** {mw.get('interpretation', '')}")


def _render_anova(report: dict[str, Any]) -> None:
    """Render FR-04 One-Way ANOVA result."""
    av = report.get("anova_result", {}) or {}
    if not av or "error" in av:
        return
    ui.section("One-Way ANOVA")
    st.write(
        f"Compares final-cycle {av.get('sensor', 'sensor')} values across engine "
        "groups (cycle-length terciles). Gated on normality and homogeneity."
    )
    ui.kpi_row(
        [
            {"label": "F statistic", "value": ui.fmt_num(av.get("statistic", 0.0), 4)},
            {"label": "p-value", "value": ui.fmt_num(av.get("p_value", 0.0), 4)},
            {"label": "Significant", "value": "Yes" if av.get("significant") else "No"},
            {"label": "Assumptions met", "value": "Yes" if av.get("assumption_passed") else "No"},
            {"label": "Eta-squared", "value": ui.fmt_num(av.get("effect_size", 0.0), 4)},
        ]
    )
    st.markdown(f"**Interpretation:** {av.get('interpretation', '')}")
    st.markdown(f"**Assumption detail:** {av.get('assumption_detail', '')}")


def _render_plots(report: dict[str, Any]) -> None:
    ui.section("Saved EDA Plots")
    plots = report.get("plots", {}) or {}
    # Prefer the well-known static files.
    candidates = [
        ("rul", plots.get("rul")),
        ("cycles_per_unit", plots.get("cycles_per_unit")),
        ("train_correlation", plots.get("train_correlation")),
        ("test_correlation", plots.get("test_correlation")),
    ]
    from pathlib import Path

    shown = 0
    for name, path in candidates:
        if not path:
            continue
        p = Path(path)
        if p.exists():
            st.image(str(p), caption=name.replace("_", " ").title(), width="stretch")
            shown += 1
    if shown == 0:
        st.info("No saved EDA plots found in reports/eda/plots.")


def render() -> None:
    """Render the Data Quality & EDA page."""
    st.set_page_config(page_title="Data Explorer & Quality", page_icon="📊", layout="wide")
    ui.page_header(
        "Data Explorer & Quality",
        "Exploratory analysis of the NASA C-MAPSS FD001 benchmark",
        "nasa",
    )
    ui.freshness_badge(dl.REPORTS_DIR / "eda" / "eda_report.json", "EDA report freshness")
    st.divider()

    report = dl.load_eda_report()
    if report is None:
        ui.missing_artifact("EDA report", "reports/eda/eda_report.json")
        return

    _render_quality(report)
    _render_rul(report)
    _render_sensors(report)
    _render_correlations(report)
    _render_chi_square(report)
    _render_mann_whitney(report)
    _render_anova(report)
    _render_plots(report)

    ui.section("Exploratory Findings")
    for item in report.get("exploratory_findings", []) or []:
        st.markdown(f"- {item}")
