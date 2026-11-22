"""Maintenance Forecasting page (synthetic MRO demand).

Renders the seasonal-naive baseline results for three forecast targets:
demand_count (work-order counts), labour_hours, and maintenance_expenditure.
Lets the user inspect the synthetic weekly data for a chosen shop/family series.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl

TARGET_LABELS = {
    "demand_count": "Work-Order Counts",
    "labour_hours": "Labour-Hour Demand",
    "maintenance_expenditure": "Maintenance Expenditure",
}

TARGET_UNITS = {
    "demand_count": "work orders / week",
    "labour_hours": "labour hours / week",
    "maintenance_expenditure": "currency / week",
}


def _render_metrics(res: dict[str, Any], target_column: str) -> None:
    label = TARGET_LABELS.get(target_column, target_column)
    ui.section(f"Test Set Metrics (Seasonal-Naive) — {label}")
    test = res.get("test_aggregate", {}) or {}
    naive = test.get("naive", {}) or {}
    seasonal = test.get("seasonal_naive", {}) or {}
    unit = TARGET_UNITS.get(target_column, "")
    ui.kpi_row(
        [
            {
                "label": f"Seasonal MAE ({unit})",
                "value": ui.fmt_num(seasonal.get("mae", 0.0), 2),
                "helper": f"Naive MAE: {ui.fmt_num(naive.get('mae', 0.0), 2)}",
            },
            {
                "label": f"Seasonal RMSE ({unit})",
                "value": ui.fmt_num(seasonal.get("rmse", 0.0), 2),
                "helper": f"Naive RMSE: {ui.fmt_num(naive.get('rmse', 0.0), 2)}",
            },
            {
                "label": "Seasonal WAPE (%)",
                "value": ui.fmt_num(seasonal.get("wape", 0.0), 3),
                "helper": f"Naive WAPE: {ui.fmt_num(naive.get('wape', 0.0), 3)}",
            },
            {
                "label": "Seasonal Period (weeks)",
                "value": res.get("selected_seasonal_period", "n/a"),
                "helper": "weeks (52 or 13)",
            },
        ]
    )
    val = res.get("validation_aggregate", {}) or {}
    if val:
        st.write("**Validation aggregate:**")
        ui.data_table(
            [
                {
                    "Model": name,
                    "MAE": round(m.get("mae", 0.0), 3),
                    "RMSE": round(m.get("rmse", 0.0), 3),
                    "WAPE": round(m.get("wape", 0.0), 3),
                }
                for name, m in val.items()
                if isinstance(m, dict)
            ]
        )


def _render_seasonal_candidates(res: dict[str, Any]) -> None:
    cands = res.get("seasonal_candidates", {}) or {}
    if not cands:
        return
    ui.section("Seasonal Period Selection")
    rows = [
        {
            "Period": period,
            "MAE": round(c.get("mae", 0.0), 3),
            "RMSE": round(c.get("rmse", 0.0), 3),
            "WAPE": round(c.get("wape", 0.0), 3),
            "Better than Naive": "Yes" if c.get("better_than_naive") else "No",
            "Selected": "Yes" if str(period) == str(res.get("selected_seasonal_period")) else "",
        }
        for period, c in cands.items()
        if isinstance(c, dict)
    ]
    ui.data_table(rows)


def _render_per_series(res: dict[str, Any], target_column: str) -> None:
    per = res.get("per_series_test_results", {}) or {}
    if not per:
        return
    ui.section(f"Per-Series Test Results — {TARGET_LABELS.get(target_column, target_column)}")
    rows = []
    for series, models in per.items():
        seasonal = models.get("seasonal_naive", {}) or {}
        naive = models.get("naive", {}) or {}
        rows.append(
            {
                "Series": series,
                "Seasonal MAE": round(seasonal.get("mae", 0.0), 2),
                "Seasonal WAPE": round(seasonal.get("wape", 0.0), 3),
                "Naive MAE": round(naive.get("mae", 0.0), 2),
                "Coverage": round(seasonal.get("empirical_coverage", 0.0), 3),
            }
        )
    ui.data_table(rows)
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        safe_name = target_column.replace(" ", "_")
        ui.csv_export_button(df, f"per_series_forecast_{safe_name}.csv")
        ui.export_section(
            {f"per_series_forecast_{safe_name}": df},
            f"forecasting_per_series_{safe_name}",
            source_lineage="Synthetic MRO weekly demand (seed=42)",
            synthetic=True,
            label=f"Export per-series forecast ({target_column})",
        )


def _render_demand_chart(target_column: str) -> None:
    ui.section(f"Weekly {TARGET_LABELS.get(target_column, target_column)} Explorer (Synthetic)")
    df = dl.load_maintenance_demand()
    if df is None or df.empty:
        st.info("maintenance_demand.parquet not available.")
        return
    if target_column not in df.columns:
        st.warning(f"Column '{target_column}' not found in demand dataset.")
        return
    shops = sorted(df["shop_id"].unique())
    families = sorted(df["part_family"].unique())
    col_shop, col_fam = st.columns(2)
    with col_shop:
        shop = st.selectbox("Shop", shops, key=f"fc_shop_{target_column}")
    with col_fam:
        family = st.selectbox("Part family", families, key=f"fc_fam_{target_column}")
    sub = df[(df["shop_id"] == shop) & (df["part_family"] == family)].copy()
    sub = sub.sort_values("date")
    unit = TARGET_UNITS.get(target_column, "")
    st.write(
        f"{shop} / {family}: {len(sub)} weeks, "
        f"mean {target_column} {sub[target_column].mean():.2f} {unit}, "
        f"max {sub[target_column].max():.2f}."
    )
    # Line chart with text/table companion so chart info is perceivable without colour.
    ui.chart_with_description(
        lambda d, **kw: st.line_chart(d.set_index("date")[target_column], height=360),
        sub,
        description=f"Weekly {TARGET_LABELS.get(target_column, target_column)} — {shop} / {family}",
        caption=f"Each point shows {target_column} for one week. Unit: {unit}.",
    )

    # PNG export for the demand chart.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(sub["date"], sub[target_column], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_column.replace("_", " ").title())
        ax.set_title(
            f"Weekly {TARGET_LABELS.get(target_column, target_column)} — {shop} / {family}"
        )
        fig.autofmt_xdate()
        ui.png_export_button(
            fig,
            f"demand_chart_{target_column}_{shop}_{family}.png",
            label="Export demand chart PNG",
        )
        plt.close(fig)
    except Exception:
        pass


def render() -> None:
    """Render the Maintenance Forecasting page."""
    st.set_page_config(page_title="Maintenance Forecasting", page_icon="📈", layout="wide")
    ui.page_header(
        "Maintenance Forecasting",
        "Seasonal-naive baselines for synthetic weekly maintenance workload",
        "synthetic_mro",
    )

    # Target selector
    target_column = st.selectbox(
        "Forecast target",
        options=list(TARGET_LABELS.keys()),
        format_func=lambda c: TARGET_LABELS[c],
        key="fc_target",
    )

    ui.freshness_badge(
        dl.REPORTS_DIR / "forecasting" / f"forecast_results_{target_column}.json",
        f"Forecast results freshness ({target_column})",
    )
    st.divider()

    res = dl.load_forecast_results(target_column=target_column)
    if res is None:
        ui.missing_artifact(
            "forecast results",
            f"reports/forecasting/forecast_results_{target_column}.json",
        )
        return

    _render_metrics(res, target_column)
    _render_seasonal_candidates(res)
    _render_per_series(res, target_column)
    _render_demand_chart(target_column)

    ui.limitations_box(res.get("limitations", []) or [])
