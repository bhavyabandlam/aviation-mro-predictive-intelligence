"""Inventory Intelligence page (synthetic MRO inventory).

Renders inventory analysis: validation status, ABC-style demand patterns,
reorder points, safety stock, stockout risk, slow-moving inventory,
inventory valuation/turnover, scenario comparison, and replenishment recommendations.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _render_overview(inv: dict[str, Any]) -> None:
    ui.section("Inventory Overview")
    meta = inv.get("metadata", {}) or {}
    validation = inv.get("validation", {}) or {}
    patterns = (inv.get("demand_analysis", {}) or {}).get("pattern_distribution", []) or []
    ui.kpi_row(
        [
            {"label": "Parts (count)", "value": meta.get("n_parts", "n/a")},
            {"label": "Families (count)", "value": len(meta.get("part_families", []) or [])},
            {"label": "Shops (count)", "value": len(meta.get("shops", []) or [])},
            {
                "label": "Valid",
                "value": "Yes" if validation.get("is_valid") else "No",
                "helper": f"{len(validation.get('errors', []) or [])} errors",
            },
        ]
    )
    if validation.get("warnings"):
        with st.expander("Validation warnings", expanded=False):
            for w in validation["warnings"]:
                st.markdown(f"- {w}")
    if patterns:
        st.write("**Demand pattern distribution:**")
        ui.data_table(patterns)


def _render_slow_moving(inv: dict[str, Any]) -> None:
    ui.section("Slow-Moving & Non-Moving Inventory")
    summary = inv.get("slow_moving_summary", {}) or {}
    if not summary:
        st.info("No slow-moving analysis available.")
        return
    ui.kpi_row(
        [
            {
                "label": "Non-Moving (count)",
                "value": summary.get("non_moving_count", 0),
                "helper": "Zero demand parts",
            },
            {
                "label": "Slow-Moving (count)",
                "value": summary.get("slow_moving_count", 0),
                "helper": "Below demand threshold",
            },
            {
                "label": "Critical Slow-Moving",
                "value": summary.get("critical_slow_moving_count", 0),
                "helper": "Criticality A or B",
            },
            {
                "label": "Slow-Moving Value",
                "value": f"${summary.get('slow_moving_inventory_value', 0.0):,.0f}",
                "helper": "Inventory value tied up",
            },
        ]
    )
    if summary.get("non_moving_parts"):
        ui.section("Non-Moving Parts")
        ui.data_table(summary["non_moving_parts"][:20])
    if summary.get("critical_slow_moving_parts"):
        ui.section("Critical Slow-Moving Parts")
        ui.data_table(summary["critical_slow_moving_parts"][:20])
    if summary.get("by_family"):
        st.write("**By Part Family:**")
        ui.data_table(summary["by_family"])


def _render_valuation(inv: dict[str, Any]) -> None:
    ui.section("Inventory Valuation & Turnover")
    summary = inv.get("valuation_summary", {}) or {}
    if not summary:
        st.info("No valuation data available.")
        return
    ui.kpi_row(
        [
            {
                "label": "Total Inventory Value",
                "value": f"${summary.get('total_inventory_value', 0.0):,.0f}",
                "helper": "Current stock × unit cost",
            },
            {
                "label": "ROP Value",
                "value": f"${summary.get('total_rop_value', 0.0):,.0f}",
                "helper": "Reorder point × unit cost",
            },
            {
                "label": "Safety Stock Value",
                "value": f"${summary.get('total_safety_stock_value', 0.0):,.0f}",
                "helper": "Safety stock × unit cost",
            },
            {
                "label": "Inventory Turnover",
                "value": f"{summary.get('overall_turnover', 0.0):.2f}×/yr",
                "helper": "Annual demand cost / avg inventory value",
            },
            {
                "label": "Days of Inventory",
                "value": f"{summary.get('overall_days_of_inventory', 0.0):.0f}",
                "helper": "365 / turnover",
            },
        ]
    )
    if summary.get("by_family"):
        st.write("**Valuation by Family:**")
        ui.data_table(summary["by_family"])
    if summary.get("by_shop"):
        st.write("**Valuation by Shop:**")
        ui.data_table(summary["by_shop"])


def _render_scenarios(inv: dict[str, Any]) -> None:
    ui.section("Inventory Scenarios (Stockout Risk vs Holding Cost)")
    summary = inv.get("scenario_summary", {}) or {}
    if not summary:
        st.info("No scenario data available.")
        return
    st.info(
        "Scenario outputs are ESTIMATES based on synthetic data — not realized savings or costs."
    )
    scenarios = summary.get("scenarios", [])
    if scenarios:
        ui.data_table(scenarios)
    col1, col2, col3 = st.columns(3)
    with col1:
        if summary.get("best_total_cost_scenario"):
            b = summary["best_total_cost_scenario"]
            st.metric(
                "Best Total Cost",
                b["scenario"],
                f"${b['total_annual_cost']:,.0f}/yr",
            )
    with col2:
        if summary.get("lowest_stockout_scenario"):
            s = summary["lowest_stockout_scenario"]
            st.metric(
                "Lowest Stockout Risk",
                s["scenario"],
                f"{s['avg_stockout_probability']:.2%}",
            )
    with col3:
        if summary.get("lowest_holding_cost_scenario"):
            h = summary["lowest_holding_cost_scenario"]
            st.metric(
                "Lowest Holding Cost",
                h["scenario"],
                f"${h['annual_holding_cost']:,.0f}/yr",
            )


def _render_stockout(inv: dict[str, Any]) -> None:
    ui.section("Stockout Risk")
    risk = inv.get("stockout_risk", []) or []
    if not risk:
        st.info("No stockout risk records.")
        return
    categories: dict[str, int] = {}
    for r in risk:
        cat = r.get("risk_category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    ui.data_table([{"Category": k, "Parts": v} for k, v in categories.items()])
    ranked = sorted(risk, key=lambda r: r.get("stockout_probability", 0.0), reverse=True)
    ui.section("Highest Stockout-Probability Parts")
    ui.data_table(
        [
            {
                "Part": r.get("part_number"),
                "Family": r.get("part_family"),
                "Shop": r.get("shop_id"),
                "Stockout Prob.": round(r.get("stockout_probability", 0.0), 4),
                "Fill Rate": round(r.get("fill_rate", 0.0), 4),
                "Risk": r.get("risk_category"),
            }
            for r in ranked[:15]
        ]
    )


def _render_reorder(inv: dict[str, Any]) -> None:
    ui.section("Reorder Points & Safety Stock")
    rop = inv.get("reorder_points", []) or []
    if not rop:
        st.info("No reorder point records.")
        return
    ui.data_table(
        [
            {
                "Part": r.get("part_number"),
                "Family": r.get("part_family"),
                "Shop": r.get("shop_id"),
                "Lead Time (wk)": round(r.get("lead_time_weeks", 0.0), 2),
                "Service Level": r.get("service_level"),
                "ROP": round(r.get("reorder_point", 0.0), 1),
                "Safety Stock": round(r.get("safety_stock", 0.0), 1),
            }
            for r in rop[:200]
        ],
        max_rows=200,
    )
    if rop:
        import pandas as pd

        df = pd.DataFrame(rop[:200])
        ui.csv_export_button(df, "reorder_points.csv")
        ui.export_section(
            {"reorder_points": df},
            "inventory_reorder_points",
            source_lineage="Synthetic MRO inventory (seed=42)",
            synthetic=True,
            label="Export reorder points (CSV + HTML)",
        )


def _render_recommendations(inv: dict[str, Any]) -> None:
    ui.section("Replenishment Recommendations")
    recs = inv.get("recommendations", []) or []
    if not recs:
        st.info("No recommendations generated.")
        return

    shops = sorted({r.get("shop_id") for r in recs if r.get("shop_id")})
    families = sorted({r.get("part_family") for r in recs if r.get("part_family")})
    priorities = sorted({r.get("priority") for r in recs if r.get("priority")})

    c1, c2, c3 = st.columns(3)
    with c1:
        f_shop = st.multiselect("Shop", shops, default=[], key="inv_shop")
    with c2:
        f_fam = st.multiselect("Family", families, default=[], key="inv_fam")
    with c3:
        f_pri = st.multiselect("Priority", priorities, default=[], key="inv_pri")

    def _match(r: dict[str, Any]) -> bool:
        if f_shop and r.get("shop_id") not in f_shop:
            return False
        if f_fam and r.get("part_family") not in f_fam:
            return False
        return not (f_pri and r.get("priority") not in f_pri)

    filtered = [r for r in recs if _match(r)]
    st.write(f"Showing {len(filtered)} of {len(recs)} recommendations.")
    ui.data_table(
        [
            {
                "Part": r.get("part_number"),
                "Shop": r.get("shop_id"),
                "Type": r.get("recommendation_type"),
                "Priority": r.get("priority"),
                "Current": round(r.get("current_value", 0.0), 1),
                "Recommended": round(r.get("recommended_value", 0.0), 1),
                "Change %": round(r.get("change_pct", 0.0), 1),
                "Rationale": (r.get("rationale", "") or "")[:80],
            }
            for r in filtered[:200]
        ],
        max_rows=200,
    )
    if filtered:
        import pandas as pd

        df = pd.DataFrame(filtered[:200])
        ui.csv_export_button(df, "replenishment_recommendations.csv")
        ui.export_section(
            {"replenishment_recommendations": df},
            "inventory_replenishment_recommendations",
            source_lineage="Synthetic MRO inventory (seed=42)",
            synthetic=True,
            label="Export replenishment recommendations (CSV + HTML)",
        )


def render() -> None:
    """Render the Inventory Analytics page."""
    st.set_page_config(page_title="Inventory Intelligence", page_icon="📦", layout="wide")
    ui.page_header(
        "Inventory Intelligence",
        "Reorder points, safety stock, stockout risk, slow-moving, valuation and scenarios",
        "synthetic_mro",
    )
    ui.freshness_badge(
        dl.REPORTS_DIR / "inventory" / "inventory_analysis.json",
        "Inventory analysis freshness",
    )
    st.divider()

    inv = dl.load_inventory_analysis()
    if inv is None:
        ui.missing_artifact("inventory analysis", "reports/inventory/inventory_analysis.json")
        return

    _render_overview(inv)
    _render_slow_moving(inv)
    _render_valuation(inv)
    _render_scenarios(inv)
    _render_stockout(inv)
    _render_reorder(inv)
    _render_recommendations(inv)
