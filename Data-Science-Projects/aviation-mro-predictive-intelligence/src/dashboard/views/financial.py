"""Financial Scenarios page (synthetic MRO financials).

Renders cost breakdown, KPIs, scenario analysis, and per-shop/family/fleet
cost tables. All values are synthetic (seed=42).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _render_cost_breakdown(fin: dict[str, Any]) -> None:
    ui.section("Annual Cost Breakdown")
    kpis = fin.get("financial_kpis", {}) or {}
    cb = kpis.get("cost_breakdown", {}) or {}
    cb_pct = kpis.get("cost_breakdown_pct", {}) or {}
    if not cb:
        st.info("No cost breakdown recorded.")
        return
    labels = {
        "labor_cost": "Labor",
        "parts_cost": "Parts",
        "downtime_cost": "Downtime",
        "holding_cost": "Holding",
        "stockout_cost": "Stockout",
    }
    rows = [
        {
            "Category": label,
            "Cost (USD)": ui.fmt_money(cb.get(key, 0.0)),
            "% of Total": cb_pct.get(key, 0.0),
        }
        for key, label in labels.items()
    ]
    rows.append(
        {
            "Category": "Total",
            "Cost (USD)": ui.fmt_money(cb.get("total_maintenance_cost", 0.0)),
            "% of Total": 100.0,
        }
    )
    ui.data_table(rows)
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        ui.csv_export_button(df, "cost_breakdown.csv")
        ui.export_section(
            {"cost_breakdown": df},
            "financial_cost_breakdown",
            source_lineage="Synthetic MRO financial data (seed=42)",
            synthetic=True,
            label="Export cost breakdown (CSV + HTML)",
        )
    # Bar chart of cost by category (excluding total).
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Category": [labels[k] for k in labels],
                "Cost": [cb.get(k, 0.0) for k in labels],
            }
        )
        st.caption("Bar chart: maintenance cost category on the x-axis, cost in USD on the y-axis.")
        st.bar_chart(df.set_index("Category"))
    except Exception:
        pass


def _render_kpis(fin: dict[str, Any]) -> None:
    ui.section("Financial KPIs")
    kpis = fin.get("financial_kpis", {}) or {}
    rows = []
    for _key, val in kpis.items():
        if not isinstance(val, dict):
            continue
        if "kpi_name" not in val:
            continue
        rows.append(
            {
                "KPI": val.get("kpi_name"),
                "Value": round(val.get("value", 0.0), 2),
                "Unit": val.get("unit"),
                "Formula": val.get("formula", ""),
            }
        )
    if rows:
        ui.data_table(rows)
    else:
        st.info("No KPIs recorded.")
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        ui.csv_export_button(df, "financial_kpis.csv")
        ui.export_section(
            {"financial_kpis": df},
            "financial_kpis",
            source_lineage="Synthetic MRO financial data (seed=42)",
            synthetic=True,
            label="Export financial KPIs (CSV + HTML)",
        )


def _render_scenarios(fin: dict[str, Any]) -> None:
    ui.section("Scenario Analysis")
    scen = fin.get("scenario_analysis", {}) or {}
    scenarios = scen.get("scenarios", {}) or {}
    if not scenarios:
        st.info("No scenario analysis recorded.")
        return
    rows = [
        {
            "Scenario": s.get("scenario_name", name),
            "Total Annual Cost": ui.fmt_money(s.get("total_annual_cost", 0.0)),
            "vs Baseline": ui.fmt_money(s.get("cost_vs_baseline", 0.0)),
            "Change %": round(s.get("pct_change_vs_baseline", 0.0), 2),
        }
        for name, s in scenarios.items()
        if isinstance(s, dict)
    ]
    ui.data_table(rows)
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        ui.csv_export_button(df, "scenario_analysis.csv")
        ui.export_section(
            {"scenario_analysis": df},
            "financial_scenario_analysis",
            source_lineage="Synthetic MRO financial data (seed=42)",
            synthetic=True,
            label="Export scenario analysis (CSV + HTML)",
        )
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Scenario": [s.get("scenario_name", name) for name, s in scenarios.items()],
                "Total Annual Cost": [s.get("total_annual_cost", 0.0) for s in scenarios.values()],
            }
        )
        ui.csv_export_button(df, "scenario_analysis.csv")
        st.caption("Bar chart: scenario on the x-axis, total annual cost in USD on the y-axis.")
        st.bar_chart(df.set_index("Scenario"))
    except Exception:
        pass
    ui.kpi_row(
        [
            {
                "label": "Best Scenario",
                "value": scen.get("best_scenario", "n/a"),
                "helper": f"Savings {ui.fmt_money(scen.get('best_savings', 0.0))}",
            },
            {
                "label": "Worst Scenario",
                "value": scen.get("worst_scenario", "n/a"),
                "helper": f"Increase {ui.fmt_money(scen.get('worst_increase', 0.0))}",
            },
        ]
    )


def _render_labor(fin: dict[str, Any]) -> None:
    ui.section("Labor Costs by Shop")
    labor = fin.get("labor_costs", {}) or {}
    summaries = labor.get("shop_summaries", []) or []
    if not summaries:
        st.info("No labor summaries recorded.")
        return
    ui.data_table(
        [
            {
                "Shop": s.get("shop_id"),
                "Technicians": s.get("technician_count"),
                "Base Rate": s.get("base_labor_rate"),
                "Overtime Rate": s.get("overtime_labor_rate"),
                "Annual Cost": ui.fmt_money(s.get("total_labor_cost", 0.0)),
                "Utilization": s.get("utilization_rate"),
            }
            for s in summaries
        ]
    )
    if summaries:
        import pandas as pd

        df = pd.DataFrame(summaries)
        ui.csv_export_button(df, "labor_costs_by_shop.csv")


def _render_parts(fin: dict[str, Any]) -> None:
    ui.section("Parts Costs by Family")
    parts = fin.get("parts_costs", {}) or {}
    families = parts.get("family_summaries", []) or []
    if not families:
        st.info("No parts summaries recorded.")
        return
    ui.data_table(
        [
            {
                "Family": f.get("group_value"),
                "Parts": f.get("total_parts"),
                "Annual Demand": ui.fmt_num(f.get("total_annual_demand", 0.0), 0),
                "Avg Unit Cost": ui.fmt_num(f.get("avg_unit_cost", 0.0), 2),
                "Annual Parts Cost": ui.fmt_money(f.get("total_annual_parts_cost", 0.0)),
                "Inventory Cost": ui.fmt_money(f.get("total_inventory_cost", 0.0)),
            }
            for f in families
        ]
    )
    if families:
        import pandas as pd

        df = pd.DataFrame(families)
        ui.csv_export_button(df, "parts_costs_by_family.csv")


def _render_downtime(fin: dict[str, Any]) -> None:
    ui.section("Downtime Costs by Fleet Type")
    dt = fin.get("downtime_costs", {}) or {}
    results = dt.get("fleet_family_results", []) or []
    if not results:
        st.info("No downtime results recorded.")
        return
    ui.data_table(
        [
            {
                "Fleet": r.get("fleet_type"),
                "Family": r.get("part_family"),
                "Aircraft": r.get("aircraft_count"),
                "Downtime/Event (hrs)": round(r.get("mean_downtime_hours", 0.0), 1),
                "Cost/Hour": ui.fmt_money(r.get("downtime_cost_per_hour", 0.0)),
                "Cost/Event": ui.fmt_money(r.get("downtime_cost_per_event", 0.0)),
                "Annual Cost": ui.fmt_money(r.get("annual_downtime_cost", 0.0)),
            }
            for r in results
        ]
    )
    if results:
        import pandas as pd

        df = pd.DataFrame(results)
        ui.csv_export_button(df, "downtime_costs_by_fleet.csv")


def _render_planned_unplanned(fin: dict[str, Any]) -> None:
    ui.section("Planned vs Unplanned Maintenance Costs")
    pu = fin.get("planned_unplanned", {}) or {}
    totals = pu.get("totals", {}) or {}
    if not totals:
        st.info("No planned/unplanned cost data recorded.")
        return
    ui.kpi_row(
        [
            {
                "label": "Total Annual Cost",
                "value": ui.fmt_money(totals.get("total_cost", 0.0)),
                "helper": "All maintenance cost categories",
            },
            {
                "label": "Planned Cost",
                "value": ui.fmt_money(totals.get("total_planned_cost", 0.0)),
                "helper": f"{totals.get('planned_share_of_total', 0):.1%} of total",
            },
            {
                "label": "Unplanned Cost",
                "value": ui.fmt_money(totals.get("total_unplanned_cost", 0.0)),
                "helper": f"{totals.get('unplanned_share_of_total', 0):.1%} of total",
            },
        ]
    )
    category_results = pu.get("category_results", []) or []
    if category_results:
        ui.data_table(
            [
                {
                    "Category": r.get("category", ""),
                    "Total Cost": ui.fmt_money(r.get("total_cost", 0.0)),
                    "Planned Cost": ui.fmt_money(r.get("planned_cost", 0.0)),
                    "Unplanned Cost": ui.fmt_money(r.get("unplanned_cost", 0.0)),
                    "Planned Share": f"{r.get('planned_share', 0):.1%}",
                }
                for r in category_results[:30]
            ]
        )
    st.caption(
        "Planned = scheduled/preventive; Unplanned = breakdown/reactive/AOG. "
        "Planned share by family (synthetic assumption): ENGINE 70%, AVIONICS 40%, AIRFRAME 50%. "
        "All data is synthetic — not real airline records."
    )


def _render_cost_per_unit(fin: dict[str, Any]) -> None:
    ui.section("Cost per Work Order and per Engine")
    cpu = fin.get("cost_per_unit", {}) or {}
    if not cpu:
        st.info("No cost-per-unit data recorded.")
        return
    cpwo = cpu.get("cost_per_work_order", {}) or {}
    cpe = cpu.get("cost_per_engine", {}) or {}
    ui.kpi_row(
        [
            {
                "label": "Cost per Work Order",
                "value": ui.fmt_money(cpwo.get("value", 0.0)),
                "helper": cpwo.get("unit", "USD/work_order"),
            },
            {
                "label": "Cost per Engine",
                "value": ui.fmt_money(cpe.get("value", 0.0)),
                "helper": cpe.get("unit", "USD/engine/year"),
            },
            {
                "label": "Total Annual Cost",
                "value": ui.fmt_money(cpu.get("total_annual_maintenance_cost", 0.0)),
                "helper": "All maintenance categories",
            },
            {
                "label": "Total Jobs / Engines",
                "value": (
                    f"{cpu.get('total_annual_job_count', 0):,} / {cpu.get('total_engines', 0):,}"
                ),
                "helper": "Annual job count / total engines",
            },
        ]
    )
    if cpwo:
        st.caption(f"Formula: {cpwo.get('formula', '')}")
    if cpe:
        st.caption(f"Formula: {cpe.get('formula', '')}")
    st.caption(
        "Engines per aircraft (synthetic assumption): NARROW_BODY 2, WIDE_BODY 2, REGIONAL 2. "
        "All data is synthetic — not real airline records."
    )


def _render_sensitivity(fin: dict[str, Any]) -> None:
    ui.section("Financial Sensitivity Analysis (Estimates)")
    sa = fin.get("sensitivity_analysis", {}) or {}
    if not sa:
        st.info("No sensitivity analysis data recorded.")
        return
    ui.kpi_row(
        [
            {
                "label": "Baseline Total Cost",
                "value": ui.fmt_money(sa.get("baseline_total_cost", 0.0)),
                "helper": "Annual maintenance cost",
            },
            {
                "label": "Most Sensitive Parameter",
                "value": sa.get("most_sensitive_parameter", "n/a"),
                "helper": "Largest cost impact at ±30%",
            },
            {
                "label": "Max Impact (±30%)",
                "value": ui.fmt_money(sa.get("max_impact_usd", 0.0)),
                "helper": "USD change in total cost",
            },
        ]
    )
    results = sa.get("results", {}) or {}
    if results:
        all_rows = []
        for param_name, param_results in results.items():
            for r in param_results:
                all_rows.append(
                    {
                        "Parameter": r.get("parameter_label", param_name),
                        "Baseline": r.get("baseline_value", 0.0),
                        "Variation": f"{r.get('variation_pct', 0):+.0f}%",
                        "Varied Total Cost": ui.fmt_money(r.get("varied_total_cost", 0.0)),
                        "Change (USD)": f"{r.get('cost_change_usd', 0):+,.0f}",
                        "Change (%)": f"{r.get('cost_change_pct', 0):+.2f}%",
                    }
                )
        ui.data_table(all_rows[:40])
    st.caption(
        "One-at-a-time sensitivity analysis; all other parameters held at baseline. "
        "Linear impact assumption. Variation levels: -30%, -20%, -10%, +10%, +20%, +30%. "
        "All outputs are ESTIMATES based on synthetic data — not realised savings."
    )


def render() -> None:
    """Render the Financial Analytics page."""
    st.set_page_config(page_title="Financial Scenarios", page_icon="💰", layout="wide")
    ui.page_header(
        "Financial Scenarios",
        "Cost breakdown, KPIs and scenario analysis for synthetic MRO operations",
        "synthetic_fin",
    )
    ui.freshness_badge(
        dl.REPORTS_DIR / "financial" / "financial_analysis.json",
        "Financial analysis freshness",
    )
    st.divider()

    fin = dl.load_financial_analysis()
    if fin is None:
        ui.missing_artifact("financial analysis", "reports/financial/financial_analysis.json")
        return

    _render_cost_breakdown(fin)
    _render_kpis(fin)
    _render_scenarios(fin)
    _render_labor(fin)
    _render_parts(fin)
    _render_downtime(fin)
    _render_planned_unplanned(fin)
    _render_cost_per_unit(fin)
    _render_sensitivity(fin)

    with st.expander("Financial assumptions & limitations", expanded=False):
        st.markdown(
            "- All data is SYNTHETIC (source: synthetic_financial_v1), not real airline records.\n"
            "- Holding cost rate fixed at 20%/yr; stockout multiplier 10x unit cost.\n"
            "- No inflation, FX, regional labour differentials, or supplier contracts.\n"
            "- Predictive maintenance assumes 20% downtime reduction, no implementation cost.\n"
            "- Planned/unplanned split (synthetic assumption): ENGINE 70% planned,\n"
            "  AVIONICS 40% planned, AIRFRAME 50% planned; default 60% planned.\n"
            "- Cost per engine assumes 2 engines per aircraft for all fleet types.\n"
            "- Sensitivity analysis uses one-at-a-time variation (-30% to +30%)\n"
            "  with linear impact.\n"
            "- All scenario and sensitivity outputs are ESTIMATES — not realised savings."
        )
