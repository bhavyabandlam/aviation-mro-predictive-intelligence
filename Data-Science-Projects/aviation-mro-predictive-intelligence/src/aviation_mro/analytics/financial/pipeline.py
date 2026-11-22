"""Financial Analytics Pipeline.

End-to-end financial analytics pipeline for MRO operations.
All calculations use SYNTHETIC data — not real airline records.

Pipeline Steps:
1. Load financial data
2. Calculate labor costs
3. Calculate parts costs
4. Calculate downtime costs
5. Calculate holding costs
6. Calculate stockout costs
7. Calculate financial KPIs
8. Run scenario analysis
9. Generate reports
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cost_per_unit import calculate_cost_per_unit
from .data_prep import (
    load_all_financial_data,
)
from .downtime_costs import calculate_downtime_costs
from .holding_costs import calculate_holding_costs
from .kpis import calculate_financial_kpis
from .labor_costs import calculate_labor_costs
from .parts_costs import calculate_parts_costs
from .planned_unplanned import analyze_planned_unplanned
from .scenarios import run_scenario_analysis
from .sensitivity import run_sensitivity_analysis
from .stockout_costs import calculate_stockout_costs

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"

RANDOM_SEED = 42


def _file_fingerprint(path: Path) -> str:
    """Return a short SHA-256 fingerprint of a file's bytes (or 'missing')."""
    try:
        if not path.exists():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unreadable"


def run_financial_pipeline(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    demand_forecast_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    safety_stock_df: pd.DataFrame | None = None,
    stockout_risk_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run complete financial analytics pipeline.

    Args:
        input_dir: Directory with financial data (defaults to data/processed)
        output_dir: Directory for output reports (defaults to reports/financial)
        demand_forecast_df: Optional demand forecast data
        inventory_df: Optional inventory data
        safety_stock_df: Optional safety stock data
        stockout_risk_df: Optional stockout risk data

    Returns:
        Dictionary with all pipeline results
    """
    if output_dir is None:
        output_dir = REPORTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load financial data
    if input_dir:
        # Load from custom directory
        financial_data = {}
        for name in [
            "shop_labor",
            "fleet",
            "maintenance_labor",
            "downtime_costs",
            "inventory_cost_params",
        ]:
            path = input_dir / f"financial_{name}.parquet"
            if path.exists():
                financial_data[name] = pd.read_parquet(path)
    else:
        financial_data = load_all_financial_data()

    # 2. Calculate labor costs
    labor_results = calculate_labor_costs(
        financial_data["shop_labor"],
        financial_data["maintenance_labor"],
        demand_forecast_df,
    )

    # 3. Calculate parts costs
    if inventory_df is None:
        # Load from default location
        try:
            inventory_df = pd.read_parquet(
                PROJECT_ROOT / "data" / "processed" / "inventory_data.parquet"
            )
        except FileNotFoundError:
            # Generate synthetic inventory if not available
            from ..inventory.data_prep import generate_synthetic_inventory

            inventory_df = generate_synthetic_inventory()

    parts_results = calculate_parts_costs(inventory_df)

    # 4. Calculate downtime costs
    downtime_results = calculate_downtime_costs(
        financial_data["fleet"],
        financial_data["downtime_costs"],
        demand_forecast_df,
    )

    # 5. Calculate holding costs
    if safety_stock_df is None:
        # Load from inventory analytics
        try:
            from ..inventory.safety_stock import calculate_safety_stock

            safety_stock_results = calculate_safety_stock(inventory_df, method="normal_constant_lt")
            safety_stock_df = pd.DataFrame(
                [
                    {"part_number": r.part_number, "safety_stock": r.safety_stock}
                    for r in safety_stock_results
                ]
            )
        except Exception:
            safety_stock_df = pd.DataFrame(columns=["part_number", "safety_stock"])

    holding_results = calculate_holding_costs(inventory_df, safety_stock_df)

    # 6. Calculate stockout costs
    if stockout_risk_df is None:
        # Load from inventory analytics
        try:
            from ..inventory.reorder_point import calculate_reorder_points
            from ..inventory.stockout_risk import analyze_stockout_risk

            rop_results = calculate_reorder_points(inventory_df)
            stockout_risk_results = analyze_stockout_risk(inventory_df, rop_results)
            stockout_risk_df = pd.DataFrame(
                [
                    {
                        "part_number": r.part_number,
                        "stockout_probability": r.stockout_probability,
                        "cycles_per_year": r.cycles_per_year,
                        "risk_category": r.risk_category,
                    }
                    for r in stockout_risk_results
                ]
            )
        except Exception:
            stockout_risk_df = pd.DataFrame(
                columns=["part_number", "stockout_probability", "cycles_per_year", "risk_category"]
            )

    stockout_results = calculate_stockout_costs(inventory_df, stockout_risk_df)

    # 7. Calculate financial KPIs
    kpi_results = calculate_financial_kpis(
        labor_results,
        parts_results,
        downtime_results,
        holding_results,
        stockout_results,
        financial_data["fleet"],
    )

    # 8. Run scenario analysis
    scenario_results = run_scenario_analysis(
        labor_results["totals"]["total_annual_labor_cost"],
        parts_results["totals"]["total_annual_parts_cost"],
        downtime_results["totals"]["total_annual_downtime_cost"],
        holding_results["totals"]["total_holding_cost"],
        stockout_results["totals"]["total_annual_stockout_cost"],
    )

    # 9. Planned vs unplanned cost analysis
    planned_unplanned_results = analyze_planned_unplanned(
        labor_results,
        parts_results,
        downtime_results,
        holding_results,
        stockout_results,
    )

    # 10. Cost per work order and per engine
    total_maintenance_cost = kpi_results["cost_breakdown"]["total_maintenance_cost"]
    total_job_count = labor_results["totals"]["total_annual_labor_hours"]  # proxy: total jobs
    # Derive total job count from labor results: sum of annual_job_count across shop-family results
    total_job_count = sum(r.annual_job_count for r in labor_results.get("shop_family_results", []))
    cost_per_unit_results = calculate_cost_per_unit(
        total_annual_maintenance_cost=total_maintenance_cost,
        total_annual_job_count=total_job_count,
        fleet_df=financial_data["fleet"],
    )

    # 11. Sensitivity analysis
    sensitivity_results = run_sensitivity_analysis(
        {
            "financial_kpis": kpi_results,
            "labor_costs": labor_results,
            "parts_costs": parts_results,
            "downtime_costs": downtime_results,
            "holding_costs": holding_results,
            "stockout_costs": stockout_results,
            "scenario_analysis": scenario_results,
        }
    )

    # 12. Save results and generate report
    results = {
        "labor_costs": labor_results,
        "parts_costs": parts_results,
        "downtime_costs": downtime_results,
        "holding_costs": holding_results,
        "stockout_costs": stockout_results,
        "financial_kpis": kpi_results,
        "scenario_analysis": scenario_results,
        "planned_unplanned": planned_unplanned_results,
        "cost_per_unit": cost_per_unit_results,
        "sensitivity_analysis": sensitivity_results,
        "metadata": {
            "source": "synthetic_financial_v1",
            "inventory_source": "synthetic_inventory_v1",
            "pipeline_version": "1.1",
            "random_seed": RANDOM_SEED,
            "data_fingerprint": _file_fingerprint(
                PROJECT_ROOT / "data" / "processed" / "financial_shop_labor.parquet"
            ),
            "inventory_fingerprint": _file_fingerprint(
                PROJECT_ROOT / "data" / "processed" / "inventory_data.parquet"
            ),
        },
    }

    # Save JSON
    output_path = output_dir / "financial_analysis.json"
    save_results(results, output_path)

    # Generate markdown report
    generate_financial_report(results, output_dir)

    return results


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save results to JSON with custom serialization."""

    def serialize(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        else:
            return obj

    serializable = serialize(results)
    output_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")


def generate_financial_report(results: dict[str, Any], output_dir: Path) -> None:
    """Generate markdown financial report."""
    lines = []

    lines.append("# Financial Analytics Report")
    lines.append("")
    lines.append("**SYNTHETIC DATA — Not Real Airline Records**")
    lines.append("")

    # Metadata
    meta = results.get("metadata", {})
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Financial Data Source:** {meta.get('source', 'synthetic_financial_v1')}")
    lines.append(
        f"**Inventory Data Source:** {meta.get('inventory_source', 'synthetic_inventory_v1')}"
    )
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    scenario = results["scenario_analysis"]
    lines.append(f"- **Baseline Annual Maintenance Cost:** ${scenario['baseline_total']:,.0f}")
    lines.append(
        f"- **Best Scenario ({scenario['best_scenario']}):** "
        f"${scenario['baseline_total'] - scenario['best_savings']:,.0f} "
        f"(${scenario['best_savings']:,.0f} savings)"
    )
    lines.append(
        f"- **Worst Scenario ({scenario['worst_scenario']}):** "
        f"${scenario['baseline_total'] + scenario['worst_increase']:,.0f} "
        f"(${scenario['worst_increase']:,.0f} increase)"
    )
    lines.append("")

    # Cost Breakdown
    lines.append("## Cost Breakdown")
    lines.append("")
    kpis = results["financial_kpis"]
    cb = kpis["cost_breakdown"]
    cb_pct = kpis["cost_breakdown_pct"]
    lines.append("| Category | Cost | % of Total |")
    lines.append("|----------|------|------------|")
    for cat in ["labor_cost", "parts_cost", "downtime_cost", "holding_cost", "stockout_cost"]:
        if cat in cb:
            lines.append(
                f"| {cat.replace('_', ' ').title()} | ${cb[cat]:,.0f} | {cb_pct.get(cat, 0):.1f}% |"
            )
    lines.append(f"| **Total** | **${cb['total_maintenance_cost']:,.0f}** | **100.0%** |")
    lines.append("")

    # Financial KPIs
    lines.append("## Financial KPIs")
    lines.append("")
    lines.append("| KPI | Value | Unit |")
    lines.append("|-----|-------|------|")
    kpi_items = [
        (
            "Cost per Flight Hour",
            kpis["cost_per_flight_hour"].value,
            kpis["cost_per_flight_hour"].unit,
        ),
        (
            "Maintenance Cost % Revenue",
            kpis["maintenance_cost_pct_revenue"].value,
            kpis["maintenance_cost_pct_revenue"].unit,
        ),
        ("Inventory Turnover", kpis["inventory_turnover"].value, kpis["inventory_turnover"].unit),
        ("Stockout Rate", kpis["stockout_rate"].value, kpis["stockout_rate"].unit),
        ("Labor Utilization", kpis["labor_utilization"].value, kpis["labor_utilization"].unit),
        (
            "Predictive Maintenance Avoidance",
            kpis["predictive_maintenance_avoidance"].value,
            kpis["predictive_maintenance_avoidance"].unit,
        ),
        (
            "Predictive Maintenance ROI (Year 1)",
            kpis["predictive_maintenance_roi"].value,
            kpis["predictive_maintenance_roi"].unit,
        ),
    ]
    for name, value, unit in kpi_items:
        lines.append(f"| {name} | {value:,.2f} | {unit} |")
    lines.append("")

    # Scenario Analysis
    lines.append("## Scenario Analysis")
    lines.append("")
    lines.append("| Scenario | Total Cost | vs Baseline | Change |")
    lines.append("|----------|------------|-------------|--------|")
    for name, s in scenario["scenarios"].items():
        if name == "baseline":
            lines.append(f"| {s.scenario_name} | ${s.total_annual_cost:,.0f} | — | — |")
        else:
            change_str = f"${s.cost_vs_baseline:+,.0f} ({s.pct_change_vs_baseline:+.1f}%)"
            lines.append(
                f"| {s.scenario_name} | ${s.total_annual_cost:,.0f} | "
                f"{change_str} | {s.description} |"
            )
    lines.append("")

    # Labor Costs Detail
    labor = results["labor_costs"]
    lines.append("## Labor Costs by Shop")
    lines.append("")
    lines.append(
        "| Shop | Technicians | Base Rate | Total Cost | Hours | Utilization | Overtime Cost |"
    )
    lines.append(
        "|------|-------------|-----------|------------|-------|-------------|---------------|"
    )
    for s in labor["shop_summaries"]:
        lines.append(
            f"| {s.shop_id} | {s.technician_count} | ${s.base_labor_rate:.2f} | "
            f"${s.total_annual_labor_cost:,.0f} | {s.total_annual_labor_hours:,.0f} | "
            f"{s.utilization_rate:.1%} | ${s.overtime_cost:,.0f} |"
        )
    lines.append("")

    # Parts Costs by Family
    parts = results["parts_costs"]
    lines.append("## Parts Costs by Family")
    lines.append("")
    lines.append(
        "| Family | Parts | Annual Demand | Avg Unit Cost | Annual Parts Cost | Inventory Cost |"
    )
    lines.append(
        "|--------|-------|---------------|---------------|-------------------|----------------|"
    )
    for s in parts["family_summaries"]:
        lines.append(
            f"| {s.group_value} | {s.total_parts} | "
            f"{s.total_annual_demand:,.0f} | "
            f"${s.avg_unit_cost:,.2f} | ${s.total_annual_parts_cost:,.0f} | "
            f"${s.total_inventory_cost:,.0f} |"
        )
    lines.append("")

    # Downtime Costs
    dt = results["downtime_costs"]
    lines.append("## Downtime Costs by Fleet Type")
    lines.append("")
    lines.append(
        "| Fleet Type | Part Family | Aircraft | "
        "Downtime/Event (hrs) | Cost/Hour | Cost/Event | "
        "Annual Events | Annual Cost |"
    )
    lines.append(
        "|------------|-------------|----------|---------------------|-----------|------------|---------------|-------------|"
    )
    for r in dt["fleet_family_results"]:
        lines.append(
            f"| {r.fleet_type} | {r.part_family} | {r.aircraft_count} | "
            f"{r.mean_downtime_hours:.1f} | ${r.downtime_cost_per_hour:,.0f} | "
            f"${r.downtime_cost_per_event:,.0f} | "
            f"{r.annual_event_count} | ${r.annual_downtime_cost:,.0f} |"
        )
    lines.append("")

    # Scenario Details
    lines.append("## Scenario Assumptions")
    lines.append("")
    for name, s in scenario["scenarios"].items():
        if name != "baseline":
            lines.append(f"### {s.scenario_name}")
            lines.append("")
            for assumption in s.key_assumptions:
                lines.append(f"- {assumption}")
            lines.append("")

    # Assumptions & Limitations
    lines.append("## Key Assumptions & Limitations")
    lines.append("")
    lines.append("1. **All data is SYNTHETIC** — generated for development/testing only")
    lines.append("2. **Labor rates** vary by shop (base + overtime), no regional differences")
    lines.append("3. **Downtime costs** include lost revenue, passenger handling, recovery costs")
    lines.append("4. **Holding cost rate** fixed at 20% annually across all parts")
    lines.append(
        "5. **Stockout cost multiplier** fixed at 10x unit cost (expediting, AOG, penalties)"
    )
    lines.append("6. **Fleet mix** assumed constant (60% narrow-body, 25% wide-body, 15% regional)")
    lines.append("7. **Demand forecast** from synthetic maintenance demand (Section 9.4)")
    lines.append("8. **Inventory parameters** from synthetic inventory (Section 9.5)")
    lines.append("9. **No supplier constraints** — unlimited capacity, no MOQs, no lead time cost")
    lines.append("10. **No inflation/escalation** — all costs in constant 2026 USD")
    lines.append(
        "11. **Predictive maintenance** assumes 20% downtime reduction, no implementation cost"
    )
    lines.append("12. **Scenarios** are independent; combined effects may differ from sum of parts")
    lines.append("")

    # Planned vs Unplanned
    pu = results.get("planned_unplanned", {})
    if pu:
        lines.append("## Planned vs Unplanned Maintenance Costs")
        lines.append("")
        totals = pu.get("totals", {})
        lines.append(f"- **Total Annual Maintenance Cost:** ${totals.get('total_cost', 0):,.0f}")
        lines.append(
            f"- **Planned Cost:** ${totals.get('total_planned_cost', 0):,.0f} "
            f"({totals.get('planned_share_of_total', 0):.1%})"
        )
        lines.append(
            f"- **Unplanned Cost:** ${totals.get('total_unplanned_cost', 0):,.0f} "
            f"({totals.get('unplanned_share_of_total', 0):.1%})"
        )
        lines.append("")
        lines.append("| Category | Total Cost | Planned Cost | Unplanned Cost | Planned Share |")
        lines.append("|----------|------------|--------------|----------------|---------------|")
        for r in pu.get("category_results", [])[:20]:
            lines.append(
                f"| {r.category} | ${r.total_cost:,.0f} | "
                f"${r.planned_cost:,.0f} | ${r.unplanned_cost:,.0f} | "
                f"{r.planned_share:.1%} |"
            )
        lines.append("")

    # Cost per work order and per engine
    cpu = results.get("cost_per_unit", {})
    if cpu:
        lines.append("## Cost per Work Order and per Engine")
        lines.append("")
        cpwo = cpu.get("cost_per_work_order", {})
        cpe = cpu.get("cost_per_engine", {})
        total_cost_val = cpu.get("total_annual_maintenance_cost", 0)
        lines.append(f"- **Total Annual Maintenance Cost:** ${total_cost_val:,.0f}")
        lines.append(f"- **Total Annual Job Count:** {cpu.get('total_annual_job_count', 0):,}")
        lines.append(f"- **Total Engines:** {cpu.get('total_engines', 0):,}")
        lines.append("")
        if cpwo:
            lines.append(f"- **Cost per Work Order:** {cpwo.value:,.2f} {cpwo.unit}")
            lines.append(f"  Formula: {cpwo.formula}")
        if cpe:
            lines.append(f"- **Cost per Engine:** {cpe.value:,.2f} {cpe.unit}")
            lines.append(f"  Formula: {cpe.formula}")
        lines.append("")

    # Sensitivity analysis
    sa = results.get("sensitivity_analysis", {})
    if sa:
        lines.append("## Sensitivity Analysis (Estimates — Synthetic Data)")
        lines.append("")
        lines.append(f"- **Baseline Total Annual Cost:** ${sa.get('baseline_total_cost', 0):,.0f}")
        lines.append(f"- **Most Sensitive Parameter:** {sa.get('most_sensitive_parameter', 'n/a')}")
        lines.append(f"- **Max Impact (±30%):** ${sa.get('max_impact_usd', 0):,.0f}")
        lines.append("")
        header = (
            "| Parameter | Baseline Value | Variation | Varied Total Cost | "
            "Change (USD) | Change (%) |"
        )
        lines.append(header)
        lines.append(
            "|-----------|----------------|-----------|-------------------|--------------|------------|"
        )
        for _param_name, param_results in sa.get("results", {}).items():
            for r in param_results:
                lines.append(
                    f"| {r['parameter_label']} | {r['baseline_value']} | "
                    f"{r['variation_pct']:+.0f}% | ${r['varied_total_cost']:,.0f} | "
                    f"${r['cost_change_usd']:+,.0f} | {r['cost_change_pct']:+.2f}% |"
                )
        lines.append("")
        lines.append(
            "*All sensitivity outputs are estimates based on synthetic data — "
            "not realised savings.*"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*End of Report*")

    report_path = output_dir / "FINANCIAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Main entry point."""
    run_financial_pipeline()


if __name__ == "__main__":
    main()
