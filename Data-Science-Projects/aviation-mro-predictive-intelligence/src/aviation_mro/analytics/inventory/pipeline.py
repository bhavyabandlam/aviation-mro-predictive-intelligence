"""Inventory Analytics Pipeline.

Main pipeline for running complete inventory analysis.
All data is SYNTHETIC — not real airline records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .data_validation import validate_inventory_data
from .demand_analysis import analyze_parts_demand
from .inventory_scenarios import (
    run_scenario_comparison,
    summarize_scenarios,
)
from .inventory_valuation import (
    calculate_inventory_valuation,
    summarize_valuation,
)
from .lead_time_demand import estimate_lead_time_demand
from .recommendations import (
    generate_inventory_recommendations,
)
from .reorder_point import calculate_reorder_points
from .safety_stock import calculate_safety_stock
from .slow_moving import (
    analyze_slow_moving_inventory,
    summarize_slow_moving,
)
from .stockout_risk import analyze_stockout_risk

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"
PLOTS_DIR = REPORTS_DIR / "plots"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def _file_fingerprint(path: Path) -> str:
    """Return a short SHA-256 fingerprint of a file's bytes (or 'missing')."""
    try:
        if not path.exists():
            return "missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest[:16]
    except OSError:
        return "unreadable"


def load_inventory_data() -> pd.DataFrame:
    """Load synthetic inventory data from Parquet."""
    path = DATA_DIR / "inventory_data.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory data not found at {path}. Run generate_synthetic_inventory() first."
        )
    return pd.read_parquet(path)


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    serializable: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, list) and value and hasattr(value[0], "__dataclass_fields__"):
            serializable[key] = [asdict(v) for v in value]
        elif hasattr(value, "__dataclass_fields__"):
            serializable[key] = asdict(value)
        elif isinstance(value, pd.DataFrame):
            serializable[key] = value.to_dict("records")
        else:
            serializable[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def run_inventory_pipeline(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run complete inventory analytics pipeline.

    Steps:
    1. Load and validate data
    2. Analyze parts demand
    3. Calculate reorder points
    4. Calculate safety stock (multiple methods)
    5. Estimate lead-time demand
    6. Analyze stockout risk
    7. Generate recommendations
    8. Save results and generate report
    """
    if output_dir is None:
        output_dir = REPORTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate data
    df = pd.read_parquet(input_path) if input_path else load_inventory_data()

    validation = validate_inventory_data(df)
    if validation.errors:
        pass
    if validation.warnings:
        pass

    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # 2. Demand analysis
    demand_results = analyze_parts_demand(df)

    # 3. Reorder points
    rop_results = calculate_reorder_points(df)
    sum(r.reorder_point for r in rop_results) / len(rop_results)

    # 4. Safety stock
    ss_results = calculate_safety_stock(df, method="normal_constant_lt")
    sum(r.safety_stock for r in ss_results) / len(ss_results)

    # 5. Lead-time demand
    ltd_results = estimate_lead_time_demand(df)
    sum(r.mean_ltd for r in ltd_results) / len(ltd_results)

    # 6. Stockout risk
    risk_results = analyze_stockout_risk(df, rop_results)
    risk_counts: dict[str, int] = {}
    for r in risk_results:
        risk_counts[r.risk_category] = risk_counts.get(r.risk_category, 0) + 1

    # 7. Slow-moving inventory analysis
    slow_moving_results = analyze_slow_moving_inventory(df)
    slow_moving_summary = summarize_slow_moving(slow_moving_results)

    # 8. Inventory valuation and turnover
    # Build lookups for reorder points and safety stocks
    rop_lookup = {r.part_number: r.reorder_point for r in rop_results}
    ss_lookup = {r.part_number: r.safety_stock for r in ss_results}
    valuation_results = calculate_inventory_valuation(df, rop_lookup, ss_lookup)
    valuation_summary = summarize_valuation(valuation_results)

    # 9. Scenario comparison
    ss_by_part = {r.part_number: r.safety_stock for r in ss_results}
    risk_by_part = {r.part_number: r for r in risk_results}
    scenario_results = run_scenario_comparison(df, ss_by_part, risk_by_part)
    scenario_summary = summarize_scenarios(scenario_results)

    # 10. Recommendations
    recommendations = generate_inventory_recommendations(df, rop_results, ss_results, risk_results)

    # Compile results
    results = {
        "validation": validation,
        "demand_analysis": demand_results,
        "reorder_points": rop_results,
        "safety_stock": ss_results,
        "lead_time_demand": ltd_results,
        "stockout_risk": risk_results,
        "slow_moving": slow_moving_results,
        "slow_moving_summary": slow_moving_summary,
        "inventory_valuation": valuation_results,
        "valuation_summary": valuation_summary,
        "scenarios": scenario_results,
        "scenario_summary": scenario_summary,
        "recommendations": recommendations,
        "metadata": {
            "source": "synthetic_inventory_v1",
            "data_fingerprint": _file_fingerprint(DATA_DIR / "inventory_data.parquet"),
            "random_seed": RANDOM_SEED,
            "n_parts": validation.n_parts,
            "part_families": sorted(df["part_family"].unique().tolist()),
            "shops": sorted(df["shop_id"].unique().tolist()),
        },
    }

    # Save results
    output_path = output_dir / "inventory_analysis.json"
    save_results(results, output_path)

    # Generate summary report
    generate_summary_report(results, output_dir)

    return results


def generate_summary_report(results: dict[str, Any], output_dir: Path) -> None:
    """Generate human-readable summary report."""
    report_path = output_dir / "INVENTORY_REPORT.md"

    val = results["validation"]
    demand = results["demand_analysis"]
    rop = results["reorder_points"]
    ss = results["safety_stock"]
    ltd = results["lead_time_demand"]
    risk = results["stockout_risk"]
    slow = results["slow_moving_summary"]
    val_summary = results["valuation_summary"]
    scenario_summary = results["scenario_summary"]
    recs = results["recommendations"]
    meta = results["metadata"]

    # Risk distribution
    risk_dist: dict[str, int] = {}
    for r in risk:
        risk_dist[r.risk_category] = risk_dist.get(r.risk_category, 0) + 1

    # Recommendation summary
    priority_dist: dict[str, int] = {}
    type_dist: dict[str, int] = {}
    total_cost = 0.0
    for r in recs:
        priority_dist[r.priority] = priority_dist.get(r.priority, 0) + 1
        type_dist[r.recommendation_type] = type_dist.get(r.recommendation_type, 0) + 1
        total_cost += r.cost_impact

    lines = [
        "# Inventory Analytics Report",
        "",
        "**SYNTHETIC DATA — Not Real Airline Records**",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source:** {meta['source']}",
        f"**Parts Analyzed:** {meta['n_parts']}",
        f"**Part Families:** {', '.join(meta['part_families'])}",
        f"**Shops:** {', '.join(meta['shops'])}",
        "",
        "## 1. Data Validation",
        "",
        f"- **Rows:** {val.n_rows}",
        f"- **Unique Parts:** {val.n_parts}",
        f"- **Valid:** {'✅ PASS' if val.is_valid else '❌ FAIL'}",
        "",
    ]

    if val.warnings:
        lines.extend(["### Warnings", ""])
        for w in val.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.extend(
        [
            "## 2. Demand Analysis",
            "",
            f"- **Total Parts:** {demand['n_parts']}",
            f"- **Families:** {len(demand['by_family'])}",
            f"- **Shops:** {len(demand['by_shop'])}",
            "",
            "### Demand Pattern Distribution",
            "",
        ]
    )

    for pattern in demand["pattern_distribution"]:
        lines.append(
            f"- {pattern['part_family']}: "
            f"Smooth={pattern['smooth_count']}, "
            f"Erratic={pattern['erratic_count']}, "
            f"Intermittent={pattern['intermittent_count']}, "
            f"Lumpy={pattern['lumpy_count']}"
        )

    lines.extend(
        [
            "",
            "## 3. Reorder Points",
            "",
            f"- **Average ROP:** {sum(r.reorder_point for r in rop) / len(rop):.1f}",
            f"- **Average Safety Stock:** {sum(r.safety_stock for r in ss) / len(ss):.1f}",
            f"- **Average Lead Time Demand:** {sum(r.mean_ltd for r in ltd) / len(ltd):.1f}",
            "",
            "## 4. Stockout Risk",
            "",
            "| Risk Category | Count |",
            "|---------------|-------|",
        ]
    )

    for cat in ["Low", "Medium", "High", "Critical"]:
        count = risk_dist.get(cat, 0)
        lines.append(f"| {cat} | {count} |")

    lines.extend(
        [
            "",
            "## 5. Slow-Moving Inventory",
            "",
            f"- **Total Parts:** {slow['total_parts']}",
            f"- **Non-Moving:** {slow['non_moving_count']}",
            f"- **Slow-Moving:** {slow['slow_moving_count']}",
            f"- **Critical Slow-Moving:** {slow['critical_slow_moving_count']}",
            f"- **Non-Moving Inventory Value:** ${slow['non_moving_inventory_value']:,.2f}",
            f"- **Slow-Moving Inventory Value:** ${slow['slow_moving_inventory_value']:,.2f}",
            "",
        ]
    )

    if slow.get("non_moving_parts"):
        lines.extend(["### Non-Moving Parts", ""])
        for p in slow["non_moving_parts"][:10]:
            lines.append(
                f"- {p['part_number']} ({p['part_family']}, {p['shop_id']}): "
                f"${p['inventory_value']:,.2f}"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. Inventory Valuation & Turnover",
            "",
            f"- **Total Inventory Value:** ${val_summary['total_inventory_value']:,.2f}",
            f"- **Total ROP Value:** ${val_summary['total_rop_value']:,.2f}",
            f"- **Total Safety Stock Value:** ${val_summary['total_safety_stock_value']:,.2f}",
            f"- **Total Annual Demand Cost:** ${val_summary['total_annual_demand_cost']:,.2f}",
            f"- **Overall Inventory Turnover:** {val_summary['overall_turnover']:.2f} times/year",
            f"- **Overall Days of Inventory:** {val_summary['overall_days_of_inventory']:.1f} days",
            "",
            "### By Family",
            "",
        ]
    )

    for fam in val_summary["by_family"]:
        lines.append(
            f"- {fam['part_family']}: {fam['n_parts']} parts, "
            f"${fam['inventory_value']:,.2f} value, "
            f"{fam['group_turnover']:.2f} turns/year"
        )

    lines.extend(
        [
            "",
            "## 7. Scenario Comparison",
            "",
            "| Scenario | Service Level | Safety Stock Value | Holding Cost | "
            "Stockout Cost | Total Cost |",
            "|----------|---------------|--------------------|--------------|---------------|------------|",
        ]
    )

    for s in scenario_summary["scenarios"]:
        lines.append(
            f"| {s['scenario']} | {s['service_level']:.0%} | "
            f"${s['total_safety_stock_value']:,.2f} | "
            f"${s['annual_holding_cost']:,.2f} | "
            f"${s['annual_stockout_cost']:,.2f} | "
            f"${s['total_annual_cost']:,.2f} |"
        )

    lines.extend(
        [
            "",
            f"- **Best Total Cost:** {scenario_summary['best_total_cost_scenario']['scenario']} "
            f"(${scenario_summary['best_total_cost_scenario']['total_annual_cost']:,.2f})",
            f"- **Lowest Stockout:** {scenario_summary['lowest_stockout_scenario']['scenario']} "
            f"({scenario_summary['lowest_stockout_scenario']['avg_stockout_probability']:.2%})",
            "",
            "## 8. Recommendations",
            "",
            f"- **Total:** {len(recs)}",
            f"- **Estimated Annual Cost Impact:** ${total_cost:,.2f}",
            "",
            "### By Priority",
            "",
        ]
    )

    for priority in ["Critical", "High", "Medium", "Low"]:
        count = priority_dist.get(priority, 0)
        if count > 0:
            lines.append(f"- {priority}: {count}")

    lines.extend(
        [
            "",
            "### By Type",
            "",
        ]
    )

    for rec_type, count in type_dist.items():
        lines.append(f"- {rec_type}: {count}")

    lines.extend(
        [
            "",
            "## 9. Key Assumptions & Limitations",
            "",
            "1. **All data is SYNTHETIC** — generated for development/testing only",
            "2. **Demand distribution:** Assumed normal (i.i.d. weekly)",
            "3. **Lead time:** Constant or normal distribution (specified per part)",
            "4. **Service level:** Target-based (95% default), z-score from normal",
            "5. **Safety stock:** Normal approximation (Z × σ_LTD)",
            "6. **Stockout probability:** P(LTD > ROP) under normal assumption",
            "7. **Fill rate:** 1 - E[shortage] / order quantity",
            "8. **EOQ:** Standard formula √(2DS/H) with assumed order cost",
            "9. **No supplier constraints:** Unlimited capacity, no MOQs",
            "10. **No perishability/obsolescence:** All parts have infinite shelf life",
            "11. **Scenario outputs are ESTIMATES** — not realized savings or costs",
            "12. **Holding cost rate:** 20% of unit cost per year (assumption)",
            "13. **Stockout cost:** 10x unit cost per stockout (assumption)",
            "",
            "---",
            "",
            "*End of Report*",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Main entry point."""
    run_inventory_pipeline()


if __name__ == "__main__":
    main()
