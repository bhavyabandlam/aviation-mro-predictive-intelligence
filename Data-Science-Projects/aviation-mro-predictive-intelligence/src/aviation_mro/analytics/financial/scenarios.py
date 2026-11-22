"""Cost Comparison and Scenario Analysis.

Performs scenario analysis and cost comparisons for MRO financial planning.
All calculations use SYNTHETIC data — not real airline records.

Scenarios include:
- Baseline (current state)
- Predictive maintenance adoption
- Inventory optimization
- Fleet growth
- Labor efficiency improvements
- Supplier cost changes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"


@dataclass
class ScenarioResult:
    """Result of a financial scenario."""

    scenario_name: str
    description: str
    total_annual_cost: float
    cost_breakdown: dict[str, float]
    cost_vs_baseline: float
    pct_change_vs_baseline: float
    key_assumptions: list[str]


class ScenarioAnalyzer:
    """Analyzes financial scenarios for MRO operations."""

    def __init__(self) -> None:
        pass

    def calculate_baseline_costs(
        self,
        labor_cost: float,
        parts_cost: float,
        downtime_cost: float,
        holding_cost: float,
        stockout_cost: float,
    ) -> dict[str, float]:
        """Calculate baseline cost breakdown."""
        return {
            "labor_cost": labor_cost,
            "parts_cost": parts_cost,
            "downtime_cost": downtime_cost,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "total": labor_cost + parts_cost + downtime_cost + holding_cost + stockout_cost,
        }

    def scenario_predictive_maintenance(
        self,
        baseline: dict[str, float],
        downtime_reduction: float = 0.20,
    ) -> ScenarioResult:
        """Scenario: Predictive maintenance reduces downtime."""
        new_downtime = baseline["downtime_cost"] * (1 - downtime_reduction)
        # Predictive maintenance may increase parts cost slightly (more proactive replacements)
        new_parts = baseline["parts_cost"] * 1.02
        # May reduce stockouts (better planning)
        new_stockout = baseline["stockout_cost"] * 0.85

        costs = {
            "labor_cost": baseline["labor_cost"],
            "parts_cost": new_parts,
            "downtime_cost": new_downtime,
            "holding_cost": baseline["holding_cost"],
            "stockout_cost": new_stockout,
        }
        total = sum(costs.values())

        return ScenarioResult(
            scenario_name="Predictive Maintenance",
            description=f"Predictive maintenance with {downtime_reduction:.0%} downtime reduction",
            total_annual_cost=total,
            cost_breakdown=costs,
            cost_vs_baseline=total - baseline["total"],
            pct_change_vs_baseline=(
                ((total - baseline["total"]) / baseline["total"]) * 100
                if baseline["total"] != 0
                else 0.0
            ),
            key_assumptions=[
                f"Downtime reduced by {downtime_reduction:.0%}",
                "Parts cost increases 2% (proactive replacements)",
                "Stockout cost reduced 15% (better planning)",
                "Labor cost unchanged",
                "No implementation cost included",
            ],
        )

    def scenario_inventory_optimization(
        self,
        baseline: dict[str, float],
        holding_cost_reduction: float = 0.15,
        stockout_cost_reduction: float = 0.25,
    ) -> ScenarioResult:
        """Scenario: Inventory optimization reduces holding and stockout costs."""
        new_holding = baseline["holding_cost"] * (1 - holding_cost_reduction)
        new_stockout = baseline["stockout_cost"] * (1 - stockout_cost_reduction)
        # May increase parts cost slightly (smaller, more frequent orders)
        new_parts = baseline["parts_cost"] * 1.03

        costs = {
            "labor_cost": baseline["labor_cost"],
            "parts_cost": new_parts,
            "downtime_cost": baseline["downtime_cost"],
            "holding_cost": new_holding,
            "stockout_cost": new_stockout,
        }
        total = sum(costs.values())

        return ScenarioResult(
            scenario_name="Inventory Optimization",
            description=(
                f"Optimized inventory with {holding_cost_reduction:.0%} holding and "
                f"{stockout_cost_reduction:.0%} stockout reduction"
            ),
            total_annual_cost=total,
            cost_breakdown=costs,
            cost_vs_baseline=total - baseline["total"],
            pct_change_vs_baseline=(
                ((total - baseline["total"]) / baseline["total"]) * 100
                if baseline["total"] != 0
                else 0.0
            ),
            key_assumptions=[
                f"Holding cost reduced by {holding_cost_reduction:.0%} (better EOQ, safety stock)",
                f"Stockout cost reduced by {stockout_cost_reduction:.0%} (improved availability)",
                "Parts cost increases 3% (more frequent orders)",
                "Labor and downtime unchanged",
            ],
        )

    def scenario_fleet_growth(
        self,
        baseline: dict[str, float],
        growth_rate: float = 0.15,
    ) -> ScenarioResult:
        """Scenario: Fleet growth increases all variable costs."""
        costs = {
            "labor_cost": baseline["labor_cost"] * (1 + growth_rate),
            "parts_cost": baseline["parts_cost"] * (1 + growth_rate),
            "downtime_cost": baseline["downtime_cost"] * (1 + growth_rate),
            "holding_cost": baseline["holding_cost"] * (1 + growth_rate),
            "stockout_cost": baseline["stockout_cost"] * (1 + growth_rate),
        }
        total = sum(costs.values())

        return ScenarioResult(
            scenario_name="Fleet Growth",
            description=f"Fleet size increases by {growth_rate:.0%}, proportional cost increase",
            total_annual_cost=total,
            cost_breakdown=costs,
            cost_vs_baseline=total - baseline["total"],
            pct_change_vs_baseline=(
                ((total - baseline["total"]) / baseline["total"]) * 100
                if baseline["total"] != 0
                else 0.0
            ),
            key_assumptions=[
                f"All variable costs scale with fleet size (+{growth_rate:.0%})",
                "No economies of scale assumed",
                "Labor, parts, downtime, holding, stockout all scale equally",
            ],
        )

    def scenario_labor_efficiency(
        self,
        baseline: dict[str, float],
        efficiency_gain: float = 0.10,
    ) -> ScenarioResult:
        """Scenario: Labor efficiency improvements reduce labor cost."""
        new_labor = baseline["labor_cost"] * (1 - efficiency_gain)
        # May reduce downtime slightly (faster turnaround)
        new_downtime = baseline["downtime_cost"] * (1 - efficiency_gain * 0.3)

        costs = {
            "labor_cost": new_labor,
            "parts_cost": baseline["parts_cost"],
            "downtime_cost": new_downtime,
            "holding_cost": baseline["holding_cost"],
            "stockout_cost": baseline["stockout_cost"],
        }
        total = sum(costs.values())

        return ScenarioResult(
            scenario_name="Labor Efficiency",
            description=f"Labor efficiency improvement of {efficiency_gain:.0%}",
            total_annual_cost=total,
            cost_breakdown=costs,
            cost_vs_baseline=total - baseline["total"],
            pct_change_vs_baseline=(
                ((total - baseline["total"]) / baseline["total"]) * 100
                if baseline["total"] != 0
                else 0.0
            ),
            key_assumptions=[
                f"Labor cost reduced by {efficiency_gain:.0%} (training, tools, processes)",
                f"Downtime reduced by {efficiency_gain * 0.3:.0%} (faster turnaround)",
                "Parts, holding, stockout costs unchanged",
            ],
        )

    def scenario_supplier_cost_change(
        self,
        baseline: dict[str, float],
        parts_cost_change: float = 0.10,
    ) -> ScenarioResult:
        """Scenario: Supplier cost changes affect parts and holding costs."""
        new_parts = baseline["parts_cost"] * (1 + parts_cost_change)
        # Holding cost scales with unit cost
        new_holding = baseline["holding_cost"] * (1 + parts_cost_change)
        # Stockout cost scales with unit cost
        new_stockout = baseline["stockout_cost"] * (1 + parts_cost_change)

        costs = {
            "labor_cost": baseline["labor_cost"],
            "parts_cost": new_parts,
            "downtime_cost": baseline["downtime_cost"],
            "holding_cost": new_holding,
            "stockout_cost": new_stockout,
        }
        total = sum(costs.values())

        direction = "increase" if parts_cost_change > 0 else "decrease"
        return ScenarioResult(
            scenario_name="Supplier Cost Change",
            description=f"Supplier parts cost {direction} by {abs(parts_cost_change):.0%}",
            total_annual_cost=total,
            cost_breakdown=costs,
            cost_vs_baseline=total - baseline["total"],
            pct_change_vs_baseline=(
                ((total - baseline["total"]) / baseline["total"]) * 100
                if baseline["total"] != 0
                else 0.0
            ),
            key_assumptions=[
                f"Parts unit cost changes by {parts_cost_change:.0%}",
                "Holding cost scales with unit cost",
                "Stockout cost scales with unit cost",
                "Labor and downtime unchanged",
            ],
        )

    def scenario_combined_optimization(
        self,
        baseline: dict[str, float],
    ) -> ScenarioResult:
        """Scenario: Combined optimization (predictive + inventory + labor)."""
        # Combined effects (not simply additive due to interactions)
        new_labor = baseline["labor_cost"] * 0.92  # 8% labor efficiency
        new_parts = baseline["parts_cost"] * 1.01  # Net +1% (predictive +2%, inventory +3%, offset)
        new_downtime = baseline["downtime_cost"] * 0.75  # 25% reduction (predictive + labor)
        new_holding = baseline["holding_cost"] * 0.80  # 20% inventory optimization
        new_stockout = baseline["stockout_cost"] * 0.70  # 30% reduction (predictive + inventory)

        costs = {
            "labor_cost": new_labor,
            "parts_cost": new_parts,
            "downtime_cost": new_downtime,
            "holding_cost": new_holding,
            "stockout_cost": new_stockout,
        }
        total = sum(costs.values())

        return ScenarioResult(
            scenario_name="Combined Optimization",
            description="Predictive maintenance + inventory optimization + labor efficiency",
            total_annual_cost=total,
            cost_breakdown=costs,
            cost_vs_baseline=total - baseline["total"],
            pct_change_vs_baseline=(
                ((total - baseline["total"]) / baseline["total"]) * 100
                if baseline["total"] != 0
                else 0.0
            ),
            key_assumptions=[
                "Labor efficiency: -8%",
                "Parts cost: +1% net (predictive +2%, inventory +3% offset)",
                "Downtime: -25% (predictive 20% + labor efficiency synergy)",
                "Holding cost: -20% (inventory optimization)",
                "Stockout cost: -30% (predictive + inventory synergy)",
                "Synergies assumed between initiatives",
            ],
        )

    def run_all_scenarios(
        self,
        baseline: dict[str, float],
    ) -> dict[str, Any]:
        """Run all predefined scenarios."""
        scenarios = {}

        scenarios["baseline"] = ScenarioResult(
            scenario_name="Baseline",
            description="Current state",
            total_annual_cost=baseline["total"],
            cost_breakdown=baseline,
            cost_vs_baseline=0.0,
            pct_change_vs_baseline=0.0,
            key_assumptions=["Current state"],
        )

        scenarios["predictive_maintenance"] = self.scenario_predictive_maintenance(baseline)
        scenarios["inventory_optimization"] = self.scenario_inventory_optimization(baseline)
        scenarios["fleet_growth"] = self.scenario_fleet_growth(baseline)
        scenarios["labor_efficiency"] = self.scenario_labor_efficiency(baseline)
        scenarios["supplier_cost_increase"] = self.scenario_supplier_cost_change(baseline, 0.10)
        scenarios["supplier_cost_decrease"] = self.scenario_supplier_cost_change(baseline, -0.05)
        scenarios["combined_optimization"] = self.scenario_combined_optimization(baseline)

        # Find best and worst scenarios (excluding baseline)
        non_baseline = {k: v for k, v in scenarios.items() if k != "baseline"}
        best = min(non_baseline.values(), key=lambda s: s.total_annual_cost)
        worst = max(non_baseline.values(), key=lambda s: s.total_annual_cost)

        return {
            "scenarios": scenarios,
            "baseline_total": baseline["total"],
            "best_scenario": best.scenario_name,
            "best_savings": baseline["total"] - best.total_annual_cost,
            "worst_scenario": worst.scenario_name,
            "worst_increase": worst.total_annual_cost - baseline["total"],
        }


def run_scenario_analysis(
    labor_cost: float,
    parts_cost: float,
    downtime_cost: float,
    holding_cost: float,
    stockout_cost: float,
) -> dict[str, Any]:
    """Convenience function to run all scenarios."""
    analyzer = ScenarioAnalyzer()
    baseline = analyzer.calculate_baseline_costs(
        labor_cost, parts_cost, downtime_cost, holding_cost, stockout_cost
    )
    return analyzer.run_all_scenarios(baseline)
