"""Inventory Scenario Analysis.

Compares inventory scenarios for stockout risk vs holding-cost tradeoffs.
All calculations use SYNTHETIC data — not real airline records.

Scenarios:
- Baseline: current service level targets
- High service: increased service level (e.g., 95% → 99%)
- Low service: decreased service level (e.g., 95% → 90%)
- Optimized: safety stock adjusted to target service level

For each scenario, calculates:
- Safety stock, reorder point
- Annual holding cost
- Expected stockout probability and cost
- Total cost (holding + stockout)
- Clearly labelled as estimates
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"

# Default scenario service levels
BASELINE_SERVICE_LEVEL = 0.95
HIGH_SERVICE_LEVEL = 0.99
LOW_SERVICE_LEVEL = 0.90

# Cost assumptions
HOLDING_COST_RATE = 0.20  # 20% of unit cost per year
STOCKOUT_COST_MULTIPLIER = 10.0  # Stockout cost = 10x unit cost
WEEKS_PER_YEAR = 52.0
ORDER_COST = 50.0  # Fixed cost per order


@dataclass
class ScenarioResult:
    """Result for a single inventory scenario."""

    scenario_name: str
    service_level: float
    z_score: float
    total_safety_stock: float
    total_safety_stock_value: float
    annual_holding_cost: float
    avg_stockout_probability: float
    expected_stockouts_per_year: float
    annual_stockout_cost: float
    total_annual_cost: float  # holding + stockout
    parts_count: int
    assumptions: list[str]
    is_estimate: bool = True


class InventoryScenarioAnalyzer:
    """Compares inventory scenarios for cost/risk tradeoffs."""

    def __init__(
        self,
        holding_cost_rate: float = HOLDING_COST_RATE,
        stockout_cost_multiplier: float = STOCKOUT_COST_MULTIPLIER,
        weeks_per_year: float = WEEKS_PER_YEAR,
        order_cost: float = ORDER_COST,
    ):
        """
        Initialize analyzer.

        Args:
            holding_cost_rate: Annual holding cost as fraction of unit cost
            stockout_cost_multiplier: Multiplier for unit cost to estimate stockout cost
            weeks_per_year: Number of weeks in a year
            order_cost: Fixed cost per order
        """
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.weeks_per_year = weeks_per_year
        self.order_cost = order_cost

    def z_score(self, service_level: float) -> float:
        """Get z-score for service level."""
        if service_level <= 0 or service_level >= 1:
            raise ValueError("Service level must be between 0 and 1")
        return float(stats.norm.ppf(service_level))

    def calculate_scenario(
        self,
        df: pd.DataFrame,
        service_level: float,
        scenario_name: str,
        safety_stocks_by_part: dict[str, float] | None = None,
        stockout_risks_by_part: dict[str, Any] | None = None,
    ) -> ScenarioResult:
        """Calculate scenario metrics for all parts."""
        z = self.z_score(service_level)

        total_safety_stock = 0.0
        total_safety_stock_value = 0.0
        total_annual_holding_cost = 0.0
        total_expected_stockouts = 0.0
        total_annual_stockout_cost = 0.0
        stockout_probs = []
        parts_count = len(df)

        for _, row in df.iterrows():
            part_num = row["part_number"]
            unit_cost = row["unit_cost"]
            demand_mean = row["demand_mean_weekly"]
            demand_std = row["demand_std_weekly"]
            lead_time_days = row["lead_time_days"]
            lead_time_std_days = row.get("lead_time_std_days", 0)

            # Lead time demand
            lead_time_weeks = lead_time_days / 7.0

            # Safety stock for this scenario (normal, constant lead time)
            std_ltd = demand_std * np.sqrt(lead_time_weeks)
            safety_stock = z * std_ltd

            # Account for variable lead time if present
            if lead_time_std_days > 0:
                lt_std_weeks = lead_time_std_days / 7.0
                var_ltd = lead_time_weeks * demand_std**2 + demand_mean**2 * lt_std_weeks**2
                std_ltd = np.sqrt(var_ltd)
                safety_stock = z * std_ltd

            # Use provided safety stock if available (override)
            if safety_stocks_by_part and part_num in safety_stocks_by_part:
                safety_stock = safety_stocks_by_part[part_num]

            ss_value = safety_stock * unit_cost
            holding_cost = ss_value * self.holding_cost_rate

            # Stockout probability at this service level
            if std_ltd > 0:
                z_actual = (safety_stock) / std_ltd  # safety stock above mean
                stockout_prob = float(1 - stats.norm.cdf(z_actual))
            else:
                stockout_prob = 0.0

            stockout_probs.append(stockout_prob)

            # Order cycles per year (approx: annual demand / (4 weeks demand))
            order_quantity = max(demand_mean * 4, 1.0)
            cycles_per_year = (demand_mean * self.weeks_per_year) / order_quantity
            expected_stockouts = stockout_prob * cycles_per_year
            stockout_cost = expected_stockouts * unit_cost * self.stockout_cost_multiplier

            total_safety_stock += safety_stock
            total_safety_stock_value += ss_value
            total_annual_holding_cost += holding_cost
            total_expected_stockouts += expected_stockouts
            total_annual_stockout_cost += stockout_cost

        avg_stockout_prob = float(np.mean(stockout_probs)) if stockout_probs else 0.0
        total_annual_cost = total_annual_holding_cost + total_annual_stockout_cost

        assumptions = [
            f"Service level: {service_level:.0%} (z-score: {z:.3f})",
            f"Holding cost rate: {self.holding_cost_rate:.0%} per year",
            f"Stockout cost multiplier: {self.stockout_cost_multiplier}x unit cost",
            f"Weeks per year: {self.weeks_per_year}",
            f"Order cost: ${self.order_cost:.2f}",
            "Lead time demand ~ Normal(mean, std)",
            "Safety stock = Z × σ_LTD (constant lead time)",
            "Stockout cost = expected stockouts × unit cost × multiplier",
            "Total cost = annual holding cost + annual stockout cost",
            "ESTIMATE — synthetic data, not realized business impact",
        ]

        return ScenarioResult(
            scenario_name=scenario_name,
            service_level=service_level,
            z_score=z,
            total_safety_stock=total_safety_stock,
            total_safety_stock_value=total_safety_stock_value,
            annual_holding_cost=total_annual_holding_cost,
            avg_stockout_probability=avg_stockout_prob,
            expected_stockouts_per_year=total_expected_stockouts,
            annual_stockout_cost=total_annual_stockout_cost,
            total_annual_cost=total_annual_cost,
            parts_count=parts_count,
            assumptions=assumptions,
            is_estimate=True,
        )

    def compare_scenarios(
        self,
        df: pd.DataFrame,
        safety_stocks_by_part: dict[str, float] | None = None,
        stockout_risks_by_part: dict[str, Any] | None = None,
    ) -> list[ScenarioResult]:
        """Compare multiple service-level scenarios."""
        scenarios = [
            (BASELINE_SERVICE_LEVEL, "Baseline (current service levels)"),
            (HIGH_SERVICE_LEVEL, "High Service (99%)"),
            (LOW_SERVICE_LEVEL, "Low Service (90%)"),
        ]

        results = []
        for service_level, name in scenarios:
            result = self.calculate_scenario(
                df,
                service_level,
                name,
                safety_stocks_by_part,
                stockout_risks_by_part,
            )
            results.append(result)

        return results


def run_scenario_comparison(
    df: pd.DataFrame,
    safety_stocks_by_part: dict[str, float] | None = None,
    stockout_risks_by_part: dict[str, Any] | None = None,
    holding_cost_rate: float = HOLDING_COST_RATE,
    stockout_cost_multiplier: float = STOCKOUT_COST_MULTIPLIER,
) -> list[ScenarioResult]:
    """Convenience function to run scenario comparison."""
    analyzer = InventoryScenarioAnalyzer(
        holding_cost_rate=holding_cost_rate,
        stockout_cost_multiplier=stockout_cost_multiplier,
    )
    return analyzer.compare_scenarios(df, safety_stocks_by_part, stockout_risks_by_part)


def summarize_scenarios(results: list[ScenarioResult]) -> dict[str, Any]:
    """Summarize scenario comparison results."""
    if not results:
        return {
            "n_scenarios": 0,
            "baseline": None,
            "best_total_cost_scenario": None,
            "lowest_stockout_scenario": None,
            "lowest_holding_cost_scenario": None,
            "scenarios": [],
        }

    # Find best scenario by total cost
    best_cost = min(results, key=lambda r: r.total_annual_cost)
    lowest_stockout = min(results, key=lambda r: r.avg_stockout_probability)
    lowest_holding = min(results, key=lambda r: r.annual_holding_cost)

    baseline = next((r for r in results if "Baseline" in r.scenario_name), results[0])

    return {
        "n_scenarios": len(results),
        "baseline": {
            "scenario": baseline.scenario_name,
            "service_level": baseline.service_level,
            "total_annual_cost": baseline.total_annual_cost,
            "annual_holding_cost": baseline.annual_holding_cost,
            "annual_stockout_cost": baseline.annual_stockout_cost,
            "avg_stockout_probability": baseline.avg_stockout_probability,
            "expected_stockouts_per_year": baseline.expected_stockouts_per_year,
        },
        "best_total_cost_scenario": {
            "scenario": best_cost.scenario_name,
            "service_level": best_cost.service_level,
            "total_annual_cost": best_cost.total_annual_cost,
        },
        "lowest_stockout_scenario": {
            "scenario": lowest_stockout.scenario_name,
            "service_level": lowest_stockout.service_level,
            "avg_stockout_probability": lowest_stockout.avg_stockout_probability,
        },
        "lowest_holding_cost_scenario": {
            "scenario": lowest_holding.scenario_name,
            "service_level": lowest_holding.service_level,
            "annual_holding_cost": lowest_holding.annual_holding_cost,
        },
        "scenarios": [
            {
                "scenario": r.scenario_name,
                "service_level": r.service_level,
                "z_score": round(r.z_score, 3),
                "total_safety_stock": round(r.total_safety_stock, 1),
                "total_safety_stock_value": round(r.total_safety_stock_value, 2),
                "annual_holding_cost": round(r.annual_holding_cost, 2),
                "avg_stockout_probability": round(r.avg_stockout_probability, 4),
                "expected_stockouts_per_year": round(r.expected_stockouts_per_year, 2),
                "annual_stockout_cost": round(r.annual_stockout_cost, 2),
                "total_annual_cost": round(r.total_annual_cost, 2),
                "is_estimate": r.is_estimate,
            }
            for r in results
        ],
    }
