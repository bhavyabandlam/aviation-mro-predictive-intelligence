"""Financial Sensitivity Analysis.

Varies key financial assumptions independently and measures the impact on
total annual maintenance cost and ROI. All outputs are estimates based on
synthetic data — not real airline results.

Parameters varied (each independently, one at a time):
- Downtime reduction % (predictive maintenance effectiveness)
- Holding cost rate (annual fraction of unit cost)
- Stockout cost multiplier (x unit cost per stockout event)
- Base labor rate ($/hour across all shops)
- Parts unit cost (across all parts)

Variations: -30%, -20%, -10%, +10%, +20%, +30% relative to baseline.

All assumptions are explicitly documented; scenario outputs are labelled
estimates, not realised savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "financial"

# Sensitivity variation levels (relative to baseline).
VARIATION_LEVELS = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]

# Parameters that can be varied, with their baseline values and descriptions.
SENSITIVITY_PARAMETERS = {
    "downtime_reduction": {
        "label": "Predictive Maintenance Downtime Reduction",
        "description": "Fraction of downtime cost avoided by predictive maintenance",
        "unit": "fraction (e.g. 0.20 = 20%)",
        "baseline": 0.20,
    },
    "holding_cost_rate": {
        "label": "Inventory Holding Cost Rate",
        "description": "Annual holding cost as fraction of unit inventory value",
        "unit": "fraction per year",
        "baseline": 0.20,
    },
    "stockout_cost_multiplier": {
        "label": "Stockout Cost Multiplier",
        "description": "Multiple of unit cost charged per stockout event",
        "unit": "multiplier (e.g. 10 = 10x unit cost)",
        "baseline": 10.0,
    },
    "base_labor_rate": {
        "label": "Base Labor Rate",
        "description": "Average shop labor rate ($/hour)",
        "unit": "USD/hour",
        "baseline": 140.0,
    },
    "parts_unit_cost": {
        "label": "Parts Unit Cost",
        "description": "Average spare parts unit cost",
        "unit": "USD/unit",
        "baseline": 1000.0,
    },
}


@dataclass
class SensitivityResult:
    """Result of varying one sensitivity parameter."""

    parameter: str
    parameter_label: str
    baseline_value: float
    varied_value: float
    variation_pct: float
    baseline_total_cost: float
    varied_total_cost: float
    cost_change_usd: float
    cost_change_pct: float
    assumption: str


class SensitivityAnalyzer:
    """Performs one-at-a-time sensitivity analysis on financial parameters."""

    def __init__(self) -> None:
        pass

    def _varied_value(self, baseline: float, variation: float) -> float:
        """Apply a relative variation to a baseline value."""
        return baseline * (1 + variation)

    def analyze_parameter(
        self,
        parameter: str,
        baseline_value: float,
        baseline_total_cost: float,
        cost_impact_factor: float,
        baseline_roi: float | None = None,
    ) -> list[SensitivityResult]:
        """Analyze sensitivity of total cost to one parameter.

        Args:
            parameter: Parameter name (must be in SENSITIVITY_PARAMETERS)
            baseline_value: Baseline parameter value
            baseline_total_cost: Baseline total annual maintenance cost
            cost_impact_factor: Fraction of total cost affected by this parameter
                (e.g. 0.20 means 20% of total cost varies with this parameter)
            baseline_roi: Optional baseline ROI for context

        Returns:
            List of SensitivityResult, one per variation level
        """
        param_info = SENSITIVITY_PARAMETERS[parameter]
        results: list[SensitivityResult] = []

        for variation in VARIATION_LEVELS:
            varied_value = self._varied_value(baseline_value, variation)
            # Linear impact: only the affected portion of cost changes
            affected_cost = baseline_total_cost * cost_impact_factor
            varied_affected_cost = affected_cost * self._varied_value(1.0, variation)
            varied_total_cost = baseline_total_cost - affected_cost + varied_affected_cost

            cost_change_usd = varied_total_cost - baseline_total_cost
            cost_change_pct = (
                (cost_change_usd / baseline_total_cost * 100) if baseline_total_cost > 0 else 0.0
            )

            results.append(
                SensitivityResult(
                    parameter=parameter,
                    parameter_label=cast(str, param_info["label"]),
                    baseline_value=round(baseline_value, 4),
                    varied_value=round(varied_value, 4),
                    variation_pct=round(variation * 100, 1),
                    baseline_total_cost=round(baseline_total_cost, 2),
                    varied_total_cost=round(varied_total_cost, 2),
                    cost_change_usd=round(cost_change_usd, 2),
                    cost_change_pct=round(cost_change_pct, 2),
                    assumption=(
                        f"{param_info['label']}: baseline {baseline_value} "
                        f"({param_info['unit']}), varied by {variation:+.0%} to {varied_value}; "
                        f"{cost_impact_factor:.0%} of total cost is affected by this parameter; "
                        "linear sensitivity assumption; synthetic data estimate"
                    ),
                )
            )

        return results

    def analyze_all(
        self,
        baseline_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Run sensitivity analysis on all parameters.

        Args:
            baseline_results: Output from run_financial_pipeline (dict with
                financial_kpis, labor_costs, parts_costs, downtime_costs,
                holding_costs, stockout_costs, scenario_analysis)

        Returns:
            Dict with sensitivity results per parameter and summary
        """
        kpis = baseline_results.get("financial_kpis", {})
        cost_breakdown = kpis.get("cost_breakdown", {})
        baseline_total_cost = cost_breakdown.get("total_maintenance_cost", 0.0)

        # Determine what fraction of total cost each parameter affects
        # (based on cost breakdown categories)
        total = baseline_total_cost if baseline_total_cost > 0 else 1.0

        parameter_impacts = {
            "downtime_reduction": cost_breakdown.get("downtime_cost", 0.0) / total,
            "holding_cost_rate": cost_breakdown.get("holding_cost", 0.0) / total,
            "stockout_cost_multiplier": cost_breakdown.get("stockout_cost", 0.0) / total,
            "base_labor_rate": cost_breakdown.get("labor_cost", 0.0) / total,
            "parts_unit_cost": cost_breakdown.get("parts_cost", 0.0) / total,
        }

        baseline_values = {
            "downtime_reduction": SENSITIVITY_PARAMETERS["downtime_reduction"]["baseline"],
            "holding_cost_rate": SENSITIVITY_PARAMETERS["holding_cost_rate"]["baseline"],
            "stockout_cost_multiplier": SENSITIVITY_PARAMETERS["stockout_cost_multiplier"][
                "baseline"
            ],
            "base_labor_rate": SENSITIVITY_PARAMETERS["base_labor_rate"]["baseline"],
            "parts_unit_cost": SENSITIVITY_PARAMETERS["parts_unit_cost"]["baseline"],
        }

        all_results: dict[str, list[SensitivityResult]] = {}
        for parameter in SENSITIVITY_PARAMETERS:
            all_results[parameter] = self.analyze_parameter(
                parameter=parameter,
                baseline_value=cast(float, baseline_values[parameter]),
                baseline_total_cost=baseline_total_cost,
                cost_impact_factor=parameter_impacts[parameter],
            )

        # Find most sensitive parameter (largest absolute cost change at ±30%)
        most_sensitive = None
        max_impact = 0.0
        for parameter, results in all_results.items():
            for r in results:
                if abs(r.variation_pct) == 30.0 and abs(r.cost_change_usd) > max_impact:
                    max_impact = abs(r.cost_change_usd)
                    most_sensitive = parameter

        return {
            "baseline_total_cost": round(baseline_total_cost, 2),
            "parameter_impacts": {k: round(v, 4) for k, v in parameter_impacts.items()},
            "results": {k: [r.__dict__ for r in v] for k, v in all_results.items()},
            "most_sensitive_parameter": most_sensitive,
            "max_impact_usd": round(max_impact, 2),
            "assumptions": [
                "One-at-a-time sensitivity analysis: each parameter varied independently",
                "All other parameters held at baseline values",
                "Linear impact assumption: cost change scales proportionally with parameter",
                "Variation levels: -30%, -20%, -10%, +10%, +20%, +30%",
                "Parameter impact fractions derived from synthetic cost breakdown",
                "All outputs are ESTIMATES based on synthetic data (seed=42)",
                "Not realised savings — do not present as actual airline financial results",
            ],
        }


def run_sensitivity_analysis(
    baseline_results: dict[str, Any],
) -> dict[str, Any]:
    """Convenience function to run full sensitivity analysis."""
    analyzer = SensitivityAnalyzer()
    return analyzer.analyze_all(baseline_results)


if __name__ == "__main__":
    pass
