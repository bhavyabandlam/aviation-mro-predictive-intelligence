"""Central KPI registry (SRS v1.1 §11).

Provides a single source of truth for every KPI used by the Aviation MRO
Predictive Intelligence platform. Each KPI has:

* A plain-language definition
* An explicit formula
* A unit of measure
* A data source (NASA synthetic, BTS public, or project-generated synthetic MRO)
* Provenance metadata (who/what produced it, how it's calculated)
* Null policy (what happens when inputs are missing or zero)
* A synthetic-data disclosure

All MRO business KPIs use SYNTHETIC data — not real airline records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "KPIDefinition",
    "KPIRegistry",
    "register_all_kpis",
    "get_kpi",
    "list_kpis",
]

# ---------------------------------------------------------------------------
# Synthetic-data disclosure
# ---------------------------------------------------------------------------

SYNTHETIC_DISCLOSURE = (
    "This KPI is computed from project-generated synthetic MRO business data "
    "(seed=42). It is an estimate for demonstration purposes, not a realised "
    "airline metric."
)

NASA_SOURCE = "NASA C-MAPSS FD001 (simulated engine degradation benchmark)"
SYNTHETIC_MRO_SOURCE = "Project-generated synthetic MRO business data (seed=42)"
SYNTHETIC_FIN_SOURCE = "Project-generated synthetic MRO financial data (seed=42)"


@dataclass
class KPIDefinition:
    """Central definition of one KPI."""

    kpi_id: str
    name: str
    definition: str
    formula: str
    unit: str
    data_source: str
    provenance: str
    null_policy: str
    synthetic_disclosure: str
    assumptions: list[str] = field(default_factory=list)
    example_value: Any = None


class KPIRegistry:
    """Central registry of all KPI definitions."""

    def __init__(self) -> None:
        self._kpis: dict[str, KPIDefinition] = {}
        self._register_defaults()

    # -- registration ------------------------------------------------------

    def _register_defaults(self) -> None:
        """Register all KPIs from SRS §11 plus financial KPIs."""

        # --- SRS §11 KPIs (maintenance / inventory) ----------------------

        self.register(
            KPIDefinition(
                kpi_id="total_engines",
                name="Total Engines",
                definition=("Count of distinct engine IDs in the selected data source and scope."),
                formula="COUNT(DISTINCT engine_id)",
                unit="engines",
                data_source=NASA_SOURCE,
                provenance=(
                    "Computed from the engines table in the SQLite metadata "
                    "store. Distinct engine_id values across all rows."
                ),
                null_policy=(
                    "Returns 0 when no engines are present. NULL engine_id "
                    "values are excluded from the count."
                ),
                synthetic_disclosure=(
                    "Engine IDs are from the NASA C-MAPSS FD001 simulated "
                    "benchmark (synthetic engine degradation data), not real "
                    "fleet records."
                ),
                assumptions=[
                    "Each engine_id uniquely identifies one engine.",
                    "Scope is determined by dashboard filters.",
                ],
                example_value=100,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="high_risk_engines",
                name="High-Risk Engines",
                definition=(
                    "Count of engines whose predicted RUL or anomaly score "
                    "exceeds the configured project risk threshold."
                ),
                formula=(
                    "COUNT(engine_id WHERE predicted_rul <= risk_threshold "
                    "OR anomaly_score <= anomaly_threshold)"
                ),
                unit="engines",
                data_source=NASA_SOURCE,
                provenance=(
                    "Computed from engine_predictions and anomaly_results "
                    "tables. Thresholds are configurable project settings, "
                    "not regulatory limits."
                ),
                null_policy=(
                    "Returns 0 when no predictions or anomalies exist. "
                    "Engines with NULL predictions are excluded."
                ),
                synthetic_disclosure=(
                    "Predictions and scores are from models trained on "
                    "simulated NASA FD001 synthetic data, not real fleet "
                    "records."
                ),
                assumptions=[
                    "Risk threshold is configurable (default: RUL ≤ 20 cycles).",
                    "Anomaly threshold from IsolationForest default (-0.07955).",
                ],
                example_value=12,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="mean_predicted_rul",
                name="Mean Predicted RUL",
                definition=(
                    "Mean of valid (non-null, non-negative) model predictions "
                    "for remaining useful life, expressed in engine cycles."
                ),
                formula="AVG(predicted_rul WHERE predicted_rul IS NOT NULL AND predicted_rul >= 0)",
                unit="cycles",
                data_source=NASA_SOURCE,
                provenance=(
                    "Computed from the engine_predictions table. Uses the "
                    "baseline_rul_model_v2 (GradientBoostingRegressor) "
                    "predictions on FD001 test engines."
                ),
                null_policy=(
                    "Returns NULL when no valid predictions exist. "
                    "Negative or NaN predictions are excluded."
                ),
                synthetic_disclosure=(
                    "RUL predictions are from a model trained on simulated "
                    "NASA FD001 synthetic data, not real fleet records."
                ),
                assumptions=[
                    "Predictions are per-engine at the final observed cycle.",
                    "RUL capped at 125 during training (modelling choice).",
                ],
                example_value=68.5,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="maintenance_expenditure",
                name="Maintenance Expenditure",
                definition=(
                    "Sum of total_cost for selected work orders within the selected time window."
                ),
                formula="SUM(total_cost WHERE status IN ('completed', 'open'))",
                unit="USD",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from the maintenance_orders table in the "
                    "SQLite store. Includes labour_cost, material_cost and "
                    "associated overhead."
                ),
                null_policy=(
                    "Returns 0 when no matching work orders exist. "
                    "NULL total_cost values are treated as 0."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Time window is configurable (default: trailing 12 months).",
                    "Only work orders with status 'completed' or 'open' counted.",
                ],
                example_value=2450000.0,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="completion_rate",
                name="Work Order Completion Rate",
                definition=(
                    "Ratio of completed work orders to eligible work orders "
                    "(opened and due within the selected window)."
                ),
                formula=(
                    "COUNT(work_order_id WHERE status = 'completed') / "
                    "COUNT(work_order_id WHERE opened_at IS NOT NULL)"
                ),
                unit="%",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from the maintenance_orders table. "
                    "Multiplied by 100 for percentage display."
                ),
                null_policy=(
                    "Returns 0 when no eligible work orders exist. Division by zero is guarded."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Eligible = opened_at is not NULL.",
                    "Completed = status equals 'completed'.",
                ],
                example_value=0.87,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="downtime_hours",
                name="Downtime Hours",
                definition=(
                    "Sum of valid downtime durations (end_at − start_at) "
                    "for selected downtime events."
                ),
                formula="SUM((end_at - start_at) / 3600)",
                unit="hours",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from the downtime_events table. Durations are "
                    "derived from start_at and end_at timestamps."
                ),
                null_policy=(
                    "Returns 0 when no downtime events exist. "
                    "Events with NULL start_at or end_at are excluded."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Durations are computed in seconds, then divided by 3600.",
                    "Only events with both timestamps present are counted.",
                ],
                example_value=184.5,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="inventory_value",
                name="Inventory Value",
                definition=(
                    "Sum of current_stock × unit_cost across all inventory "
                    "parts in the selected scope."
                ),
                formula="SUM(current_stock × unit_cost)",
                unit="USD",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from the inventory_parts table. Multiplies "
                    "current stock quantity by unit cost for each part."
                ),
                null_policy=(
                    "Returns 0 when no parts exist. NULL current_stock or unit_cost treated as 0."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "current_stock is the on-hand quantity at snapshot time.",
                    "unit_cost is the last known purchase price.",
                ],
                example_value=1850000.0,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="stockout_risk",
                name="Stockout Risk",
                definition=(
                    "Count of parts whose projected stock (current_stock − "
                    "expected demand during lead time) breaches the configured "
                    "minimum_stock requirement."
                ),
                formula=(
                    "COUNT(part_id WHERE current_stock - demand_during_lead_time <= minimum_stock)"
                ),
                unit="parts",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from inventory_parts and demand forecasts. "
                    "Uses the same safety-stock logic as FR-09 reorder point "
                    "calculations."
                ),
                null_policy=(
                    "Returns 0 when no parts exist. Parts with NULL current_stock are excluded."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Demand during lead time = weekly demand × lead_time_weeks.",
                    "minimum_stock is configurable per part.",
                ],
                example_value=23,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="technician_utilisation",
                name="Technician Utilisation",
                definition=(
                    "Ratio of assigned labour hours to available technician capacity hours."
                ),
                formula=("SUM(assigned_labour_hours) / SUM(available_hours) × 100"),
                unit="%",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from the technicians table and maintenance order labour assignments."
                ),
                null_policy=(
                    "Returns 0 when no technicians exist or capacity is 0. "
                    "Division by zero is guarded."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Available hours include regular + overtime capacity.",
                    "Assigned hours from completed work orders only.",
                ],
                example_value=0.78,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="inventory_turnover",
                name="Inventory Turnover",
                definition=(
                    "Ratio of annual parts consumption value to average "
                    "inventory value, over a defined period."
                ),
                formula="Annual Parts Cost / Average Inventory Value",
                unit="times/year",
                data_source=SYNTHETIC_MRO_SOURCE,
                provenance=(
                    "Computed from inventory_transactions (annual parts "
                    "cost) and inventory_parts (average inventory value). "
                    "Matches financial.kpis.calculate_inventory_turnover."
                ),
                null_policy=(
                    "Returns 0 when average inventory value is 0. Division by zero is guarded."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Period is trailing 12 months.",
                    "Average inventory value = (opening + closing) / 2.",
                ],
                example_value=3.2,
            )
        )

        # --- Financial KPIs (from src/aviation_mro/analytics/financial/kpis.py) ----

        self.register(
            KPIDefinition(
                kpi_id="cost_per_flight_hour",
                name="Cost per Flight Hour",
                definition=("Total annual maintenance cost divided by total annual flight hours."),
                formula="Total Annual Maintenance Cost / Total Annual Flight Hours",
                unit="USD/flight_hour",
                data_source=SYNTHETIC_FIN_SOURCE,
                provenance=(
                    "Computed from financial.kpis.FinancialKPICalculator. "
                    "Aggregates labour, parts, downtime, holding and "
                    "stockout costs."
                ),
                null_policy=(
                    "Returns 0 when total flight hours is 0. Division by zero is guarded."
                ),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Flight hours from synthetic fleet data.",
                    "Maintenance cost includes all five cost categories.",
                ],
                example_value=47.50,
            )
        )

        self.register(
            KPIDefinition(
                kpi_id="maintenance_cost_pct_revenue",
                name="Maintenance Cost % Revenue",
                definition=(
                    "Total annual maintenance cost as a percentage of total annual revenue."
                ),
                formula="(Total Annual Maintenance Cost / Total Annual Revenue) × 100",
                unit="%",
                data_source=SYNTHETIC_FIN_SOURCE,
                provenance=(
                    "Computed from financial.kpis.FinancialKPICalculator. "
                    "Revenue from synthetic fleet revenue potential."
                ),
                null_policy=("Returns 0 when total revenue is 0. Division by zero is guarded."),
                synthetic_disclosure=SYNTHETIC_DISCLOSURE,
                assumptions=[
                    "Revenue based on fleet flight hours × revenue per hour.",
                    "Maintenance cost includes all cost categories.",
                ],
                example_value=12.3,
            )
        )

    # -- API --------------------------------------------------------------

    def register(self, kpi: KPIDefinition) -> None:
        """Register or replace a KPI definition."""
        self._kpis[kpi.kpi_id] = kpi

    def get(self, kpi_id: str) -> KPIDefinition | None:
        """Return the KPI definition for *kpi_id*, or None if not registered."""
        return self._kpis.get(kpi_id)

    def list(self) -> list[KPIDefinition]:
        """Return all registered KPI definitions."""
        return list(self._kpis.values())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return all KPI definitions as dictionaries (for export/reporting)."""
        return {
            kpi_id: {
                "name": kpi.name,
                "definition": kpi.definition,
                "formula": kpi.formula,
                "unit": kpi.unit,
                "data_source": kpi.data_source,
                "provenance": kpi.provenance,
                "null_policy": kpi.null_policy,
                "synthetic_disclosure": kpi.synthetic_disclosure,
                "assumptions": kpi.assumptions,
                "example_value": kpi.example_value,
            }
            for kpi_id, kpi in self._kpis.items()
        }


# Module-level singleton registry.
_registry: KPIRegistry | None = None


def get_registry() -> KPIRegistry:
    """Return the module-level KPI registry singleton."""
    global _registry
    if _registry is None:
        _registry = KPIRegistry()
    return _registry


def get_kpi(kpi_id: str) -> KPIDefinition | None:
    """Convenience: return a KPI definition by ID."""
    return get_registry().get(kpi_id)


def list_kpis() -> list[KPIDefinition]:
    """Convenience: return all registered KPI definitions."""
    return get_registry().list()


def register_all_kpis() -> KPIRegistry:
    """Convenience: ensure all default KPIs are registered and return registry."""
    return get_registry()
