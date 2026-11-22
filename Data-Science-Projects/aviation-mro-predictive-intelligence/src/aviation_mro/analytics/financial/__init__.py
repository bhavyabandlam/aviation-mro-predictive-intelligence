"""Financial Analytics Module.

Financial analytics for MRO operations including:
- Maintenance labor costs
- Spare parts costs
- Downtime cost scenarios
- Inventory holding costs
- Stockout-related costs
- Financial KPIs
- Cost comparisons and scenario analysis
"""

from __future__ import annotations

from .cost_per_unit import (
    CostPerUnitCalculator,
    CostPerUnitResult,
    calculate_cost_per_unit,
)
from .data_prep import (
    generate_all_financial_data,
    generate_downtime_costs,
    generate_fleet_data,
    generate_inventory_cost_params,
    generate_maintenance_labor_costs,
    generate_shop_labor_data,
    load_all_financial_data,
    load_financial_data,
    save_financial_data,
)
from .downtime_costs import (
    DowntimeCostCalculator,
    DowntimeCostResult,
    DowntimeScenarioResult,
    calculate_downtime_costs,
)
from .holding_costs import (
    HoldingCostCalculator,
    HoldingCostResult,
    HoldingCostSummary,
    calculate_holding_costs,
)
from .kpis import (
    FinancialKPICalculator,
    FinancialKPIResult,
    calculate_financial_kpis,
)
from .labor_costs import (
    LaborCostCalculator,
    LaborCostResult,
    ShopLaborSummary,
    calculate_labor_costs,
)
from .parts_costs import (
    PartsCostCalculator,
    PartsCostResult,
    PartsCostSummary,
    calculate_parts_costs,
)
from .pipeline import (
    generate_financial_report,
    run_financial_pipeline,
    save_results,
)
from .planned_unplanned import (
    PlannedUnplannedAnalyzer,
    PlannedUnplannedResult,
    analyze_planned_unplanned,
)
from .scenarios import (
    ScenarioAnalyzer,
    ScenarioResult,
    run_scenario_analysis,
)
from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    run_sensitivity_analysis,
)
from .stockout_costs import (
    StockoutCostCalculator,
    StockoutCostResult,
    StockoutCostSummary,
    calculate_stockout_costs,
)

__all__ = [
    # Data prep
    "generate_all_financial_data",
    "generate_shop_labor_data",
    "generate_fleet_data",
    "generate_maintenance_labor_costs",
    "generate_downtime_costs",
    "generate_inventory_cost_params",
    "save_financial_data",
    "load_financial_data",
    "load_all_financial_data",
    # Labor costs
    "LaborCostCalculator",
    "LaborCostResult",
    "ShopLaborSummary",
    "calculate_labor_costs",
    # Parts costs
    "PartsCostCalculator",
    "PartsCostResult",
    "PartsCostSummary",
    "calculate_parts_costs",
    # Downtime costs
    "DowntimeCostCalculator",
    "DowntimeCostResult",
    "DowntimeScenarioResult",
    "calculate_downtime_costs",
    # Holding costs
    "HoldingCostCalculator",
    "HoldingCostResult",
    "HoldingCostSummary",
    "calculate_holding_costs",
    # Stockout costs
    "StockoutCostCalculator",
    "StockoutCostResult",
    "StockoutCostSummary",
    "calculate_stockout_costs",
    # KPIs
    "FinancialKPICalculator",
    "FinancialKPIResult",
    "calculate_financial_kpis",
    # Scenarios
    "ScenarioAnalyzer",
    "ScenarioResult",
    "run_scenario_analysis",
    # Planned vs Unplanned
    "PlannedUnplannedAnalyzer",
    "PlannedUnplannedResult",
    "analyze_planned_unplanned",
    # Cost per Unit
    "CostPerUnitCalculator",
    "CostPerUnitResult",
    "calculate_cost_per_unit",
    # Sensitivity Analysis
    "SensitivityAnalyzer",
    "SensitivityResult",
    "run_sensitivity_analysis",
    # Pipeline
    "run_financial_pipeline",
    "save_results",
    "generate_financial_report",
]
