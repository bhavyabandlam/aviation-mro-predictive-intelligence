"""Inventory Analytics Module.

Implements inventory management analytics for Aviation MRO:
- Inventory data validation
- Parts demand analysis
- Reorder point calculations
- Safety stock calculations
- Lead-time demand estimation
- Stockout-risk analysis
- Inventory recommendations with documented assumptions

All data is SYNTHETIC (source: synthetic_inventory_v1) — not real airline records.
"""

from .data_prep import (
    generate_synthetic_inventory,
    load_inventory_data,
    save_inventory_data,
)
from .data_validation import (
    InventoryDataValidator,
    ValidationResult,
    validate_inventory_data,
)
from .demand_analysis import (
    DemandStatistics,
    PartsDemandAnalyzer,
    analyze_parts_demand,
)
from .lead_time_demand import (
    LeadTimeDemandEstimator,
    LeadTimeDemandResult,
    estimate_lead_time_demand,
)
from .pipeline import (
    run_inventory_pipeline,
)
from .recommendations import (
    InventoryRecommendation,
    InventoryRecommendationEngine,
    generate_inventory_recommendations,
)
from .reorder_point import (
    ReorderPointCalculator,
    ReorderPointResult,
    calculate_reorder_points,
)
from .safety_stock import (
    SafetyStockCalculator,
    SafetyStockResult,
    calculate_safety_stock,
)
from .stockout_risk import (
    StockoutRiskAnalyzer,
    StockoutRiskResult,
    analyze_stockout_risk,
)

__all__ = [
    # Data validation
    "InventoryDataValidator",
    "ValidationResult",
    "validate_inventory_data",
    # Data preparation
    "generate_synthetic_inventory",
    "save_inventory_data",
    "load_inventory_data",
    # Demand analysis
    "PartsDemandAnalyzer",
    "DemandStatistics",
    "analyze_parts_demand",
    # Reorder point
    "ReorderPointCalculator",
    "ReorderPointResult",
    "calculate_reorder_points",
    # Safety stock
    "SafetyStockCalculator",
    "SafetyStockResult",
    "calculate_safety_stock",
    # Lead-time demand
    "LeadTimeDemandEstimator",
    "LeadTimeDemandResult",
    "estimate_lead_time_demand",
    # Stockout risk
    "StockoutRiskAnalyzer",
    "StockoutRiskResult",
    "analyze_stockout_risk",
    # Recommendations
    "InventoryRecommendationEngine",
    "InventoryRecommendation",
    "generate_inventory_recommendations",
    # Pipeline
    "run_inventory_pipeline",
]
