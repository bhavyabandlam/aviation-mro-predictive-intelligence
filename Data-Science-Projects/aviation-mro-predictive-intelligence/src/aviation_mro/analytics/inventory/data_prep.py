"""Synthetic Inventory Data Generator.

Generates synthetic MRO inventory data for development and testing.
All data is SYNTHETIC — not real airline records.

Data includes:
- Part master data (part_number, family, shop, cost, lead time)
- Demand parameters (mean, std, service level target)
- Current stock levels
- Criticality classification
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Configuration
RANDOM_SEED = 42
N_PARTS = 150  # 150 parts across 5 shops × 3 families = 10 per shop-family
SHOPS = ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"]
PART_FAMILIES = ["ENGINE", "AVIONICS", "AIRFRAME"]
CRITICALITY_LEVELS = ["A", "B", "C"]  # A=Critical, B=Essential, C=Standard

# Part family demand characteristics (base weekly demand)
FAMILY_DEMAND_PARAMS = {
    "ENGINE": {"mean": 8.0, "cv": 0.4},  # Higher demand, moderate variability
    "AVIONICS": {"mean": 4.0, "cv": 0.6},  # Medium demand, higher variability
    "AIRFRAME": {"mean": 2.0, "cv": 0.8},  # Lower demand, high variability
}

# Shop-specific multipliers
SHOP_MULTIPLIERS = {
    "SHOP_01": 1.2,  # Large shop
    "SHOP_02": 1.0,  # Medium
    "SHOP_03": 0.9,  # Medium-small
    "SHOP_04": 0.8,  # Small
    "SHOP_05": 1.1,  # Medium-large
}

# Lead time by family (days)
FAMILY_LEAD_TIME = {
    "ENGINE": {"mean": 21, "std": 5},  # 3 weeks ± 5 days
    "AVIONICS": {"mean": 14, "std": 3},  # 2 weeks ± 3 days
    "AIRFRAME": {"mean": 10, "std": 2},  # 1.5 weeks ± 2 days
}

# Unit cost by family ($)
FAMILY_UNIT_COST = {
    "ENGINE": {"mean": 5000, "cv": 0.3},
    "AVIONICS": {"mean": 2000, "cv": 0.4},
    "AIRFRAME": {"mean": 500, "cv": 0.5},
}

# Service level targets by criticality
CRITICALITY_SERVICE_LEVEL = {
    "A": 0.99,  # Critical - 99%
    "B": 0.95,  # Essential - 95%
    "C": 0.90,  # Standard - 90%
}

# Current stock multipliers (relative to ROP)
CRITICALITY_STOCK_MULTIPLIER = {
    "A": 1.3,  # Critical parts well-stocked
    "B": 1.0,  # Essential at ROP
    "C": 0.8,  # Standard slightly below ROP
}


def generate_synthetic_inventory(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic inventory data.

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic inventory data
    """
    np.random.seed(seed)

    rows = []
    part_counter = 0

    for shop in SHOPS:
        for family in PART_FAMILIES:
            # Number of parts per shop-family
            n_parts_family = N_PARTS // (len(SHOPS) * len(PART_FAMILIES))
            # Distribute remainder
            if part_counter < N_PARTS % (len(SHOPS) * len(PART_FAMILIES)):
                n_parts_family += 1

            for _i in range(n_parts_family):
                part_counter += 1
                part_number = f"PART-{shop[-2:]}-{family[:3].upper()}-{part_counter:04d}"

                # Demand parameters
                fam_params = FAMILY_DEMAND_PARAMS[family]
                shop_mult = SHOP_MULTIPLIERS[shop]
                demand_mean = fam_params["mean"] * shop_mult
                demand_cv = fam_params["cv"]
                demand_std = demand_mean * demand_cv

                # Lead time
                lt_params = FAMILY_LEAD_TIME[family]
                lead_time_days = max(1, int(np.random.normal(lt_params["mean"], lt_params["std"])))
                lead_time_std_days = max(
                    0, int(np.random.normal(lt_params["std"] * 0.5, lt_params["std"] * 0.2))
                )

                # Unit cost
                cost_params = FAMILY_UNIT_COST[family]
                unit_cost = max(
                    10,
                    np.random.lognormal(
                        np.log(cost_params["mean"]) - 0.5 * np.log(1 + cost_params["cv"] ** 2),
                        np.sqrt(np.log(1 + cost_params["cv"] ** 2)),
                    ),
                )

                # Criticality (weighted: 20% A, 50% B, 30% C)
                criticality = np.random.choice(CRITICALITY_LEVELS, p=[0.2, 0.5, 0.3])
                service_level = CRITICALITY_SERVICE_LEVEL[criticality]

                # Calculate ROP for stock level reference
                z = stats.norm.ppf(service_level)
                lead_time_weeks = lead_time_days / 7.0
                mean_ltd = demand_mean * lead_time_weeks
                std_ltd = demand_std * np.sqrt(lead_time_weeks)
                rop = mean_ltd + z * std_ltd

                # Current stock (with some noise around target)
                stock_mult = CRITICALITY_STOCK_MULTIPLIER[criticality]
                current_stock = max(0, int(np.random.normal(rop * stock_mult, rop * 0.15)))

                rows.append(
                    {
                        "part_number": part_number,
                        "part_family": family,
                        "shop_id": shop,
                        "current_stock": current_stock,
                        "unit_cost": round(unit_cost, 2),
                        "lead_time_days": lead_time_days,
                        "lead_time_std_days": lead_time_std_days,
                        "demand_mean_weekly": round(demand_mean, 2),
                        "demand_std_weekly": round(demand_std, 2),
                        "service_level_target": service_level,
                        "criticality": criticality,
                        "source": "synthetic_inventory_v1",
                    }
                )

    df = pd.DataFrame(rows)
    return df


def save_inventory_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save inventory data to Parquet."""
    if path is None:
        path = DATA_DIR / "inventory_data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_inventory_data(path: Path | None = None) -> pd.DataFrame:
    """Load inventory data from Parquet."""
    if path is None:
        path = DATA_DIR / "inventory_data.parquet"
    return pd.read_parquet(path)


# Need to import stats for norm.ppf


def main() -> None:
    """Generate and save synthetic inventory data."""
    df = generate_synthetic_inventory()

    # Validate
    from .data_validation import validate_inventory_data

    validation = validate_inventory_data(df)
    if validation.warnings:
        for _w in validation.warnings:
            pass

    save_inventory_data(df)


if __name__ == "__main__":
    main()
