"""Synthetic Financial Data Generator.

Generates synthetic MRO financial data for development and testing.
All data is SYNTHETIC — not real airline records.

Data includes:
- Shop labor rates and capacity
- Aircraft fleet composition
- Maintenance event costs
- Downtime cost parameters
- Inventory cost parameters
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Configuration
RANDOM_SEED = 42
N_SHOPS = 5
N_FLEET_TYPES = 3  # Narrow-body, Wide-body, Regional

SHOPS = ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"]
PART_FAMILIES = ["ENGINE", "AVIONICS", "AIRFRAME"]
FLEET_TYPES = ["NARROW_BODY", "WIDE_BODY", "REGIONAL"]

# Shop labor rates ($/hour) - varies by shop size/complexity
SHOP_LABOR_RATES = {
    "SHOP_01": {"base_rate": 150.0, "overtime_multiplier": 1.5, "technicians": 45},
    "SHOP_02": {"base_rate": 140.0, "overtime_multiplier": 1.5, "technicians": 38},
    "SHOP_03": {"base_rate": 135.0, "overtime_multiplier": 1.5, "technicians": 32},
    "SHOP_04": {"base_rate": 130.0, "overtime_multiplier": 1.5, "technicians": 28},
    "SHOP_05": {"base_rate": 145.0, "overtime_multiplier": 1.5, "technicians": 40},
}

# Fleet composition
FLEET_COMPOSITION = {
    "NARROW_BODY": {"count": 45, "avg_annual_flight_hours": 2800, "revenue_per_hour": 8500},
    "WIDE_BODY": {"count": 18, "avg_annual_flight_hours": 3200, "revenue_per_hour": 18000},
    "REGIONAL": {"count": 22, "avg_annual_flight_hours": 2200, "revenue_per_hour": 3200},
}

# Maintenance labor hours by part family (mean hours per job)
LABOR_HOURS_BY_FAMILY = {
    "ENGINE": {"mean": 48.0, "cv": 0.35},  # Complex, high variability
    "AVIONICS": {"mean": 12.0, "cv": 0.25},  # Medium complexity
    "AIRFRAME": {"mean": 8.0, "cv": 0.30},  # Lower complexity
}

# Downtime cost parameters ($/hour of aircraft ground time)
DOWNTIME_COST_PER_HOUR = {
    "NARROW_BODY": 12000,  # Includes lost revenue, passenger handling, etc.
    "WIDE_BODY": 28000,
    "REGIONAL": 5000,
}

# Downtime duration by part family (hours)
DOWNTIME_BY_FAMILY = {
    "ENGINE": {"mean": 72, "cv": 0.40},  # 3 days average
    "AVIONICS": {"mean": 24, "cv": 0.30},  # 1 day average
    "AIRFRAME": {"mean": 16, "cv": 0.35},  # <1 day average
}

# Inventory holding cost rate (annual fraction of unit cost)
HOLDING_COST_RATE = 0.20  # 20% per year

# Stockout cost multiplier (times unit cost)
STOCKOUT_COST_MULTIPLIER = 10.0

# Order cost per purchase order
ORDER_COST = 75.0


def generate_shop_labor_data(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate shop labor rate and capacity data."""
    np.random.seed(seed)

    rows = []
    for shop in SHOPS:
        params = SHOP_LABOR_RATES[shop]
        base_rate = params["base_rate"]
        # Add some variation
        actual_rate = max(100, np.random.normal(base_rate, base_rate * 0.05))
        overtime_rate = actual_rate * params["overtime_multiplier"]

        # Annual capacity (hours)
        regular_hours_per_tech = 2080  # 52 weeks * 40 hours
        overtime_hours_per_tech = 400  # ~10% overtime
        total_regular_capacity = params["technicians"] * regular_hours_per_tech
        total_overtime_capacity = params["technicians"] * overtime_hours_per_tech

        rows.append(
            {
                "shop_id": shop,
                "base_labor_rate": round(actual_rate, 2),
                "overtime_labor_rate": round(overtime_rate, 2),
                "technician_count": params["technicians"],
                "annual_regular_capacity_hours": total_regular_capacity,
                "annual_overtime_capacity_hours": total_overtime_capacity,
                "total_annual_capacity_hours": total_regular_capacity + total_overtime_capacity,
                "source": "synthetic_financial_v1",
            }
        )

    return pd.DataFrame(rows)


def generate_fleet_data(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate fleet composition and revenue data."""
    np.random.seed(seed + 1)

    rows = []
    for fleet_type in FLEET_TYPES:
        params = FLEET_COMPOSITION[fleet_type]
        # Add variation to counts
        count = max(1, int(np.random.normal(params["count"], params["count"] * 0.1)))
        flight_hours = max(
            1000,
            int(
                np.random.normal(
                    params["avg_annual_flight_hours"], params["avg_annual_flight_hours"] * 0.1
                )
            ),
        )
        revenue_per_hour = max(
            1000, np.random.normal(params["revenue_per_hour"], params["revenue_per_hour"] * 0.08)
        )

        rows.append(
            {
                "fleet_type": fleet_type,
                "aircraft_count": count,
                "avg_annual_flight_hours_per_aircraft": flight_hours,
                "total_annual_flight_hours": count * flight_hours,
                "revenue_per_flight_hour": round(revenue_per_hour, 2),
                "total_annual_revenue_potential": count * flight_hours * revenue_per_hour,
                "source": "synthetic_financial_v1",
            }
        )

    return pd.DataFrame(rows)


def generate_maintenance_labor_costs(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate maintenance labor cost parameters by shop and part family."""
    np.random.seed(seed + 2)

    rows = []
    for shop in SHOPS:
        shop_rate = SHOP_LABOR_RATES[shop]["base_rate"]
        for family in PART_FAMILIES:
            labor_params = LABOR_HOURS_BY_FAMILY[family]
            mean_hours = labor_params["mean"]
            cv = labor_params["cv"]
            # Log-normal distribution for labor hours
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(mean_hours) - 0.5 * sigma**2
            hours = max(1, np.random.lognormal(mu, sigma))

            labor_cost = hours * shop_rate

            rows.append(
                {
                    "shop_id": shop,
                    "part_family": family,
                    "mean_labor_hours_per_job": round(hours, 1),
                    "labor_cost_per_job": round(labor_cost, 2),
                    "labor_rate_used": round(shop_rate, 2),
                    "source": "synthetic_financial_v1",
                }
            )

    return pd.DataFrame(rows)


def generate_downtime_costs(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate downtime cost parameters by fleet type and part family."""
    np.random.seed(seed + 3)

    rows = []
    for fleet_type in FLEET_TYPES:
        dt_cost_per_hour = DOWNTIME_COST_PER_HOUR[fleet_type]
        for family in PART_FAMILIES:
            dt_params = DOWNTIME_BY_FAMILY[family]
            mean_hours = dt_params["mean"]
            cv = dt_params["cv"]
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(mean_hours) - 0.5 * sigma**2
            downtime_hours = max(1, np.random.lognormal(mu, sigma))

            downtime_cost = downtime_hours * dt_cost_per_hour

            rows.append(
                {
                    "fleet_type": fleet_type,
                    "part_family": family,
                    "mean_downtime_hours": round(downtime_hours, 1),
                    "downtime_cost_per_hour": dt_cost_per_hour,
                    "downtime_cost_per_event": round(downtime_cost, 2),
                    "source": "synthetic_financial_v1",
                }
            )

    return pd.DataFrame(rows)


def generate_inventory_cost_params(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate inventory cost parameters."""
    return pd.DataFrame(
        [
            {
                "holding_cost_rate": HOLDING_COST_RATE,
                "stockout_cost_multiplier": STOCKOUT_COST_MULTIPLIER,
                "order_cost": ORDER_COST,
                "currency": "USD",
                "source": "synthetic_financial_v1",
            }
        ]
    )


def generate_all_financial_data(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Generate all synthetic financial datasets."""
    return {
        "shop_labor": generate_shop_labor_data(seed),
        "fleet": generate_fleet_data(seed),
        "maintenance_labor": generate_maintenance_labor_costs(seed),
        "downtime_costs": generate_downtime_costs(seed),
        "inventory_cost_params": generate_inventory_cost_params(seed),
    }


def save_financial_data(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Save financial data to parquet files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in data.items():
        path = DATA_DIR / f"financial_{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path

    return paths


def load_financial_data(name: str) -> pd.DataFrame:
    """Load financial data from parquet."""
    path = DATA_DIR / f"financial_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Financial data file not found: {path}")
    return pd.read_parquet(path)


def load_all_financial_data() -> dict[str, pd.DataFrame]:
    """Load all financial datasets."""
    names = ["shop_labor", "fleet", "maintenance_labor", "downtime_costs", "inventory_cost_params"]
    return {name: load_financial_data(name) for name in names}


if __name__ == "__main__":
    data = generate_all_financial_data()
    paths = save_financial_data(data)
    for _name, _path in paths.items():
        pass
