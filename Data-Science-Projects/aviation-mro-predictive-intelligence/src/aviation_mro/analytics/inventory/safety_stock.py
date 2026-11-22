"""Safety Stock Calculator.

Calculates safety stock using multiple methods for inventory management.
All calculations use SYNTHETIC data — not real airline records.

Methods:
1. Normal distribution (standard)
2. Normal with lead time variability
3. Log-normal (for positive demand only)
4. Empirical quantile (non-parametric)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports" / "inventory"


@dataclass
class SafetyStockResult:
    """Result of safety stock calculation."""

    part_number: str
    part_family: str
    shop_id: str
    method: str
    service_level: float
    safety_stock: float
    assumptions: list[str]
    parameters: dict[str, float]


class SafetyStockCalculator:
    """Calculates safety stock using multiple methods."""

    def __init__(self, default_service_level: float = 0.95):
        self.default_service_level = default_service_level

    def z_score(self, service_level: float) -> float:
        """Get z-score for given service level."""
        return float(stats.norm.ppf(service_level))

    def method_normal_constant_lt(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
        service_level: float,
    ) -> float:
        """
        Method 1: Normal distribution, constant lead time.

        SS = Z × σ_D × √L
        """
        lead_time_weeks = lead_time_days / 7.0
        std_ltd = std_weekly_demand * np.sqrt(lead_time_weeks)
        z = self.z_score(service_level)
        return float(z * std_ltd)

    def method_normal_variable_lt(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
        lead_time_std_days: float,
        service_level: float,
    ) -> float:
        """
        Method 2: Normal distribution, variable lead time.

        SS = Z × √(L × σ_D² + D² × σ_L²)
        Where L = mean lead time (weeks), σ_L = std lead time (weeks)
        """
        lead_time_weeks = lead_time_days / 7.0
        lead_time_std_weeks = lead_time_std_days / 7.0

        var_ltd = (
            lead_time_weeks * std_weekly_demand**2 + mean_weekly_demand**2 * lead_time_std_weeks**2
        )
        std_ltd = np.sqrt(var_ltd)

        z = self.z_score(service_level)
        return float(z * std_ltd)

    def method_lognormal(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
        service_level: float,
    ) -> float:
        """
        Method 3: Log-normal demand distribution.

        For strictly positive demand. Uses log-normal parameters.
        """
        if mean_weekly_demand <= 0:
            return self.method_normal_constant_lt(
                mean_weekly_demand, std_weekly_demand, lead_time_days, service_level
            )

        # Convert to log-normal parameters
        cv = std_weekly_demand / mean_weekly_demand
        sigma_ln = np.sqrt(np.log(1 + cv**2))
        mu_ln = np.log(mean_weekly_demand) - 0.5 * sigma_ln**2

        lead_time_weeks = lead_time_days / 7.0

        # Sum of log-normals approximated by log-normal (Fenton-Wilkinson)
        mu_sum = lead_time_weeks * mu_ln + 0.5 * lead_time_weeks * sigma_ln**2
        sigma_sum = np.sqrt(lead_time_weeks * sigma_ln**2)

        z = self.z_score(service_level)
        # Quantile of log-normal
        q = np.exp(mu_sum + z * sigma_sum)
        mean_ltd = np.exp(mu_sum + 0.5 * sigma_sum**2)

        return float(max(0.0, q - mean_ltd))

    def method_empirical_quantile(
        self,
        demand_samples: np.ndarray,
        lead_time_days: float,
        service_level: float,
        n_simulations: int = 10000,
    ) -> float:
        """
        Method 4: Empirical quantile via simulation.

        Simulates lead time demand from empirical demand distribution.
        """
        if len(demand_samples) < 2:
            return 0.0

        lead_time_weeks = lead_time_days / 7.0
        n_periods = int(np.ceil(lead_time_weeks))

        # Bootstrap simulation
        np.random.seed(42)
        simulated_ltd: list[float] = []

        for _ in range(n_simulations):
            # Sample demand for each week in lead time
            weekly_demands = np.random.choice(demand_samples, size=n_periods, replace=True)
            # Scale by fraction if lead time not integer weeks
            if lead_time_weeks != n_periods:
                weekly_demands[-1] *= lead_time_weeks - (n_periods - 1)
            simulated_ltd.append(float(weekly_demands.sum()))

        simulated_ltd_arr = np.array(simulated_ltd)
        mean_ltd = np.mean(simulated_ltd_arr)
        q = np.percentile(simulated_ltd_arr, service_level * 100)

        return float(max(0.0, q - mean_ltd))

    def calculate_all_methods(
        self,
        df: pd.DataFrame,
    ) -> dict[str, list[SafetyStockResult]]:
        """Calculate safety stock using all methods for comparison."""
        results: dict[str, list[SafetyStockResult]] = {
            "normal_constant_lt": [],
            "normal_variable_lt": [],
            "lognormal": [],
        }

        for _, row in df.iterrows():
            sl = row.get("service_level_target", self.default_service_level)

            # Method 1: Normal, constant lead time
            ss1 = self.method_normal_constant_lt(
                row["demand_mean_weekly"],
                row["demand_std_weekly"],
                row["lead_time_days"],
                sl,
            )
            results["normal_constant_lt"].append(
                SafetyStockResult(
                    part_number=row["part_number"],
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    method="normal_constant_lt",
                    service_level=sl,
                    safety_stock=ss1,
                    assumptions=[
                        "Weekly demand ~ Normal(μ, σ)",
                        "Lead time is constant",
                        "i.i.d. weekly demands",
                    ],
                    parameters={"z_score": self.z_score(sl)},
                )
            )

            # Method 2: Normal, variable lead time
            ss2 = self.method_normal_variable_lt(
                row["demand_mean_weekly"],
                row["demand_std_weekly"],
                row["lead_time_days"],
                row.get("lead_time_std_days", 0),
                sl,
            )
            results["normal_variable_lt"].append(
                SafetyStockResult(
                    part_number=row["part_number"],
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    method="normal_variable_lt",
                    service_level=sl,
                    safety_stock=ss2,
                    assumptions=[
                        "Weekly demand ~ Normal(μ, σ)",
                        "Lead time ~ Normal(L, σ_L)",
                        "Demand and lead time independent",
                    ],
                    parameters={"z_score": self.z_score(sl)},
                )
            )

            # Method 3: Log-normal
            ss3 = self.method_lognormal(
                row["demand_mean_weekly"],
                row["demand_std_weekly"],
                row["lead_time_days"],
                sl,
            )
            results["lognormal"].append(
                SafetyStockResult(
                    part_number=row["part_number"],
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    method="lognormal",
                    service_level=sl,
                    safety_stock=ss3,
                    assumptions=[
                        "Weekly demand ~ LogNormal",
                        "Lead time is constant",
                        "Demand strictly positive",
                    ],
                    parameters={"z_score": self.z_score(sl)},
                )
            )

        return results


def calculate_safety_stock(
    df: pd.DataFrame,
    method: str = "normal_constant_lt",
    default_service_level: float = 0.95,
) -> list[SafetyStockResult]:
    """Convenience function to calculate safety stock."""
    calculator = SafetyStockCalculator(default_service_level)
    all_results = calculator.calculate_all_methods(df)
    return all_results.get(method, [])
