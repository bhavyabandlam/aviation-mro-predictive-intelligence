"""Lead-Time Demand Estimation.

Estimates demand during lead time for inventory management.
All calculations use SYNTHETIC data — not real airline records.

Key concepts:
- Lead Time Demand (LTD) = Demand over the replenishment lead time
- For constant lead time L (weeks): LTD = Σ D_i for i=1 to L
- Mean LTD = μ_D × L
- Var(LTD) = L × σ_D² (if demand i.i.d.)
- With variable lead time: Var(LTD) = E[L] × σ_D² + μ_D² × Var(L)
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
class LeadTimeDemandResult:
    """Result of lead-time demand estimation."""

    part_number: str
    part_family: str
    shop_id: str
    lead_time_weeks: float
    mean_weekly_demand: float
    std_weekly_demand: float
    mean_ltd: float
    std_ltd: float
    distribution: str
    parameters: dict[str, float]
    percentiles: dict[str, float]
    assumptions: list[str]


class LeadTimeDemandEstimator:
    """Estimates lead-time demand distributions."""

    def __init__(self) -> None:
        pass

    def estimate_normal_constant_lt(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
    ) -> tuple[float, float]:
        """Estimate LTD mean and std assuming normal demand, constant lead time."""
        lead_time_weeks = lead_time_days / 7.0
        mean_ltd = mean_weekly_demand * lead_time_weeks
        std_ltd = std_weekly_demand * np.sqrt(lead_time_weeks)
        return mean_ltd, std_ltd

    def estimate_normal_variable_lt(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
        lead_time_std_days: float,
    ) -> tuple[float, float]:
        """Estimate LTD mean and std with variable lead time."""
        lead_time_weeks = lead_time_days / 7.0
        lead_time_std_weeks = lead_time_std_days / 7.0

        mean_ltd = mean_weekly_demand * lead_time_weeks

        # Var(LTD) = E[L] × Var(D) + E[D]² × Var(L)
        var_ltd = (
            lead_time_weeks * std_weekly_demand**2 + mean_weekly_demand**2 * lead_time_std_weeks**2
        )
        std_ltd = np.sqrt(var_ltd)

        return mean_ltd, std_ltd

    def estimate_lognormal(
        self,
        mean_weekly_demand: float,
        std_weekly_demand: float,
        lead_time_days: float,
    ) -> tuple[float, float, float, float]:
        """
        Estimate log-normal LTD parameters.

        Returns: (mu_ln, sigma_ln, mean_ltd, std_ltd) in normal space
        """
        if mean_weekly_demand <= 0:
            mean_ltd, std_ltd = self.estimate_normal_constant_lt(
                mean_weekly_demand, std_weekly_demand, lead_time_days
            )
            return 0.0, 0.0, mean_ltd, std_ltd

        cv = std_weekly_demand / mean_weekly_demand
        sigma_ln_weekly = np.sqrt(np.log(1 + cv**2))
        mu_ln_weekly = np.log(mean_weekly_demand) - 0.5 * sigma_ln_weekly**2

        lead_time_weeks = lead_time_days / 7.0

        # Fenton-Wilkinson approximation for sum of log-normals
        mu_sum = lead_time_weeks * mu_ln_weekly + 0.5 * lead_time_weeks * sigma_ln_weekly**2
        sigma_sum = np.sqrt(lead_time_weeks * sigma_ln_weekly**2)

        mean_ltd = np.exp(mu_sum + 0.5 * sigma_sum**2)
        std_ltd = np.sqrt((np.exp(sigma_sum**2) - 1) * np.exp(2 * mu_sum + sigma_sum**2))

        return mu_sum, sigma_sum, mean_ltd, std_ltd

    def compute_percentiles(
        self,
        distribution: str,
        params: dict[str, float],
        percentiles: list[float] | None = None,
    ) -> dict[str, float]:
        """Compute percentiles for the given distribution."""
        if percentiles is None:
            percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        result = {}

        for p in percentiles:
            if distribution == "normal":
                val = stats.norm.ppf(p, loc=params["mean"], scale=params["std"])
            elif distribution == "lognormal":
                val = stats.lognorm.ppf(p, s=params["sigma"], scale=np.exp(params["mu"]))
            else:
                val = float("nan")
            result[f"p{int(p * 100)}"] = float(val)

        return result

    def estimate_for_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[LeadTimeDemandResult]:
        """Estimate LTD for all parts in DataFrame."""
        results = []

        for _, row in df.iterrows():
            lt_days = row["lead_time_days"]
            lt_std_days = row.get("lead_time_std_days", 0)

            # Choose method based on data characteristics
            mean_d = row["demand_mean_weekly"]
            std_d = row["demand_std_weekly"]

            # Method 1: Normal, constant lead time
            mean_ltd, std_ltd = self.estimate_normal_constant_lt(mean_d, std_d, lt_days)
            norm_params = {"mean": mean_ltd, "std": std_ltd}
            norm_pctls = self.compute_percentiles("normal", norm_params)

            # Method 2: Normal, variable lead time (if lead time std provided)
            if lt_std_days > 0:
                mean_ltd_var, std_ltd_var = self.estimate_normal_variable_lt(
                    mean_d, std_d, lt_days, lt_std_days
                )
                var_params = {"mean": mean_ltd_var, "std": std_ltd_var}
                var_pctls = self.compute_percentiles("normal", var_params)
            else:
                mean_ltd_var, std_ltd_var = mean_ltd, std_ltd
                var_pctls = norm_pctls

            # Method 3: Log-normal (if demand strictly positive)
            if mean_d > 0:
                mu_ln, sigma_ln, mean_ltd_ln, std_ltd_ln = self.estimate_lognormal(
                    mean_d, std_d, lt_days
                )
                ln_params = {"mu": mu_ln, "sigma": sigma_ln}
                ln_pctls = self.compute_percentiles("lognormal", ln_params)
            else:
                mu_ln = sigma_ln = 0.0
                ln_pctls = {}

            # Combine results (use normal constant LT as primary)
            combined_pctls = norm_pctls.copy()
            for k, v in var_pctls.items():
                combined_pctls[f"{k}_var_lt"] = v
            for k, v in ln_pctls.items():
                combined_pctls[f"{k}_lognormal"] = v

            assumptions = [
                "Weekly demand is i.i.d. (assumption for synthetic data)",
                "Lead time in days converted to weeks (÷7)",
                f"Constant lead time: {lt_days} days = {lt_days / 7:.1f} weeks",
                "Demand variance adds over lead time periods",
            ]
            if lt_std_days > 0:
                assumptions.append(f"Variable lead time: σ={lt_std_days} days")

            results.append(
                LeadTimeDemandResult(
                    part_number=row["part_number"],
                    part_family=row["part_family"],
                    shop_id=row["shop_id"],
                    lead_time_weeks=lt_days / 7.0,
                    mean_weekly_demand=mean_d,
                    std_weekly_demand=std_d,
                    mean_ltd=mean_ltd,
                    std_ltd=std_ltd,
                    distribution="normal",
                    parameters={
                        "mean": mean_ltd,
                        "std": std_ltd,
                        "mean_var_lt": mean_ltd_var,
                        "std_var_lt": std_ltd_var,
                        "lognormal_mu": mu_ln,
                        "lognormal_sigma": sigma_ln,
                    },
                    percentiles=combined_pctls,
                    assumptions=assumptions,
                )
            )

        return results


def estimate_lead_time_demand(df: pd.DataFrame) -> list[LeadTimeDemandResult]:
    """Convenience function to estimate lead-time demand."""
    estimator = LeadTimeDemandEstimator()
    return estimator.estimate_for_dataframe(df)
