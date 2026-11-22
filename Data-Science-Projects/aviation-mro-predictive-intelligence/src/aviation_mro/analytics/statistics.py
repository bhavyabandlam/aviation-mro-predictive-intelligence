"""Statistical analysis methods for NASA C-MAPSS FD001 (SRS v1.1 FR-04).

Implements inferential statistics with explicit assumption checks:

- Pearson and Spearman correlation.
- Mann--Whitney U test for two independent groups.
- Chi-square test of independence for categorical associations.
- One-way ANOVA, gated on normality (Shapiro--Wilk) and homogeneity of
  variances (Levene); reports assumption violations instead of silently
  proceeding.
- Confidence intervals for means and medians (bootstrap).
- Effect sizes: Cohen's d, eta-squared, Cramér's V.
- Plain-language interpretation of results.

Every test returns a structured dict with the test name, statistic, p-value,
assumption results and a human-readable interpretation. No causal claims are
made from association measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """Structured result of a statistical test."""

    test: str
    statistic: float | None
    p_value: float | None
    alpha: float
    significant: bool | None
    assumption_passed: bool | None
    assumption_detail: str | None
    effect_size: float | None
    effect_size_interpretation: str | None
    interpretation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""

    method: str
    statistic: float
    p_value: float
    n: int
    interpretation: str


# ---------------------------------------------------------------------------
# Effect-size helpers
# ---------------------------------------------------------------------------
def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Cohen's d for two independent groups (pooled standard deviation)."""
    a = np.asarray(group_a.dropna().values, dtype=float)
    b = np.asarray(group_b.dropna().values, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude (standard thresholds)."""
    if np.isnan(d):
        return "undefined"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def cramers_v(contingency: pd.DataFrame) -> float:
    """Cramér's V for a contingency table (association strength)."""
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.values.sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0.0
    r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    k_corr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V (rough heuristic by df=1 benchmarks)."""
    if np.isnan(v):
        return "undefined"
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def eta_squared(groups: list[pd.Series]) -> float:
    """Eta-squared for one-way ANOVA (proportion of variance explained)."""
    all_vals = np.concatenate([np.asarray(g.dropna().values, dtype=float) for g in groups])
    grand_mean = np.mean(all_vals)
    ss_between = 0.0
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))
    for g in groups:
        vals = np.asarray(g.dropna().values, dtype=float)
        if len(vals) == 0:
            continue
        ss_between += len(vals) * (np.mean(vals) - grand_mean) ** 2
    if ss_total <= 0:
        return 0.0
    return float(ss_between / ss_total)


def interpret_eta_squared(eta2: float) -> str:
    """Interpret eta-squared magnitude."""
    if np.isnan(eta2):
        return "undefined"
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def mean_ci(
    series: pd.Series, alpha: float = 0.05, bootstrap_reps: int = 2000
) -> dict[str, float | None]:
    """Bootstrap confidence interval for the mean."""
    vals = np.asarray(series.dropna().values, dtype=float)
    n = len(vals)
    if n < 2:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n": n}
    mean_val = float(np.mean(vals))
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(bootstrap_reps):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"mean": mean_val, "ci_lower": lower, "ci_upper": upper, "n": n}


def median_ci(
    series: pd.Series, alpha: float = 0.05, bootstrap_reps: int = 2000
) -> dict[str, float | None]:
    """Bootstrap confidence interval for the median."""
    vals = np.asarray(series.dropna().values, dtype=float)
    n = len(vals)
    if n < 2:
        return {"median": None, "ci_lower": None, "ci_upper": None, "n": n}
    median_val = float(np.median(vals))
    rng = np.random.default_rng(42)
    boot_medians = []
    for _ in range(bootstrap_reps):
        sample = rng.choice(vals, size=n, replace=True)
        boot_medians.append(np.median(sample))
    lower = float(np.percentile(boot_medians, 100 * alpha / 2))
    upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return {"median": median_val, "ci_lower": lower, "ci_upper": upper, "n": n}


# ---------------------------------------------------------------------------
# Normality and homogeneity assumption checks
# ---------------------------------------------------------------------------
def check_normality(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """Normality check via Shapiro--Wilk (n <= 5000) or KS test (n > 5000).

    Returns whether the normality assumption appears met.
    """
    vals = np.asarray(series.dropna().values, dtype=float)
    n = len(vals)
    if n < 3:
        return {
            "test": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "passed": None,
            "detail": f"n={n} < 3; normality cannot be assessed",
        }
    if n <= 5000:
        stat, p = stats.shapiro(vals)
        test_name = "shapiro_wilk"
    else:
        stat, p = stats.kstest(
            vals, "norm", args=(float(np.mean(vals)), float(np.std(vals, ddof=1)))
        )
        test_name = "kolmogorov_smirnov"
    passed = bool(p > alpha)
    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "passed": passed,
        "detail": f"{test_name} p={p:.4g} vs alpha={alpha} -> "
        f"{'normal' if passed else 'non-normal'}",
    }


def check_homogeneity(groups: list[pd.Series], alpha: float = 0.05) -> dict[str, Any]:
    """Homogeneity of variances via Levene's test."""
    clean = [np.asarray(g.dropna().values, dtype=float) for g in groups if len(g.dropna()) > 1]
    if len(clean) < 2:
        return {
            "test": "insufficient_groups",
            "statistic": None,
            "p_value": None,
            "passed": None,
            "detail": "fewer than 2 groups with >1 observation",
        }
    stat, p = stats.levene(*clean)
    passed = bool(p > alpha)
    return {
        "test": "levene",
        "statistic": float(stat),
        "p_value": float(p),
        "passed": passed,
        "detail": f"Levene p={p:.4g} vs alpha={alpha} -> "
        f"{'homogeneous' if passed else 'heterogeneous'}",
    }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def pearson_correlation(x: pd.Series, y: pd.Series) -> CorrelationResult:
    """Pearson linear correlation with p-value and interpretation."""
    xv = x.dropna()
    yv = y.dropna()
    common = xv.index.intersection(yv.index)
    xa = np.asarray(xv.loc[common].values, dtype=float)
    ya = np.asarray(yv.loc[common].values, dtype=float)
    n = len(xa)
    if n < 3:
        return CorrelationResult("pearson", float("nan"), float("nan"), n, "insufficient data")
    if np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return CorrelationResult(
            "pearson",
            float("nan"),
            float("nan"),
            n,
            "constant input; correlation undefined",
        )
    stat, p = stats.pearsonr(xa, ya)
    return CorrelationResult(
        "pearson",
        float(stat),
        float(p),
        n,
        f"r={stat:.4g}, p={p:.4g}; "
        f"{'strong' if abs(stat) >= 0.7 else 'moderate' if abs(stat) >= 0.4 else 'weak'} "
        f"{'positive' if stat > 0 else 'negative'} linear association",
    )


def spearman_correlation(x: pd.Series, y: pd.Series) -> CorrelationResult:
    """Spearman rank correlation (monotonic association, robust to non-normality)."""
    xv = x.dropna()
    yv = y.dropna()
    common = xv.index.intersection(yv.index)
    xa = np.asarray(xv.loc[common].values, dtype=float)
    ya = np.asarray(yv.loc[common].values, dtype=float)
    n = len(xa)
    if n < 3:
        return CorrelationResult("spearman", float("nan"), float("nan"), n, "insufficient data")
    if np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return CorrelationResult(
            "spearman",
            float("nan"),
            float("nan"),
            n,
            "constant input; correlation undefined",
        )
    stat, p = stats.spearmanr(xa, ya)
    return CorrelationResult(
        "spearman",
        float(stat),
        float(p),
        n,
        f"rho={stat:.4g}, p={p:.4g}; "
        f"{'strong' if abs(stat) >= 0.7 else 'moderate' if abs(stat) >= 0.4 else 'weak'} "
        f"{'positive' if stat > 0 else 'negative'} monotonic association",
    )


# ---------------------------------------------------------------------------
# Two-group tests
# ---------------------------------------------------------------------------
def mann_whitney_u(group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05) -> TestResult:
    """Mann--Whitney U test for two independent samples (non-parametric).

    Appropriate when normality is not assumed. Tests whether one group tends
    to have larger values than the other.
    """
    a = np.asarray(group_a.dropna().values, dtype=float)
    b = np.asarray(group_b.dropna().values, dtype=float)
    if len(a) < 1 or len(b) < 1:
        return _empty_result("mann_whitney_u", alpha, "insufficient data in one or both groups")
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(group_a, group_b)
    passed = bool(p < alpha)
    sig_text = (
        "Statistically significant difference"
        if passed
        else "No statistically significant difference"
    )
    return TestResult(
        test="mann_whitney_u",
        statistic=float(stat),
        p_value=float(p),
        alpha=alpha,
        significant=passed,
        assumption_passed=True,  # non-parametric: no normality assumption required
        assumption_detail="non-parametric; no normality assumption required",
        effect_size=float(d) if not np.isnan(d) else None,
        effect_size_interpretation=interpret_cohens_d(d) if not np.isnan(d) else None,
        interpretation=(
            f"U={stat:.4g}, p={p:.4g} vs alpha={alpha}. {sig_text} "
            f"between groups (Mann--Whitney U). Effect size: "
            f"{interpret_cohens_d(d) if not np.isnan(d) else 'undefined'} (Cohen's d={d:.4g})."
        ),
    )


def independent_ttest(
    group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05, equal_var: bool = False
) -> TestResult:
    """Welch's t-test (default) or Student's t-test for two independent groups.

    Assumes approximate normality; check normality before use.
    """
    a = np.asarray(group_a.dropna().values, dtype=float)
    b = np.asarray(group_b.dropna().values, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return _empty_result("independent_ttest", alpha, "insufficient data")
    stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    d = cohens_d(group_a, group_b)
    passed = bool(p < alpha)
    sig_text = (
        "Statistically significant difference"
        if passed
        else "No statistically significant difference"
    )
    return TestResult(
        test="independent_ttest",
        statistic=float(stat),
        p_value=float(p),
        alpha=alpha,
        significant=passed,
        assumption_passed=None,  # caller must check normality separately
        assumption_detail="assumes approximate normality; verify with check_normality",
        effect_size=float(d) if not np.isnan(d) else None,
        effect_size_interpretation=interpret_cohens_d(d) if not np.isnan(d) else None,
        interpretation=(
            f"t={stat:.4g}, p={p:.4g} vs alpha={alpha} "
            f"({'Welch' if not equal_var else 'Student'} t-test, equal_var={equal_var}). "
            f"{sig_text} between groups. Effect size: "
            f"{interpret_cohens_d(d) if not np.isnan(d) else 'undefined'} (Cohen's d={d:.4g})."
        ),
    )


# ---------------------------------------------------------------------------
# Categorical association
# ---------------------------------------------------------------------------
def chi_square_independence(contingency: pd.DataFrame, alpha: float = 0.05) -> TestResult:
    """Chi-square test of independence for a contingency table.

    Warns if expected counts are small (< 5 in any cell), which violates
    the chi-square approximation.
    """
    observed = contingency.values
    n = observed.sum()
    if n == 0 or observed.size == 0:
        return _empty_result("chi_square_independence", alpha, "empty contingency table")
    chi2, p, dof, expected = stats.chi2_contingency(observed, correction=False)
    min_expected = float(np.min(expected))
    expected_ok = bool(min_expected >= 5)
    passed = bool(p < alpha)
    v = cramers_v(contingency)
    if expected_ok:
        assumption_text = "adequate expected counts"
    else:
        assumption_text = "small expected counts - chi-square approximation may be unreliable"
    assumption_detail = f"min expected count={min_expected:.2f}; {assumption_text}"
    sig_text = (
        "Statistically significant association"
        if passed
        else "No statistically significant association"
    )
    return TestResult(
        test="chi_square_independence",
        statistic=float(chi2),
        p_value=float(p),
        alpha=alpha,
        significant=passed,
        assumption_passed=expected_ok,
        assumption_detail=assumption_detail,
        effect_size=float(v) if not np.isnan(v) else None,
        effect_size_interpretation=interpret_cramers_v(v) if not np.isnan(v) else None,
        interpretation=(
            f"chi2={chi2:.4g}, p={p:.4g}, dof={dof} vs alpha={alpha}. {sig_text} "
            f"between variables (Chi-square). Effect size: "
            f"{interpret_cramers_v(v) if not np.isnan(v) else 'undefined'} (Cramér's V={v:.4g}). "
            f"{assumption_detail}."
        ),
        metadata={"dof": int(dof), "min_expected_count": min_expected},
    )


# ---------------------------------------------------------------------------
# ANOVA (one-way) with assumption gates
# ---------------------------------------------------------------------------
def one_way_anova(
    groups: list[pd.Series], group_names: list[str] | None = None, alpha: float = 0.05
) -> TestResult:
    """One-way ANOVA with normality and homogeneity assumption checks.

    If normality or homogeneity is violated, the result reports
    ``assumption_passed=False`` and the p-value should be treated with
    caution; consider a non-parametric alternative (e.g. Kruskal--Wallis).
    """
    clean = [np.asarray(g.dropna().values, dtype=float) for g in groups if len(g) > 0]
    if len(clean) < 2:
        return _empty_result("one_way_anova", alpha, "fewer than 2 non-empty groups")
    names = group_names or [f"group_{i}" for i in range(len(clean))]

    # Assumption checks
    normality = [check_normality(pd.Series(g), alpha) for g in clean]
    normality_passed = all(n["passed"] for n in normality if n["passed"] is not None)
    homogeneity = check_homogeneity([pd.Series(g) for g in clean], alpha)
    homogeneity_passed = bool(homogeneity["passed"]) if homogeneity["passed"] is not None else None

    assumption_passed = bool(normality_passed and homogeneity_passed)

    stat, p = stats.f_oneway(*clean)
    passed = bool(p < alpha)
    eta2 = eta_squared([pd.Series(g) for g in clean])
    sig_text = (
        "Statistically significant differences among groups"
        if passed
        else "No statistically significant differences among groups"
    )

    detail_parts = []
    for n_res, name in zip(normality, names, strict=False):
        detail_parts.append(f"{name}: {n_res['detail']}")
    detail_parts.append(homogeneity["detail"])
    assumption_detail = "; ".join(detail_parts)

    return TestResult(
        test="one_way_anova",
        statistic=float(stat),
        p_value=float(p),
        alpha=alpha,
        significant=passed,
        assumption_passed=assumption_passed,
        assumption_detail=assumption_detail,
        effect_size=float(eta2) if not np.isnan(eta2) else None,
        effect_size_interpretation=interpret_eta_squared(eta2) if not np.isnan(eta2) else None,
        interpretation=(
            f"F={stat:.4g}, p={p:.4g} vs alpha={alpha} "
            f"(one-way ANOVA, {len(clean)} groups). {sig_text}. "
            f"Effect size: "
            f"{interpret_eta_squared(eta2) if not np.isnan(eta2) else 'undefined'} "
            f"(eta-squared={eta2:.4g}). "
            f"Assumptions {'met' if assumption_passed else 'violated'}; {assumption_detail}."
        ),
    )


def kruskal_wallis(groups: list[pd.Series], alpha: float = 0.05) -> TestResult:
    """Kruskal--Wallis H test (non-parametric alternative to one-way ANOVA)."""
    clean = [np.asarray(g.dropna().values, dtype=float) for g in groups if len(g) > 0]
    if len(clean) < 2:
        return _empty_result("kruskal_wallis", alpha, "fewer than 2 non-empty groups")
    stat, p = stats.kruskal(*clean)
    passed = bool(p < alpha)
    sig_text = (
        "Statistically significant differences among groups"
        if passed
        else "No statistically significant differences among groups"
    )
    return TestResult(
        test="kruskal_wallis",
        statistic=float(stat),
        p_value=float(p),
        alpha=alpha,
        significant=passed,
        assumption_passed=True,
        assumption_detail="non-parametric; no normality or homogeneity assumption",
        effect_size=None,
        effect_size_interpretation=None,
        interpretation=(f"H={stat:.4g}, p={p:.4g} vs alpha={alpha} (Kruskal--Wallis). {sig_text}."),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _empty_result(test: str, alpha: float, detail: str) -> TestResult:
    return TestResult(
        test=test,
        statistic=None,
        p_value=None,
        alpha=alpha,
        significant=None,
        assumption_passed=None,
        assumption_detail=detail,
        effect_size=None,
        effect_size_interpretation=None,
        interpretation=f"Could not run {test}: {detail}",
    )


__all__ = [
    "TestResult",
    "CorrelationResult",
    "cohens_d",
    "interpret_cohens_d",
    "cramers_v",
    "interpret_cramers_v",
    "eta_squared",
    "interpret_eta_squared",
    "mean_ci",
    "median_ci",
    "check_normality",
    "check_homogeneity",
    "pearson_correlation",
    "spearman_correlation",
    "mann_whitney_u",
    "independent_ttest",
    "chi_square_independence",
    "one_way_anova",
    "kruskal_wallis",
]
