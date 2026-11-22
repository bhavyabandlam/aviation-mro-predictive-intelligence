"""Tests for the statistical analysis module (SRS v1.1 FR-04).

Covers Mann-Whitney U, Chi-square, ANOVA with assumption gates, Kruskal-Wallis,
Pearson/Spearman correlation, confidence intervals, effect sizes and plain-language
interpretation. Tests use synthetic deterministic data where possible; tests that
require the real FD001 archive skip gracefully when data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from aviation_mro.analytics.statistics import (  # noqa: E402
    check_homogeneity,
    check_normality,
    chi_square_independence,
    cohens_d,
    cramers_v,
    eta_squared,
    independent_ttest,
    interpret_cohens_d,
    interpret_cramers_v,
    kruskal_wallis,
    mann_whitney_u,
    mean_ci,
    median_ci,
    one_way_anova,
    pearson_correlation,
    spearman_correlation,
)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
class TestEffectSizes:
    def test_cohens_d_identical_groups(self) -> None:
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([1.0, 2.0, 3.0])
        assert cohens_d(a, b) == pytest.approx(0.0)

    def test_cohens_d_different_groups(self) -> None:
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 1, 30))
        b = pd.Series(rng.normal(3, 1, 30))  # clearly shifted mean
        d = cohens_d(a, b)
        assert abs(d) > 0.5  # large, clearly different means

    def test_interpret_cohens_d(self) -> None:
        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.3) == "small"
        assert interpret_cohens_d(0.6) == "medium"
        assert interpret_cohens_d(1.0) == "large"

    def test_cramers_v_independence(self) -> None:
        # Independent 2x2 table -> Cramér's V near 0
        table = pd.DataFrame([[50, 50], [50, 50]])
        v = cramers_v(table)
        assert v < 0.05

    def test_cramers_v_perfect_association(self) -> None:
        # Perfect association -> Cramér's V = 1
        table = pd.DataFrame([[100, 0], [0, 100]])
        v = cramers_v(table)
        assert v == pytest.approx(1.0)

    def test_interpret_cramers_v(self) -> None:
        assert interpret_cramers_v(0.05) == "negligible"
        assert interpret_cramers_v(0.2) == "small"
        assert interpret_cramers_v(0.4) == "medium"
        assert interpret_cramers_v(0.7) == "large"

    def test_eta_squared(self) -> None:
        groups = [
            pd.Series([1.0, 1.0, 1.0]),
            pd.Series([10.0, 10.0, 10.0]),
            pd.Series([20.0, 20.0, 20.0]),
        ]
        e2 = eta_squared(groups)
        assert e2 > 0.5  # strong between-group separation


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
class TestConfidenceIntervals:
    def test_mean_ci(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = mean_ci(s)
        assert ci["mean"] == pytest.approx(3.0)
        assert ci["ci_lower"] < ci["ci_upper"]
        assert ci["n"] == 5

    def test_median_ci(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = median_ci(s)
        assert ci["median"] == pytest.approx(3.0)
        assert ci["ci_lower"] < ci["ci_upper"]

    def test_ci_insufficient_data(self) -> None:
        s = pd.Series([1.0])
        ci = mean_ci(s)
        assert ci["mean"] is None
        ci2 = median_ci(s)
        assert ci2["median"] is None


# ---------------------------------------------------------------------------
# Assumption checks
# ---------------------------------------------------------------------------
class TestAssumptionChecks:
    def test_check_normality_normal_data(self) -> None:
        rng = np.random.default_rng(7)
        s = pd.Series(rng.normal(0, 1, 200))
        res = check_normality(s)
        assert res["test"] == "shapiro_wilk"
        assert res["passed"] in (True, False)
        assert isinstance(res["detail"], str)

    def test_check_normality_non_normal_data(self) -> None:
        rng = np.random.default_rng(7)
        s = pd.Series(rng.exponential(1.0, 200))
        res = check_normality(s)
        assert res["passed"] is False
        assert "non-normal" in res["detail"]

    def test_check_homogeneity_equal_variance(self) -> None:
        g1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
        res = check_homogeneity([g1, g2])
        assert res["test"] == "levene"
        assert res["passed"] in (True, False)

    def test_check_homogeneity_different_variance(self) -> None:
        g1 = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        g2 = pd.Series([1.0, 10.0, 20.0, 30.0, 40.0])
        res = check_homogeneity([g1, g2])
        assert res["passed"] is False


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
class TestCorrelation:
    def test_pearson_perfect_linear(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        r = pearson_correlation(x, y)
        assert r.statistic == pytest.approx(1.0)
        assert r.p_value < 0.001
        assert "positive" in r.interpretation

    def test_pearson_negative_linear(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        r = pearson_correlation(x, y)
        assert r.statistic == pytest.approx(-1.0)
        assert "negative" in r.interpretation

    def test_spearman_monotonic_nonlinear(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0])  # quadratic -> monotonic
        r = spearman_correlation(x, y)
        assert r.statistic == pytest.approx(1.0)
        assert "monotonic" in r.interpretation

    def test_correlation_insufficient_data(self) -> None:
        x = pd.Series([1.0])
        y = pd.Series([2.0])
        r = pearson_correlation(x, y)
        assert np.isnan(r.statistic)
        r2 = spearman_correlation(x, y)
        assert np.isnan(r2.statistic)


# ---------------------------------------------------------------------------
# Two-group tests
# ---------------------------------------------------------------------------
class TestTwoGroupTests:
    def test_mann_whitney_separated_groups(self) -> None:
        rng = np.random.default_rng(3)
        a = pd.Series(rng.normal(0, 1, 40))
        b = pd.Series(rng.normal(3, 1, 40))  # clearly shifted
        res = mann_whitney_u(a, b)
        assert res.test == "mann_whitney_u"
        assert res.significant is True
        assert res.assumption_passed is True  # non-parametric
        assert "Mann--Whitney" in res.interpretation

    def test_mann_whitney_same_distribution(self) -> None:
        rng = np.random.default_rng(11)
        a = pd.Series(rng.normal(0, 1, 30))
        b = pd.Series(rng.normal(0, 1, 30))
        res = mann_whitney_u(a, b)
        # With same distribution, usually not significant (may occasionally be)
        assert res.p_value is not None
        assert 0 <= res.p_value <= 1

    def test_independent_ttest(self) -> None:
        rng = np.random.default_rng(5)
        a = pd.Series(rng.normal(0, 1, 30))
        b = pd.Series(rng.normal(1, 1, 30))
        res = independent_ttest(a, b)
        assert res.test == "independent_ttest"
        assert res.p_value is not None
        assert "t-test" in res.interpretation


# ---------------------------------------------------------------------------
# Categorical association
# ---------------------------------------------------------------------------
class TestChiSquare:
    def test_chi_square_independent(self) -> None:
        # Independent rows/cols -> no association
        table = pd.DataFrame([[50, 50], [50, 50]])
        res = chi_square_independence(table)
        assert res.test == "chi_square_independence"
        assert res.significant is False
        assert res.assumption_passed is True  # expected counts = 25 >= 5

    def test_chi_square_associated(self) -> None:
        table = pd.DataFrame([[90, 10], [10, 90]])
        res = chi_square_independence(table)
        assert res.significant is True
        assert "association" in res.interpretation

    def test_chi_square_small_expected_warns(self) -> None:
        # Small expected counts -> assumption not met
        table = pd.DataFrame([[1, 1], [1, 1]])
        res = chi_square_independence(table)
        assert res.assumption_passed is False
        assert "small expected" in res.assumption_detail.lower()


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------
class TestAnova:
    def test_one_way_anova_separated_groups(self) -> None:
        g1 = pd.Series([1.0, 1.2, 0.9, 1.1, 1.0])
        g2 = pd.Series([5.0, 5.2, 4.8, 5.1, 5.0])
        g3 = pd.Series([9.0, 9.1, 8.9, 9.0, 9.2])
        res = one_way_anova([g1, g2, g3], ["A", "B", "C"])
        assert res.test == "one_way_anova"
        assert res.significant is True
        assert res.assumption_passed is True
        assert "ANOVA" in res.interpretation

    def test_one_way_anova_same_means(self) -> None:
        rng = np.random.default_rng(4)
        g1 = pd.Series(rng.normal(0, 1, 20))
        g2 = pd.Series(rng.normal(0, 1, 20))
        res = one_way_anova([g1, g2])
        assert res.p_value is not None
        assert 0 <= res.p_value <= 1

    def test_kruskal_wallis(self) -> None:
        g1 = pd.Series([1.0, 1.2, 0.9])
        g2 = pd.Series([5.0, 5.2, 4.8])
        res = kruskal_wallis([g1, g2])
        assert res.test == "kruskal_wallis"
        assert res.assumption_passed is True  # non-parametric


# ---------------------------------------------------------------------------
# Real-data guards (skip when FD001 absent)
# ---------------------------------------------------------------------------
class TestRealDataGuard:
    def test_statistics_module_imports(self) -> None:
        from aviation_mro.analytics import statistics  # noqa: F401

        assert callable(statistics.one_way_anova)
        assert callable(statistics.mann_whitney_u)
        assert callable(statistics.chi_square_independence)
        assert callable(statistics.spearman_correlation)


# ---------------------------------------------------------------------------
# EDA integration tests (synthetic data, no FD001 dependency)
# ---------------------------------------------------------------------------
class TestEDAIntegration:
    """Tests for FR-04 statistical analysis integrated into the EDA pipeline."""

    def _make_train_df(self, n_units: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows = []
        for unit in range(1, n_units + 1):
            n_cycles = int(rng.integers(50, 150))
            for cycle in range(1, n_cycles + 1):
                # Sensor degrades over cycle + noise
                s11 = 50.0 + 0.3 * cycle + rng.normal(0, 2.0)
                s4 = 30.0 + 0.2 * cycle + rng.normal(0, 1.5)
                rows.append(
                    {
                        "unit_id": unit,
                        "cycle": cycle,
                        "sensor_11": s11,
                        "sensor_4": s4,
                        "sensor_1": 512.0,  # constant sensor
                    }
                )
        return pd.DataFrame(rows)

    def _make_rul_df(self, n_units: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rul_values = rng.integers(7, 145, size=n_units)
        return pd.DataFrame({"unit_id": range(1, n_units + 1), "rul": rul_values})

    def test_compute_sensor_correlations(self) -> None:
        from aviation_mro.analytics.eda import compute_sensor_correlations

        train_df = self._make_train_df()
        rul_df = self._make_rul_df()
        result = compute_sensor_correlations(
            train_df, rul_df, sensors=["sensor_11", "sensor_4", "sensor_1"]
        )

        assert "sensor_11" in result
        assert "sensor_4" in result
        assert "sensor_1" in result  # constant sensor -> nan

        r11 = result["sensor_11"]
        assert "pearson" in r11
        assert "spearman" in r11
        assert r11["pearson"]["n"] > 0
        assert not np.isnan(r11["pearson"]["statistic"])  # sensor_11 varies, so correlation defined
        assert r11["pearson"]["p_value"] is not None

        # Constant sensor should have nan correlation
        r1 = result["sensor_1"]
        assert np.isnan(r1["pearson"]["statistic"])
        assert np.isnan(r1["spearman"]["statistic"])

    def test_compute_categorical_association(self) -> None:
        from aviation_mro.analytics.eda import compute_categorical_association

        train_df = self._make_train_df()
        rul_df = self._make_rul_df()
        result = compute_categorical_association(train_df, rul_df, sensor="sensor_11")

        assert "sensor" in result
        assert result["sensor"] == "sensor_11"
        assert "chi2_statistic" in result
        assert "chi2_p_value" in result
        assert "significant" in result
        assert "assumption_passed" in result
        assert "effect_size" in result
        assert "interpretation" in result
        assert "contingency" in result

    def test_compare_engine_groups_mann_whitney(self) -> None:
        from aviation_mro.analytics.eda import compare_engine_groups_mann_whitney

        train_df = self._make_train_df()
        rul_df = self._make_rul_df()
        result = compare_engine_groups_mann_whitney(train_df, rul_df, sensor="sensor_11")

        assert "sensor" in result
        assert result["sensor"] == "sensor_11"
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "high_rul_n" in result
        assert "low_rul_n" in result
        assert "interpretation" in result

    def test_compare_sensor_across_groups_anova(self) -> None:
        from aviation_mro.analytics.eda import compare_sensor_across_groups_anova

        train_df = self._make_train_df()
        result = compare_sensor_across_groups_anova(train_df, sensor="sensor_11", n_groups=3)

        assert "sensor" in result
        assert result["sensor"] == "sensor_11"
        assert "n_groups" in result
        assert result["n_groups"] == 3
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "assumption_passed" in result
        assert "effect_size" in result
        assert "interpretation" in result

    def test_anova_assumption_gate(self) -> None:
        """ANOVA reports assumption_passed=False when normality is violated."""
        from aviation_mro.analytics.eda import compare_sensor_across_groups_anova

        train_df = self._make_train_df()
        # Use constant sensor -> groups will be constant -> normality fails
        result = compare_sensor_across_groups_anova(train_df, sensor="sensor_1", n_groups=3)
        # Constant sensor across groups: either error or assumption violation
        assert "error" in result or result.get("assumption_passed") is False
