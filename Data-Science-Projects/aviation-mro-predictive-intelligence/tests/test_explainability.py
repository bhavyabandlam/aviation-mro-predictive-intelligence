"""Focused tests for the RUL SHAP explainability implementation (Section 9.6).

These tests assert SHAP model compatibility, exact feature-name/order
alignment, prediction/explanation numerical consistency, graceful handling of
missing/invalid inputs, and that the RUL evaluation metrics are NOT modified.

SHAP is treated as a hard requirement: if it is not installed the acceptance
suite FAILS (it is not silently skipped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aviation_mro.explainability import rul_explainer as ex  # noqa: E402

# Snapshot the model card at import time to prove it is never modified.
_CARD_PATH = PROJECT_ROOT / "reports" / "model_card_baseline_v2.json"
_CARD_BEFORE = _CARD_PATH.read_text(encoding="utf-8") if _CARD_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Session-scoped fixtures (compute expensive artefacts once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifact():
    return ex._load_rul_artifact()


@pytest.fixture(scope="session")
def card():
    return ex._load_model_card()


@pytest.fixture(scope="session")
def reconstruction(artifact):
    return ex._reconstruct_features(artifact)


@pytest.fixture(scope="session")
def result():
    """Run the full SHAP explanation once for the whole test session."""
    return ex.run()


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------
class TestCompatibility:
    def test_shap_installed(self) -> None:
        # Hard requirement: SHAP must be present (no skip-as-pass).
        assert ex.shap_available(), "SHAP must be installed for explainability"
        assert ex.shap_version() != "not-installed"

    def test_explainer_matches_model(self, result) -> None:
        assert result.model_algorithm == "GradientBoostingRegressor"
        assert result.explainer_type == "TreeExplainer"
        assert result.n_features == 152


# ---------------------------------------------------------------------------
# Feature name / order alignment
# ---------------------------------------------------------------------------
class TestFeatureAlignment:
    def test_artifact_card_result_order_match(self, artifact, card, result) -> None:
        assert artifact.feature_names == card["feature_names"]
        assert result.feature_names == card["feature_names"]
        assert len(result.feature_names) == result.n_features

    def test_reconstruction_matches_artifact_order(self, artifact, reconstruction) -> None:
        # The reconstruction pipeline must yield the exact trained feature order.
        assert reconstruction["feature_cols"] == artifact.feature_names

    def test_reconstruction_matches_card_feature_config(self, artifact, card) -> None:
        # FeatureConfig rebuilt from the model card must match the saved model's
        # feature space (no lagged features unless explicitly stored in the card).
        from aviation_mro.features.rul_features import FeatureConfig

        fc = card.get("feature_config", {}) or {}
        expected = FeatureConfig(
            window_sizes=tuple(fc.get("window_sizes", (3, 5))),
            include_op_settings=fc.get("include_op_settings", True),
            include_cycle_norm=fc.get("include_cycle_norm", False),
            rul_cap=fc.get("rul_cap", 125),
            random_seed=fc.get("random_seed", 42),
            lags=tuple(fc.get("lags", ())),
        )
        actual = ex._feature_config_from_card(card)
        assert actual.window_sizes == expected.window_sizes
        assert actual.include_op_settings == expected.include_op_settings
        assert actual.include_cycle_norm == expected.include_cycle_norm
        assert actual.rul_cap == expected.rul_cap
        assert actual.random_seed == expected.random_seed
        assert actual.lags == expected.lags
        assert actual.lags == ()


# ---------------------------------------------------------------------------
# Numerical consistency
# ---------------------------------------------------------------------------
class TestConsistency:
    def test_global_consistency_passed(self, result) -> None:
        assert result.consistency["passed"] is True
        assert result.consistency["max_abs_error"] < 1e-3
        assert result.consistency["n_checked"] > 0

    def test_local_base_plus_shap_equals_prediction(self, result) -> None:
        # Verify additivity from the persisted per-feature values.
        for le in result.local_explanations:
            total = le.base_value + sum(le.shap_values.values())
            assert abs(total - le.prediction) < 1e-2, (
                f"engine {le.unit_id}: {total} != {le.prediction}"
            )
            assert le.consistency_max_abs_error < 1e-3

    def test_global_importance_sorted_and_finite(self, result) -> None:
        import math

        gi = result.global_importance
        assert len(gi) == result.n_features
        vals = [g["mean_abs_shap"] for g in gi]
        assert all(math.isfinite(v) and v >= 0 for v in vals)
        assert vals == sorted(vals, reverse=True)
        assert [g["rank"] for g in gi] == list(range(1, len(gi) + 1))


# ---------------------------------------------------------------------------
# Provenance & metadata
# ---------------------------------------------------------------------------
class TestProvenance:
    def test_metadata_present(self, result) -> None:
        assert result.model_version == "baseline_rul_model_v2"
        assert result.dataset_fingerprint, "expected dataset fingerprint"
        assert result.background_size > 0
        assert "causation" in result.causation_disclaimer.lower()
        assert "scaled" in result.feature_space_note.lower()
        assert result.shap_version != "not-installed"

    def test_local_explanations_have_feature_values(self, result) -> None:
        for le in result.local_explanations:
            assert set(le.shap_values.keys()) == set(result.feature_names)
            assert set(le.feature_values_scaled.keys()) == set(result.feature_names)
            assert set(le.feature_values_raw.keys()) == set(result.feature_names)
            assert len(le.top_contributions) > 0


# ---------------------------------------------------------------------------
# Missing / invalid inputs & robustness
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert ex.load(tmp_path / "nope.json") is None

    def test_load_malformed_returns_none(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert ex.load(bad) is None

    def test_json_roundtrip(self, result, tmp_path: Path) -> None:
        data = ex.to_json_dict(result)
        # Must be serialisable (no numpy scalars leak through).
        encoded = json.dumps(data)
        assert len(encoded) > 0
        path = ex.save(result, tmp_path / "summary.json")
        assert path.exists()
        reloaded = ex.load(path)
        assert reloaded is not None
        assert reloaded["model_version"] == result.model_version

    def test_select_local_engines_handles_small_input(self) -> None:
        # per_group larger than available -> returns all unique ids, no crash.
        ids = [1, 2, 3]
        errs = np.array([0.1, 0.5, 0.2])
        selected = ex._select_local_engines(ids, errs, per_group=10)
        assert sorted(selected) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Non-regression: evaluation metrics unchanged
# ---------------------------------------------------------------------------
class TestNoMetricChange:
    def test_model_card_unchanged(self) -> None:
        current = _CARD_PATH.read_text(encoding="utf-8") if _CARD_PATH.exists() else ""
        assert current == _CARD_BEFORE, "model card was modified by explainability"

    def test_test_metrics_intact(self, card) -> None:
        tm = card["test_metrics"]
        assert round(tm["mae"], 4) == 13.4948
        assert round(tm["r2"], 4) == 0.8071
