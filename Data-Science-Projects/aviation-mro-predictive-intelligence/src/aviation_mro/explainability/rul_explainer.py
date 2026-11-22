"""SHAP explanations for the Section 9.2 RUL prediction model.

Design
------
The RUL model is a scikit-learn ``GradientBoostingRegressor`` wrapped with a
median imputer and standard scaler (see ``models/baseline.py``). Tree models
are exactly and efficiently explained by SHAP's ``TreeExplainer``.

This module:
- Loads the EXISTING trained artefact (``baseline_rul_model_v2.pkl``). It does
  NOT retrain or replace the model.
- Reconstructs the exact model-input features (same ``FeatureConfig``, same
  imputer + scaler, same sorted feature order) used during training.
- Computes global feature importance (mean |SHAP|) and local explanations for a
  representative set of test engines.
- Verifies numerical consistency: ``base_value + sum(shap) == prediction``.
- Persists everything (model version, dataset fingerprint, feature names,
  scaled + raw feature values, SHAP values, metadata) to JSON.

Feature-space note
------------------
SHAP values are computed in the SCALED model-input space (post imputer +
StandardScaler). Feature names are the engineered rolling-feature names (e.g.
``sensor_11_mean_3``). Values are standardised (z-score-like), NOT original
sensor units. This is stated explicitly in the saved metadata and the UI.

Causation disclaimer
--------------------
SHAP attributions reflect learned correlations in the (simulated NASA FD001)
training data. They DO NOT establish causation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPLAIN_DIR = REPORTS_DIR / "explainability"
DEFAULT_OUTPUT = EXPLAIN_DIR / "rul_shap_summary.json"

RUL_MODEL_VERSION = "baseline_rul_model_v2"
CAUSATION_DISCLAIMER = (
    "SHAP values show how each feature pushes the model prediction higher or "
    "lower for a specific engine. They reflect correlations learned from the "
    "NASA FD001 simulated training data and DO NOT establish causation."
)
FEATURE_SPACE_NOTE = (
    "SHAP values and feature values are in the SCALED model-input space "
    "(median-imputed, then StandardScaler-transformed). They are standardised "
    "values, NOT original sensor units. Feature names are engineered rolling "
    "statistics (window 3 and 5) over the 15 informative sensors plus "
    "op_setting_1 and op_setting_2."
)


# ---------------------------------------------------------------------------
# SHAP availability
# ---------------------------------------------------------------------------
def shap_available() -> bool:
    """Return True if SHAP can be imported."""
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def shap_version() -> str:
    """Return the installed SHAP version string, or 'not-installed'."""
    try:
        import shap

        return str(getattr(shap, "__version__", "unknown"))
    except Exception:
        return "not-installed"


# ---------------------------------------------------------------------------
# Configuration & result containers
# ---------------------------------------------------------------------------
@dataclass
class RulExplainerConfig:
    """Configuration for RUL SHAP explanation generation."""

    background_size: int = 100
    n_local_per_group: int = 6
    random_seed: int = 42


@dataclass
class LocalExplanation:
    """A single prediction explanation (one test engine's last cycle)."""

    unit_id: int
    cycle: int
    prediction: float
    base_value: float
    absolute_error: float
    shap_values: dict[str, float]
    feature_values_scaled: dict[str, float]
    feature_values_raw: dict[str, float]
    top_contributions: list[dict[str, Any]]
    consistency_max_abs_error: float


@dataclass
class RulExplanationResult:
    """Full explanation artefact, serialisable to JSON."""

    model_version: str
    model_algorithm: str
    dataset_fingerprint: dict[str, str]
    feature_names: list[str]
    n_features: int
    explainer_type: str
    model_output: str
    background_size: int
    global_importance: list[dict[str, Any]]
    local_explanations: list[LocalExplanation]
    consistency: dict[str, Any]
    shap_version: str
    feature_space_note: str
    causation_disclaimer: str
    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Artifact / data loading (with graceful, informative failures)
# ---------------------------------------------------------------------------
def _load_rul_artifact() -> Any:
    """Load the trained RUL artefact, aliasing __main__ dataclasses first."""
    path = MODELS_DIR / "baseline_rul_model_v2.pkl"
    if not path.exists():
        path = MODELS_DIR / "baseline_rul_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"RUL model artefact not found under {MODELS_DIR}")

    import joblib

    import __main__
    from aviation_mro.models import baseline as baseline_module

    for name in ("TrainingArtifacts", "ModelResult"):
        cls = getattr(baseline_module, name, None)
        if cls is not None and not hasattr(__main__, name):
            setattr(__main__, name, cls)
    return joblib.load(path)


def _load_model_card() -> dict[str, Any]:
    """Load the RUL model card (v2 preferred, v1 fallback)."""
    for name in ("model_card_baseline_v2.json", "model_card_baseline.json"):
        path = REPORTS_DIR / name
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                card: dict[str, Any] = json.load(fh)
                return card
    raise FileNotFoundError(f"RUL model card not found under {REPORTS_DIR}")


def _feature_config_from_card(card: dict[str, Any]) -> Any:
    """Rebuild the FeatureConfig used during training from the model card."""
    from aviation_mro.features.rul_features import FeatureConfig

    fc = card.get("feature_config", {}) or {}
    kwargs: dict[str, Any] = {}
    if "window_sizes" in fc:
        kwargs["window_sizes"] = tuple(fc["window_sizes"])
    for key in ("include_op_settings", "include_cycle_norm", "rul_cap", "random_seed"):
        if key in fc:
            kwargs[key] = fc[key]
    # Reconstruct the same lag configuration the trained model used. The saved
    # model card stores no `lags` key (model trained without lagged features),
    # so default to an empty tuple to preserve the original feature space and
    # feature order used at inference time.
    kwargs["lags"] = tuple(fc.get("lags", ()))
    return FeatureConfig(**kwargs)


def _reconstruct_features(artifact: Any) -> dict[str, Any]:
    """Reconstruct train/test feature frames with the exact training config."""
    from aviation_mro.features.rul_features import load_and_prepare_data

    card = _load_model_card()
    config = _feature_config_from_card(card)
    data: dict[str, Any] = load_and_prepare_data(config)

    feature_names: list[str] = list(artifact.feature_names)
    # Align column order exactly to the trained model's feature order.
    if data["feature_cols"] != feature_names:
        data["train_df"] = data["train_df"][feature_names]
        data["test_last"] = data["test_last"][feature_names]
    return data


def _transform_features(artifact: Any, df: pd.DataFrame) -> np.ndarray:
    """Apply the fitted imputer + scaler to a feature DataFrame.

    Passing a DataFrame preserves feature names for the imputer (which was fit
    on named columns) and avoids spurious sklearn warnings.
    """
    imputed = artifact.imputer.transform(df)
    return np.asarray(artifact.scaler.transform(imputed), dtype=float)


# ---------------------------------------------------------------------------
# Core explanation logic
# ---------------------------------------------------------------------------
def _select_local_engines(unit_ids: list[int], abs_errors: np.ndarray, per_group: int) -> list[int]:
    """Pick well-predicted and poorly-predicted engines for local explanations."""
    order = np.argsort(abs_errors)
    best = [unit_ids[i] for i in order[:per_group]]
    worst = [unit_ids[i] for i in order[-per_group:]]
    # De-duplicate while preserving order.
    seen: set[int] = set()
    selected: list[int] = []
    for uid in best + worst:
        if uid not in seen:
            seen.add(uid)
            selected.append(uid)
    return selected


def run(cfg: RulExplainerConfig | None = None) -> RulExplanationResult:
    """Compute SHAP explanations and return the result object.

    Raises a clear error if SHAP is unavailable or required artefacts are
    missing. Does not modify the model, model card, or evaluation metrics.
    """
    cfg = cfg or RulExplainerConfig()

    if not shap_available():
        raise RuntimeError(
            "SHAP is not installed. Install a Python 3.13-compatible SHAP "
            "(e.g. shap==0.54.1) before running RUL explanations."
        )

    import shap

    artifact = _load_rul_artifact()
    card = _load_model_card()
    data = _reconstruct_features(artifact)
    feature_names: list[str] = list(artifact.feature_names)

    train_df: pd.DataFrame = data["train_df"]
    test_last: pd.DataFrame = data["test_last"]

    X_train_scaled = _transform_features(artifact, train_df[feature_names])
    X_test_scaled = _transform_features(artifact, test_last[feature_names])

    # Deterministic background sample of training data.
    rng = np.random.RandomState(cfg.random_seed)
    bg_n = min(cfg.background_size, len(X_train_scaled))
    bg_idx = rng.choice(len(X_train_scaled), size=bg_n, replace=False)
    background = X_train_scaled[bg_idx]

    explainer = shap.TreeExplainer(artifact.best_model, data=background, model_output="raw")
    expected_value = float(np.asarray(explainer.expected_value).ravel()[0])

    shap_values = np.asarray(explainer.shap_values(X_test_scaled), dtype=float)
    predictions = np.asarray(artifact.best_model.predict(X_test_scaled), dtype=float)

    # Raw (unscaled) engineered feature values for display/provenance.
    raw_engineered = test_last[feature_names].astype(float)

    unit_ids = [int(u) for u in test_last["unit_id"].to_numpy()]
    cycles = [int(c) for c in test_last["cycle"].to_numpy()]
    y_true = data["rul_df"].set_index("unit_id").loc[unit_ids, "rul"].to_numpy().astype(float)
    abs_errors = np.abs(predictions - y_true)

    # Global importance: mean |SHAP| across all test rows.
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_importance = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 6),
        }
        for rank, idx in enumerate(order)
    ]

    # Local explanations for a representative subset of engines.
    selected = _select_local_engines(unit_ids, abs_errors, cfg.n_local_per_group)
    unit_to_row = {uid: i for i, uid in enumerate(unit_ids)}

    local: list[LocalExplanation] = []
    consistency_errors: list[float] = []
    for uid in selected:
        i = unit_to_row[uid]
        sv = shap_values[i]
        pred = float(predictions[i])
        reconstructed = expected_value + float(np.sum(sv))
        row_err = abs(reconstructed - pred)
        consistency_errors.append(row_err)

        contrib_order = np.argsort(np.abs(sv))[::-1]
        top = [
            {
                "feature": feature_names[j],
                "shap_value": round(float(sv[j]), 6),
                "direction": "increases" if sv[j] >= 0 else "decreases",
            }
            for j in contrib_order[:10]
        ]

        local.append(
            LocalExplanation(
                unit_id=uid,
                cycle=cycles[i],
                prediction=round(pred, 4),
                base_value=round(expected_value, 4),
                absolute_error=round(float(abs_errors[i]), 4),
                shap_values={
                    feature_names[j]: round(float(sv[j]), 6) for j in range(len(feature_names))
                },
                feature_values_scaled={
                    feature_names[j]: round(float(X_test_scaled[i, j]), 6)
                    for j in range(len(feature_names))
                },
                feature_values_raw={
                    feature_names[j]: round(float(np.asarray(raw_engineered.iloc[i, j]).item()), 6)
                    for j in range(len(feature_names))
                },
                top_contributions=top,
                consistency_max_abs_error=round(float(row_err), 8),
            )
        )

    max_consistency_error = float(max(consistency_errors)) if consistency_errors else 0.0
    consistency = {
        "method": "base_value + sum(shap_values) == prediction",
        "tolerance": 1e-3,
        "max_abs_error": max_consistency_error,
        "n_checked": len(consistency_errors),
        "passed": bool(max_consistency_error < 1e-3),
    }

    return RulExplanationResult(
        model_version=RUL_MODEL_VERSION,
        model_algorithm=type(artifact.best_model).__name__,
        dataset_fingerprint=dict(card.get("dataset_fingerprint", {}) or {}),
        feature_names=feature_names,
        n_features=len(feature_names),
        explainer_type=type(explainer).__name__,
        model_output="raw",
        background_size=int(bg_n),
        global_importance=global_importance,
        local_explanations=local,
        consistency=consistency,
        shap_version=shap_version(),
        feature_space_note=FEATURE_SPACE_NOTE,
        causation_disclaimer=CAUSATION_DISCLAIMER,
        limitations=[
            "Explanations use the NASA C-MAPSS FD001 simulated benchmark, not real fleet data.",
            "SHAP values are in the scaled model-input space (standardised), not original units.",
            "Correlations do not imply causation.",
            "Background sample is a deterministic subset of training engines.",
        ],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def to_json_dict(result: RulExplanationResult) -> dict[str, Any]:
    """Convert the result to a JSON-serialisable dict."""
    data = asdict(result)
    data["local_explanations"] = [asdict(local) for local in result.local_explanations]
    data["generated_by"] = "aviation_mro.explainability.rul_explainer"
    return data


def save(result: RulExplanationResult, path: Path | None = None) -> Path:
    """Persist the explanation result to JSON."""
    path = path or DEFAULT_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_json_dict(result), fh, indent=2)
    return path


def load(path: Path | None = None) -> dict[str, Any] | None:
    """Load a previously generated explanation summary, or None if absent."""
    path = path or DEFAULT_OUTPUT
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
            return data
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    """Generate and save RUL SHAP explanations (offline batch entry point)."""
    if not shap_available():
        raise SystemExit("SHAP not installed. Install shap==0.54.1 to generate explanations.")
    result = run()
    path = save(result)
    if not result.consistency["passed"]:
        raise SystemExit(
            f"Consistency check failed: max_abs_error={result.consistency['max_abs_error']}"
        )
    # Emitted for logs; no return value needed for CLI usage.
    _ = path


if __name__ == "__main__":
    main()
