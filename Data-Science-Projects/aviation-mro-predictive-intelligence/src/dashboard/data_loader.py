"""Safe loaders for dashboard artefacts.

Every loader degrades gracefully: if an artefact is missing or cannot be
parsed the loader returns ``None`` (or an empty result) instead of raising,
so a page can render a warning rather than crashing the whole app.

Data provenance is explicit. The NASA C-MAPSS FD001 benchmark powers the
EDA, RUL prediction and anomaly-detection pages; all MRO business data
(forecasting, inventory, financial) is synthetic and clearly labelled.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aviation_mro.services.caching import ttl_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EDA_PLOTS_DIR = REPORTS_DIR / "eda" / "plots"

# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------
SOURCE_NASA = "NASA C-MAPSS FD001 (simulated benchmark)"
SOURCE_SYNTHETIC_MRO = "Synthetic MRO business data"
SOURCE_SYNTHETIC_FIN = "Synthetic MRO financial data"

SOURCE_LEGEND = {
    "nasa": {
        "label": "NASA FD001",
        "detail": SOURCE_NASA,
        "description": (
            "Turbofan engine run-to-failure simulation published by NASA PCoE. "
            "Simulated benchmark data, not real fleet records."
        ),
    },
    "synthetic_mro": {
        "label": "Synthetic MRO",
        "detail": SOURCE_SYNTHETIC_MRO,
        "description": (
            "Synthetic weekly maintenance demand, inventory and shop parameters "
            "generated locally (seed=42). Not real airline records."
        ),
    },
    "synthetic_fin": {
        "label": "Synthetic Financial",
        "detail": SOURCE_SYNTHETIC_FIN,
        "description": (
            "Synthetic labour, downtime, holding and stockout cost parameters "
            "generated locally (seed=42). Not real airline financials."
        ),
    },
}


def artefact_freshness(path: Path | str | None) -> str | None:
    """Return the ISO-8601 UTC modification time of *path*, or ``None``.

    Always reads the live filesystem mtime (never cached) so stale data
    is visible in freshness badges.
    """
    if path is None:
        return None
    p = Path(path)
    try:
        ts = p.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning ``None`` if missing or unparsable."""
    try:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return dict(json.load(fh))
    except (OSError, json.JSONDecodeError):
        return None


def load_parquet(path: Path) -> pd.DataFrame | None:
    """Load a Parquet file, returning ``None`` if missing or unreadable."""
    try:
        if not path.exists():
            return None
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def artifact_exists(path: Path) -> bool:
    """Return whether a file exists on disk."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Report loaders (JSON)
# ---------------------------------------------------------------------------
def load_eda_report() -> dict[str, Any] | None:
    return load_json(REPORTS_DIR / "eda" / "eda_report.json")


def load_rul_model_card() -> dict[str, Any] | None:
    # Prefer the v2 card (leak-free model); fall back to the v1 card.
    card = load_json(REPORTS_DIR / "model_card_baseline_v2.json")
    if card is None:
        card = load_json(REPORTS_DIR / "model_card_baseline.json")
    return card


def load_anomaly_model_card() -> dict[str, Any] | None:
    return load_json(REPORTS_DIR / "anomaly_model_card.json")


def load_forecast_results(target_column: str = "demand_count") -> dict[str, Any] | None:
    """Load forecast results for a given target column.

    Args:
        target_column: Target column name (demand_count, labour_hours, maintenance_expenditure)

    Returns:
        Forecast results dict, or None if not available
    """
    if target_column == "demand_count":
        return load_json(REPORTS_DIR / "forecasting" / "forecast_results.json")
    return load_json(REPORTS_DIR / "forecasting" / f"forecast_results_{target_column}.json")


def load_inventory_analysis() -> dict[str, Any] | None:
    return load_json(REPORTS_DIR / "inventory" / "inventory_analysis.json")


def load_financial_analysis() -> dict[str, Any] | None:
    return load_json(REPORTS_DIR / "financial" / "financial_analysis.json")


def load_rul_explanation() -> dict[str, Any] | None:
    return load_json(REPORTS_DIR / "explainability" / "rul_shap_summary.json")


# ---------------------------------------------------------------------------
# Dataset loaders (Parquet)
# ---------------------------------------------------------------------------
@ttl_cache(ttl_seconds=300, max_size=32)
def load_maintenance_demand() -> pd.DataFrame | None:
    return load_parquet(DATA_DIR / "maintenance_demand.parquet")


@ttl_cache(ttl_seconds=300, max_size=32)
def load_inventory_data() -> pd.DataFrame | None:
    return load_parquet(DATA_DIR / "inventory_data.parquet")


@ttl_cache(ttl_seconds=300, max_size=32)
def load_future_buffer() -> pd.DataFrame | None:
    return load_parquet(DATA_DIR / "future_buffer.parquet")


# ---------------------------------------------------------------------------
# Model artefact loaders
# ---------------------------------------------------------------------------
def load_rul_artifact() -> Any | None:
    """Load the RUL ``TrainingArtifacts`` bundle.

    The v2 artefact was pickled from a ``__main__`` context during training,
    so we alias the dataclasses onto ``__main__`` before unpickling. If the
    shim or unpickling fails we return ``None`` and callers fall back to the
    model-card JSON (which contains all reported metrics).
    """
    path = MODELS_DIR / "baseline_rul_model_v2.pkl"
    if not path.exists():
        path = MODELS_DIR / "baseline_rul_model.pkl"
    if not artifact_exists(path):
        return None
    try:
        import joblib

        import __main__

        try:
            from aviation_mro.models import baseline as baseline_module
        except Exception:
            return None
        for name in ("TrainingArtifacts", "ModelResult"):
            cls = getattr(baseline_module, name, None)
            if cls is not None and not hasattr(__main__, name):
                setattr(__main__, name, cls)
        return joblib.load(path)
    except Exception:
        return None


def load_anomaly_detector() -> Any | None:
    """Load the fitted IsolationForest anomaly detector, or ``None``."""
    path = MODELS_DIR / "anomaly_model.pkl"
    if not artifact_exists(path):
        return None
    try:
        from aviation_mro.models.anomaly_model import (
            IsolationForestAnomalyDetector,
        )

        return IsolationForestAnomalyDetector.load(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model artefact status (for the executive overview health check)
# ---------------------------------------------------------------------------
def model_artifact_status() -> dict[str, dict[str, Any]]:
    """Report existence and size of each saved model artefact."""
    artefacts = {
        "RUL model (GradientBoosting)": MODELS_DIR / "baseline_rul_model_v2.pkl",
        "Anomaly model (IsolationForest)": MODELS_DIR / "anomaly_model.pkl",
        "Seasonal-naive baseline": REPORTS_DIR
        / "forecasting"
        / "models"
        / "seasonal_naive_m52_baseline.pkl",
        "Naive baseline": REPORTS_DIR / "forecasting" / "models" / "naive_baseline.pkl",
    }
    status: dict[str, dict[str, Any]] = {}
    for name, path in artefacts.items():
        exists = artifact_exists(path)
        status[name] = {
            "exists": exists,
            "path": str(path),
            "size_kb": round(path.stat().st_size / 1024, 1) if exists else 0.0,
        }
    return status
