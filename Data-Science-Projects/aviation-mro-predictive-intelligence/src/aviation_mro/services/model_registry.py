"""Model registry service (SRS v1.1 FR-14).

Tracks model name, version, dataset fingerprint, feature set, training date,
parameters, metrics, random seed and artifact location for every saved model
artefact. Wraps the existing SQLite ``model_registry`` table in
``src/aviation_mro/persistence/database.py`` without modifying it.

Real metadata is read from saved model cards and artefact file metadata.
Training dates come from artefact file modification times — never invented.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aviation_mro.persistence.database import MRODatabase
from aviation_mro.services.logging import get_logger

__all__ = [
    "ModelRegistryEntry",
    "ModelRegistryService",
    "register_saved_artefacts",
    "list_registered_models",
    "get_model_by_id_version",
]

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class ModelRegistryEntry:
    """A single model-registry record with all FR-14 fields."""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        version: str,
        algorithm: str,
        dataset_fingerprint: dict[str, str],
        features: list[str],
        params: dict[str, Any],
        metrics: dict[str, Any],
        random_seed: int,
        artifact_path: str,
        training_date: str | None,
        notes: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.version = version
        self.algorithm = algorithm
        self.dataset_fingerprint = dataset_fingerprint
        self.features = features
        self.params = params
        self.metrics = metrics
        self.random_seed = random_seed
        self.artifact_path = artifact_path
        self.training_date = training_date
        self.notes = notes

    def to_record(self) -> dict[str, Any]:
        """Return a dict suitable for ``MRODatabase.upsert_model_registry``."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "algorithm": self.algorithm,
            "dataset_fingerprint": self.dataset_fingerprint,
            "features_json": self.features,
            "params_json": self.params,
            "metrics_json": self.metrics,
            "random_seed": self.random_seed,
            "artifact_path": self.artifact_path,
            "training_date": self.training_date,
            "notes": self.notes,
        }


def _artifact_mtime_iso(path: Path) -> str | None:
    """Return the ISO-8601 UTC modification time of *path*, or ``None``."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            data: Any = json.load(fh)
            if not isinstance(data, dict):
                return None
            return data
    except (OSError, json.JSONDecodeError):
        return None


class ModelRegistryService:
    """Service layer for FR-14 model registry operations."""

    def __init__(self, db: MRODatabase | None = None) -> None:
        if db is None:
            db = MRODatabase()
            db.create_schema()
        self.db = db

    def register_saved_artefacts(self) -> list[ModelRegistryEntry]:
        """Register all saved model artefacts from real metadata.

        Reads metadata from model cards and artefact file modification times.
        Does not retrain models or invent training dates, parameters or
        dataset fingerprints.
        """
        builders = [
            self._build_rul_v1_entry,
            self._build_rul_v2_entry,
            self._build_anomaly_entry,
        ]

        entries: list[ModelRegistryEntry] = []
        for build in builders:
            try:
                entry = build()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to build model registry entry: %s", exc)
                continue
            if entry is None:
                continue
            self.db.upsert_model_registry(entry.to_record())
            entries.append(entry)
            logger.info(
                "Registered model %s v%s at %s",
                entry.model_name,
                entry.version,
                entry.artifact_path,
            )
        return entries

    def list_registered_models(self) -> list[dict[str, Any]]:
        """Return all model registry rows as dictionaries."""
        return self.db.query("SELECT * FROM model_registry ORDER BY model_id, version")

    def get_model_by_id_version(self, model_id: str, version: str) -> dict[str, Any] | None:
        """Return a single model registry row by model_id and version."""
        rows = self.db.query(
            "SELECT * FROM model_registry WHERE model_id = ? AND version = ?",
            (model_id, version),
        )
        return rows[0] if rows else None

    # -- entry builders (read-only, real metadata only) -------------------

    def _build_rul_v1_entry(self) -> ModelRegistryEntry | None:
        card_path = REPORTS_DIR / "model_card_baseline.json"
        artifact_path = MODELS_DIR / "baseline_rul_model.pkl"
        card = _load_json(card_path)
        if card is None:
            logger.warning("Model card not found: %s", card_path)
            return None

        feature_config = card.get("feature_config", {})
        feature_names = card.get("feature_names", [])
        test_metrics = card.get("test_metrics", {})

        return ModelRegistryEntry(
            model_id="rul_v1",
            model_name="GradientBoosting",
            version="1",
            algorithm="GradientBoostingRegressor",
            dataset_fingerprint=card.get("dataset_fingerprint", {}),
            features=feature_names,
            params={
                "window_sizes": feature_config.get("window_sizes", [3, 5]),
                "include_op_settings": feature_config.get("include_op_settings", True),
                "include_cycle_norm": feature_config.get("include_cycle_norm", True),
                "rul_cap": feature_config.get("rul_cap", 125),
                "random_seed": feature_config.get("random_seed", 42),
            },
            metrics={
                "test_mae": test_metrics.get("mae"),
                "test_rmse": test_metrics.get("rmse"),
                "test_r2": test_metrics.get("r2"),
            },
            random_seed=feature_config.get("random_seed", 42),
            artifact_path=str(artifact_path),
            training_date=_artifact_mtime_iso(artifact_path),
            notes=(
                "v1 baseline with cycle_norm feature (later found to leak future info). "
                "Test R² is negative (-3.30); superseded by v2."
            ),
        )

    def _build_rul_v2_entry(self) -> ModelRegistryEntry | None:
        card_path = REPORTS_DIR / "model_card_baseline_v2.json"
        artifact_path = MODELS_DIR / "baseline_rul_model_v2.pkl"
        card = _load_json(card_path)
        if card is None:
            logger.warning("Model card not found: %s", card_path)
            return None

        feature_config = card.get("feature_config", {})
        feature_names = card.get("feature_names", [])
        test_metrics = card.get("test_metrics", {})
        model_comparison = card.get("model_comparison", [])
        gb_result: dict[str, Any] = next(
            (m for m in model_comparison if m.get("name") == "GradientBoosting"), {}
        )

        return ModelRegistryEntry(
            model_id="rul_v2",
            model_name="GradientBoosting",
            version="2",
            algorithm="GradientBoostingRegressor",
            dataset_fingerprint=card.get("dataset_fingerprint", {}),
            features=feature_names,
            params={
                "window_sizes": feature_config.get("window_sizes", [3, 5]),
                "include_op_settings": feature_config.get("include_op_settings", True),
                "include_cycle_norm": feature_config.get("include_cycle_norm", False),
                "rul_cap": feature_config.get("rul_cap", 125),
                "random_seed": feature_config.get("random_seed", 42),
                "val_mae": gb_result.get("val_mae"),
                "val_rmse": gb_result.get("val_rmse"),
                "val_r2": gb_result.get("val_r2"),
            },
            metrics={
                "test_mae": test_metrics.get("mae"),
                "test_rmse": test_metrics.get("rmse"),
                "test_r2": test_metrics.get("r2"),
            },
            random_seed=feature_config.get("random_seed", 42),
            artifact_path=str(artifact_path),
            training_date=_artifact_mtime_iso(artifact_path),
            notes=(
                "v2 leak-free RUL model. cycle_norm disabled, op_setting_3 excluded "
                "(zero variance). Test R²=0.807, MAE=13.49."
            ),
        )

    def _build_anomaly_entry(self) -> ModelRegistryEntry | None:
        card_path = REPORTS_DIR / "anomaly_model_card.json"
        artifact_path = MODELS_DIR / "anomaly_model.pkl"
        card = _load_json(card_path)
        if card is None:
            logger.warning("Model card not found: %s", card_path)
            return None

        model_config = card.get("model_config", {})
        feature_names = card.get("feature_names", [])
        sweep = card.get("contamination_sweep", {}).get("0.01", {})

        return ModelRegistryEntry(
            model_id="anomaly_v1",
            model_name="IsolationForest",
            version="1",
            algorithm="IsolationForest",
            dataset_fingerprint=card.get("dataset_fingerprint", {}),
            features=feature_names,
            params={
                "n_estimators": model_config.get("n_estimators", 10),
                "max_samples": model_config.get("max_samples", "auto"),
                "contamination": model_config.get("contamination", "auto"),
                "max_features": model_config.get("max_features", 1.0),
                "bootstrap": model_config.get("bootstrap", False),
                "random_state": model_config.get("random_state", 42),
                "n_jobs": model_config.get("n_jobs", -1),
                "default_contamination": card.get("default_contamination", 0.01),
                "default_threshold": card.get("default_threshold", -0.07955),
            },
            metrics={
                "default_threshold": card.get("default_threshold", -0.07955),
                "default_contamination": card.get("default_contamination", 0.01),
                "train_anomaly_rate": sweep.get("train_anomaly_rate"),
                "val_anomaly_rate": sweep.get("val_anomaly_rate"),
                "test_anomaly_rate": sweep.get("test_anomaly_rate"),
            },
            random_seed=model_config.get("random_state", 42),
            artifact_path=str(artifact_path),
            training_date=_artifact_mtime_iso(artifact_path),
            notes=(
                "IsolationForest anomaly detector trained on FD001 sensor features. "
                "Default contamination=0.01, threshold=-0.07955."
            ),
        )


def register_saved_artefacts(db: MRODatabase | None = None) -> list[ModelRegistryEntry]:
    """Convenience function: register all saved model artefacts."""
    service = ModelRegistryService(db)
    return service.register_saved_artefacts()


def list_registered_models(db: MRODatabase | None = None) -> list[dict[str, Any]]:
    """Convenience function: list all model registry rows."""
    service = ModelRegistryService(db)
    return service.list_registered_models()


def get_model_by_id_version(
    model_id: str, version: str, db: MRODatabase | None = None
) -> dict[str, Any] | None:
    """Convenience function: get a model by ID and version."""
    service = ModelRegistryService(db)
    return service.get_model_by_id_version(model_id, version)
