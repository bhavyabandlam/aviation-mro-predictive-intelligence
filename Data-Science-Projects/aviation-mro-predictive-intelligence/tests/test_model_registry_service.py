"""Tests for the model registry service (SRS v1.1 FR-14).

Covers registration of saved artefacts, retrieval by model_id/version,
all 12 FR-14 fields, and real metadata from model cards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aviation_mro.persistence.database import MRODatabase
from aviation_mro.services.logging import setup_logging
from aviation_mro.services.model_registry import (
    ModelRegistryEntry,
    ModelRegistryService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@pytest.fixture
def db(tmp_path) -> MRODatabase:
    """Create a temporary MRODatabase with schema."""
    db_path = tmp_path / "test_mro.db"
    database = MRODatabase(db_path)
    database.create_schema()
    return database


class TestModelRegistryServiceRegistration:
    """Tests for registering saved model artefacts."""

    def test_setup_logging_once(self) -> None:
        """setup_logging is idempotent."""
        setup_logging()
        setup_logging()

    def test_register_saved_artefacts_creates_entries(self, db) -> None:
        """register_saved_artefacts() creates model registry entries."""
        service = ModelRegistryService(db)
        entries = service.register_saved_artefacts()
        assert len(entries) >= 1

    def test_registered_models_have_all_12_fields(self, db) -> None:
        """Every registered model has all 12 FR-14 fields."""
        service = ModelRegistryService(db)
        entries = service.register_saved_artefacts()

        required_fields = {
            "model_id",
            "model_name",
            "version",
            "algorithm",
            "dataset_fingerprint",
            "features_json",
            "params_json",
            "metrics_json",
            "random_seed",
            "artifact_path",
            "training_date",
            "notes",
        }

        for entry in entries:
            record = entry.to_record()
            for field in required_fields:
                assert field in record, f"Missing field {field} in {entry.model_id}"

    def test_rul_v2_has_real_metrics(self, db) -> None:
        """The v2 RUL model entry has real metrics from the model card."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()

        row = service.get_model_by_id_version("rul_v2", "2")
        assert row is not None
        metrics = json.loads(row["metrics_json"])
        # Real values from model_card_baseline_v2.json
        assert metrics["test_mae"] == 13.494793746770101
        assert metrics["test_rmse"] == 18.25096571600896
        assert metrics["test_r2"] == 0.8071089157126082

    def test_rul_v2_has_real_fingerprint(self, db) -> None:
        """The v2 RUL model entry has real dataset fingerprints."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()

        row = service.get_model_by_id_version("rul_v2", "2")
        assert row is not None
        fingerprint = json.loads(row["dataset_fingerprint"])
        assert "train_FD001.txt" in fingerprint
        assert "test_FD001.txt" in fingerprint
        assert "RUL_FD001.txt" in fingerprint
        assert fingerprint["train_FD001.txt"] == "963b5e22825b34d8"

    def test_rul_v2_has_real_feature_count(self, db) -> None:
        """The v2 RUL model entry has 152 features from the model card."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()

        row = service.get_model_by_id_version("rul_v2", "2")
        assert row is not None
        features = json.loads(row["features_json"])
        assert len(features) == 152

    def test_anomaly_has_real_params(self, db) -> None:
        """The anomaly model entry has real params from its model card."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()

        row = service.get_model_by_id_version("anomaly_v1", "1")
        assert row is not None
        params = json.loads(row["params_json"])
        assert params["n_estimators"] == 10
        assert params["random_state"] == 42
        assert params["contamination"] == "auto"

    def test_training_date_from_file_mtime_not_invented(self, db) -> None:
        """Training dates come from artefact file mtimes, not invented."""
        service = ModelRegistryService(db)
        entries = service.register_saved_artefacts()

        for entry in entries:
            assert entry.training_date is not None
            # Must be a valid ISO-8601 timestamp
            assert "T" in entry.training_date
            assert "+00:00" in entry.training_date

    def test_artifact_paths_point_to_real_files(self, db) -> None:
        """All artifact paths in registry entries point to existing files."""
        service = ModelRegistryService(db)
        entries = service.register_saved_artefacts()

        for entry in entries:
            assert Path(entry.artifact_path).exists(), (
                f"Artifact {entry.artifact_path} does not exist"
            )


class TestModelRegistryQueries:
    """Tests for querying the model registry."""

    def test_list_registered_models(self, db) -> None:
        """list_registered_models() returns all rows."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()
        rows = service.list_registered_models()
        assert len(rows) >= 1

    def test_get_model_by_id_version_found(self, db) -> None:
        """get_model_by_id_version() returns the correct row."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()

        row = service.get_model_by_id_version("rul_v2", "2")
        assert row is not None
        assert row["model_name"] == "GradientBoosting"
        assert row["version"] == "2"

    def test_get_model_by_id_version_not_found(self, db) -> None:
        """get_model_by_id_version() returns None for missing model."""
        service = ModelRegistryService(db)
        row = service.get_model_by_id_version("nonexistent", "1")
        assert row is None

    def test_upsert_is_idempotent(self, db) -> None:
        """Calling register_saved_artefacts() twice does not duplicate rows."""
        service = ModelRegistryService(db)
        service.register_saved_artefacts()
        count_after_first = len(service.list_registered_models())

        service.register_saved_artefacts()
        count_after_second = len(service.list_registered_models())

        assert count_after_first == count_after_second


class TestModelRegistryEntry:
    """Tests for the ModelRegistryEntry dataclass."""

    def test_to_record_has_all_fields(self) -> None:
        """to_record() returns a dict with all FR-14 fields."""
        entry = ModelRegistryEntry(
            model_id="test_model",
            model_name="TestModel",
            version="1",
            algorithm="TestAlgorithm",
            dataset_fingerprint={"file.txt": "abc123"},
            features=["feat1", "feat2"],
            params={"param1": "value1"},
            metrics={"metric1": 0.5},
            random_seed=42,
            artifact_path="models/test.pkl",
            training_date="2026-01-01T00:00:00+00:00",
            notes="test notes",
        )
        record = entry.to_record()
        assert record["model_id"] == "test_model"
        assert record["model_name"] == "TestModel"
        assert record["version"] == "1"
        assert record["algorithm"] == "TestAlgorithm"
        assert record["dataset_fingerprint"] == {"file.txt": "abc123"}
        assert record["features_json"] == ["feat1", "feat2"]
        assert record["params_json"] == {"param1": "value1"}
        assert record["metrics_json"] == {"metric1": 0.5}
        assert record["random_seed"] == 42
        assert record["artifact_path"] == "models/test.pkl"
        assert record["training_date"] == "2026-01-01T00:00:00+00:00"
        assert record["notes"] == "test notes"
