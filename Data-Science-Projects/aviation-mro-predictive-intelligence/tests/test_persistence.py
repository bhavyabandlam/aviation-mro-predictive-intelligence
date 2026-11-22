"""Tests for the SQLite persistence layer (SRS v1.1 §10).

Covers schema creation, insert/read, foreign-key enforcement, model-registry
upsert, and Parquet export.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aviation_mro.persistence import TABLE_NAMES, MRODatabase  # noqa: E402


@pytest.fixture()
def db(tmp_path: Path) -> MRODatabase:
    database = MRODatabase(tmp_path / "test_mro.db")
    database.create_schema()
    yield database
    database.close()


class TestSchema:
    def test_creates_all_tables(self, db: MRODatabase) -> None:
        assert set(TABLE_NAMES) <= set(db.existing_tables())
        assert len(TABLE_NAMES) == 13

    def test_schema_idempotent(self, tmp_path: Path) -> None:
        database = MRODatabase(tmp_path / "m.db")
        database.create_schema()
        database.create_schema()  # second call must not raise
        database.close()


class TestInsertRead:
    def test_insert_and_read_technicians(self, db: MRODatabase) -> None:
        df = pd.DataFrame(
            [{"technician_id": "T1", "shop_id": "S1", "role": "MECH", "hourly_rate": 120.0}]
        )
        assert db.insert_dataframe("technicians", df) == 1
        rows = db.read_table("technicians")
        assert len(rows) == 1
        assert rows.iloc[0]["technician_id"] == "T1"
        assert db.count("technicians") == 1

    def test_unknown_table_raises(self, db: MRODatabase) -> None:
        with pytest.raises(ValueError):
            db.insert_dataframe("not_a_table", pd.DataFrame({"a": [1]}))


class TestForeignKeys:
    def test_fk_violation_rejected(self, db: MRODatabase) -> None:
        # Insert aircraft first, then an engine referencing a missing aircraft.
        db.insert_dataframe(
            "aircraft",
            pd.DataFrame(
                [{"aircraft_id": "A1", "registration": "N1", "fleet_type": "NARROW_BODY"}]
            ),
        )
        bad_engine = pd.DataFrame(
            [
                {
                    "engine_id": "E1",
                    "aircraft_id": "MISSING",
                    "shop_id": "S1",
                    "part_family": "ENGINE",
                }
            ]
        )
        with pytest.raises(pd.errors.DatabaseError):
            db.insert_dataframe("engines", bad_engine)

    def test_valid_fk_accepted(self, db: MRODatabase) -> None:
        db.insert_dataframe(
            "aircraft",
            pd.DataFrame(
                [{"aircraft_id": "A1", "registration": "N1", "fleet_type": "NARROW_BODY"}]
            ),
        )
        good_engine = pd.DataFrame(
            [{"engine_id": "E1", "aircraft_id": "A1", "shop_id": "S1", "part_family": "ENGINE"}]
        )
        assert db.insert_dataframe("engines", good_engine) == 1


class TestModelRegistry:
    def test_upsert_and_read(self, db: MRODatabase) -> None:
        db.upsert_model_registry(
            {
                "model_id": "rul_v2",
                "model_name": "GradientBoosting",
                "version": "2",
                "algorithm": "GradientBoostingRegressor",
                "dataset_fingerprint": "abc123",
                "features_json": ["a", "b"],
                "params_json": {"n_estimators": 200},
                "metrics_json": {"r2": 0.807},
                "random_seed": 42,
                "artifact_path": "models/baseline_rul_model_v2.pkl",
                "training_date": None,
            }
        )
        rows = db.query("SELECT * FROM model_registry")
        assert len(rows) == 1
        assert rows[0]["model_name"] == "GradientBoosting"
        assert rows[0]["random_seed"] == 42


class TestParquetExport:
    def test_export_table(self, db: MRODatabase, tmp_path: Path) -> None:
        db.insert_dataframe(
            "aircraft",
            pd.DataFrame(
                [{"aircraft_id": "A1", "registration": "N1", "fleet_type": "NARROW_BODY"}]
            ),
        )
        out = db.export_table_parquet("aircraft", tmp_path / "aircraft.parquet")
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 1

    def test_export_all(self, db: MRODatabase, tmp_path: Path) -> None:
        db.insert_dataframe(
            "aircraft",
            pd.DataFrame(
                [{"aircraft_id": "A1", "registration": "N1", "fleet_type": "NARROW_BODY"}]
            ),
        )
        written = db.export_all_parquet(tmp_path / "exports")
        assert "aircraft" in written
        assert written["aircraft"].exists()
