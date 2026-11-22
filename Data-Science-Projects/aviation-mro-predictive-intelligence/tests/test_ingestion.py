"""Tests for the synthetic MRO generator + SQLite/Parquet integration.

Covers schema shape, primary/foreign-key integrity, synthetic labelling,
determinism, and end-to-end load into the SQLite business database with
Parquet exports (SRS v1.1 §5.3, FR-03; §10).
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

from aviation_mro.ingestion import (  # noqa: E402
    SyntheticConfig,
    SyntheticGenerator,
    build_business_database,
)
from aviation_mro.persistence import MRODatabase  # noqa: E402

EXPECTED_TABLES = {
    "technicians",
    "aircraft",
    "engines",
    "inventory_parts",
    "maintenance_orders",
    "inventory_transactions",
    "downtime_events",
}


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return SyntheticGenerator(SyntheticConfig(random_seed=42)).generate_all()


class TestGeneration:
    def test_all_tables_present(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables.keys()) == EXPECTED_TABLES

    def test_primary_keys_unique(self, tables: dict[str, pd.DataFrame]) -> None:
        pk = {
            "technicians": "technician_id",
            "aircraft": "aircraft_id",
            "engines": "engine_id",
            "inventory_parts": "part_id",
            "maintenance_orders": "order_id",
            "inventory_transactions": "txn_id",
            "downtime_events": "event_id",
        }
        for name, col in pk.items():
            assert tables[name][col].is_unique, f"{name}.{col} not unique"

    def test_foreign_keys_valid(self, tables: dict[str, pd.DataFrame]) -> None:
        aircraft_ids = set(tables["aircraft"]["aircraft_id"])
        engine_ids = set(tables["engines"]["engine_id"])
        tech_ids = set(tables["technicians"]["technician_id"])
        part_ids = set(tables["inventory_parts"]["part_id"])
        order_ids = set(tables["maintenance_orders"]["order_id"])

        orders = tables["maintenance_orders"]
        assert set(orders["aircraft_id"]) <= aircraft_ids
        assert set(orders["technician_id"]) <= tech_ids
        assert set(orders["part_id"]) <= part_ids
        assert set(orders["engine_id"].dropna()) <= engine_ids

        txns = tables["inventory_transactions"]
        assert set(txns["part_id"]) <= part_ids
        assert set(txns["order_id"].dropna()) <= order_ids

        dt = tables["downtime_events"]
        assert set(dt["aircraft_id"]) <= aircraft_ids
        assert set(dt["order_id"].dropna()) <= order_ids

    def test_parts_are_synthetic(self, tables: dict[str, pd.DataFrame]) -> None:
        assert (tables["inventory_parts"]["is_synthetic"] == 1).all()

    def test_deterministic(self) -> None:
        a = SyntheticGenerator(SyntheticConfig(random_seed=42)).generate_all()
        b = SyntheticGenerator(SyntheticConfig(random_seed=42)).generate_all()
        pd.testing.assert_frame_equal(a["aircraft"], b["aircraft"])
        pd.testing.assert_frame_equal(a["maintenance_orders"], b["maintenance_orders"])


class TestDatabaseIntegration:
    def test_build_business_database(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mro.db"
        out_dir = tmp_path / "synthetic"
        tables, manifest = build_business_database(db_path=db_path, out_dir=out_dir)

        assert manifest["is_synthetic"] is True
        assert manifest["random_seed"] == 42
        assert db_path.exists()

        db = MRODatabase(db_path)
        db.connect()
        # All 7 business tables + registry tables exist and are populated.
        for name in EXPECTED_TABLES:
            assert db.count(name) == len(tables[name])
        # Lineage + source registry.
        assert db.count("ingestion_runs") == len(EXPECTED_TABLES)
        sources = db.read_table("data_sources")
        assert "SYN-MRO-BIZ" in set(sources["source_id"])
        db.close()

        # Parquet exports exist.
        for name in EXPECTED_TABLES:
            assert (out_dir / f"{name}.parquet").exists()
        assert (out_dir / "manifest.json").exists()

    def test_sqlite_fk_enforced(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mro.db"
        build_business_database(db_path=db_path, out_dir=tmp_path / "s")
        db = MRODatabase(db_path)
        db.connect()
        # Every order's aircraft exists (FK integrity held at insert time).
        orphans = db.query(
            "SELECT COUNT(*) AS n FROM maintenance_orders o "
            "LEFT JOIN aircraft a ON o.aircraft_id = a.aircraft_id "
            "WHERE a.aircraft_id IS NULL"
        )
        assert orphans[0]["n"] == 0
        db.close()
