"""Synthetic MRO data generator (SRS v1.1 §5.3, FR-03).

Generates a coherent, reproducible set of business tables with primary/foreign
key integrity: technicians, aircraft, engines, inventory parts, maintenance
orders, inventory transactions and downtime events. All records are clearly
labelled synthetic via an ``is_synthetic`` flag on parts and a dataset manifest.

The generator is deterministic for a fixed seed and can be loaded into the
SQLite business database and exported to Parquet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aviation_mro.persistence import DEFAULT_DB_PATH, MRODatabase

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

SHOPS = [f"SHOP_{i:02d}" for i in range(1, 6)]
FLEET_TYPES = ["NARROW_BODY", "WIDE_BODY", "REGIONAL"]
PART_FAMILIES = ["ENGINE", "LANDING_GEAR", "AVIONICS"]
ROLES = ["MECH", "AVIONICS", "STRUCT"]
ORDER_TYPES = ["PLANNED", "UNPLANNED"]
ORDER_STATUSES = ["COMPLETED", "OPEN", "IN_PROGRESS"]

# Approximate unit cost ranges by family (synthetic).
FAMILY_COST = {
    "ENGINE": (8000.0, 40000.0),
    "LANDING_GEAR": (3000.0, 15000.0),
    "AVIONICS": (1500.0, 12000.0),
}


@dataclass
class SyntheticConfig:
    """Configuration for synthetic business-data generation."""

    random_seed: int = 42
    n_shops: int = 5
    technicians_per_shop: int = 3
    n_aircraft: int = 30
    parts_per_shop_family: int = 2
    n_orders: int = 200
    start_date: date = field(default_factory=lambda: date(2023, 1, 2))
    days_span: int = 700


class SyntheticGenerator:
    """Generate coherent synthetic MRO business tables."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    # -- individual tables ------------------------------------------------
    def generate_technicians(self) -> pd.DataFrame:
        rows = []
        tid = 0
        for shop in SHOPS[: self.config.n_shops]:
            for _ in range(self.config.technicians_per_shop):
                tid += 1
                rows.append(
                    {
                        "technician_id": f"T{tid:04d}",
                        "shop_id": shop,
                        "role": ROLES[tid % len(ROLES)],
                        "hourly_rate": round(float(self.rng.uniform(90, 160)), 2),
                    }
                )
        return pd.DataFrame(rows)

    def generate_aircraft(self) -> pd.DataFrame:
        rows = []
        for i in range(self.config.n_aircraft):
            rows.append(
                {
                    "aircraft_id": f"A{i + 1:03d}",
                    "registration": f"N{100 + i:03d}",
                    "fleet_type": FLEET_TYPES[i % len(FLEET_TYPES)],
                    "year_built": int(self.rng.integers(2005, 2023)),
                }
            )
        return pd.DataFrame(rows)

    def generate_engines(self, aircraft: pd.DataFrame) -> pd.DataFrame:
        rows = []
        eid = 0
        for _, ac in aircraft.iterrows():
            n_engines = int(self.rng.integers(1, 3))  # 1-2 per aircraft
            for _ in range(n_engines):
                eid += 1
                install = self.config.start_date + timedelta(
                    days=int(self.rng.integers(0, self.config.days_span))
                )
                rows.append(
                    {
                        "engine_id": f"E{eid:04d}",
                        "aircraft_id": ac["aircraft_id"],
                        "shop_id": SHOPS[eid % self.config.n_shops],
                        "part_family": PART_FAMILIES[eid % len(PART_FAMILIES)],
                        "install_date": install.isoformat(),
                    }
                )
        return pd.DataFrame(rows)

    def generate_inventory_parts(self) -> pd.DataFrame:
        rows = []
        pid = 0
        for shop in SHOPS[: self.config.n_shops]:
            for family in PART_FAMILIES:
                lo, hi = FAMILY_COST[family]
                for _ in range(self.config.parts_per_shop_family):
                    pid += 1
                    rows.append(
                        {
                            "part_id": f"P{pid:04d}",
                            "part_number": f"PN{pid:05d}",
                            "part_family": family,
                            "shop_id": shop,
                            "unit_cost": round(float(self.rng.uniform(lo, hi)), 2),
                            "lead_time_days": round(float(self.rng.uniform(7, 90)), 1),
                            "lead_time_std_days": round(float(self.rng.uniform(0, 10)), 1),
                            "current_stock": round(float(self.rng.uniform(0, 50)), 1),
                            "is_synthetic": 1,
                        }
                    )
        return pd.DataFrame(rows)

    def generate_maintenance_orders(
        self,
        aircraft: pd.DataFrame,
        engines: pd.DataFrame,
        technicians: pd.DataFrame,
        parts: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        engines_by_ac: dict[str, list[str]] = {
            str(k): list(v)
            for k, v in engines.groupby("aircraft_id")["engine_id"].apply(list).items()
        }
        tech_ids = technicians["technician_id"].tolist()
        part_ids = parts["part_id"].tolist()
        for i in range(self.config.n_orders):
            ac = aircraft.iloc[int(self.rng.integers(0, len(aircraft)))]
            ac_engines = engines_by_ac.get(ac["aircraft_id"], [])
            engine_id = (
                ac_engines[int(self.rng.integers(0, len(ac_engines)))] if ac_engines else None
            )
            order_date = self.config.start_date + timedelta(
                days=int(self.rng.integers(0, self.config.days_span))
            )
            downtime = round(float(self.rng.uniform(0, 36)), 1)
            rows.append(
                {
                    "order_id": f"O{i + 1:05d}",
                    "aircraft_id": ac["aircraft_id"],
                    "engine_id": engine_id,
                    "technician_id": tech_ids[int(self.rng.integers(0, len(tech_ids)))],
                    "part_id": part_ids[int(self.rng.integers(0, len(part_ids)))],
                    "order_date": order_date.isoformat(),
                    "order_type": ORDER_TYPES[int(self.rng.integers(0, len(ORDER_TYPES)))],
                    "status": ORDER_STATUSES[int(self.rng.integers(0, len(ORDER_STATUSES)))],
                    "labor_hours": round(float(self.rng.uniform(1, 40)), 1),
                    "downtime_hours": downtime,
                }
            )
        return pd.DataFrame(rows)

    def generate_inventory_transactions(
        self, orders: pd.DataFrame, parts: pd.DataFrame
    ) -> pd.DataFrame:
        cost_by_part = dict(zip(parts["part_id"], parts["unit_cost"], strict=False))
        rows = []
        tid = 0
        for _, order in orders.iterrows():
            if order["part_id"] is None or pd.isna(order["part_id"]):
                continue
            tid += 1
            rows.append(
                {
                    "txn_id": f"TX{tid:06d}",
                    "part_id": order["part_id"],
                    "order_id": order["order_id"],
                    "txn_date": order["order_date"],
                    "quantity": -round(float(self.rng.uniform(1, 5)), 1),  # consumption
                    "unit_cost": float(cost_by_part.get(order["part_id"], 0.0)),
                    "txn_type": "CONSUMPTION",
                }
            )
            # Some replenishment transactions.
            if self.rng.random() < 0.3:
                tid += 1
                rows.append(
                    {
                        "txn_id": f"TX{tid:06d}",
                        "part_id": order["part_id"],
                        "order_id": None,
                        "txn_date": order["order_date"],
                        "quantity": round(float(self.rng.uniform(5, 20)), 1),
                        "unit_cost": float(cost_by_part.get(order["part_id"], 0.0)),
                        "txn_type": "REPLENISHMENT",
                    }
                )
        return pd.DataFrame(rows)

    def generate_downtime_events(self, orders: pd.DataFrame) -> pd.DataFrame:
        rows = []
        eid = 0
        for _, order in orders.iterrows():
            if order["downtime_hours"] <= 0:
                continue
            eid += 1
            start = date.fromisoformat(order["order_date"])
            hours = float(order["downtime_hours"])
            end = start + timedelta(hours=int(round(hours)))
            rows.append(
                {
                    "event_id": f"D{eid:05d}",
                    "aircraft_id": order["aircraft_id"],
                    "order_id": order["order_id"],
                    "start_time": f"{start.isoformat()}T08:00:00",
                    "end_time": f"{end.isoformat()}T17:00:00",
                    "downtime_hours": hours,
                    "cost_per_hour": round(float(self.rng.uniform(5000, 150000)), 2),
                }
            )
        return pd.DataFrame(rows)

    # -- orchestration ----------------------------------------------------
    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate every business table with referential integrity."""
        technicians = self.generate_technicians()
        aircraft = self.generate_aircraft()
        engines = self.generate_engines(aircraft)
        parts = self.generate_inventory_parts()
        orders = self.generate_maintenance_orders(aircraft, engines, technicians, parts)
        transactions = self.generate_inventory_transactions(orders, parts)
        downtime = self.generate_downtime_events(orders)
        return {
            "technicians": technicians,
            "aircraft": aircraft,
            "engines": engines,
            "inventory_parts": parts,
            "maintenance_orders": orders,
            "inventory_transactions": transactions,
            "downtime_events": downtime,
        }

    def manifest(self, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Build a dataset manifest with row counts, seed and synthetic flag."""
        return {
            "is_synthetic": True,
            "random_seed": self.config.random_seed,
            "generator": "aviation_mro.ingestion.synthetic_generator",
            "table_row_counts": {name: int(len(df)) for name, df in tables.items()},
            "notes": "Synthetic MRO business data. Not real airline records.",
        }


# ---------------------------------------------------------------------------
# Persistence helpers: SQLite load, Parquet export, lineage
# ---------------------------------------------------------------------------
SYNTHETIC_BUSINESS_SOURCE_ID = "SYN-MRO-BIZ"


def _checksum(text: str) -> str:
    """Short SHA-256 checksum of a text payload (for lineage records)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def register_data_sources(db: MRODatabase) -> None:
    """Register the synthetic source rows in ``data_sources``."""
    rows = [
        {
            "source_id": "DS-05",
            "title": "Synthetic MRO weekly demand",
            "source_type": "synthetic",
            "status": "active",
            "local_path": "data/processed/maintenance_demand.parquet",
        },
        {
            "source_id": "DS-06",
            "title": "Synthetic inventory part master",
            "source_type": "synthetic",
            "status": "active",
            "local_path": "data/processed/inventory_data.parquet",
        },
        {
            "source_id": "DS-07",
            "title": "Synthetic financial cost parameters",
            "source_type": "synthetic",
            "status": "active",
            "local_path": "data/processed",
        },
        {
            "source_id": SYNTHETIC_BUSINESS_SOURCE_ID,
            "title": "Synthetic MRO business tables",
            "source_type": "synthetic",
            "status": "active",
            "local_path": "data/synthetic",
        },
    ]
    db.insert_dataframe("data_sources", pd.DataFrame(rows))


def record_ingestion_run(db: MRODatabase, source_id: str, table: str, df: pd.DataFrame) -> None:
    """Record an ingestion run with row count and checksum for lineage."""
    row = {
        "run_id": f"{source_id}:{table}",
        "source_id": source_id,
        "run_timestamp": datetime.now(UTC).isoformat(),
        "file_name": f"{table}.parquet",
        "rows_ingested": int(len(df)),
        "checksum": _checksum(df.to_csv(index=False)),
        "status": "success",
    }
    db.insert_dataframe("ingestion_runs", pd.DataFrame([row]))


def build_business_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    out_dir: str | Path = SYNTHETIC_DIR,
    random_seed: int = 42,
    reset: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Generate synthetic tables, load them into SQLite, and export Parquet.

    Returns ``(tables, manifest)``.
    """
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    if reset and db_path.exists():
        db_path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticGenerator(SyntheticConfig(random_seed=random_seed))
    tables = generator.generate_all()
    manifest = generator.manifest(tables)

    db = MRODatabase(db_path)
    try:
        db.create_schema()
        register_data_sources(db)
        for name, df in tables.items():
            db.insert_dataframe(name, df)
            record_ingestion_run(db, SYNTHETIC_BUSINESS_SOURCE_ID, name, df)
        # Parquet exports (one per table) + manifest.
        for name, df in tables.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        db.export_all_parquet(out_dir / "db_export")
    finally:
        db.close()

    return tables, manifest
