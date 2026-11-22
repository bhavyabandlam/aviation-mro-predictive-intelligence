"""SQLite access layer for the MRO business database (SRS v1.1 §10).

Provides :class:`MRODatabase`, a thin, safe wrapper over the standard-library
``sqlite3`` module. It creates the schema, inserts/reads data, upserts model
registry rows, and exports tables to Parquet.

All connections enable ``PRAGMA foreign_keys = ON`` so referential integrity
is enforced by the database.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from aviation_mro.persistence.schema import TABLE_DDL, TABLE_NAMES

__all__ = ["MRODatabase", "TABLE_NAMES"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "database" / "mro.db"


class MRODatabase:
    """Connection + schema management for the MRO SQLite database."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # -- connection management --------------------------------------------
    def connect(self) -> sqlite3.Connection:
        """Open (or reuse) the connection with foreign keys enforced."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> MRODatabase:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- schema -----------------------------------------------------------
    def create_schema(self) -> None:
        """Create all tables if they do not exist."""
        conn = self.connect()
        cur = conn.cursor()
        for name in TABLE_NAMES:
            cur.execute(TABLE_DDL[name])
        conn.commit()

    def table_names(self) -> list[str]:
        """Return the names of all tables defined by the schema."""
        return list(TABLE_NAMES)

    def existing_tables(self) -> list[str]:
        """Return table names that currently exist in the database."""
        conn = self.connect()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    # -- writes -----------------------------------------------------------
    def insert_dataframe(self, table: str, df: pd.DataFrame, if_exists: str = "append") -> int:
        """Insert a DataFrame into a table. Returns rows written.

        Dict/list cells are serialised to JSON strings so they can be stored in
        TEXT columns (e.g. ``params_json``, ``metrics_json``, ``features_json``).
        """
        if table not in TABLE_DDL:
            raise ValueError(f"Unknown table: {table}")
        conn = self.connect()
        df = df.copy()
        for col in df.columns:
            if df[col].map(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].map(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        df.to_sql(table, conn, if_exists=if_exists, index=False)  # type: ignore[arg-type]
        conn.commit()
        return len(df)

    def upsert_model_registry(self, record: dict[str, Any]) -> None:
        """Insert or replace a single model-registry row."""
        conn = self.connect()
        columns = [
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
        ]
        placeholders = ", ".join(["?"] * len(columns))
        values = [self._serialise(record.get(c)) for c in columns]
        conn.execute(
            f"INSERT OR REPLACE INTO model_registry ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    @staticmethod
    def _serialise(value: Any) -> Any:
        """Serialise dict/list values to JSON strings for storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return value

    # -- reads ------------------------------------------------------------
    def query(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """Run a SELECT and return rows as dictionaries."""
        conn = self.connect()
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params or ())
        return [dict(row) for row in cur.fetchall()]

    def read_table(self, table: str) -> pd.DataFrame:
        """Read an entire table into a DataFrame."""
        return pd.read_sql_query(f"SELECT * FROM {table}", self.connect())

    def count(self, table: str) -> int:
        """Return row count for a table."""
        conn = self.connect()
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    # -- exports ----------------------------------------------------------
    def export_table_parquet(self, table: str, out_path: str | Path) -> Path:
        """Export one table to a Parquet file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_table(table).to_parquet(out_path, index=False)
        return out_path

    def export_all_parquet(self, out_dir: str | Path) -> dict[str, Path]:
        """Export every existing table to Parquet; returns name->path map."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for table in self.existing_tables():
            path = out_dir / f"{table}.parquet"
            self.read_table(table).to_parquet(path, index=False)
            written[table] = path
        return written
