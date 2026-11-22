"""SQLite schema for the Aviation MRO business-data layer (SRS v1.1 §10).

Defines the 13 tables required by the SRS. All tables are created with primary
keys and foreign keys; ``PRAGMA foreign_keys`` is enabled per connection so the
database enforces referential integrity.

This module is pure DDL/data definitions (no side effects).
"""

from __future__ import annotations

# Ordered list of table names (creation order respects FK dependencies).
TABLE_NAMES: list[str] = [
    "data_sources",
    "technicians",
    "aircraft",
    "engines",
    "inventory_parts",
    "maintenance_orders",
    "inventory_transactions",
    "downtime_events",
    "ingestion_runs",
    "engine_predictions",
    "anomaly_results",
    "model_registry",
    "experiments",
]

# Table -> CREATE statement. Types: TEXT, INTEGER, REAL.
TABLE_DDL: dict[str, str] = {
    "data_sources": """
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id   TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            source_type TEXT NOT NULL,
            status      TEXT NOT NULL,
            local_path  TEXT
        )
    """,
    "technicians": """
        CREATE TABLE IF NOT EXISTS technicians (
            technician_id TEXT PRIMARY KEY,
            shop_id       TEXT NOT NULL,
            role          TEXT NOT NULL,
            hourly_rate   REAL NOT NULL
        )
    """,
    "aircraft": """
        CREATE TABLE IF NOT EXISTS aircraft (
            aircraft_id  TEXT PRIMARY KEY,
            registration TEXT NOT NULL,
            fleet_type   TEXT NOT NULL,
            year_built   INTEGER
        )
    """,
    "engines": """
        CREATE TABLE IF NOT EXISTS engines (
            engine_id    TEXT PRIMARY KEY,
            aircraft_id  TEXT NOT NULL,
            shop_id      TEXT NOT NULL,
            part_family  TEXT,
            install_date TEXT,
            FOREIGN KEY (aircraft_id) REFERENCES aircraft(aircraft_id)
        )
    """,
    "inventory_parts": """
        CREATE TABLE IF NOT EXISTS inventory_parts (
            part_id            TEXT PRIMARY KEY,
            part_number        TEXT NOT NULL,
            part_family        TEXT NOT NULL,
            shop_id            TEXT NOT NULL,
            unit_cost          REAL NOT NULL,
            lead_time_days     REAL NOT NULL,
            lead_time_std_days REAL,
            current_stock      REAL NOT NULL,
            is_synthetic       INTEGER NOT NULL DEFAULT 1
        )
    """,
    "maintenance_orders": """
        CREATE TABLE IF NOT EXISTS maintenance_orders (
            order_id       TEXT PRIMARY KEY,
            aircraft_id    TEXT NOT NULL,
            engine_id      TEXT,
            technician_id  TEXT,
            part_id        TEXT,
            order_date     TEXT NOT NULL,
            order_type     TEXT NOT NULL,
            status         TEXT NOT NULL,
            labor_hours    REAL NOT NULL,
            downtime_hours REAL NOT NULL,
            FOREIGN KEY (aircraft_id)   REFERENCES aircraft(aircraft_id),
            FOREIGN KEY (engine_id)     REFERENCES engines(engine_id),
            FOREIGN KEY (technician_id) REFERENCES technicians(technician_id),
            FOREIGN KEY (part_id)       REFERENCES inventory_parts(part_id)
        )
    """,
    "inventory_transactions": """
        CREATE TABLE IF NOT EXISTS inventory_transactions (
            txn_id    TEXT PRIMARY KEY,
            part_id   TEXT NOT NULL,
            order_id  TEXT,
            txn_date  TEXT NOT NULL,
            quantity  REAL NOT NULL,
            unit_cost REAL NOT NULL,
            txn_type  TEXT NOT NULL,
            FOREIGN KEY (part_id)  REFERENCES inventory_parts(part_id),
            FOREIGN KEY (order_id) REFERENCES maintenance_orders(order_id)
        )
    """,
    "downtime_events": """
        CREATE TABLE IF NOT EXISTS downtime_events (
            event_id       TEXT PRIMARY KEY,
            aircraft_id    TEXT NOT NULL,
            order_id       TEXT,
            start_time     TEXT,
            end_time       TEXT,
            downtime_hours REAL NOT NULL,
            cost_per_hour  REAL NOT NULL,
            FOREIGN KEY (aircraft_id) REFERENCES aircraft(aircraft_id),
            FOREIGN KEY (order_id)    REFERENCES maintenance_orders(order_id)
        )
    """,
    "ingestion_runs": """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id         TEXT PRIMARY KEY,
            source_id      TEXT NOT NULL,
            run_timestamp  TEXT NOT NULL,
            file_name      TEXT,
            rows_ingested  INTEGER,
            checksum       TEXT,
            status         TEXT NOT NULL,
            FOREIGN KEY (source_id) REFERENCES data_sources(source_id)
        )
    """,
    "engine_predictions": """
        CREATE TABLE IF NOT EXISTS engine_predictions (
            prediction_id   TEXT PRIMARY KEY,
            engine_id       TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            model_version   TEXT,
            predicted_rul   REAL NOT NULL,
            actual_rul      REAL,
            prediction_date TEXT,
            FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
        )
    """,
    "anomaly_results": """
        CREATE TABLE IF NOT EXISTS anomaly_results (
            result_id     TEXT PRIMARY KEY,
            engine_id     TEXT NOT NULL,
            model_name    TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            is_anomaly    INTEGER NOT NULL,
            threshold     REAL NOT NULL,
            run_date      TEXT,
            FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
        )
    """,
    "model_registry": """
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id            TEXT PRIMARY KEY,
            model_name          TEXT NOT NULL,
            version             TEXT,
            algorithm           TEXT,
            dataset_fingerprint TEXT,
            features_json       TEXT,
            params_json         TEXT,
            metrics_json        TEXT,
            random_seed         INTEGER,
            artifact_path       TEXT,
            training_date       TEXT,
            notes               TEXT
        )
    """,
    "experiments": """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            model_name    TEXT NOT NULL,
            run_date      TEXT,
            params_json   TEXT,
            metrics_json  TEXT,
            notes         TEXT
        )
    """,
}
