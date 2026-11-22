"""Data validation for the synthetic business tables (SRS v1.1 §6, FR-02).

Provides schema, uniqueness, foreign-key and range checks and a consolidated
quality report. Validation never mutates data; it returns structured errors.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Required columns per table (must match the SQLite schema).
EXPECTED_SCHEMAS: dict[str, list[str]] = {
    "technicians": ["technician_id", "shop_id", "role", "hourly_rate"],
    "aircraft": ["aircraft_id", "registration", "fleet_type", "year_built"],
    "engines": ["engine_id", "aircraft_id", "shop_id", "part_family", "install_date"],
    "inventory_parts": [
        "part_id",
        "part_number",
        "part_family",
        "shop_id",
        "unit_cost",
        "lead_time_days",
        "lead_time_std_days",
        "current_stock",
        "is_synthetic",
    ],
    "maintenance_orders": [
        "order_id",
        "aircraft_id",
        "engine_id",
        "technician_id",
        "part_id",
        "order_date",
        "order_type",
        "status",
        "labor_hours",
        "downtime_hours",
    ],
    "inventory_transactions": [
        "txn_id",
        "part_id",
        "order_id",
        "txn_date",
        "quantity",
        "unit_cost",
        "txn_type",
    ],
    "downtime_events": [
        "event_id",
        "aircraft_id",
        "order_id",
        "start_time",
        "end_time",
        "downtime_hours",
        "cost_per_hour",
    ],
}

PRIMARY_KEYS: dict[str, str] = {
    "technicians": "technician_id",
    "aircraft": "aircraft_id",
    "engines": "engine_id",
    "inventory_parts": "part_id",
    "maintenance_orders": "order_id",
    "inventory_transactions": "txn_id",
    "downtime_events": "event_id",
}

# Foreign keys: table -> list of (column, referenced_table, referenced_column, nullable)
FOREIGN_KEYS: dict[str, list[tuple[str, str, str, bool]]] = {
    "engines": [("aircraft_id", "aircraft", "aircraft_id", False)],
    "maintenance_orders": [
        ("aircraft_id", "aircraft", "aircraft_id", False),
        ("engine_id", "engines", "engine_id", True),
        ("technician_id", "technicians", "technician_id", True),
        ("part_id", "inventory_parts", "part_id", True),
    ],
    "inventory_transactions": [
        ("part_id", "inventory_parts", "part_id", False),
        ("order_id", "maintenance_orders", "order_id", True),
    ],
    "downtime_events": [
        ("aircraft_id", "aircraft", "aircraft_id", False),
        ("order_id", "maintenance_orders", "order_id", True),
    ],
}


def check_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return a list of missing-column errors."""
    return [f"missing column: {c}" for c in required if c not in df.columns]


def check_unique(df: pd.DataFrame, column: str) -> list[str]:
    """Return an error if the primary key is not unique."""
    if column in df.columns and not df[column].is_unique:
        return [f"duplicate values in primary key: {column}"]
    return []


def check_foreign_keys(
    tables: dict[str, pd.DataFrame],
    table: str,
    column: str,
    ref_table: str,
    ref_column: str,
    nullable: bool,
) -> list[str]:
    """Return an error if any non-null FK value is missing from the parent table."""
    if table not in tables or ref_table not in tables:
        return []
    child = tables[table][column].dropna() if nullable else tables[table][column]
    parent = set(tables[ref_table][ref_column])
    orphans = set(child) - parent
    if orphans:
        return [f"{table}.{column} has {len(orphans)} orphan(s) not in {ref_table}.{ref_column}"]
    return []


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate all tables and return a structured quality report."""
    report: dict[str, Any] = {}
    for name, df in tables.items():
        errors: list[str] = []
        if name in EXPECTED_SCHEMAS:
            errors.extend(check_required_columns(df, EXPECTED_SCHEMAS[name]))
        if name in PRIMARY_KEYS:
            errors.extend(check_unique(df, PRIMARY_KEYS[name]))
        for col, ref_table, ref_col, nullable in FOREIGN_KEYS.get(name, []):
            errors.extend(check_foreign_keys(tables, name, col, ref_table, ref_col, nullable))
        report[name] = {
            "rows": int(len(df)),
            "n_errors": len(errors),
            "errors": errors,
            "passed": len(errors) == 0,
        }
    all_passed = all(v["passed"] for v in report.values())
    return {"passed": all_passed, "tables": report}
