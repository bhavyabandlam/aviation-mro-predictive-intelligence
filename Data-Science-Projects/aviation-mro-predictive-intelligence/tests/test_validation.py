"""Tests for the validation layer (SRS v1.1 §6, FR-02)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aviation_mro.ingestion import SyntheticConfig, SyntheticGenerator  # noqa: E402
from aviation_mro.validation import (  # noqa: E402
    EXPECTED_SCHEMAS,
    check_foreign_keys,
    check_required_columns,
    check_unique,
    validate_tables,
)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return SyntheticGenerator(SyntheticConfig(random_seed=42)).generate_all()


class TestPositiveValidation:
    def test_clean_data_passes(self, tables: dict[str, pd.DataFrame]) -> None:
        report = validate_tables(tables)
        assert report["passed"] is True
        for name, result in report["tables"].items():
            assert result["passed"], f"{name}: {result['errors']}"
            assert result["n_errors"] == 0

    def test_all_expected_tables_validated(self, tables: dict[str, pd.DataFrame]) -> None:
        report = validate_tables(tables)
        assert set(report["tables"].keys()) == set(EXPECTED_SCHEMAS.keys())


class TestNegativeValidation:
    def test_missing_column_detected(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        errors = check_required_columns(df, ["a", "b", "c"])
        assert errors == ["missing column: c"]

    def test_duplicate_pk_detected(self) -> None:
        df = pd.DataFrame({"id": [1, 1, 2]})
        errors = check_unique(df, "id")
        assert errors and "primary key" in errors[0]

    def test_unique_ok(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3]})
        assert check_unique(df, "id") == []

    def test_orphan_fk_detected(self, tables: dict[str, pd.DataFrame]) -> None:
        broken = {k: v.copy() for k, v in tables.items()}
        broken["maintenance_orders"].loc[0, "aircraft_id"] = "ac_MISSING"
        errors = check_foreign_keys(
            broken, "maintenance_orders", "aircraft_id", "aircraft", "aircraft_id", False
        )
        assert errors and "orphan" in errors[0]

    def test_report_marks_failures(self, tables: dict[str, pd.DataFrame]) -> None:
        broken = {k: v.copy() for k, v in tables.items()}
        broken["aircraft"] = broken["aircraft"].drop(columns=["fleet_type"])
        report = validate_tables(broken)
        assert report["passed"] is False
        assert report["tables"]["aircraft"]["passed"] is False
        assert any("fleet_type" in e for e in report["tables"]["aircraft"]["errors"])
