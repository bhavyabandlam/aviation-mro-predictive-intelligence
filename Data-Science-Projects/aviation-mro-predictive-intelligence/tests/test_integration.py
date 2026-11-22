"""End-to-end persistence integration test (SRS v1.1 §10).

Exercises the full SQLite business layer: schema creation, inserts across the
13 tables with foreign-key enforcement, model-registry upsert, and Parquet
export round-trip. Uses only synthetic in-memory data (no external downloads,
no model training). Fast; runs in the default suite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from aviation_mro.persistence import TABLE_NAMES, MRODatabase  # noqa: E402


@pytest.fixture()
def db(tmp_path: Path) -> MRODatabase:
    database = MRODatabase(tmp_path / "integration_mro.db")
    database.create_schema()
    yield database
    database.close()


class TestEndToEndPersistence:
    def test_all_thirteen_tables_created(self, db: MRODatabase) -> None:
        assert set(db.existing_tables()) == set(TABLE_NAMES)
        assert len(TABLE_NAMES) == 13

    def test_full_business_flow_and_parquet_round_trip(
        self, db: MRODatabase, tmp_path: Path
    ) -> None:
        # 1. Reference data.
        aircraft = pd.DataFrame(
            [
                {
                    "aircraft_id": "A1",
                    "registration": "N100AA",
                    "fleet_type": "NARROW_BODY",
                    "year_built": 2018,
                }
            ]
        )
        assert db.insert_dataframe("aircraft", aircraft) == 1

        technicians = pd.DataFrame(
            [
                {
                    "technician_id": "T1",
                    "shop_id": "S1",
                    "role": "MECH",
                    "hourly_rate": 120.0,
                }
            ]
        )
        assert db.insert_dataframe("technicians", technicians) == 1

        parts = pd.DataFrame(
            [
                {
                    "part_id": "P1",
                    "part_number": "PN-100",
                    "part_family": "ENGINE",
                    "shop_id": "S1",
                    "unit_cost": 500.0,
                    "lead_time_days": 3.0,
                    "lead_time_std_days": 0.5,
                    "current_stock": 10.0,
                    "is_synthetic": 1,
                }
            ]
        )
        assert db.insert_dataframe("inventory_parts", parts) == 1

        # 2. Engine referencing aircraft (FK).
        engines = pd.DataFrame(
            [
                {
                    "engine_id": "E1",
                    "aircraft_id": "A1",
                    "shop_id": "S1",
                    "part_family": "ENGINE",
                    "install_date": "2020-01-01",
                }
            ]
        )
        assert db.insert_dataframe("engines", engines) == 1

        # 3. Maintenance order referencing aircraft/engine/technician/part (FKs).
        orders = pd.DataFrame(
            [
                {
                    "order_id": "O1",
                    "aircraft_id": "A1",
                    "engine_id": "E1",
                    "technician_id": "T1",
                    "part_id": "P1",
                    "order_date": "2024-01-10",
                    "order_type": "REPAIR",
                    "status": "CLOSED",
                    "labor_hours": 4.0,
                    "downtime_hours": 8.0,
                }
            ]
        )
        assert db.insert_dataframe("maintenance_orders", orders) == 1

        # 4. Inventory transaction referencing part + order (FKs).
        txns = pd.DataFrame(
            [
                {
                    "txn_id": "TX1",
                    "part_id": "P1",
                    "order_id": "O1",
                    "txn_date": "2024-01-10",
                    "quantity": 1.0,
                    "unit_cost": 500.0,
                    "txn_type": "ISSUE",
                }
            ]
        )
        assert db.insert_dataframe("inventory_transactions", txns) == 1

        # 5. Downtime event referencing aircraft + order (FKs).
        downtime = pd.DataFrame(
            [
                {
                    "event_id": "D1",
                    "aircraft_id": "A1",
                    "order_id": "O1",
                    "start_time": "2024-01-10T08:00:00Z",
                    "end_time": "2024-01-10T16:00:00Z",
                    "downtime_hours": 8.0,
                    "cost_per_hour": 1500.0,
                }
            ]
        )
        assert db.insert_dataframe("downtime_events", downtime) == 1

        # 6. Ingestion run referencing a data source (FK).
        db.insert_dataframe(
            "data_sources",
            pd.DataFrame(
                [
                    {
                        "source_id": "SYN-MRO-BIZ",
                        "title": "Synthetic MRO business data",
                        "source_type": "synthetic",
                        "status": "active",
                        "local_path": "data/processed/inventory_data.parquet",
                    }
                ]
            ),
        )
        runs = pd.DataFrame(
            [
                {
                    "run_id": "RUN1",
                    "source_id": "SYN-MRO-BIZ",
                    "run_timestamp": "2024-01-10T12:00:00Z",
                    "file_name": "inventory_data.parquet",
                    "rows_ingested": 200,
                    "checksum": "abc123",
                    "status": "OK",
                }
            ]
        )
        assert db.insert_dataframe("ingestion_runs", runs) == 1

        # 7. Model registry row.
        db.upsert_model_registry(
            {
                "model_id": "rul_v2",
                "model_name": "GradientBoosting",
                "version": "2",
                "algorithm": "GradientBoostingRegressor",
                "dataset_fingerprint": "fp123",
                "features_json": ["a", "b"],
                "params_json": {"n_estimators": 100},
                "metrics_json": {"r2": 0.8},
                "random_seed": 42,
                "artifact_path": "models/baseline_rul_model_v2.pkl",
                "training_date": "2024-01-10",
                "notes": "integration test row",
            }
        )
        rows = db.query("SELECT * FROM model_registry")
        assert len(rows) == 1
        assert rows[0]["random_seed"] == 42

        # 8. Prediction + anomaly result referencing engine (FK).
        assert (
            db.insert_dataframe(
                "engine_predictions",
                pd.DataFrame(
                    [
                        {
                            "prediction_id": "PR1",
                            "engine_id": "E1",
                            "model_name": "GradientBoosting",
                            "model_version": "2",
                            "predicted_rul": 45.0,
                            "actual_rul": 50.0,
                            "prediction_date": "2024-01-10",
                        }
                    ]
                ),
            )
            == 1
        )
        assert (
            db.insert_dataframe(
                "anomaly_results",
                pd.DataFrame(
                    [
                        {
                            "result_id": "AN1",
                            "engine_id": "E1",
                            "model_name": "IsolationForest",
                            "anomaly_score": -0.08,
                            "is_anomaly": 1,
                            "threshold": -0.07955,
                            "run_date": "2024-01-10",
                        }
                    ]
                ),
            )
            == 1
        )

        # 9. Experiment row.
        assert (
            db.insert_dataframe(
                "experiments",
                pd.DataFrame(
                    [
                        {
                            "experiment_id": "EXP1",
                            "model_name": "GradientBoosting",
                            "run_date": "2024-01-10",
                            "params_json": {"n_estimators": 100},
                            "metrics_json": {"mae": 13.5},
                            "notes": "integration test",
                        }
                    ]
                ),
            )
            == 1
        )

        # 10. Referential integrity: a prediction referencing a missing engine
        #     must be rejected.
        bad = pd.DataFrame(
            [
                {
                    "prediction_id": "PR_BAD",
                    "engine_id": "NO_SUCH_ENGINE",
                    "model_name": "X",
                    "predicted_rul": 1.0,
                    "prediction_date": "2024-01-10",
                }
            ]
        )
        with pytest.raises(pd.errors.DatabaseError):
            db.insert_dataframe("engine_predictions", bad)

        # 11. Parquet export round-trip: every table written and readable back.
        export_dir = tmp_path / "parquet_export"
        written = db.export_all_parquet(export_dir)
        assert set(written.keys()) == set(db.existing_tables())
        for table, path in written.items():
            assert path.exists()
            back = pd.read_parquet(path)
            # Row counts must match what is in the database.
            assert len(back) == db.count(table), (
                f"row-count mismatch for {table}: {len(back)} vs {db.count(table)}"
            )
