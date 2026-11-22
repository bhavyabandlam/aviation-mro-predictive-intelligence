"""Pytest configuration and shared fixtures for aviation-mro tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aviation_mro.features.rul_features import FeatureConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-integration`` flag.

    Integration tests train models on the full FD001 dataset and take many
    minutes; they are opt-in so the default suite stays fast.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that train models on the real FD001 dataset (slow)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``integration`` marker."""
    config.addinivalue_line(
        "markers",
        "integration: test trains a model on the real FD001 dataset "
        "(slow; opt-in via --run-integration)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless ``--run-integration`` was passed.

    The skip is explicit (shows as ``SKIPPED`` with a reason), not silent.
    """
    if config.getoption("--run-integration"):
        return
    skip_marker = pytest.mark.skip(
        reason="integration test (trains on real FD001 data); run with --run-integration"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


# Column names matching the FD001 dataset
COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


@pytest.fixture(scope="session")
def constant_sensors() -> list[str]:
    """Sensors identified as constant (zero variance) in EDA."""
    return ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


@pytest.fixture(scope="session")
def informative_sensors() -> list[str]:
    """Sensors that vary with degradation (non-constant)."""
    return [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21",
    ]


@pytest.fixture(scope="function")
def synthetic_train_data() -> pd.DataFrame:
    """Create small synthetic train data for unit testing.

    3 engines, varying life lengths, with known degradation trends.
    """
    np.random.seed(42)
    rows = []

    # Engine 1: 10 cycles, sensor_3 increases by ~0.5 per cycle
    for cycle in range(1, 11):
        rows.append(
            {
                "unit_id": 1,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,  # constant
                "sensor_2": 640.0 + 0.1 * cycle + np.random.normal(0, 0.05),
                "sensor_3": 1580.0 + 0.5 * cycle + np.random.normal(0, 0.1),  # strong trend
                "sensor_4": 1400.0 - 0.2 * cycle + np.random.normal(0, 0.1),
                "sensor_5": 14.62,  # constant
                "sensor_6": 21.61 + np.random.normal(0, 0.01),
                "sensor_7": 550.0 - 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_8": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_9": 9040.0 - 0.3 * cycle + np.random.normal(0, 1.0),
                "sensor_10": 1.30,  # constant
                "sensor_11": 47.0 - 0.02 * cycle + np.random.normal(0, 0.05),
                "sensor_12": 520.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_13": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_14": 8130.0 + 0.2 * cycle + np.random.normal(0, 1.0),
                "sensor_15": 8.4 + np.random.normal(0, 0.01),
                "sensor_16": 0.03,  # constant
                "sensor_17": 390.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_18": 2388.0,  # constant
                "sensor_19": 100.0,  # constant
                "sensor_20": 39.0 - 0.01 * cycle + np.random.normal(0, 0.05),
                "sensor_21": 23.4 + np.random.normal(0, 0.01),
            }
        )

    # Engine 2: 8 cycles, different starting point
    for cycle in range(1, 9):
        rows.append(
            {
                "unit_id": 2,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 642.0 + 0.1 * cycle + np.random.normal(0, 0.05),
                "sensor_3": 1585.0 + 0.5 * cycle + np.random.normal(0, 0.1),
                "sensor_4": 1395.0 - 0.2 * cycle + np.random.normal(0, 0.1),
                "sensor_5": 14.62,
                "sensor_6": 21.61 + np.random.normal(0, 0.01),
                "sensor_7": 552.0 - 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_8": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_9": 9050.0 - 0.3 * cycle + np.random.normal(0, 1.0),
                "sensor_10": 1.30,
                "sensor_11": 47.5 - 0.02 * cycle + np.random.normal(0, 0.05),
                "sensor_12": 522.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_13": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_14": 8140.0 + 0.2 * cycle + np.random.normal(0, 1.0),
                "sensor_15": 8.4 + np.random.normal(0, 0.01),
                "sensor_16": 0.03,
                "sensor_17": 392.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 39.2 - 0.01 * cycle + np.random.normal(0, 0.05),
                "sensor_21": 23.5 + np.random.normal(0, 0.01),
            }
        )

    # Engine 3: 12 cycles
    for cycle in range(1, 13):
        rows.append(
            {
                "unit_id": 3,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 638.0 + 0.1 * cycle + np.random.normal(0, 0.05),
                "sensor_3": 1575.0 + 0.5 * cycle + np.random.normal(0, 0.1),
                "sensor_4": 1405.0 - 0.2 * cycle + np.random.normal(0, 0.1),
                "sensor_5": 14.62,
                "sensor_6": 21.61 + np.random.normal(0, 0.01),
                "sensor_7": 548.0 - 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_8": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_9": 9030.0 - 0.3 * cycle + np.random.normal(0, 1.0),
                "sensor_10": 1.30,
                "sensor_11": 46.5 - 0.02 * cycle + np.random.normal(0, 0.05),
                "sensor_12": 518.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_13": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_14": 8120.0 + 0.2 * cycle + np.random.normal(0, 1.0),
                "sensor_15": 8.4 + np.random.normal(0, 0.01),
                "sensor_16": 0.03,
                "sensor_17": 388.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 38.8 - 0.01 * cycle + np.random.normal(0, 0.05),
                "sensor_21": 23.3 + np.random.normal(0, 0.01),
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture(scope="function")
def synthetic_test_data() -> pd.DataFrame:
    """Create small synthetic test data (truncated trajectories)."""
    np.random.seed(123)
    rows = []

    # Engine 1: truncated at cycle 7 (true RUL = 3)
    for cycle in range(1, 8):
        rows.append(
            {
                "unit_id": 1,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 640.0 + 0.1 * cycle + np.random.normal(0, 0.05),
                "sensor_3": 1580.0 + 0.5 * cycle + np.random.normal(0, 0.1),
                "sensor_4": 1400.0 - 0.2 * cycle + np.random.normal(0, 0.1),
                "sensor_5": 14.62,
                "sensor_6": 21.61 + np.random.normal(0, 0.01),
                "sensor_7": 550.0 - 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_8": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_9": 9040.0 - 0.3 * cycle + np.random.normal(0, 1.0),
                "sensor_10": 1.30,
                "sensor_11": 47.0 - 0.02 * cycle + np.random.normal(0, 0.05),
                "sensor_12": 520.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_13": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_14": 8130.0 + 0.2 * cycle + np.random.normal(0, 1.0),
                "sensor_15": 8.4 + np.random.normal(0, 0.01),
                "sensor_16": 0.03,
                "sensor_17": 390.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 39.0 - 0.01 * cycle + np.random.normal(0, 0.05),
                "sensor_21": 23.4 + np.random.normal(0, 0.01),
            }
        )

    # Engine 2: truncated at cycle 5 (true RUL = 3)
    for cycle in range(1, 6):
        rows.append(
            {
                "unit_id": 2,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 642.0 + 0.1 * cycle + np.random.normal(0, 0.05),
                "sensor_3": 1585.0 + 0.5 * cycle + np.random.normal(0, 0.1),
                "sensor_4": 1395.0 - 0.2 * cycle + np.random.normal(0, 0.1),
                "sensor_5": 14.62,
                "sensor_6": 21.61 + np.random.normal(0, 0.01),
                "sensor_7": 552.0 - 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_8": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_9": 9050.0 - 0.3 * cycle + np.random.normal(0, 1.0),
                "sensor_10": 1.30,
                "sensor_11": 47.5 - 0.02 * cycle + np.random.normal(0, 0.05),
                "sensor_12": 522.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_13": 2388.0 + 0.05 * cycle + np.random.normal(0, 0.1),
                "sensor_14": 8140.0 + 0.2 * cycle + np.random.normal(0, 1.0),
                "sensor_15": 8.4 + np.random.normal(0, 0.01),
                "sensor_16": 0.03,
                "sensor_17": 392.0 + 0.1 * cycle + np.random.normal(0, 0.1),
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 39.2 - 0.01 * cycle + np.random.normal(0, 0.05),
                "sensor_21": 23.5 + np.random.normal(0, 0.01),
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture(scope="function")
def synthetic_rul_data() -> pd.DataFrame:
    """True RUL values for synthetic test engines."""
    return pd.DataFrame(
        {
            "unit_id": [1, 2],
            "rul": [3, 3],  # cycles remaining after truncation
        }
    )


@pytest.fixture(scope="function")
def feature_config() -> FeatureConfig:
    """Default feature configuration for testing (cycle_norm disabled by default)."""
    return FeatureConfig(
        window_sizes=(3, 5),  # smaller windows for small synthetic data
        include_op_settings=True,
        include_cycle_norm=False,
        rul_cap=125,
        random_seed=42,
    )
