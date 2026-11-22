"""Tests for the typed configuration layer (SRS v1.1 §5.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aviation_mro import config  # noqa: E402


class TestConfigLoading:
    def test_data_sources_loads(self) -> None:
        data = config.load_data_sources()
        assert "sources" in data
        assert "DS-01" in data["sources"]
        assert data["sources"]["DS-01"]["type"] == "simulated_benchmark"

    def test_model_config_loads(self) -> None:
        cfg = config.load_model_config()
        assert cfg["random_seed"] == 42
        assert cfg["rul"]["rul_cap"] == 125
        assert cfg["anomaly"]["n_estimators"] == 200
        assert cfg["forecasting"]["seasonal_candidate_periods"] == [52, 13]

    def test_business_rules_load(self) -> None:
        rules = config.load_business_rules()
        assert rules["inventory"]["default_service_level"] == 0.95
        assert rules["financial"]["holding_cost_rate"] == 0.20
        assert rules["financial"]["order_cost"] == 75.0

    def test_get_accessor(self) -> None:
        assert config.get("inventory", "default_service_level") == 0.95
        assert config.get("inventory", "does_not_exist", "fallback") == "fallback"

    def test_missing_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "CONFIG_DIR", PROJECT_ROOT / "does_not_exist")
        with pytest.raises(FileNotFoundError):
            config.load_business_rules()
