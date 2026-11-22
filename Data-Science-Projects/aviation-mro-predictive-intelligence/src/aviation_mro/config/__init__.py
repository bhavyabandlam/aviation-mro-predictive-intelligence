"""Typed configuration loading for the Aviation MRO platform (SRS v1.1 §5.4).

Configuration lives in ``config/*.yaml`` at the project root and is loaded once
and cached. Loaders validate the presence of required keys and raise clear
errors on malformed files rather than failing deep inside business logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"

__all__ = [
    "CONFIG_DIR",
    "load_data_sources",
    "load_model_config",
    "load_business_rules",
    "get",
]


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level")
    return data


def load_data_sources() -> dict[str, Any]:
    """Load ``config/data_sources.yaml`` and validate the ``sources`` mapping."""
    data = _load_yaml("data_sources.yaml")
    if "sources" not in data or not isinstance(data["sources"], dict):
        raise ValueError("data_sources.yaml must define a top-level 'sources' mapping")
    return data


def load_model_config() -> dict[str, Any]:
    """Load ``config/model_config.yaml``."""
    return _load_yaml("model_config.yaml")


def load_business_rules() -> dict[str, Any]:
    """Load ``config/business_rules.yaml``."""
    return _load_yaml("business_rules.yaml")


def get(section: str, key: str, default: Any = None) -> Any:
    """Convenience accessor for a nested business-rule value.

    Example: ``get("inventory", "default_service_level", 0.95)``.
    """
    rules = load_business_rules()
    return rules.get(section, {}).get(key, default)
