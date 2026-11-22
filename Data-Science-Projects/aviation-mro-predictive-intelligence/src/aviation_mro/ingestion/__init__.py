"""Ingestion package: synthetic business-data generation (SRS §5.3, FR-03)."""

from __future__ import annotations

from aviation_mro.ingestion.synthetic_generator import (
    SyntheticConfig,
    SyntheticGenerator,
    build_business_database,
)

__all__ = ["SyntheticConfig", "SyntheticGenerator", "build_business_database"]
