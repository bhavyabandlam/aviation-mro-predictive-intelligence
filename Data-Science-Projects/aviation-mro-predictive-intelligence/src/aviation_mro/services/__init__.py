"""Aviation MRO application services (SRS v1.1 Phase 5).

Provides:
- Structured logging (NFR-07): ``logging`` module
- Safe TTL caching (NFR-08): ``caching`` module
- Model registry (FR-14): ``model_registry`` module
- Central KPI registry (§11): ``kpi_registry`` module
"""

from aviation_mro.services.caching import cache_stats, clear_all_caches, ttl_cache
from aviation_mro.services.kpi_registry import (
    KPIDefinition,
    KPIRegistry,
    get_kpi,
    get_registry,
    list_kpis,
    register_all_kpis,
)
from aviation_mro.services.logging import get_logger, setup_logging
from aviation_mro.services.model_registry import (
    ModelRegistryEntry,
    ModelRegistryService,
    get_model_by_id_version,
    list_registered_models,
    register_saved_artefacts,
)

__all__ = [
    "KPIDefinition",
    "KPIRegistry",
    "ModelRegistryEntry",
    "ModelRegistryService",
    "cache_stats",
    "clear_all_caches",
    "get_kpi",
    "get_logger",
    "get_model_by_id_version",
    "get_registry",
    "list_kpis",
    "list_registered_models",
    "register_all_kpis",
    "register_saved_artefacts",
    "setup_logging",
    "ttl_cache",
]
