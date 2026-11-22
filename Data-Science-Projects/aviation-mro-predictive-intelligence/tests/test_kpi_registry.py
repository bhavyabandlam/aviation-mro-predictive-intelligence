"""Tests for the central KPI registry (SRS v1.1 §11).

Covers KPI registration, lookup, all required fields (definition, formula,
unit, source, provenance, synthetic-data disclosure), and the full set of
§11 KPIs.
"""

from __future__ import annotations

from aviation_mro.services.kpi_registry import (
    KPIDefinition,
    KPIRegistry,
    get_kpi,
    register_all_kpis,
)

# All KPI IDs required by SRS §11
SRS_KPI_IDS = [
    "total_engines",
    "high_risk_engines",
    "mean_predicted_rul",
    "maintenance_expenditure",
    "completion_rate",
    "downtime_hours",
    "inventory_value",
    "stockout_risk",
    "technician_utilisation",
    "inventory_turnover",
]

# Additional financial KPIs from the existing code
FINANCIAL_KPI_IDS = [
    "cost_per_flight_hour",
    "maintenance_cost_pct_revenue",
]


class TestKPIRegistryRegistration:
    """Tests for KPI registration and lookup."""

    def test_register_all_creates_srs_kpis(self) -> None:
        """register_all_kpis() creates all SRS §11 KPIs."""
        registry = register_all_kpis()
        registered_ids = {k.kpi_id for k in registry.list()}
        for kpi_id in SRS_KPI_IDS:
            assert kpi_id in registered_ids, f"Missing SRS KPI: {kpi_id}"

    def test_register_all_creates_financial_kpis(self) -> None:
        """register_all_kpis() creates financial KPIs."""
        registry = register_all_kpis()
        registered_ids = {k.kpi_id for k in registry.list()}
        for kpi_id in FINANCIAL_KPI_IDS:
            assert kpi_id in registered_ids, f"Missing financial KPI: {kpi_id}"

    def test_get_kpi_returns_definition(self) -> None:
        """get_kpi() returns the correct KPIDefinition."""
        registry = KPIRegistry()
        kpi = registry.get("total_engines")
        assert kpi is not None
        assert kpi.name == "Total Engines"
        assert kpi.unit == "engines"

    def test_get_kpi_unknown_returns_none(self) -> None:
        """get_kpi() returns None for unknown KPI ID."""
        registry = KPIRegistry()
        assert registry.get("nonexistent_kpi") is None

    def test_list_returns_all_registered(self) -> None:
        """list() returns all registered KPIs."""
        registry = KPIRegistry()
        kpis = registry.list()
        assert len(kpis) >= len(SRS_KPI_IDS)


class TestKPIFieldCompleteness:
    """Tests that every KPI has all required fields."""

    def test_every_kpi_has_definition(self) -> None:
        """Every KPI has a non-empty definition."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.definition, f"KPI {kpi.kpi_id} missing definition"

    def test_every_kpi_has_formula(self) -> None:
        """Every KPI has a non-empty formula."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.formula, f"KPI {kpi.kpi_id} missing formula"

    def test_every_kpi_has_unit(self) -> None:
        """Every KPI has a non-empty unit."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.unit, f"KPI {kpi.kpi_id} missing unit"

    def test_every_kpi_has_data_source(self) -> None:
        """Every KPI has a non-empty data source."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.data_source, f"KPI {kpi.kpi_id} missing data source"

    def test_every_kpi_has_provenance(self) -> None:
        """Every KPI has non-empty provenance."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.provenance, f"KPI {kpi.kpi_id} missing provenance"

    def test_every_kpi_has_null_policy(self) -> None:
        """Every KPI has a non-empty null policy."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.null_policy, f"KPI {kpi.kpi_id} missing null policy"

    def test_every_kpi_has_synthetic_disclosure(self) -> None:
        """Every KPI has a synthetic-data disclosure."""
        registry = KPIRegistry()
        for kpi in registry.list():
            assert kpi.synthetic_disclosure, f"KPI {kpi.kpi_id} missing synthetic disclosure"
            assert "synthetic" in kpi.synthetic_disclosure.lower(), (
                f"KPI {kpi.kpi_id} disclosure does not mention synthetic"
            )


class TestIndividualKPIs:
    """Tests for specific KPI definitions."""

    def test_total_engines_fields(self) -> None:
        """total_engines KPI has correct fields."""
        kpi = get_kpi("total_engines")
        assert kpi is not None
        assert kpi.unit == "engines"
        assert "COUNT(DISTINCT engine_id)" in kpi.formula
        assert "NASA" in kpi.data_source

    def test_mean_predicted_rul_fields(self) -> None:
        """mean_predicted_rul KPI has correct fields."""
        kpi = get_kpi("mean_predicted_rul")
        assert kpi is not None
        assert kpi.unit == "cycles"
        assert "AVG(predicted_rul" in kpi.formula

    def test_maintenance_expenditure_fields(self) -> None:
        """maintenance_expenditure KPI has correct fields."""
        kpi = get_kpi("maintenance_expenditure")
        assert kpi is not None
        assert kpi.unit == "USD"
        assert "SUM(total_cost" in kpi.formula

    def test_completion_rate_fields(self) -> None:
        """completion_rate KPI has correct fields."""
        kpi = get_kpi("completion_rate")
        assert kpi is not None
        assert kpi.unit == "%"

    def test_inventory_turnover_fields(self) -> None:
        """inventory_turnover KPI has correct fields."""
        kpi = get_kpi("inventory_turnover")
        assert kpi is not None
        assert kpi.unit == "times/year"

    def test_cost_per_flight_hour_fields(self) -> None:
        """cost_per_flight_hour KPI has correct fields."""
        kpi = get_kpi("cost_per_flight_hour")
        assert kpi is not None
        assert kpi.unit == "USD/flight_hour"
        assert "Total Annual Maintenance Cost" in kpi.formula


class TestKPIRegistryToDict:
    """Tests for KPI registry serialization."""

    def test_to_dict_returns_all_kpis(self) -> None:
        """to_dict() returns all KPIs as dictionaries."""
        registry = KPIRegistry()
        result = registry.to_dict()
        assert isinstance(result, dict)
        assert len(result) >= len(SRS_KPI_IDS)
        for kpi_id in SRS_KPI_IDS:
            assert kpi_id in result
            entry = result[kpi_id]
            assert "name" in entry
            assert "definition" in entry
            assert "formula" in entry
            assert "unit" in entry
            assert "data_source" in entry
            assert "provenance" in entry
            assert "null_policy" in entry
            assert "synthetic_disclosure" in entry
            assert "assumptions" in entry
            assert "example_value" in entry

    def test_register_is_idempotent(self) -> None:
        """Registering the same KPI twice replaces rather than duplicates."""
        registry = KPIRegistry()
        initial_count = len(registry.list())

        kpi = KPIDefinition(
            kpi_id="total_engines",
            name="Updated Total Engines",
            definition="Updated definition",
            formula="COUNT(DISTINCT engine_id)",
            unit="engines",
            data_source="test",
            provenance="test provenance",
            null_policy="Returns 0",
            synthetic_disclosure="Synthetic data",
        )
        registry.register(kpi)
        assert len(registry.list()) == initial_count
        updated = registry.get("total_engines")
        assert updated is not None
        assert updated.name == "Updated Total Engines"
