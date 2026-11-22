"""Tests for inventory recommendations module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aviation_mro.analytics.inventory.recommendations import (
    InventoryRecommendation,
    InventoryRecommendationEngine,
    generate_inventory_recommendations,
)
from src.aviation_mro.analytics.inventory.reorder_point import ReorderPointResult
from src.aviation_mro.analytics.inventory.safety_stock import SafetyStockResult
from src.aviation_mro.analytics.inventory.stockout_risk import StockoutRiskResult


class TestInventoryRecommendationEngine:
    """Test InventoryRecommendationEngine class."""

    def test_calculate_eoq(self):
        """Test EOQ calculation."""
        engine = InventoryRecommendationEngine()
        # D = 520/year, S = 50, H = 0.20 * 1000 = 200
        # EOQ = sqrt(2 * 520 * 50 / 200) = sqrt(260) = 16.1
        eoq = engine.calculate_eoq(annual_demand=520.0, unit_cost=1000.0, order_cost=50.0)
        assert abs(eoq - 16.12) < 0.1

    def test_calculate_eoq_zero_demand(self):
        """Test EOQ with zero demand."""
        engine = InventoryRecommendationEngine()
        eoq = engine.calculate_eoq(annual_demand=0.0, unit_cost=1000.0)
        assert eoq == 1.0  # min_order_quantity

    def test_calculate_eoq_zero_cost(self):
        """Test EOQ with zero unit cost."""
        engine = InventoryRecommendationEngine()
        eoq = engine.calculate_eoq(annual_demand=520.0, unit_cost=0.0)
        assert eoq == 1.0

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        engine = InventoryRecommendationEngine()
        df = self._create_test_df(5)
        rop_results = self._create_rop_results(5)
        ss_results = self._create_ss_results(5)
        risk_results = self._create_risk_results(5)

        recs = engine.generate_recommendations(df, rop_results, ss_results, risk_results)

        assert len(recs) > 0
        assert all(isinstance(r, InventoryRecommendation) for r in recs)
        # Should have Critical/High priority for low stock
        priorities = [r.priority for r in recs]
        assert "Critical" in priorities or "High" in priorities

    def test_generate_summary_report(self):
        """Test summary report generation."""
        engine = InventoryRecommendationEngine()
        recs = [
            InventoryRecommendation(
                part_number="PART-0001",
                part_family="ENGINE",
                shop_id="SHOP_01",
                recommendation_type="adjust_rop",
                priority="Critical",
                current_value=10.0,
                recommended_value=25.0,
                change_pct=150.0,
                rationale="Test",
                cost_impact=1000.0,
                assumptions=[],
            ),
            InventoryRecommendation(
                part_number="PART-0002",
                part_family="AVIONICS",
                shop_id="SHOP_02",
                recommendation_type="adjust_ss",
                priority="Medium",
                current_value=5.0,
                recommended_value=8.0,
                change_pct=60.0,
                rationale="Test",
                cost_impact=500.0,
                assumptions=[],
            ),
        ]

        summary = engine.generate_summary_report(recs)

        assert summary["total"] == 2
        assert summary["by_priority"]["Critical"] == 1
        assert summary["by_priority"]["Medium"] == 1
        assert summary["by_type"]["adjust_rop"] == 1
        assert summary["by_type"]["adjust_ss"] == 1
        assert summary["total_cost_impact"] == 1500.0
        assert summary["critical_parts"] == ["PART-0001"]

    def _create_test_df(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(["ENGINE", "AVIONICS", "AIRFRAME"], n),
                "shop_id": np.random.choice(
                    ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"], n
                ),
                "current_stock": np.random.randint(5, 30, n),  # Low stock to trigger recs
                "unit_cost": np.random.uniform(100, 10000, n),
                "demand_mean_weekly": np.random.uniform(5, 20, n),
                "lead_time_days": np.random.randint(7, 21, n),
            }
        )

    def _create_rop_results(self, n: int) -> list[ReorderPointResult]:
        return [
            ReorderPointResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                lead_time_weeks=2.0,
                mean_weekly_demand=10.0,
                std_weekly_demand=2.0,
                service_level=0.95,
                z_score=1.645,
                lead_time_demand=20.0,
                safety_stock=5.0,
                reorder_point=25.0,  # Higher than current stock
                assumptions=[],
            )
            for i in range(n)
        ]

    def _create_ss_results(self, n: int) -> list[SafetyStockResult]:
        return [
            SafetyStockResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                method="normal_constant_lt",
                service_level=0.95,
                safety_stock=5.0,
                assumptions=[],
                parameters={},
            )
            for i in range(n)
        ]

    def _create_risk_results(self, n: int) -> list[StockoutRiskResult]:
        return [
            StockoutRiskResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                current_stock=10.0,
                reorder_point=25.0,
                mean_ltd=20.0,
                std_ltd=5.0,
                service_level_target=0.95,
                stockout_probability=0.20,  # High risk
                expected_shortage_per_cycle=2.0,
                fill_rate=0.80,
                cycles_per_year=13.0,
                expected_stockouts_per_year=2.6,
                risk_category="High",
                assumptions=[],
            )
            for i in range(n)
        ]


class TestGenerateInventoryRecommendations:
    """Test convenience function."""

    def test_function(self):
        """Test generate_inventory_recommendations function."""
        df = self._create_test_df(5)
        rop_results = self._create_rop_results(5)
        ss_results = self._create_ss_results(5)
        risk_results = self._create_risk_results(5)

        recs = generate_inventory_recommendations(df, rop_results, ss_results, risk_results)

        assert len(recs) > 0
        assert all(isinstance(r, InventoryRecommendation) for r in recs)

    def _create_test_df(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "part_number": [f"PART-{i:04d}" for i in range(n)],
                "part_family": np.random.choice(["ENGINE", "AVIONICS", "AIRFRAME"], n),
                "shop_id": np.random.choice(
                    ["SHOP_01", "SHOP_02", "SHOP_03", "SHOP_04", "SHOP_05"], n
                ),
                "current_stock": np.random.randint(5, 30, n),
                "unit_cost": np.random.uniform(100, 10000, n),
                "demand_mean_weekly": np.random.uniform(5, 20, n),
                "lead_time_days": np.random.randint(7, 21, n),
            }
        )

    def _create_rop_results(self, n: int) -> list:
        from src.aviation_mro.analytics.inventory.reorder_point import ReorderPointResult

        return [
            ReorderPointResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                lead_time_weeks=2.0,
                mean_weekly_demand=10.0,
                std_weekly_demand=2.0,
                service_level=0.95,
                z_score=1.645,
                lead_time_demand=20.0,
                safety_stock=5.0,
                reorder_point=25.0,
                assumptions=[],
            )
            for i in range(n)
        ]

    def _create_ss_results(self, n: int) -> list:
        from src.aviation_mro.analytics.inventory.safety_stock import SafetyStockResult

        return [
            SafetyStockResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                method="normal_constant_lt",
                service_level=0.95,
                safety_stock=5.0,
                assumptions=[],
                parameters={},
            )
            for i in range(n)
        ]

    def _create_risk_results(self, n: int) -> list:
        from src.aviation_mro.analytics.inventory.stockout_risk import StockoutRiskResult

        return [
            StockoutRiskResult(
                part_number=f"PART-{i:04d}",
                part_family="ENGINE",
                shop_id="SHOP_01",
                current_stock=10.0,
                reorder_point=25.0,
                mean_ltd=20.0,
                std_ltd=5.0,
                service_level_target=0.95,
                stockout_probability=0.20,
                expected_shortage_per_cycle=2.0,
                fill_rate=0.80,
                cycles_per_year=13.0,
                expected_stockouts_per_year=2.6,
                risk_category="High",
                assumptions=[],
            )
            for i in range(n)
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
