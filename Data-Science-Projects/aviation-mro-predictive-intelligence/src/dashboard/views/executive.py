"""Executive Overview page.

Aggregates headline results from Sections 9.1-9.6 into a single landing
screen. All figures are read from saved artefacts; missing artefacts are
reported rather than estimated.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _build_rul_summary(card: dict[str, Any] | None) -> dict[str, Any]:
    if not card:
        return {"available": False}
    test = card.get("test_metrics", {}) or {}
    return {
        "available": True,
        "best_model": card.get("best_model", "n/a"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_r2": test.get("r2"),
        "n_features": card.get("n_features"),
    }


def _build_anomaly_summary(card: dict[str, Any] | None) -> dict[str, Any]:
    if not card:
        return {"available": False}
    test = card.get("test_flagged_summary", {}) or {}
    val = card.get("val_flagged_summary", {}) or {}
    return {
        "available": True,
        "threshold": card.get("default_threshold"),
        "test_rate": test.get("anomaly_rate"),
        "test_flagged": test.get("n_flagged"),
        "val_rate": val.get("anomaly_rate"),
    }


def _build_forecast_summary(res: dict[str, Any] | None) -> dict[str, Any]:
    if not res:
        return {"available": False}
    test = res.get("test_aggregate", {}) or {}
    seasonal = test.get("seasonal_naive", {}) or {}
    return {
        "available": True,
        "selected_period": res.get("selected_seasonal_period"),
        "n_series": res.get("n_series"),
        "test_mae": seasonal.get("mae"),
        "test_rmse": seasonal.get("rmse"),
        "test_wape": seasonal.get("wape"),
    }


def _build_inventory_summary(inv: dict[str, Any] | None) -> dict[str, Any]:
    if not inv:
        return {"available": False}
    meta = inv.get("metadata", {}) or {}
    validation = inv.get("validation", {}) or {}
    return {
        "available": True,
        "n_parts": meta.get("n_parts"),
        "n_recommendations": len(inv.get("recommendations", []) or []),
        "is_valid": validation.get("is_valid"),
        "families": meta.get("part_families", []),
    }


def _build_financial_summary(fin: dict[str, Any] | None) -> dict[str, Any]:
    if not fin:
        return {"available": False}
    kpis = fin.get("financial_kpis", {}) or {}
    cb = kpis.get("cost_breakdown", {}) or {}
    scen = fin.get("scenario_analysis", {}) or {}
    avoid = kpis.get("predictive_maintenance_avoidance", {}) or {}
    return {
        "available": True,
        "total_cost": cb.get("total_maintenance_cost"),
        "cost_per_flight_hour": (kpis.get("cost_per_flight_hour", {}) or {}).get("value"),
        "avoidance": avoid.get("value"),
        "best_scenario": scen.get("best_scenario"),
        "best_savings": scen.get("best_savings"),
    }


def render() -> None:
    """Render the Executive Overview page."""
    st.set_page_config(page_title="Executive Overview", page_icon="✈️", layout="wide")
    ui.page_header(
        "Executive Overview",
        "Aggregates headline results from Sections 9.1-9.6 into a single landing screen.",
        "mixed",
    )
    st.info(
        "This platform blends the **NASA C-MAPSS FD001** simulated turbofan "
        "benchmark (EDA, RUL prediction, anomaly detection) with **synthetic "
        "MRO business data** (forecasting, inventory, financial analytics). "
        "All business figures are synthetic and not real airline records."
    )

    # ---- Data freshness ----------------------------------------------------
    ui.section("Data Freshness")
    ui.freshness_badge(dl.REPORTS_DIR / "model_card_baseline_v2.json", "RUL model card")
    ui.freshness_badge(dl.REPORTS_DIR / "anomaly_model_card.json", "Anomaly model card")
    ui.freshness_badge(dl.REPORTS_DIR / "forecasting" / "forecast_results.json", "Forecast results")
    ui.freshness_badge(
        dl.REPORTS_DIR / "inventory" / "inventory_analysis.json", "Inventory analysis"
    )
    ui.freshness_badge(
        dl.REPORTS_DIR / "financial" / "financial_analysis.json", "Financial analysis"
    )
    st.divider()

    rul = _build_rul_summary(dl.load_rul_model_card())
    anomaly = _build_anomaly_summary(dl.load_anomaly_model_card())
    forecast = _build_forecast_summary(dl.load_forecast_results())
    inventory = _build_inventory_summary(dl.load_inventory_analysis())
    financial = _build_financial_summary(dl.load_financial_analysis())

    # ---- Model & analytics health cards -----------------------------------
    ui.section("Model & Analytics Health")
    cards: list[dict[str, Any]] = []
    if rul["available"]:
        cards.append(
            {
                "label": "RUL Test R2",
                "value": ui.fmt_num(rul["test_r2"]),
                "helper": f"Best model: {rul['best_model']} (NASA FD001)",
            }
        )
    if anomaly["available"]:
        cards.append(
            {
                "label": "Anomaly Rate (test)",
                "value": ui.fmt_pct((anomaly["test_rate"] or 0) * 100, 2),
                "helper": f"{anomaly['test_flagged']} flagged obs (NASA FD001)",
            }
        )
    if forecast["available"]:
        cards.append(
            {
                "label": "Forecast WAPE (test)",
                "value": ui.fmt_num(forecast["test_wape"]),
                "helper": f"Seasonal-naive, period={forecast['selected_period']} (synthetic)",
            }
        )
    if inventory["available"]:
        cards.append(
            {
                "label": "Inventory Parts",
                "value": ui.fmt_num(inventory["n_parts"], 0),
                "helper": f"{inventory['n_recommendations']} recommendations (synthetic)",
            }
        )
    if cards:
        ui.kpi_row(cards)
    else:
        ui.missing_artifact("model cards and analytical reports", "reports/")

    # ---- Financial headline ----------------------------------------------
    if financial["available"]:
        ui.section("Financial Headline (Synthetic)")
        fin_cards = [
            {
                "label": "Annual Maintenance Cost",
                "value": ui.fmt_money(financial["total_cost"]),
                "helper": "Synthetic MRO cost model",
            },
            {
                "label": "Cost per Flight Hour",
                "value": ui.fmt_num(financial["cost_per_flight_hour"]),
                "helper": "USD per flight hour",
            },
            {
                "label": "Predictive Maint. Avoidance",
                "value": ui.fmt_money(financial["avoidance"]),
                "helper": "USD / year (20% downtime reduction)",
            },
            {
                "label": "Best Scenario",
                "value": ui.fmt_money(financial["best_savings"]),
                "helper": f"{financial['best_scenario']} savings",
            },
        ]
        ui.kpi_row(fin_cards)

    # ---- Artefact health --------------------------------------------------
    ui.section("Saved Model Artefacts")
    status = dl.model_artifact_status()
    rows = [
        {
            "Artefact": name,
            "Present": "Yes" if info["exists"] else "No",
            "Size (KB)": info["size_kb"],
        }
        for name, info in status.items()
    ]
    ui.data_table(rows)
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        ui.csv_export_button(df, "artefact_health.csv")
        ui.export_section(
            {"artefact_health": df},
            "executive_artefact_health",
            source_lineage="NASA FD001 + synthetic MRO business data",
            synthetic=True,
            label="Export artefact health (CSV + HTML)",
        )

    # ---- Section index ----------------------------------------------------
    ui.section("Dashboard Sections")
    st.markdown(
        "1. **Data Quality & EDA** - NASA FD001 profiling and sensor analysis\n"
        "2. **RUL Prediction** - GradientBoosting remaining-useful-life model\n"
        "3. **Anomaly Detection** - IsolationForest anomaly scoring\n"
        "4. **Maintenance Forecasting** - seasonal-naive demand baselines\n"
        "5. **Inventory Analytics** - reorder points, safety stock, stockout risk\n"
        "6. **Financial Analytics** - cost breakdown, KPIs and scenarios"
    )
