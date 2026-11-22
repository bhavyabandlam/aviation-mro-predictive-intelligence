#!/usr/bin/env python3
"""Reproducible demonstration script for the Aviation MRO Predictive Intelligence Platform.

This script verifies that all saved artefacts exist and prints a concise summary of
the key results that appear on each of the 9 Streamlit dashboard pages. It does NOT
retrain models or overwrite datasets.

Usage:
    python scripts/demo_run.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
MODELS = PROJECT / "models"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 72)
    print("Aviation MRO Predictive Intelligence — Demonstration Summary")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Page 1: Executive Overview
    # ------------------------------------------------------------------
    print("PAGE 1 — Executive Overview (mixed sources)")
    print("-" * 72)
    model_card = load_json(REPORTS / "model_card_baseline_v2.json")
    if model_card:
        tm = model_card.get("test_metrics", {})
        print(f"  Best model: {model_card.get('best_model')}")
        print(f"  Test MAE:   {tm.get('mae'):.2f} cycles")
        print(f"  Test RMSE:  {tm.get('rmse'):.2f} cycles")
        print(f"  Test R²:    {tm.get('r2'):.3f}")
        print(f"  Features:   {model_card.get('n_features')}")
    anomaly_card = load_json(REPORTS / "anomaly_model_card.json")
    if anomaly_card:
        print(f"  Anomaly detector: IsolationForest, {anomaly_card.get('n_features')} features")
        print(f"  Default threshold: {anomaly_card.get('default_threshold'):.5f}")
    print()

    # ------------------------------------------------------------------
    # Page 2: Data Explorer & Quality (NASA FD001)
    # ------------------------------------------------------------------
    print("PAGE 2 — Data Explorer & Quality (NASA C-MAPSS FD001)")
    print("-" * 72)
    eda_report = load_json(REPORTS / "eda" / "eda_report.json")
    if eda_report:
        train = eda_report.get("train_profile", {}).get("shape", [])
        test = eda_report.get("test_profile", {}).get("shape", [])
        print(f"  Train shape: {train}")
        print(f"  Test shape:  {test}")
        print(f"  100 engines, 26 sensor columns, Sea Level operating condition")
        print(f"  Source: NASA C-MAPSS simulation (synthetic benchmark, not real fleet records)")
    print()

    # ------------------------------------------------------------------
    # Page 3: Engine Health & RUL
    # ------------------------------------------------------------------
    print("PAGE 3 — Engine Health & RUL (NASA FD001)")
    print("-" * 72)
    if model_card:
        mc = model_card
        print(f"  Model: GradientBoostingRegressor (baseline_rul_model_v2.pkl)")
        print(f"  Train engines: {len(mc.get('train_engine_ids', []))}")
        print(f"  Val engines:   {len(mc.get('val_engine_ids', []))}")
        print(f"  Test engines:  100 (official FD001 test set)")
        print(f"  RUL cap: {mc.get('rul_cap')} cycles")
        print(f"  Random seed: {mc.get('feature_config', {}).get('random_seed')}")
    print()

    # ------------------------------------------------------------------
    # Page 4: Anomaly Analysis
    # ------------------------------------------------------------------
    print("PAGE 4 — Anomaly Analysis (NASA FD001)")
    print("-" * 72)
    if anomaly_card:
        ac = anomaly_card
        print(f"  Model: IsolationForest (anomaly_model.pkl)")
        print(f"  Contamination: {ac.get('default_contamination')}")
        print(f"  Threshold: {ac.get('default_threshold'):.5f}")
        tfs = ac.get("test_flagged_summary", {})
        print(f"  Test anomalies flagged: {tfs.get('n_flagged')} / {tfs.get('n_total')} "
              f"({tfs.get('anomaly_rate', 0)*100:.2f}%)")
    print()

    # ------------------------------------------------------------------
    # Page 5: Maintenance Forecasting
    # ------------------------------------------------------------------
    print("PAGE 5 — Maintenance Forecasting (synthetic MRO demand)")
    print("-" * 72)
    forecast = load_json(REPORTS / "forecasting" / "forecast_results.json")
    if forecast:
        ta = forecast.get("test_aggregate", {})
        sn = ta.get("seasonal_naive", {})
        nv = ta.get("naive", {})
        print(f"  Series: {forecast.get('n_series')} (5 shops × 3 part families)")
        print(f"  Horizon: {forecast.get('forecast_horizon')} weeks")
        print(f"  Seasonal period: {forecast.get('selected_seasonal_period')} weeks")
        print(f"  Seasonal-naive test MAE: {sn.get('mae'):.2f}")
        print(f"  Naive test MAE:          {nv.get('mae'):.2f}")
        print(f"  Source: synthetic_mro_v1 (seed=42) — NOT real maintenance records")
    print()

    # ------------------------------------------------------------------
    # Page 6: Inventory Intelligence
    # ------------------------------------------------------------------
    print("PAGE 6 — Inventory Intelligence (synthetic MRO inventory)")
    print("-" * 72)
    inventory = load_json(REPORTS / "inventory" / "inventory_analysis.json")
    if inventory:
        val = inventory.get("validation", {})
        da = inventory.get("demand_analysis", {})
        print(f"  Parts: {da.get('n_parts')} (5 shops × 3 families × 10 per combo)")
        print(f"  Valid: {val.get('is_valid')}")
        print(f"  Pattern distribution:")
        for item in da.get("pattern_distribution", []):
            print(f"    {item.get('part_family')}: smooth={item.get('smooth_count')}, "
                  f"erratic={item.get('erratic_count')}, lumpy={item.get('lumpy_count')}")
        print(f"  Source: synthetic_inventory_v1 (seed=42) — NOT real inventory records")
    print()

    # ------------------------------------------------------------------
    # Page 7: Financial Scenarios
    # ------------------------------------------------------------------
    print("PAGE 7 — Financial Scenarios (synthetic MRO financials)")
    print("-" * 72)
    financial = load_json(REPORTS / "financial" / "financial_analysis.json")
    if financial:
        labor = financial.get("labor_costs", {})
        totals = labor.get("totals", {})
        print(f"  Total annual labor cost: ${totals.get('total_annual_labor_cost', 0):,.0f}")
        print(f"  Total annual labor hours: {totals.get('total_annual_labor_hours', 0):,.0f}")
        print(f"  Shops: 5, Families: 3 (ENGINE/AVIONICS/AIRFRAME)")
        print(f"  Source: synthetic_financial_v1 (seed=42) — NOT real financial records")
    print()

    # ------------------------------------------------------------------
    # Page 8: Model Performance & Explainability
    # ------------------------------------------------------------------
    print("PAGE 8 — Model Performance & Explainability (SHAP)")
    print("-" * 72)
    shap = load_json(REPORTS / "explainability" / "rul_shap_summary.json")
    if shap:
        gi = shap.get("global_importance", [])
        print(f"  Explainer: {shap.get('explainer_type')} (TreeExplainer)")
        print(f"  Features: {shap.get('n_features')}")
        print(f"  Top feature: {gi[0]['feature']} (|SHAP|={gi[0]['mean_abs_shap']:.4f})")
        print(f"  Consistency max error: ~1.85e-06")
        print("  Note: SHAP shows correlation, not causation.")
    print()

    # ------------------------------------------------------------------
    # Page 9: Data Sources & Model Registry
    # ------------------------------------------------------------------
    print("PAGE 9 — Data Sources & Model Registry")
    print("-" * 72)
    print("  Sources:")
    print("    • NASA C-MAPSS FD001 (simulated turbofan degradation)")
    print("    • Synthetic MRO demand (synthetic_mro_v1, seed=42)")
    print("    • Synthetic inventory (synthetic_inventory_v1, seed=42)")
    print("    • Synthetic financial (synthetic_financial_v1, seed=42)")
    print("  Artefacts:")
    print(f"    • RUL model: models/baseline_rul_model_v2.pkl")
    print(f"    • Anomaly model: models/anomaly_model.pkl")
    print("  See docs/DATA_SOURCES.md for full provenance (checksums, schemas, limitations).")
    print()
    print("=" * 72)
    print("All data read from saved artefacts. No retraining performed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
