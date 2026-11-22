# Software Requirements Specification (SRS)

> ⚠️ **NON-AUTHORITATIVE — as-built reconstruction v1.0.**
> The authoritative specification is **`docs/SRS.md` (v1.1)**, provided by the
> project owner. This root file is a *reconstruction* of what the repository
> currently implements; it is retained for traceability only and **does not**
> override `docs/SRS.md`. Where they differ, `docs/SRS.md` (v1.1) governs.
> See `docs/RECONCILIATION_REPORT.md` for the gap analysis.

| Field | Value |
|---|---|
| Project | Aviation MRO Predictive Intelligence Platform |
| Document | SRS |
| Version | **1.0** |
| Status | **As-built (reconstructed)** — see note below |
| Last updated | 2026-11-16 |

> **Reconstruction note:** No original `SRS.md` was present in the repository at
> audit time. This document is reconstructed **as-built** from the implemented
> code, saved artefacts and tests. It records requirements the system actually
> satisfies; it does not invent unbuilt features. Version 1.0 is assigned to
> this reconstructed baseline. If an authoritative SRS exists elsewhere, it
> supersedes this file and the version must be reconciled.

## 1. Purpose & Scope

A local-first, reproducible analytics platform for aviation Maintenance, Repair
and Overhaul (MRO) that combines:

- the **NASA C-MAPSS FD001** simulated turbofan degradation benchmark, and
- **synthetic MRO business data** (demand, inventory, costs),

to deliver descriptive analytics, remaining-useful-life (RUL) prediction, anomaly
detection, demand forecasting, inventory analytics, financial analytics, and
explainability, surfaced through a Streamlit dashboard.

**Out of scope:** cloud deployment, real airline data ingestion, production MLOps.

## 2. Data Sources

| ID | Source | Type | Used by |
|---|---|---|---|
| `nasa_cmapss_fd001` | NASA C-MAPSS FD001 Turbofan Degradation | Simulated benchmark | EDA, RUL, Anomaly |
| `synthetic_mro_v1` | Weekly maintenance demand (5 shops × 3 families) | Synthetic, seed=42 | Forecasting, Inventory |
| `synthetic_inventory_v1` | Inventory part master (150 parts) | Synthetic, seed=42 | Inventory, Financial |
| `synthetic_financial_v1` | Labour / downtime / cost parameters | Synthetic, seed=42 | Financial |

See `docs/DATA_SOURCES.md` for checksums, schemas and known limitations.

## 3. Functional Requirements

### 3.1 Exploratory Data Analysis (Section 9.1) — IMPLEMENTED
- Profile train/test FD001: shape, schema, missingness, duplicates, ranges.
- Identify constant vs informative sensors; RUL statistics.
- Normality (Shapiro-Wilk), variance (Levene), distribution shift (KS),
  stationarity (ADF), and train/test trend comparison.
- Persist `reports/eda/eda_report.json` + `reports/eda/plots/*.png`.

### 3.2 RUL Prediction (Section 9.2) — IMPLEMENTED
- Engine-level (unit) train/validation split — no row-level leakage.
- Past-only rolling features (windows 3 & 5) over 15 informative sensors + 2 op-settings.
- Median imputer + standard scaler fit on **train only**; RUL cap = 125.
- Compare ≥4 models (Dummy, Linear, Ridge, RandomForest, GradientBoosting);
  select on validation RMSE; evaluate once on the official truncated test set.
- Metrics: MAE, RMSE, R² + per-engine error distribution.
- Persist `models/baseline_rul_model_v2.pkl`, `reports/model_card_baseline_v2.json`.
- Verified test performance: **MAE 13.49, RMSE 18.25, R² 0.807** (GradientBoosting).

### 3.3 Anomaly Detection (Section 9.3) — IMPLEMENTED
- Unsupervised `IsolationForest` over rolling + residual features; no RUL labels.
- Contamination threshold at train-score percentile; contamination sweep.
- Sensor-context attribution via training z-score deviation.
- Persist `models/anomaly_model.pkl`, `reports/anomaly_model_card.json`.

### 3.4 Maintenance Forecasting (Section 9.4) — IMPLEMENTED
- Chronological train/validation/test split of synthetic weekly demand.
- Seasonal-naive baselines (period 52 vs 13) vs naive; WAPE/RMSE/MAE + empirical
  prediction-interval coverage.
- Persist `reports/forecasting/forecast_results.json`, split metadata, baseline models.

### 3.5 Inventory Analytics (Section 9.5) — IMPLEMENTED
- Validate inventory data; demand-pattern classification; lead-time demand;
  reorder points; safety stock (normal, variable-LT, lognormal);
  stockout risk; replenishment recommendations (EOQ-aware).
- Persist `data/processed/inventory_data.parquet`,
  `reports/inventory/inventory_analysis.json`, `INVENTORY_REPORT.md`.

### 3.6 Explainability (Section 9.6) — IMPLEMENTED
- SHAP `TreeExplainer` for the existing RUL `GradientBoostingRegressor` (no retrain).
- Exact training feature order and preprocessing; additive consistency verified.
- Global mean|SHAP| importance + per-engine local explanations with scaled/raw values.
- Persist model version, dataset fingerprint, feature names, background size,
  SHAP version, consistency result and causation disclaimer to
  `reports/explainability/rul_shap_summary.json`.
- Regeneration: `python -m aviation_mro.explainability.rul_explainer`.

### 3.7 Dashboard (Streamlit) — IMPLEMENTED (9 pages)
Entry point: `streamlit run src/dashboard/app.py`. Required pages:
1. Executive Overview
2. Data Quality & EDA
3. RUL Prediction & Explainability
4. Anomaly Detection
5. Maintenance Forecasting
6. Inventory Analytics
7. Financial Analytics
8. Explainability (SHAP)
9. Data Provenance & Sources

Each page reads saved artefacts only (JSON/Parquet/model cards); missing artefacts
render a non-fatal warning.

## 4. Non-Functional Requirements

- **Reproducibility:** fixed `random_seed=42`; dataset fingerprints recorded.
- **Provenance:** every page shows a source badge (NASA / Synthetic / Mixed);
  sources documented in `docs/DATA_SOURCES.md`.
- **No fabrication:** all displayed values come from saved artefacts.
- **Explainability caveats:** SHAP contributions are labelled as correlation, not causation.
- **Preservation:** trained models, model cards and evaluation metrics are immutable
  to analytics/dashboard code.

## 5. Acceptance Criteria

- `pytest` passes for all implemented sections (excluding known-slow training suites
  unless run explicitly).
- `ruff check` and `ruff format --check` clean.
- Streamlit AppTest renders all 9 pages without exceptions.
- SHAP additivity check `max_abs_error < 1e-3`.
- RUL metrics remain MAE 13.49 / R² 0.807 (no regression).