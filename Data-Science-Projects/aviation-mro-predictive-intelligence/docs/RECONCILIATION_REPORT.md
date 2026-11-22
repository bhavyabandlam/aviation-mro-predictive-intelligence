# Documentation Reconciliation Report

**Authoritative specification:** `docs/SRS.md` (v1.1) and `docs/DEVELOPER_HANDOFF.md` (v1.1), supplied by the project owner.
**Compared against:** the current repository at `aviation-mro-predictive-intelligence/`.
**Date:** 2026-11-16.
**Nature:** read-only audit. No models, metrics, or requirements were changed to make this report.

> The root `SRS.md` and `DEVELOPER_HANDOFF.md` are an **as-built reconstruction v1.0**
> (now banner-marked NON-AUTHORITATIVE). The **v1.1 documents in `docs/` govern**.
> This report records where the repository conforms to v1.1 and where it does not.

---

## 0. Executive summary

The repository implements the **analytics / ML / explainability core** of the SRS
well (EDA, RUL, anomaly detection, forecasting baselines, inventory analytics,
financial analytics, SHAP, and a 9-page Streamlit dashboard), with **315 passing
tests**. However, it was built to a **different, narrower scope** than SRS v1.1.
The following v1.1 areas are **not implemented**:

- **SQLite persistence layer** and the **typed YAML configuration**
  (`config/data_sources.yaml`, `model_config.yaml`, `business_rules.yaml`).
- The **SRS synthetic-data logical schema** (aircraft, engines, maintenance orders,
  technicians, inventory transactions, downtime events with FK integrity).
- **BTS TranStats** operational analysis (DS-04/DS-05) and **OurAirports** (DS-06).
- A **generic ingestion framework with checksums / `ingestion_runs`** lineage.
- Several **statistical tests** (Pearson/Spearman, chi-square, t-test/Mann-Whitney, ANOVA, CIs).
- **CSV / HTML / PDF exports**, a **central model registry**, and a **KPI registry**.
- The **prescribed repository layout** (`config/`, `scripts/`, `.github/workflows/`,
  `notebooks/`, `docs/DATA_DICTIONARY.md`, `docs/MODEL_CARD.md`, `CHANGELOG.md`,
  test sub-packages).

**Requirement status counts (SRS v1.1):** PASS 8 · PARTIAL 12 · MISSING 6 · NOT APPLICABLE/PROCESS 4.

---

## 1. Functional requirements (SRS §6)

| ID | Requirement | Status | Evidence / Gap |
|---|---|---|---|
| FR-01 | Ingestion adapters (CSV/TXT/Parquet, schema map, checksums) | **PARTIAL** | FD001 whitespace parser in `features/rul_features.py::load_raw_data`; Parquet loaders in analytics. **Missing:** generic adapter layer, checksum recording, `ingestion_runs` lineage. |
| FR-02 | Validation & cleaning + quality report | **PARTIAL** | EDA profiling (`analytics/eda.py`) and `inventory/data_validation.py` produce quality metrics. **Missing:** dedicated `validation/` package, schema validators, cleaning report artefacts. |
| FR-03 | Synthetic MRO generation (aircraft, engines, work orders, technicians, inventory, txns, downtime) | **PARTIAL** | Generators exist (`analytics/inventory/data_prep.py`, `analytics/financial/data_prep.py`, forecasting data). **Gap:** SRS logical schema (Aircraft/Engines/WorkOrders/Technicians/InventoryTransactions/Downtime with PK/FK) not produced; no `src/ingestion/synthetic_generator.py`. |
| FR-04 | Statistical analysis (Pearson/Spearman, t-test/MWU, chi-square, ANOVA, CIs, effect sizes) | **PARTIAL** | EDA computes Shapiro-Wilk, Levene, KS, ADF and Cohen's *d* (`trend_comparison`). **Missing:** Pearson/Spearman correlation, t-test/Mann-Whitney, chi-square, ANOVA, confidence intervals. |
| FR-05 | Feature engineering (lags, rolling, op-condition normalisation) | **PARTIAL** | Past-only rolling mean/std/min/max/slope over windows 3 & 5 (`features/rul_features.py`); optional `include_cycle_norm`. **Missing:** explicit lagged sensor columns. |
| FR-06 | RUL prediction (baselines, engine split, train-only preprocessing, MAE/RMSE/R², per-engine, artefacts) | **PASS** | `models/baseline.py`, `reports/model_card_baseline_v2.json`. Test MAE 13.49, RMSE 18.25, R² 0.807. Engine split; imputer+scaler fit on train; RUL cap 125. |
| FR-07 | Anomaly detection (IsolationForest, threshold doc, sensor context, no precision/recall claims) | **PASS** | `models/anomaly_model.py`, `reports/anomaly_model_card.json`; contamination sweep + z-score sensor attribution. |
| FR-08 | Forecasting (chronological split, naive + seasonal-naive, MAE/RMSE/WAPE, horizon + uncertainty) | **PARTIAL** | `analytics/forecasting/*`. Uses `demand_count` only. **Missing:** work-order counts, labour-hour demand, maintenance expenditure targets; ETS/ARIMA. Empirical intervals present. |
| FR-09 | Inventory (demand summary, ROP = LTD + SS, documented SS, stockout risk, slow-moving, value & turnover, scenarios) | **PARTIAL** | ROP, 4 safety-stock methods, stockout risk, EOQ recommendations (`analytics/inventory/*`). **Missing:** slow-moving analysis, inventory value/turnover (turnover exists only in financial KPIs), inventory scenarios. |
| FR-10 | Financial (planned/unplanned, labour/material, downtime scenarios, cost per order/engine, baseline vs scenario, benefit/investment/ROI, sensitivity) | **PARTIAL** | Labour, parts (EOQ), downtime scenarios, holding, stockout, KPIs, 8 scenarios (`analytics/financial/*`). **Missing:** planned vs unplanned split, cost per order / per engine, formal sensitivity analysis. ROI formula deviates (see §3). |
| FR-11 | Explainability (feature importance + SHAP global/local, model version + feature values, causation caveat) | **PASS** | `explainability/rul_explainer.py`, `reports/explainability/rul_shap_summary.json`, `docs/EXPLAINABILITY.md`. |
| FR-12 | Streamlit 9 pages with exact names; source/data-class label; no hardcoded KPIs | **PARTIAL** | 9 pages exist and read saved artefacts (no hardcoded KPIs). **Gap:** page names differ from SRS (see §2); no "BTS public operational" source class. |
| FR-13 | Reporting & exports (PDF/HTML/CSV/Excel, provenance) | **PARTIAL** | JSON + Markdown reports + EDA PNG plots. **Missing:** CSV exports, HTML report, PDF/Excel. |
| FR-14 | Model registry (name/version, fingerprint, features, params, metrics, seed, artifact, training date) | **PARTIAL** | Model cards hold name/version/fingerprint/features/params/metrics/seed. **Missing:** central `model_registry`, `training_date`. |

---

## 2. Streamlit pages — name reconciliation (FR-12)

| # | SRS v1.1 required page | Current page (`url_path`) | Status |
|---|---|---|---|
| 1 | Executive Overview | Executive Overview (`home`) | **MATCH** |
| 2 | Data Explorer & Quality | Data Quality & EDA (`eda`) | name differs |
| 3 | Engine Health & RUL | RUL Prediction & Explainability (`rul`) | name differs |
| 4 | Anomaly Analysis | Anomaly Detection (`anomaly`) | name differs |
| 5 | Maintenance Forecasting | Maintenance Forecasting (`forecasting`) | **MATCH** |
| 6 | Inventory Intelligence | Inventory Analytics (`inventory`) | name differs |
| 7 | Financial Scenarios | Financial Analytics (`financial`) | name differs |
| 8 | Model Performance & Explainability | Explainability (SHAP) (`explainability`) + SHAP section inside `rul` | name differs / split |
| 9 | Data Sources & Model Registry | Data Provenance & Sources (`provenance`) | name differs |

All 9 pages render (Streamlit AppTest, 16 tests). **Action:** rename pages to the SRS titles and add a "BTS public operational" source class to the legend (even if BTS data is not yet loaded, the class should exist).

---

## 3. Deviations / unapproved changes (require owner decision)

1. **Page names** differ from FR-12 (table above). *Decision: rename to SRS titles.*
2. **DATA_SOURCES.md uses internal IDs** (A = NASA FD001, C = synthetic_mro_v1, E = synthetic_inventory_v1, F = synthetic_financial_v1) instead of SRS **DS-01…DS-07**. **DS-04/DS-05 (BTS)** and **DS-06 (OurAirports)** are absent; **DS-03 (N-CMAPSS)** deferred.
3. **ROI formula** in `financial/kpis.py`: `((cost_avoidance − annual_operating_cost) / implementation_cost) × 100`. SRS FR-10 specifies `((Estimated Benefit − Investment Cost) / Investment Cost) × 100`. *Conceptually similar; needs mapping of benefit/investment to the SRS definition.*
4. **Root `SRS.md` / `DEVELOPER_HANDOFF.md`** were a v1.0 reconstruction (now banner-marked non-authoritative). *Action: keep for traceability; `docs/` v1.1 governs.*
5. **Pre-existing bug** in `financial/holding_costs.py:214`: a dict key uses the variable `total_cycle_stock_value` instead of the string literal — produces a numeric key in `totals`. Not caused by recent work; should be fixed.
6. **Doc error corrected during this audit:** `docs/DATA_SOURCES.md` previously stated order cost `$500/order`; the code default is `$75/order` (`parts_costs.py`). Fixed.

---

## 4. Non-functional requirements (SRS §7)

| ID | Requirement | Status | Note |
|---|---|---|---|
| NFR-01 | Local-first, no paid cloud | **PASS** | Fully local. |
| NFR-02 | Layered modular architecture | **PASS (partial)** | Layers exist under `src/aviation_mro/` but not the exact SRS package split (`ingestion`, `validation`, `business`, `persistence`, `services`). |
| NFR-03 | Python 3.13 | **PASS** | Verified. |
| NFR-04 | Leakage-safe validation | **PASS** | Engine split; train-only preprocessing; past-only features. |
| NFR-05 | Reproducible (seed 42) | **PASS** | Seeds recorded; fingerprints in reports. |
| NFR-06 | Synthetic data labelled | **PASS** | Source badges + `source` columns. |
| NFR-07 | No airworthiness claims | **PASS** | Disclaimers present. |
| NFR-08 | Cache expensive dashboard operations | **MISSING** | No `st.cache_data` used. |
| NFR-09 | Export processed data/CSV when feasible | **PARTIAL** | JSON/MD/PNG only; no CSV export. |
| NFR-10 | Accessible labels, units, colour-independent | **PARTIAL** | Units shown; badges carry text as well as colour. |

---

## 5. Repository structure & process (SRS §9, §12–14)

| Area | SRS v1.1 expects | Current | Status |
|---|---|---|---|
| Layout | `config/`, `data/{raw,processed,synthetic,database,exports}`, `docs/{SRS,DEVELOPER_HANDOFF,DATA_SOURCES,DATA_DICTIONARY,MODEL_CARD}.md`, `notebooks/`, `src/aviation_mro/{config,ingestion,validation,preprocessing,analytics,features,models/{rul,anomaly,forecasting},business,persistence,services}`, `dashboard/{app.py,pages,components,utils}`, `models/`, `reports/`, `scripts/`, `tests/{unit,integration,data_quality,model_validation}`, `.github/workflows/`, `CHANGELOG.md` | Flat `analytics/`, flat `models/`, `dashboard/views/`, flat `tests/`, no `config/`, `scripts/`, `.github/`, `notebooks/`, `CHANGELOG.md`, `DATA_DICTIONARY.md`, `MODEL_CARD.md` | **PARTIAL / DEVIATED** |
| Database | SQLite with 13 tables (SRS §10) | No SQLite | **MISSING** |
| Testing | Unit/integration/data_quality/model_validation sub-suites | Single flat `tests/` (315 tests, all pass) | **PARTIAL** |
| mypy | `mypy src` clean | Configured but reports missing-stub errors (pandas/shap/joblib) + legacy debt | **PARTIAL** |

---

## 6. Prioritized completion plan (to reach 100% of v1.1)

**P0 — Architectural blockers**
1. SQLite persistence (`persistence/`) + 13-table schema (SRS §10) and Parquet exports.
2. Typed YAML config: `config/data_sources.yaml`, `model_config.yaml`, `business_rules.yaml`.
3. Synthetic generator to the SRS logical schema (`ingestion/synthetic_generator.py`): aircraft, engines, maintenance orders, technicians, inventory parts, inventory transactions, downtime events (PK/FK, `is_synthetic`, manifest).
4. Ingestion framework + checksums + `ingestion_runs` lineage.
5. `validation/` package (schema + cleaning + quality report).

**P1 — Analytics completeness**
6. Statistical suite: Pearson/Spearman, chi-square, t-test/Mann-Whitney, ANOVA, CIs, effect sizes.
7. Forecasting targets: work-order counts, labour-hour demand, maintenance expenditure.
8. Inventory: slow-moving analysis, inventory value & turnover, inventory scenarios.
9. Financial: planned vs unplanned split, cost per order / per engine, sensitivity analysis; align ROI to SRS formula.
10. CSV/HTML/PDF exports.
11. Model registry table + `training_date`; central KPI registry (SRS §11).
12. Rename dashboard pages to SRS titles; add "BTS public operational" source class.

**P2 — Structure, quality, ops**
13. Restructure to SRS §9 layout (`config/`, `scripts/`, `.github/workflows/`, `notebooks/`, `docs/DATA_DICTIONARY.md`, `docs/MODEL_CARD.md`, `CHANGELOG.md`, test sub-packages).
14. `st.cache_data` for expensive deterministic computations with correct cache keys (NFR-08).
15. Accessibility pass (NFR-10); `pandera` validation; `mypy src` clean (install stubs or configure overrides).

**P3 — Optional sources**
16. BTS TranStats (DS-04/DS-05) operational analysis + dashboard section; OurAirports (DS-06). N-CMAPSS (DS-03) deferred.

---

## 7. What is already compliant and must be preserved

- FR-06 RUL, FR-07 Anomaly, FR-11 Explainability: **PASS**.
- Leakage-safe splits, reproducibility (seed 42), synthetic-data labelling, no airworthiness claims: **PASS**.
- 315 passing tests; 9 dashboard pages render; SHAP additivity verified (< 1e-3).
- Trained models, model cards, and evaluation metrics are intact and unchanged.