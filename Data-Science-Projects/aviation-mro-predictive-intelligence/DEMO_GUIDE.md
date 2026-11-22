# Demonstration Guide — Aviation MRO Predictive Intelligence Platform

Beginner-friendly guide to launching and demonstrating the 9-page Streamlit
dashboard, with an explanation of each page and the data it uses.

> **Data notice:** NASA C-MAPSS FD001 is a *simulated* benchmark dataset; all MRO
> business data (forecasting, inventory, financial) is *synthetic*, generated
> locally with a fixed random seed (seed=42). This project is for development and
> demonstration only — not for production or operational decisions.

---

## 1. Prerequisites

- Python 3.13
- The virtual environment already exists at `.venv` (or create one)
- Dependencies installed via `pip install -r requirements.txt`

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

---

## 2. Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`). Open it in your
browser. The dashboard uses `st.navigation` with a sidebar containing all 9 pages.

---

## 3. The Nine Pages

| # | Page | Primary Data Source | What You See |
|---|------|---------------------|--------------|
| 1 | **Executive Overview** | All reports (mixed) | Headline KPIs across RUL, anomaly, forecasting, inventory, financial |
| 2 | **Data Explorer & Quality** | NASA C-MAPSS FD001 | Data quality summary, sensor informativeness, train/test distribution shift |
| 3 | **Engine Health & RUL** | NASA C-MAPSS FD001 + model card | GradientBoosting RUL performance, per-engine errors, model comparison |
| 4 | **Anomaly Analysis** | NASA C-MAPSS FD001 + anomaly model card | IsolationForest scores, threshold sweep, per-engine anomaly rates, sensor attributions |
| 5 | **Maintenance Forecasting** | Synthetic MRO demand (seed=42) | Seasonal-naive baselines, per-series forecast results, demand charts with PNG export |
| 6 | **Inventory Intelligence** | Synthetic MRO inventory (seed=42) | Reorder points, safety stock, stockout risk, replenishment recommendations, CSV/HTML export |
| 7 | **Financial Scenarios** | Synthetic MRO financials (seed=42) | Cost breakdown, KPIs, scenario analysis, per-shop/family tables, CSV/HTML export |
| 8 | **Model Performance & Explainability** | SHAP summary (precomputed) | Global feature importance (mean |SHAP|), per-engine explanations, causation caveat |
| 9 | **Data Sources & Model Registry** | DATA_SOURCES.md + artefact status | Provenance legend, source tables, reproducibility metadata, model registry with export |

Each page shows **source/data type**, units, filters, freshness badges, and
empty/error states. Missing artefacts produce a non-fatal warning instead of
crashing.

---

## 4. Demonstration Walkthrough

Follow this order for a complete demo:

### Step 1 — Executive Overview
Confirm the platform blends two data worlds:
- **NASA C-MAPSS FD001** (simulated turbofan) powers EDA, RUL, anomaly detection.
- **Synthetic MRO** (seed=42) powers forecasting, inventory, financial analytics.

Key numbers shown: RUL Test MAE 13.49, RMSE 18.25, R² 0.807; anomaly threshold −0.0796.

### Step 2 — Data Explorer & Quality
Explore NASA FD001: 100 engines, 26 sensor columns, Sea Level operating condition.
Review the data quality summary, sensor informativeness chart, and train/test
distribution-shift analysis. Notice the sensor noise and constant channels.

### Step 3 — Engine Health & RUL
Review the GradientBoostingRUL model card:
- Model comparison table (DummyMean → GradientBoosting)
- Test metrics: MAE 13.49, RMSE 18.25, R² 0.807
- Per-engine error distribution
- Root-cause analysis of the previous v1 negative R² (leaky `cycle_norm` feature)

### Step 4 — Anomaly Analysis
Review the IsolationForest anomaly detector:
- Default threshold −0.0796, contamination 1%
- Train/validation/test score distributions
- Per-engine anomaly rates and top contributing sensors

### Step 5 — Maintenance Forecasting
Review seasonal-naive demand forecasts for 15 series (5 shops × 3 families):
- Test MAE 2.77 vs naive 3.89 (seasonal period 52 weeks)
- Select a shop/family from the dropdown to view its demand chart
- **Export:** Click "Export demand chart PNG" to download a PNG of the chart

### Step 6 — Inventory Intelligence
Review reorder points and replenishment plans for 150 parts:
- Demand patterns: smooth (ENGINE), erratic (AVIONICS), lumpy (AIRFRAME)
- Reorder points with safety stock at 90/95/99% service levels
- Stockout risk indicators
- **Export:** Click "Export reorder points (CSV + HTML)" to download both formats

### Step 7 — Financial Scenarios
Review cost breakdown and scenario analysis:
- Labor costs by shop and part family (total ≈ $897.9M/yr baseline)
- Parts costs with EOQ calculations
- Scenario savings (best scenario ≈ $187.5M)
- **Export:** Click "Export financial KPIs (CSV + HTML)" or "Export scenario analysis (CSV + HTML)"

### Step 8 — Model Performance & Explainability
Review SHAP explanations for the RUL model:
- Global feature importance ranking (top: sensor_11_mean_5, sensor_4_mean_5)
- Per-engine local explanations (unit 88, cycle 68 example shown)
- Causation caveat: SHAP shows correlation, not causation

### Step 9 — Data Sources & Model Registry
Review the provenance registry:
- Source table for NASA FD001 and all synthetic MRO sources
- SHA-256 checksums, fingerprints, generation timestamps
- Model artefact status (exists/size)
- **Export:** Click "Export model registry (CSV + HTML)"

---

## 5. Data Distinction: NASA FD001 vs Synthetic MRO

| Aspect | NASA C-MAPSS FD001 | Synthetic MRO Business Data |
|--------|-------------------|----------------------------|
| **Origin** | NASA PCoE C-MAPSS simulation (PHM08) | Generated locally by this project |
| **Real-world?** | No — simulated turbofan degradation | No — entirely synthetic business data |
| **Purpose** | EDA, RUL prediction, anomaly detection | Forecasting, inventory, financial analytics |
| **Random seed** | N/A (fixed dataset) | seed=42 for all generators |
| **Rows** | 20,631 train / 13,096 test engine-cycles | 2,340 demand rows; 150 inventory parts |
| **Checksums** | Recorded in `docs/DATA_SOURCES.md` | Fingerprints recorded per source |
| **Use constraint** | Simulation, not physical measurements | Not for production decisions |
| **Source ID** | `nasa_cmapss_fd001` | `synthetic_mro_v1`, `synthetic_inventory_v1`, `synthetic_financial_v1` |

Both are clearly labeled on every page with source lineage and synthetic-data
warnings where applicable (FR-13).

---

## 6. Export Functionality (FR-13)

Every analytics page offers exports:

| Page | Export Buttons |
|------|---------------|
| Maintenance Forecasting | Export demand chart PNG |
| Inventory Intelligence | Export reorder points (CSV + HTML) |
| Financial Scenarios | Export financial KPIs (CSV + HTML), Export scenario analysis (CSV + HTML) |
| Data Sources & Model Registry | Export model registry (CSV + HTML) |

Exports are written to `data/exports/` with:
- UTC generation timestamp
- Source lineage
- Synthetic-data warning (where applicable)

Files are offered as Streamlit downloads immediately after generation.

---

## 7. Verification Commands

Run the full quality gate (no retraining, no data modification):

```bash
pytest -q                    # 537 tests pass, 12 skipped, 0 failed
ruff check .                 # lint clean
ruff format --check .        # formatting clean
mypy src                     # 69 source files, no type errors
pytest tests/test_dashboard.py -v   # all 9 pages render via Streamlit AppTest
pytest tests/test_dashboard_export.py -v   # export buttons functional
```

---

## 8. Artefacts & Screenshots

- **Screenshots:** `reports/demo_screenshots/` — 9 genuine screenshots, one per
  dashboard page, captured via Playwright + Chromium against the live Streamlit
  server.
- **Demonstration script:** `scripts/demo_run.py` — prints key results from all
  saved artefacts without retraining or modifying data.
- **Reports:** `reports/` — JSON model cards and analysis results consumed by
  the dashboard.

---

## 9. Limitations

- NASA FD001 is simulated; MRO business data is synthetic — neither represents
  real airline operations.
- SHAP values are in scaled model-input space; local explanations cover 12
  representative engines, not all 100.
- Not intended for production or safety-critical decisions.
- CSV/HTML/PNG exports are available per FR-13; PDF is optional and not
  implemented.
