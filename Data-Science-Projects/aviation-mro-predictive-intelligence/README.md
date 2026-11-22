# Aviation MRO Predictive Intelligence Platform

A local-first, reproducible analytics platform for aviation **M**aintenance,
**R**epair and **O**verhaul (MRO). It blends the **NASA C-MAPSS FD001** simulated
turbofan degradation benchmark with **synthetic MRO business data** to deliver
exploratory analysis, remaining-useful-life (RUL) prediction, anomaly detection,
demand forecasting, inventory analytics, financial analytics and explainability —
all surfaced through a Streamlit dashboard.

> **Data notice:** NASA FD001 is a *simulation*; all MRO business data is
> *synthetic* (generated with seed=42). This project is for development and
> demonstration, not production decision-making.

## Key Results

| Area | Result |
|---|---|
| RUL prediction (GradientBoosting) | Test **MAE 13.49**, **RMSE 18.25**, **R² 0.807** |
| Anomaly detection (IsolationForest) | Unsupervised; 182 features; default threshold −0.0796 |
| Demand forecasting (seasonal-naive) | Test MAE 2.77, WAPE 0.55 (period 52) |
| Inventory analytics | 150 parts; 421 replenishment recommendations |
| Financial analytics | Baseline ≈ **$897.9M/yr**; best scenario saves ≈ **$187.5M** |
| Explainability (SHAP) | TreeExplainer; additivity max error ≈ 1.85e-06 |

## Quick Start

```bash
# 1. Create environment (Python 3.13)
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
#   or: pip install -e .

# 3. Launch the dashboard
streamlit run src/dashboard/app.py
```

The dashboard opens in your browser at the URL Streamlit prints (default
`http://localhost:8501`).

## Dashboard Pages (9)

1. **Executive Overview** — headline KPIs across all sections (mixed sources)
2. **Data Quality & EDA** — NASA FD001 profiling and sensor analysis
3. **RUL Prediction & Explainability** — GradientBoosting performance + SHAP summary
4. **Anomaly Detection** — IsolationForest scores, thresholds, attributions
5. **Maintenance Forecasting** — seasonal-naive baselines + demand explorer
6. **Inventory Analytics** — reorder points, safety stock, stockout risk, recommendations
7. **Financial Analytics** — cost breakdown, KPIs, scenarios
8. **Explainability (SHAP)** — global importance + per-engine explanations
9. **Data Provenance & Sources** — sources, seeds, fingerprints, artefact status

Each page reads saved artefacts only; missing files produce a non-fatal warning.

## Project Structure

```
src/aviation_mro/        Core package (analytics, features, models, explainability)
src/dashboard/           Streamlit app (app.py + views/ for the 9 pages)
data/raw/fd001/          NASA FD001 raw data
data/processed/          Parquet datasets
models/                  Saved model artefacts (*.pkl)
reports/                 JSON/Markdown reports and plots
tests/                   Pytest suites
docs/                    DATA_SOURCES.md, PROFILING_FD001.md, EXPLAINABILITY.md
SRS.md                   Software Requirements Specification (v1.0, as-built)
DEVELOPER_HANDOFF.md     Developer handoff (v1.0, as-built)
```

## Regenerating Analytics

| Section | Command |
|---|---|
| EDA | `python -m aviation_mro.analytics.eda` |
| RUL model | `python -m aviation_mro.models.baseline` |
| Anomaly model | `python -m aviation_mro.models.anomaly_train` |
| Forecasting | `python -m aviation_mro.analytics.forecasting.train` |
| Inventory | `python -m aviation_mro.analytics.inventory.pipeline` |
| Financial | `python -m aviation_mro.analytics.financial.pipeline` |
| SHAP explanations | `python -m aviation_mro.explainability.rul_explainer` |

The SHAP step and the dashboard **never retrain** models.

## Testing & Quality

```bash
pytest -q                              # full suite
pytest tests/test_explainability.py -v # focused SHAP tests
pytest tests/test_dashboard.py -v      # Streamlit smoke (all 9 pages)
ruff check .                           # lint
ruff format --check .                  # formatting
mypy src                               # type checks (configured)
```

## Data Sources

- **NASA C-MAPSS FD001** — simulated turbofan run-to-failure (EDA, RUL, Anomaly).
- **Synthetic MRO demand** (`synthetic_mro_v1`, seed=42) — forecasting, inventory.
- **Synthetic inventory** (`synthetic_inventory_v1`, seed=42) — inventory, financial.
- **Synthetic financial** (`synthetic_financial_v1`, seed=42) — financial.

Full detail (checksums, schemas, limitations): `docs/DATA_SOURCES.md`.

## Explainability

SHAP `TreeExplainer` explains the existing RUL `GradientBoostingRegressor` without
retraining. Explanations are precomputed offline to
`reports/explainability/rul_shap_summary.json` and displayed in the dashboard.
Methodology, validation and regeneration: `docs/EXPLAINABILITY.md`.

> SHAP contributions reflect **correlation, not causation**.

## Limitations

- NASA FD001 is simulated; MRO/financial data is synthetic — not real airline records.
- SHAP values are in the **scaled** model-input space, not original sensor units.
- Local SHAP explanations cover 12 representative engines, not all 100.
- Not intended for production or safety-critical decisions.

## Documentation

- **`docs/SRS.md`** — authoritative Software Requirements Specification (**v1.1**)
- **`docs/DEVELOPER_HANDOFF.md`** — authoritative developer handoff (**v1.1**)
- `docs/RECONCILIATION_REPORT.md` — gap analysis of this repo vs SRS v1.1
- `docs/TECHNICAL_REPORT.md` — technical explanation of every implemented component
- `docs/BEGINNERS_GUIDE.md` — beginner-friendly, step-by-step implementation manual
- `docs/DATA_SOURCES.md` — data provenance
- `docs/EXPLAINABILITY.md` — SHAP methodology
- `docs/PROFILING_FD001.md` — FD001 profiling

> **Specification authority:** The governing specification is **SRS v1.1** and
> **DEVELOPER_HANDOFF v1.1** in `docs/`. The root `SRS.md` and
> `DEVELOPER_HANDOFF.md` are a **non-authoritative as-built reconstruction (v1.0)**
> kept for traceability. See `docs/RECONCILIATION_REPORT.md` for the current
> compliance status against v1.1 (this build covers the analytics/ML core; several
> v1.1 items such as the SQLite persistence layer, BTS operational data, the full
> statistical-test suite, and CSV/HTML/PDF exports remain to be implemented).