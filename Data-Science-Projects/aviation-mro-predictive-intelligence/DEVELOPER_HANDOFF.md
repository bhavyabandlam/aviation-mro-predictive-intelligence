# Developer Handoff

> ⚠️ **NON-AUTHORITATIVE — as-built reconstruction v1.0.**
> The authoritative handoff is **`docs/DEVELOPER_HANDOFF.md` (v1.1)**, provided
> by the project owner. This root file is a *reconstruction* retained for
> traceability only and **does not** override `docs/DEVELOPER_HANDOFF.md`.
> See `docs/RECONCILIATION_REPORT.md` for the gap analysis.

| Field | Value |
|---|---|
| Project | Aviation MRO Predictive Intelligence Platform |
| Document | DEVELOPER_HANDOFF |
| Version | **1.0** |
| Status | **As-built (reconstructed)** — see note |
| Last updated | 2026-11-16 |

> **Reconstruction note:** No original handoff document was present at audit time.
> This is reconstructed **as-built** from the current code and artefacts and is
> assigned version 1.0. It describes the workflow the repository actually follows
> (READ → AUDIT → PLAN → AUTHORIZE → IMPLEMENT → VERIFY → CHECKPOINT → COMMIT) and
> the concrete commands to build, regenerate and verify each section.

## 1. Environment

- Python **3.13**, Windows.
- Install: `pip install -r requirements.txt` (or `pip install -e .`).
- Key deps: pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1,
  matplotlib 3.11.2, streamlit 1.67.0, pyarrow 26.0.1, joblib 1.4.2,
  statsmodels 0.14.4, **shap 0.54.1**, pytest 9.2.0, ruff 0.17.4, mypy (configured).

## 2. Repository Layout

```
aviation-mro-predictive-intelligence/
  src/aviation_mro/            # core package
    analytics/eda.py           # 9.1 EDA
    features/                  # 9.2 RUL features, 9.3 anomaly features
    models/                    # 9.2 baseline.py, 9.3 anomaly_model.py / anomaly_train.py
    analytics/forecasting/     # 9.4 forecasting
    analytics/inventory/       # 9.5 inventory
    analytics/financial/       # 9.6 financial analytics
    explainability/            # 9.6 SHAP (rul_explainer.py)
  src/dashboard/               # Streamlit app (9 pages)
    app.py                     # entry point (st.navigation)
    data_loader.py             # safe artefact loaders
    components.py              # UI helpers (badges, KPIs, tables)
    views/                     # one module per page
  data/raw/fd001/              # NASA FD001 raw data
  data/processed/              # parquet datasets
  models/                      # saved model artefacts (*.pkl)
  reports/                     # JSON/MD reports + plots
    eda/ forecasting/ inventory/ financial/ explainability/
  tests/                       # pytest suites
  docs/                        # DATA_SOURCES.md, PROFILING_FD001.md, EXPLAINABILITY.md
  SRS.md, DEVELOPER_HANDOFF.md, README.md
```

## 3. Regeneration Commands

| Section | Command |
|---|---|
| 9.1 EDA | `python -m aviation_mro.analytics.eda` |
| 9.2 RUL | `python -m aviation_mro.models.baseline` |
| 9.3 Anomaly | `python -m aviation_mro.models.anomaly_train` |
| 9.4 Forecasting | `python -m aviation_mro.analytics.forecasting.train` |
| 9.5 Inventory | `python -m aviation_mro.analytics.inventory.pipeline` |
| 9.6 Financial | `python -m aviation_mro.analytics.financial.pipeline` |
| 9.6 Explainability | `python -m aviation_mro.explainability.rul_explainer` |
| Dashboard | `streamlit run src/dashboard/app.py` |

**Do not retrain models unnecessarily.** The SHAP step and the dashboard never
retrain; they consume the saved artefacts.

## 4. Dashboard Pages (9 required)

| # | Page | Module | Primary source |
|---|---|---|---|
| 1 | Executive Overview | `views/executive.py` | all reports (mixed) |
| 2 | Data Quality & EDA | `views/eda.py` | NASA FD001 + `eda_report.json` |
| 3 | RUL Prediction & Explainability | `views/rul.py` | `baseline_rul_model_v2.pkl`, model card |
| 4 | Anomaly Detection | `views/anomaly.py` | `anomaly_model.pkl`, model card |
| 5 | Maintenance Forecasting | `views/forecasting.py` | `forecast_results.json` |
| 6 | Inventory Analytics | `views/inventory.py` | `inventory_analysis.json` |
| 7 | Financial Analytics | `views/financial.py` | `financial_analysis.json` |
| 8 | Explainability (SHAP) | `views/explainability.py` | `rul_shap_summary.json` |
| 9 | Data Provenance & Sources | `views/provenance.py` | `DATA_SOURCES.md`, artefact status |

All pages read saved artefacts only; missing files produce a non-fatal warning.

## 5. Testing & Quality

```bash
pytest -q                                   # full suite
pytest tests/test_explainability.py -v      # focused SHAP (15 tests)
pytest tests/test_dashboard.py -v           # Streamlit AppTest (all 9 pages)
ruff check .                                # lint
ruff format --check .                       # formatting
mypy src                                   # types (configured in pyproject.toml)
```

Known-slow suites: `tests/test_baseline.py` (trains models, ~6 min) and
`tests/test_anomaly_train.py` (~7 min). They are correct but slow by design.

## 6. Workflow (developer handoff)

READ → AUDIT → PLAN → AUTHORIZE → IMPLEMENT → VERIFY → CHECKPOINT → COMMIT.

- **READ/AUDIT:** inspect code, artefacts, tests before changing anything.
- **PLAN/AUTHORIZE:** propose minimal change; wait for approval on scope changes.
- **IMPLEMENT:** preserve models, model cards, metrics; no fabrication.
- **VERIFY:** run the relevant pytest + ruff + Streamlit smoke; report actual results.
- **CHECKPOINT/COMMIT:** commit only when explicitly asked. **Do not push to
  GitHub until the user approves publication.**

## 7. Invariants (do not break)

- `models/*.pkl` and `reports/model_card_*.json` are produced by training only.
- RUL evaluation metrics (MAE 13.49, RMSE 18.25, R² 0.807) must not change.
- All synthetic generators use `random_seed = 42`.
- Every dataset/report records provenance; SHAP output states correlation ≠ causation.