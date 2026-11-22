# Beginner-Friendly Implementation Manual

This guide explains **what the code actually does** and **how to run it**, step by
step, in plain language. It mirrors the real repository (see
`docs/TECHNICAL_REPORT.md` for the formal version and `docs/RECONCILIATION_REPORT.md`
for SRS compliance).

---

## 1. What is this project, in one paragraph?

It is a program that studies aero-engine health and airline maintenance. It uses a
NASA **simulation** of jet engines that slowly wear out, plus **made-up (synthetic)**
maintenance and money data, to answer questions like: *How much life is left in an
engine? Which engines behave strangely? How many spare parts will we need? What does
maintenance cost, and what could we save?* The answers appear in a web dashboard.

Two kinds of data are used and **always kept separate**:
- **NASA FD001** — a *simulation*, used for engine health (RUL) and anomaly detection.
- **Synthetic MRO data** — *invented* maintenance/finance numbers, used for forecasting,
  inventory and money.

---

## 2. The big picture (how data flows)

```
NASA FD001 (engine sensor logs)            Synthetic MRO data
        |                                        |
   [EDA + statistics]                     [forecast demand]
        |                                        |
   [build features]                       [inventory maths]
        |                                        |
   [train RUL model]                      [cost maths]
        |                                        |
   [detect anomalies]                     [SHAP explain RUL]
        \________________ [Streamlit dashboard] ___________/
```

Every step **saves its output to disk** (`reports/`, `models/`, `data/processed/`).
The dashboard only **reads** those saved files — it never re-trains anything.

---

## 3. One-time setup

```powershell
# from the project folder
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check it works:
```powershell
python -c "import streamlit, shap, sklearn; print('ok')"
```

---

## 4. Folder map (what lives where)

| Folder | What it holds |
|---|---|
| `src/aviation_mro/` | The Python "engine" of the project (all the logic). |
| `src/aviation_mro/analytics/` | EDA, forecasting, inventory, financial calculations. |
| `src/aviation_mro/features/` | Turns raw sensor logs into model-ready features. |
| `src/aviation_mro/models/` | RUL model training + anomaly detection. |
| `src/aviation_mro/explainability/` | SHAP explanations of the RUL model. |
| `src/dashboard/` | The Streamlit web app (`app.py` + `views/`). |
| `data/raw/fd001/` | The original NASA files (untouched). |
| `data/processed/` | Cleaned/synthetic datasets (`.parquet`). |
| `models/` | Saved trained models (`.pkl`). |
| `reports/` | Saved results (JSON/Markdown/PNG). |
| `tests/` | Automated checks. |
| `docs/` | Specifications and guides (start with `SRS.md`, then this file). |

---

## 5. Step-by-step: what each stage does

### 5.1 EDA — "get to know the data" (Section 9.1)
- **File:** `src/aviation_mro/analytics/eda.py`
- **Run:** `python -m aviation_mro.analytics.eda`
- **Does:** counts rows/engines, finds missing values and duplicates, spots sensors that
  never change, runs normality/variance/distribution tests, and draws charts.
- **Produces:** `reports/eda/eda_report.json` and `reports/eda/plots/*.png`.

### 5.2 RUL prediction — "how much life is left?" (Section 9.2)
- **Files:** `features/rul_features.py` (features) + `models/baseline.py` (training).
- **Run:** `python -m aviation_mro.models.baseline`
- **Does:** builds past-only rolling sensor features; splits by **engine** (never mixes rows
  of the same engine across train/test); trains several models; picks the best on validation;
  evaluates once on the test set.
- **Produces:** `models/baseline_rul_model_v2.pkl`, `reports/model_card_baseline_v2.json`.
- **Result:** MAE ≈ 13.5 cycles, R² ≈ 0.81.

### 5.3 Anomaly detection — "which engines look odd?" (Section 9.3)
- **Files:** `models/anomaly_model.py` + `models/anomaly_train.py`.
- **Run:** `python -m aviation_mro.models.anomaly_train`
- **Does:** an IsolationForest scores each engine's sensor patterns; unusual ones are flagged
  as **candidates** (not confirmed faults).
- **Produces:** `models/anomaly_model.pkl`, `reports/anomaly_model_card.json`.

### 5.4 Maintenance forecasting — "how many jobs next week?" (Section 9.4)
- **Folder:** `src/aviation_mro/analytics/forecasting/`
- **Run:** `python -m aviation_mro.analytics.forecasting.train`
- **Does:** splits history chronologically; compares a naive forecast with a seasonal-naive
  forecast; reports error (MAE/RMSE/WAPE) and interval coverage.
- **Produces:** `reports/forecasting/forecast_results.json`.

### 5.5 Inventory analytics — "when and how much to reorder?" (Section 9.5)
- **Folder:** `src/aviation_mro/analytics/inventory/`
- **Run:** `python -m aviation_mro.analytics.inventory.pipeline`
- **Does:** computes lead-time demand, reorder points (ROP = lead-time demand + safety stock),
  safety stock (several methods), stockout risk, and replenishment recommendations (EOQ).
- **Produces:** `reports/inventory/inventory_analysis.json`, `INVENTORY_REPORT.md`.

### 5.6 Financial analytics — "what does it cost / save?" (Section 9.6)
- **Folder:** `src/aviation_mro/analytics/financial/`
- **Run:** `python -m aviation_mro.analytics.financial.pipeline`
- **Does:** adds up labour + parts + downtime + holding + stockout costs; computes KPIs and
  scenario comparisons (e.g. predictive maintenance, inventory optimisation).
- **Produces:** `reports/financial/financial_analysis.json`, `FINANCIAL_REPORT.md`.

### 5.7 SHAP explainability — "why did the model say that?" (Section 9.6 XAI)
- **File:** `src/aviation_mro/explainability/rul_explainer.py`
- **Run:** `python -m aviation_mro.explainability.rul_explainer`
- **Does:** explains the existing RUL model **without retraining it**. For each engine it shows
  which sensors pushed the prediction up or down, and checks that the pieces add back to the
  prediction.
- **Produces:** `reports/explainability/rul_shap_summary.json`.

### 5.8 Dashboard — "look at everything in a browser"
- **File:** `src/dashboard/app.py`
- **Run:** `streamlit run src/dashboard/app.py`
- **Does:** 9 pages that read the saved reports/models. If a file is missing, a page shows a
  friendly warning instead of crashing.

---

## 6. How to read the key numbers

- **MAE (cycles):** average size of the RUL error. Smaller is better.
- **RMSE:** like MAE but punishes big misses more.
- **R²:** 1.0 = perfect; 0 = no better than guessing the average.
- **WAPE:** forecast error as a fraction of total demand (0.55 ≈ 55%).
- **Reorder point (ROP):** stock level that triggers a new order.
- **Safety stock:** buffer stock for demand/lead-time surprises.
- **SHAP value:** how much one sensor reading pushed a single prediction up (+) or down (−).

---

## 7. Testing and quality checks

```powershell
pytest -q                         # run all tests
pytest tests/test_explainability.py -v   # just the SHAP tests
ruff check src tests              # style/lint
ruff format --check src tests     # formatting
```

---

## 8. Common questions

- **Will running the dashboard change the models?** No — it only reads saved files.
- **Are the money numbers real?** No — they are synthetic (invented) for demonstration.
- **Do SHAP values prove what caused an engine to fail?** No — they show correlation/contribution,
  not causation.
- **Why do some pages warn about missing files?** Because a pipeline hasn't been run yet; run the
  command in §5 for that section.

---

## 9. Glossary

- **RUL:** Remaining Useful Life (how many cycles an engine can still run).
- **FD001:** a NASA simulated dataset of engines run to failure.
- **IsolationForest:** an algorithm that flags rare/odd patterns.
- **EOQ:** Economic Order Quantity — the order size that minimises ordering + holding cost.
- **SHAP:** a method to explain a model's prediction by splitting it across inputs.
- **Streamlit:** the library that turns Python into a web app.