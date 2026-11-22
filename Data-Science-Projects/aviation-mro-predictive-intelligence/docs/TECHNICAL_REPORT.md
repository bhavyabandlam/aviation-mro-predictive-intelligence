# Technical Project Report — Aviation MRO Predictive Intelligence Platform

**Scope:** This report documents the **actually implemented** code in this repository
(it describes the current build, not the full SRS v1.1 target; see
`docs/RECONCILIATION_REPORT.md` for the gap analysis). Every formula and metric
below is taken from the source.

---

## 1. Technology stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.13 |
| Data | pandas, numpy, pyarrow | 3.0.6 / 2.5.3 / 26.0.1 |
| Statistics | scipy, statsmodels | 1.18.1 / 0.14.4 |
| ML | scikit-learn | 1.9.1 |
| Persistence | joblib (models), Parquet (data) | 1.4.2 |
| Explainability | SHAP | 0.54.1 |
| Dashboard | Streamlit | 1.67.0 |
| Plotting | matplotlib | 3.11.2 |
| Quality | pytest, ruff, mypy | 9.2.0 / 0.17.4 / configured |

---

## 2. Data sources & datasets

### 2.1 NASA C-MAPSS FD001 (simulated benchmark)
Turbofan engine run-to-failure simulation. Three files: `train_FD001.txt` (100 units),
`test_FD001.txt` (100 units, truncated), `RUL_FD001.txt` (ground-truth RUL for test).
Each record: `unit_number, time_in_cycles, op_setting_1..3, sensor_1..21`. Parsed with
a **whitespace** parser (no header assumption) in `features/rul_features.py::load_raw_data`.

### 2.2 Synthetic MRO business data (seed=42)
- `synthetic_mro_v1`: weekly maintenance demand, 5 shops × 3 part families
  (ENGINE, LANDING_GEAR, AVIONICS) → `data/processed/maintenance_demand.parquet`
  (`date, shop_id, part_family, demand_count`).
- `synthetic_inventory_v1`: 150-part master (`part_number, part_family, shop_id,
  criticality, unit_cost, lead_time_days, lead_time_std_days, current_stock,
  demand_mean_weekly, demand_std_weekly, service_level_target`) →
  `data/processed/inventory_data.parquet`.
- `synthetic_financial_v1`: shop labour (5), fleet (3), maintenance labour (15),
  downtime costs (9), inventory cost params (1) → `data/processed/financial_*.parquet`.

All synthetic generators use `numpy.random.default_rng(42)` / `np.random.seed(42)`.

---

## 3. Statistical methods (EDA — `analytics/eda.py`)

Descriptive profiling (shape, schema, missingness, duplicates, ranges), constant-sensor
detection (variance), RUL statistics, and inferential checks:
- **Shapiro-Wilk** (normality), **Levene** (equal variance), **Kolmogorov-Smirnov**
  (train/test distribution shift), **Augmented Dickey-Fuller** (stationarity),
  and **Cohen's d** effect size for train/test degradation-slope comparison.
Output: `reports/eda/eda_report.json` + `reports/eda/plots/*.png`.

---

## 4. Feature engineering (`features/rul_features.py`)

- **RUL target:** `RUL = max(0, rul_cap − time_in_cycles)`, cap = **125**.
- **Rolling aggregates** (past-only, windows **3** and **5**): mean, std, min, max,
  and a slope (linear trend) per informative sensor over 15 sensors + 2 op-settings.
- Optional cycle normalisation (`include_cycle_norm`).
- **Engine-level split** (seed 42): train / validation / test are disjoint sets of
  `unit_id`s — sensor rows are never split across sets. Test uses the official
  truncated FD001 test set. Final feature count = **152**, sorted.

---

## 5. RUL prediction (`models/baseline.py`)

- **Preprocessing:** `SimpleImputer(strategy="median")` → `StandardScaler`, **fit on
  train only**.
- **Models compared (validation):** Dummy, Linear, Ridge, RandomForest, GradientBoosting
  (and others; 6 total). Best selected on **validation RMSE** → **GradientBoostingRegressor**.
- **Evaluation (once, on test):** MAE, RMSE, R² + per-engine absolute-error distribution.
- **Verified results:** Test **MAE 13.49**, **RMSE 18.25**, **R² 0.807**.
- Artefacts: `models/baseline_rul_model_v2.pkl` (imputer, scaler, feature names, best
  model, metrics), `reports/model_card_baseline_v2.json`.

---

## 6. Anomaly detection (`models/anomaly_model.py`)

- **Algorithm:** `IsolationForest` (unsupervised; no RUL labels).
- **Features:** rolling + residual features (182).
- **Threshold:** contamination = train-score percentile (default 2%); contamination
  sweep for sensitivity; sensor context via training **z-score deviation**.
- Output is treated as **anomaly candidates, not confirmed faults** (no precision/recall).
- Artefacts: `models/anomaly_model.pkl`, `reports/anomaly_model_card.json`.

---

## 7. Maintenance forecasting (`analytics/forecasting/`)

- **Chronological** train/validation/test split of synthetic weekly demand.
- **Baselines:** `Naive` (last value) and `SeasonalNaive` (seasonal period **52** vs **13**,
  chosen by validation WAPE).
- **Metrics:** MAE, RMSE, **WAPE** + **empirical prediction-interval coverage**.
- **Verified (test, seasonal-naive):** MAE 2.77, WAPE 0.55.
- Artefacts: `reports/forecasting/forecast_results.json`, `split_metadata.json`, baseline `.pkl`s.

---

## 8. Inventory formulas (`analytics/inventory/`)

Let `D̄` = mean weekly demand, `σ_D` = weekly demand std, `L` = lead time (weeks) = lead_time_days/7, `z` = `Φ⁻¹(service_level)`.

- **Lead-time demand (LTD):** `mean_LTD = D̄ · L`; `std_LTD = σ_D · √L` (constant lead time).
- **Reorder Point:** `ROP = mean_LTD + SafetyStock`.
- **Safety stock (normal, constant LT):** `SS = z · std_LTD`.
- **Safety stock (variable LT):** `SS = z · √(L·σ_D² + D̄²·σ_L²)`.
- **Safety stock (log-normal):** Fenton-Wilkinson approximation; `SS = q − mean_LTD`.
- **Safety stock (empirical):** bootstrap simulation (seed 42) of LTD; `SS = percentile − mean`.
- **Stockout probability:** `P(LTD > ROP) = 1 − Φ((ROP − mean_LTD)/std_LTD)`.
- **Expected shortage per cycle:** `σ_LTD · [φ(z) − z·(1−Φ(z))]`.
- **Fill rate:** `1 − expected_shortage / order_quantity` (order qty ≈ 4 weeks demand).
- **Expected stockouts/year:** `stockout_probability × cycles_per_year`.
- **EOQ (recommendations):** `Q* = √(2·D·S / H)` with `D` = annual demand, `S` = order cost
  ($75), `H = unit_cost × holding_rate` (20%).

Verified output: 150 parts, **421** replenishment recommendations.

---

## 9. Financial calculations (`analytics/financial/`)

- **Labor:** `cost_per_job = hours × rate`; `annual_labor = cost_per_job × job_count`;
  overtime premium `= overtime_hours × (overtime_rate − base_rate)`; `utilization =
  required_hours / capacity`.
- **Parts:** `annual_parts_cost = annual_demand × unit_cost`; **EOQ** as above;
  `annual_ordering_cost = (D/Q*)·S`; `annual_holding_cost = (Q*/2)·unit_cost·holding_rate`.
- **Holding:** `avg_inventory = EOQ/2 + safety_stock`;
  `annual_holding_cost = avg_inventory × unit_cost × holding_rate`.
- **Downtime:** `cost_per_event = downtime_hours × cost_per_hour`;
  `annual_downtime = cost_per_event × event_count`. Scenarios: baseline, −10%, −20%
  (predictive), +15% fleet growth.
- **Stockout:** `expected_stockouts = stockout_probability × cycles_per_year`;
  `cost_per_event = unit_cost × multiplier` (10×); `annual = expected × cost_per_event`.
- **KPIs:** cost per flight hour `= total_cost / flight_hours`; maintenance cost % revenue;
  inventory turnover `= parts_cost / avg_inventory_value`; stockout rate; labor utilization.
- **Cost avoidance (predictive):** `baseline_downtime − predictive_downtime` (20% reduction).
- **ROI (predictive maintenance):** `((cost_avoidance − annual_operating_cost) / implementation_cost) × 100`
  (implementation $500k, operating $100k hardcoded assumptions).
- **Scenarios (8):** baseline, predictive maintenance, inventory optimization, fleet growth,
  labor efficiency, supplier cost +10%, supplier cost −5%, combined optimization.

Verified baseline ≈ **$897.9M/yr**; best scenario (Combined Optimization) saves ≈ **$187.5M**.

---

## 10. Explainability — SHAP (`explainability/rul_explainer.py`)

- **Explainer:** `shap.TreeExplainer(best_model, data=background, model_output="raw")`
  (exact for the GradientBoosting tree ensemble; model reused, **not retrained**).
- **Exact preprocessing/order:** features reconstructed with the model-card `FeatureConfig`
  and transformed with the artefact's own imputer + scaler; order = `artifact.feature_names`.
- **Background:** deterministic 100-row training sample (seed 42).
- **Global importance:** mean `|SHAP|` per feature across 100 test engines.
- **Local explanations:** 12 representative engines (6 best + 6 worst predicted); per-feature
  SHAP values, top-10 contributions, scaled + raw feature values.
- **Additivity check:** `base_value + Σ SHAP ≈ prediction`; observed max abs error ≈ **1.85e-06**.
- Artefact: `reports/explainability/rul_shap_summary.json` (model version, dataset fingerprint,
  feature names, SHAP version, consistency, causation disclaimer).
- **Caveat:** SHAP values are in the **scaled** feature space; contributions are **correlation,
  not causation**.

---

## 11. Streamlit dashboard (`src/dashboard/`)

Entry point: `streamlit run src/dashboard/app.py` (uses `st.navigation`, 9 pages).
Shared modules: `data_loader.py` (safe artefact loaders, graceful on missing files),
`components.py` (source badges, KPI cards, tables, formatters). Views (one per page):

| Page | Module | Reads |
|---|---|---|
| Executive Overview | `views/executive.py` | all reports (mixed) |
| Data Quality & EDA | `views/eda.py` | `eda_report.json` + plots |
| RUL Prediction & Explainability | `views/rul.py` | RUL model card + SHAP summary |
| Anomaly Detection | `views/anomaly.py` | anomaly model card |
| Maintenance Forecasting | `views/forecasting.py` | `forecast_results.json` |
| Inventory Analytics | `views/inventory.py` | `inventory_analysis.json` |
| Financial Analytics | `views/financial.py` | `financial_analysis.json` |
| Explainability (SHAP) | `views/explainability.py` | `rul_shap_summary.json` |
| Data Provenance & Sources | `views/provenance.py` | `DATA_SOURCES.md`, artefact status |

All pages read saved artefacts only (no on-the-fly model inference, no fabricated KPIs);
missing artefacts render a non-fatal warning.

---

## 12. Reproducibility & provenance

- Fixed `random_seed = 42` across generators; dataset fingerprints recorded in report metadata.
- Every page shows a provenance badge (NASA / Synthetic / Mixed).
- Sources documented in `docs/DATA_SOURCES.md`; SHAP methodology in `docs/EXPLAINABILITY.md`.

## 13. Known limitations

- NASA FD001 is a **simulation**; MRO/financial data is **synthetic** — not real airline records.
- SHAP values are in **scaled** feature space; **not causation**.
- Local SHAP explanations cover 12 representative engines, not all 100.
- Not for production or safety-critical/airworthiness decisions.
- This build covers the analytics/ML core; several SRS v1.1 items (SQLite, BTS, full
  statistical suite, CSV/HTML/PDF exports, central registry, prescribed layout) remain — see
  `docs/RECONCILIATION_REPORT.md`.