# Model Cards

Model documentation for the Aviation MRO Predictive Intelligence Platform
(SRS §11, §13). Machine-generated JSON cards in `reports/` are authoritative;
this file is the human-readable summary. Dataset licensing is in
[`DATA_SOURCES.md`](DATA_SOURCES.md).

---

## 1. RUL Regression — `baseline_rul_model_v2`

| Field | Value |
|---|---|
| Artifact | `models/baseline_rul_model_v2.pkl` |
| Algorithm | scikit-learn `GradientBoostingRegressor` |
| Target | `rul_capped = min(RUL, rul_cap)`, `rul_cap = 125` |
| Features | 152 (rolling sensor stats + op settings; see §RUL features) |
| Random seed | 42 |
| Task | Engine Remaining Useful Life (cycles) regression |
| Dataset | NASA C-MAPSS FD001 |
| Dataset fingerprint | `train_FD001.txt=963b5e22825b34d8`, `test_FD001.txt=3cda7109ce17bafb`, `RUL_FD001.txt=a19c8ec94931949d` |

### Data split
- **Train:** 60 engines (e.g. 1, 4, 5, 6, …)
- **Validation:** 20 engines (2, 3, 15, 21, 22, 24, 30, 38, 52, 53, 61, 64, 72, 75, 83, 85, 87, 88, 92, 93)
- **Test:** official FD001 `test_FD001.txt` engines
- Split is **engine-level** (no leakage across splits).

### Model selection
Candidates compared on validation MAE: `DummyMean`, `DummyMedian`, and
gradient boosting. Best model: **GradientBoosting**. Rationale: beats dummy
baselines on MAE/RMSE/R². A v1 negative-R² defect (leaky `cycle_norm`) was
removed; v2 uses only past-only rolling features.

### Test performance (held-out)
| Metric | Value |
|---|---|
| MAE | 13.4948 |
| RMSE | 18.2510 |
| R² | 0.8071 |

Per-engine MAE range: ~0.086 (best) to ~49.96 (worst); mean 13.49, median 9.19.

### Explainability (SHAP 0.54.1)
- `reports/explainability/rul_shap_summary.json`: 152 global feature attributions
  + 12 local (per-engine) explanations.
- TreeSHAP consistency error ≤ 1.85e-06 (numerically exact).
- Dashboard: **RUL Prediction & Explainability** view.

### Intended use / limitations
- Designed for engine RUL trend monitoring, not safety-of-flight certification.
- C-MAPSS is simulated; degradation patterns are idealized.
- Per-engine error variance is high (some engines poorly predicted) — review local
  SHAP before acting on any single prediction.

---

## 2. Anomaly Detection — `anomaly_model`

| Field | Value |
|---|---|
| Artifact | `models/anomaly_model.pkl` |
| Algorithm | scikit-learn `IsolationForest` |
| Features | 182 (152 RUL features + residual features, `residual_windows=[3,5]`) |
| `n_estimators` | 10 |
| `contamination` | 0.01 (default) |
| Default threshold | −0.07955 (score below ⇒ anomaly) |
| Random seed | 42 |
| Dataset | NASA C-MAPSS FD001 |
| Dataset fingerprint | same as RUL card |

### Validation / test flag rates
| Split | Flagged | Total | Rate |
|---|---|---|---|
| Validation | 40 | 4291 | 0.93% |
| Test | 48 | 13096 | 0.37% |

Per-engine attribution lists top contributing sensors for flagged cycles
(`reports/anomaly_model_card.json`).

### Limitations
- **No ground-truth fault labels.** `contamination` is a threshold percentile,
  not a measured fault rate.
- IsolationForest flags statistical outliers; late-life degradation naturally
  appears as anomalies.
- Not a maintenance decision system — use as a screening signal with human review.

---

## 3. Reproduction & provenance
- All artifacts are reproducible with `random_seed = 42` and pinned dependencies
  (`requirements.txt`).
- Model cards in `reports/` (JSON) record `feature_names`, `params`, `metrics`,
  `random_seed`, and `dataset_fingerprint`.
- Dataset fingerprints (16-hex-digit hashes) detect data drift: a changed
  fingerprint means the input data changed and the model should be re-validated.