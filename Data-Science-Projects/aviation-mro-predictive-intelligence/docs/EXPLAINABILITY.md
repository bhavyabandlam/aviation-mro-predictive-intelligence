# Explainability (SHAP) — RUL Model

**Status:** Implemented (Section 9.6)
**Model explained:** `models/baseline_rul_model_v2.pkl` — scikit-learn `GradientBoostingRegressor`
**Explainer:** SHAP `TreeExplainer` (`shap==0.54.1`)
**Artefact:** `reports/explainability/rul_shap_summary.json` (precomputed offline)

> **Causation disclaimer:** SHAP values show how each feature *pushes* a model
> prediction higher or lower for a specific engine. They reflect correlations
> learned from the **NASA C-MAPSS FD001 simulated** training data and **do not
> establish causation**.

---

## 1. Why SHAP

The RUL model predicts remaining useful life (in cycles) for a turbofan engine.
A single number is not enough for maintenance decisions — engineers need to know
*which sensor readings* drove the prediction. SHAP (SHapley Additive exPlanations)
attributes the prediction to input features in a way that is:

- **Additive (faithful):** `base_value + Σ SHAP values == prediction` (verified).
- **Consistent:** the same feature does not flip attribution direction arbitrarily.
- **Model-agnostic in principle, exact for trees:** `TreeExplainer` computes exact
  Shapley values for tree ensembles efficiently.

## 2. Compatibility (why TreeExplainer is valid here)

| Property | Value |
|---|---|
| Trained algorithm | `GradientBoostingRegressor` (best of 6 compared in Section 9.2) |
| Preprocessing | `SimpleImputer(median)` → `StandardScaler`, fit on **train only** |
| Feature count / order | 152 engineered rolling features, **sorted** (windows 3 & 5) |
| Explainer | `shap.TreeExplainer(model, data=background, model_output="raw")` |
| Background | Deterministic sample of 100 training rows (seed=42) |

Because the model is a tree ensemble, `TreeExplainer` is **exact** (not a sampling
approximation), fast, and requires no retraining.

## 3. Exact feature order & preprocessing

Explanations use the **same** `FeatureConfig` (from the model card) and the **same**
fitted imputer + scaler as training:

1. Reconstruct features with `aviation_mro.features.rul_features.load_and_prepare_data(config)`.
2. Align columns to `artifact.feature_names` (the exact trained order).
3. Apply `artifact.imputer.transform` then `artifact.scaler.transform`.

Verified: reconstructed feature order == artifact order == model-card order, and
re-predicting reproduces the model card test metrics **exactly**
(MAE 13.4948, R² 0.8071). Therefore explanations correspond to the displayed
prediction.

## 4. What is produced

- **Global importance:** mean `|SHAP|` per feature across all 100 test engines,
  ranked. Shows which sensors drive predictions fleet-wide.
- **Local explanations:** for 12 representative engines (6 best-predicted + 6
  worst-predicted), per-feature SHAP values, top-10 contributions with direction
  (increases/decreases RUL), and both **scaled** and **raw** feature values.
- **Consistency check:** per-engine and overall `max |base + ΣSHAP − prediction|`.

## 5. Feature space and units

SHAP values and the accompanying `feature_values_scaled` are in the **scaled
model-input space** (median-imputed, then standardised). They are **standardised
values, not original sensor units**. `feature_values_raw` holds the original
engineered rolling-feature values for reference. Interpret **direction and
relative magnitude**, not absolute sensor units.

## 6. Numerical consistency (validation)

For every explained engine:

```
base_value + Σ_j shap_value_j  ≈  model_prediction
```

Observed maximum absolute error: **~1.85 × 10⁻⁶** (float32 precision), well under
the 1e-3 tolerance. This is asserted in `tests/test_explainability.py`.

## 7. How to regenerate

```bash
# from the project root
python -m aviation_mro.explainability.rul_explainer
# -> writes reports/explainability/rul_shap_summary.json
```

The dashboard reads the saved JSON only — **no SHAP computation happens at page
load**. If the file is missing, the dashboard shows instructions instead of
failing.

## 8. Limitations

1. Explanations use the **NASA FD001 simulated** benchmark, not real fleet data.
2. SHAP values are in **scaled** feature space (standardised), not original units.
3. **Correlation, not causation** — attributions must not be read as causal effects.
4. Local explanations cover 12 representative engines, not all 100.
5. Background distribution is a deterministic 100-row training sample; different
   backgrounds can shift attributions slightly.

## 9. Tests

`tests/test_explainability.py` (15 tests, all passing) covers: SHAP availability
(hard requirement), explainer/model match, feature-name/order alignment,
prediction↔explanation additivity, global-importance finiteness/ordering,
provenance metadata, missing/malformed-input handling, JSON round-trip, and
non-regression of the RUL evaluation metrics.