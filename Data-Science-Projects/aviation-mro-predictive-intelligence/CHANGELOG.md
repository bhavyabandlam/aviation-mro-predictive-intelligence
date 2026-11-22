# Changelog

All notable changes to the Aviation MRO Predictive Intelligence Platform are
documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-18

### Added

- Repository, package and tooling scaffold (Python 3.13, Ruff, mypy, pytest).
- NASA C-MAPSS FD001 ingestion (whitespace parser), profiling and EDA reports.
- Synthetic MRO business-data generator with deterministic seed and manifest.
- Typed configuration loaders and business-rule parameters.
- SQLite persistence layer (schema + `MRODatabase`) and Parquet exports.
- Data validation layer (schema / referential-integrity / quality checks).
- RUL regression pipeline (engine-level split, leakage-safe) and model cards.
- IsolationForest anomaly detection with contamination sweep and attribution.
- SHAP explainability report for the RUL model.
- Maintenance forecasting baselines with chronological backtesting and
  prediction intervals.
- Inventory analytics (ROP, safety stock, stockout risk, EOQ recommendations).
- Financial analytics (labour/parts/downtime/holding/stockout costs, scenarios).
- Nine-page Streamlit dashboard with provenance badges.
- Licensing (`LICENSE`), changelog, dependency documentation and CI workflow.

### Notes

- NASA C-MAPSS FD001 remains the primary external dataset; synthetic MRO data is
  the business-analytics dataset. N-CMAPSS is optional; BTS and OurAirports are
  deferred. See `docs/DATA_SOURCES.md`.
- Third-party datasets are not covered by the repository `LICENSE`; see
  `docs/DATA_SOURCES.md` and `docs/DEPENDENCIES.md`.