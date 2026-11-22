# Developer Handoff

## Aviation MRO Predictive Intelligence Platform

  Field                     Value
  ------------------------- ---------------------------------------
  Version                   1.1
  Companion specification   `SRS.md`
  Language                  Python only
  Runtime                   Python 3.13.x
  UI                        Streamlit
  Storage                   SQLite + Parquet
  Development style         Incremental, test-first, reproducible
  Initial environment       Windows 11, local-first

------------------------------------------------------------------------

## 1. Handoff objective

Deliver a complete Python-only data science and business analytics
application based on `SRS.md`.

This is not a request to generate a few notebooks or a static dashboard
mock-up. The deliverable must be a runnable, modular, tested and
documented repository.

**The SRS is authoritative for scope, requirements, data sources and
acceptance.** If this handoff appears to conflict with the SRS, stop and
report the conflict rather than silently choosing.

## 2. Developer role

Act as a senior Python data scientist, data engineer, machine learning
engineer and Streamlit application developer.

Responsibilities: - Inspect source datasets and document their actual
schemas. - Build robust ingestion and validation pipelines. - Perform
defensible statistical analysis. - Train and evaluate ML models without
leakage. - Implement business calculations with tests. - Deliver an
interactive Streamlit application. - Maintain data provenance,
reproducibility and clear documentation. - Provide evidence for every
phase.

## 3. Non-negotiable constraints

1.  Python is the only implementation language.
2.  Use Python 3.13.x.
3.  Streamlit is the UI; no Angular, React, JavaScript application, .NET
    or separate FastAPI service.
4.  Local-first; do not require paid cloud services.
5.  Do not alter or mix the user's separate AI-BOS project.
6.  Do not download data into an unrelated project directory.
7.  Preserve original downloaded data.
8.  Do not commit raw datasets, secrets, large generated artifacts or
    private information.
9.  Do not fabricate results, model scores, business outcomes or
    data-source claims.
10. Do not treat simulated data as real airline records.
11. Do not claim engine predictions are airworthiness decisions.
12. No implementation phase is complete without validation evidence.

## 4. Initial environment and directory

Target Windows location:

`D:\Data-Science-Projects\aviation-mro-predictive-intelligence`

Before creating files: - Confirm the D: drive exists and has adequate
free space. - Confirm Python 3.13 is available. - Confirm Git is
available. - Confirm the target directory does not contain unrelated
work. - Do not delete or overwrite existing user files. - Report
conflicts before proceeding.

Create an isolated virtual environment:

``` powershell
Set-Location D:\Data-Science-Projects\aviation-mro-predictive-intelligence
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
```

If the Python launcher does not recognise `-3.13`, inspect installed
interpreters and use the verified Python 3.13 executable. Do not install
another Python version without approval.

## 5. Required project structure

Follow the structure in `SRS.md`. The minimum structure is:

``` text
aviation-mro-predictive-intelligence/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   ├── database/
│   └── exports/
├── docs/
│   ├── SRS.md
│   ├── DEVELOPER_HANDOFF.md
│   ├── DATA_SOURCES.md
│   ├── DATA_DICTIONARY.md
│   └── MODEL_CARD.md
├── notebooks/
├── src/aviation_mro/
├── dashboard/
├── models/
├── reports/
├── scripts/
├── tests/
├── pyproject.toml
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

Keep raw data and large artifacts out of Git using `.gitignore`. Add
small schema examples or generated sample data only when useful and
clearly labelled.

## 6. Data source onboarding --- do this before modelling

Create `docs/DATA_SOURCES.md` and `config/data_sources.yaml`.

For every source, record: - Source ID and title. - Official landing
page. - Direct download URL, if available. - Dataset file name and
format. - Retrieval date/time. - Version or reporting period. - SHA-256
checksum. - Publisher and original authors. - Citation. - Licence/terms
status. - Known limitations. - Local storage path. - Parser/loader used.

### Required source order

**A. NASA C-MAPSS --- mandatory primary dataset**

Official NASA PCoE repository:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

NASA PCoE archive:
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

Tasks: 1. Open the official repository page. 2. Verify the turbofan
degradation dataset entry. 3. Attempt the official archive download. 4.
Verify HTTP/download success and archive integrity. 5. Inspect archive
members before extraction. 6. Read the included README. 7. Record exact
files, columns, row counts and file hashes. 8. Preserve the archive
under `data/raw/`. 9. Extract to a separate raw subdirectory. 10. Parse
whitespace-delimited text without assuming headers. 11. Compare observed
columns against the expected schema in the SRS. 12. Produce a data
profiling report.

Start with FD001. Add FD002--FD004 only after FD001 passes ingestion and
quality tests.

Do not substitute a third-party re-hosted dataset without documenting
the source and obtaining approval.

**B. BTS TranStats --- optional separate operational analysis**

Official field selection/download:
https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ

Official downloads: https://www.bts.gov/airline-data-downloads

Tasks: - Select a manageable time period and only required columns. -
Save selected fields and reporting period. - Read current field
definitions and reporting notes. - Record download metadata. - Analyse
delays/cancellations as a separate operational dataset. - Do not join
flight delays to NASA engine IDs or infer engine causation.

**C. OurAirports --- optional reference data**

https://ourairports.com/data/

Use only if airport-level descriptive analysis or mapping is
implemented. Record the source's public-domain statement and its
accuracy disclaimer.

**D. Synthetic MRO data --- mandatory for business modules**

Generate reproducible, clearly labelled data for aircraft, engines,
maintenance orders, technicians, inventory, inventory transactions and
downtime.

Requirements: - Fixed configurable random seed. - Configurable row
counts. - Consistent primary/foreign keys. - Valid dates and numeric
ranges. - Coherent stock movements and costs. - Explicit `is_synthetic`
or equivalent metadata. - Data dictionary and generation manifest.

Synthetic records must never be blended invisibly with NASA or BTS data.

### Source failure policy

If an official URL is unavailable, the file is corrupt, or terms are
unclear: - Stop that source's ingestion. - Record the URL, date and
error. - Do not silently use a mirror. - Continue unrelated work only if
it does not depend on the failed source. - Request approval for any
substitute.

## 7. Architecture and code boundaries

Use a layered modular design:

-   `ingestion`: source-specific readers and synthetic generation.
-   `validation`: schemas, quality rules and source checks.
-   `preprocessing`: cleaning and transformations.
-   `features`: reusable feature pipelines.
-   `analytics`: EDA and statistical methods.
-   `models`: training, prediction, evaluation and explainability.
-   `business`: inventory, maintenance and financial rules.
-   `persistence`: SQLite and Parquet access.
-   `services`: orchestration for dashboard use.
-   `dashboard`: Streamlit pages and presentation components.

Rules: - Streamlit pages must not contain complex business logic. -
Notebooks are for exploration, not the only place where functionality
exists. - Reusable calculations belong in `src/`. - Configuration
belongs in YAML or typed settings, not scattered literals. - Use
explicit types and meaningful exceptions. - Avoid unnecessary
abstractions and dependency-heavy frameworks.

## 8. Dependency strategy

Use only packages justified by requirements. Suggested groups:

**Core:** pandas, numpy, pyarrow\
**Statistics:** scipy, statsmodels\
**ML:** scikit-learn; xgboost only if justified\
**Visualisation:** matplotlib, seaborn, plotly\
**Dashboard:** streamlit\
**Explainability:** shap\
**Validation:** pandera\
**Testing/quality:** pytest, pytest-cov, ruff, mypy\
**Experiments:** mlflow optional\
**Notebooks:** jupyterlab

Before adding a dependency: 1. Confirm Python 3.13 compatibility. 2.
Confirm licence and maintenance status. 3. Assess installation size and
hardware impact. 4. Explain why the standard library or an existing
dependency is insufficient. 5. Update dependency files and
documentation.

Pin or constrain versions in a reproducible way. Do not blindly install
every suggested package at once.

## 9. Data science implementation rules

### 9.1 EDA and statistics

-   Start with data shape, schema, missingness, duplicates and ranges.
-   Check whether measurements are constant or near-constant.
-   Visualise distributions and trends.
-   State the question each chart answers.
-   Test assumptions before selecting inferential methods.
-   Report effect sizes and uncertainty, not only p-values.
-   Separate exploratory findings from confirmatory conclusions.

### 9.2 RUL prediction

-   Establish a simple baseline first.
-   Split by engine/unit; never randomly split sensor rows across train
    and test.
-   Use validation data for model selection.
-   Preserve final test data for final evaluation.
-   Fit imputation, scaling and feature selection on training data only.
-   Use past-only rolling features.
-   Document RUL capping.
-   Evaluate MAE, RMSE and R², plus per-engine error distributions.
-   Compare models on identical splits and feature definitions.
-   Save dataset fingerprint, model parameters, feature names and
    metrics.

### 9.3 Anomaly detection

-   Begin with IsolationForest.
-   Treat output as anomaly candidates, not confirmed faults.
-   Document contamination/threshold assumptions.
-   Provide sensor context for flagged observations.
-   Do not claim precision/recall without defensible labels.

### 9.4 Forecasting

-   Use chronological train/validation/test windows.
-   Compare with naive and seasonal-naive baselines.
-   Prevent future observations from entering features.
-   Report MAE, RMSE and WAPE.
-   Show horizon and uncertainty.
-   Do not extrapolate far beyond the support of the historical series
    without warning.

### 9.5 Inventory and finance

-   Centralise formulas and assumptions.
-   Test units, edge cases and zero-demand cases.
-   Explain safety-stock service level and demand variability
    assumptions.
-   Make lead time and cost units explicit.
-   Distinguish actual recorded values in synthetic data from scenario
    assumptions.
-   Label projected savings and ROI as estimates.

### 9.6 Explainability

-   Use SHAP only with a supported and validated model setup.
-   Record model version and feature values.
-   Do not interpret association or feature contribution as causation.
-   Make explanations understandable to a business user.

## 10. Streamlit dashboard handoff

Implement these pages: 1. Executive Overview. 2. Data Explorer &
Quality. 3. Engine Health & RUL. 4. Anomaly Analysis. 5. Maintenance
Forecasting. 6. Inventory Intelligence. 7. Financial Scenarios. 8. Model
Performance & Explainability. 9. Data Sources & Model Registry.

Every page must: - Use real computed outputs from the selected
dataset. - Display the source/data class: NASA simulated, BTS public
operational, or synthetic MRO. - Show units and relevant time window. -
Provide clear empty/loading/error states. - Avoid invented default
KPIs. - Use consistent filters and labels. - Support CSV export where
useful.

Cache only safe, deterministic computations. Cache keys must account for
dataset identity, selected filters and model version.

## 11. Testing strategy

### Unit tests

-   Parsers and schema mapping.
-   Cleaning and validation rules.
-   Feature calculations.
-   RUL target construction.
-   KPI formulas.
-   Inventory reorder calculations.
-   Financial and ROI calculations.
-   Metric functions.

### Integration tests

-   Archive/file ingestion to processed storage.
-   Synthetic generator to SQLite.
-   Training pipeline to saved model.
-   Saved model to prediction.
-   Dashboard service to data retrieval.
-   Export generation.

### ML/data tests

-   Engine-level split integrity.
-   Temporal split integrity.
-   No target leakage.
-   Reproducibility with fixed seed.
-   Prediction shape and finite values.
-   Baseline comparison.
-   Data schema drift detection.

### Quality commands

Document and run the actual configured commands, for example:

``` powershell
python -m pytest
ruff check .
ruff format --check .
mypy src
streamlit run dashboard/app.py
```

If a command is not configured or a tool is not installed, report that
accurately. Do not claim a check passed when it was skipped.

## 12. Development workflow and approval gates

Follow this workflow for every phase:

**READ → AUDIT → PLAN → AUTHORIZE → IMPLEMENT → VERIFY → CHECKPOINT →
COMMIT**

Before implementation: - Read `SRS.md`, this handoff and relevant
docs. - Inspect the repository and working tree. - Identify dependencies
and risks. - Propose the smallest coherent change. - Define tests and
acceptance evidence. - Wait for authorization where required by the
project owner.

During implementation: - Work in small, reviewable increments. - Do not
rewrite unrelated files. - Preserve existing work. - Do not skip data
inspection. - Do not suppress errors to make a demo appear successful.

After implementation: - Run targeted tests, then relevant full tests. -
Run lint and type checks. - Inspect generated data and reports. - Review
the diff. - Update documentation and changelog. - Provide a checkpoint
report. - Commit only after validation and approval.

If a conflict, missing requirement, destructive operation or unexpected
repository state is found: **STOP → REPORT → CHANGE CONTROL**.

## 13. Phase plan and developer handoff checklist

### Phase 0 --- Foundation

-   [ ] Inspect target directory and environment.
-   [ ] Create isolated Python environment.
-   [ ] Create package and repository structure.
-   [ ] Configure pyproject, dependencies, Ruff, Mypy and Pytest.
-   [ ] Add logging and typed configuration.
-   [ ] Add Streamlit skeleton and smoke test.
-   [ ] Add `.gitignore`.
-   [ ] Verify clean installation.

### Phase 1 --- Data sources and ingestion

-   [ ] Create source registry and `DATA_SOURCES.md`.
-   [ ] Verify official NASA C-MAPSS source.
-   [ ] Download and inspect archive.
-   [ ] Implement whitespace parser.
-   [ ] Validate FD001 and produce profiling report.
-   [ ] Implement synthetic MRO generator.
-   [ ] Create SQLite schema and Parquet outputs.
-   [ ] Add source lineage and checksums.
-   [ ] Add ingestion and data-quality tests.

### Phase 2 --- EDA and statistics

-   [ ] Create EDA notebooks.
-   [ ] Implement reusable EDA functions.
-   [ ] Implement descriptive statistics.
-   [ ] Implement correlation and hypothesis tests.
-   [ ] Document assumptions and limitations.
-   [ ] Produce reproducible reports and charts.

### Phase 3 --- Predictive modelling

-   [ ] Build RUL target pipeline.
-   [ ] Create engine-based train/validation/test strategy.
-   [ ] Train baseline model.
-   [ ] Train candidate regressors.
-   [ ] Compare metrics and residuals.
-   [ ] Implement anomaly detection.
-   [ ] Add SHAP where compatible.
-   [ ] Save model metadata and model card.
-   [ ] Add leakage and reproducibility tests.

### Phase 4 --- Business analytics

-   [ ] Build maintenance workload series.
-   [ ] Add forecast baselines and backtesting.
-   [ ] Implement inventory KPIs and reorder logic.
-   [ ] Implement cost and downtime scenarios.
-   [ ] Implement sensitivity analysis.
-   [ ] Test all business formulas.

### Phase 5 --- Dashboard

-   [ ] Implement all nine pages.
-   [ ] Connect pages to application services.
-   [ ] Add filters, units and provenance labels.
-   [ ] Add exports.
-   [ ] Test empty, invalid and normal states.
-   [ ] Verify no hardcoded demonstration metrics.

### Phase 6 --- Release

-   [ ] Run full test suite.
-   [ ] Run lint and type checks.
-   [ ] Verify fresh environment installation.
-   [ ] Complete README and setup guide.
-   [ ] Complete source register and data dictionary.
-   [ ] Complete model card and limitations.
-   [ ] Capture genuine screenshots.
-   [ ] Record a reproducible demo.
-   [ ] Review Git status and licensing.
-   [ ] Prepare public GitHub release.

## 14. Required phase completion report

At the end of every phase, provide:

### A. Summary

-   Phase name and status.
-   Requirements completed.
-   Requirements not completed.

### B. File changes

-   Created files.
-   Modified files.
-   Deleted files, if any, with prior approval.

### C. Evidence

-   Commands executed.
-   Test counts and results.
-   Lint/type-check results.
-   Data row counts and schema checks.
-   Model metrics where relevant.
-   Screenshot/report paths.

### D. Risks and limitations

-   Source availability issues.
-   Licence questions.
-   Data quality concerns.
-   Model limitations.
-   Performance constraints.

### E. Next phase

-   Remaining work.
-   Dependencies.
-   Proposed plan.
-   Required approval.

Never report a phase as complete based solely on code generation.

## 15. Final release acceptance

The project is accepted only when: - All mandatory SRS requirements are
implemented or formally deferred. - The official input sources and
actual downloaded file formats are documented. - Data provenance and
synthetic-data labelling are visible. - RUL evaluation is leakage-safe
and reproducible. - Forecasts use chronological validation. - Business
calculations have tests. - Dashboard values are derived from data. -
Tests, lint and configured type checks pass. - Installation works in a
clean Python 3.13 environment. - Documentation and demo are complete. -
No unsupported claims or restricted data are published.

## 16. First instruction to the coding agent

**Do not start implementing the entire application.**

Begin with Phase 0 only.

1.  Read `SRS.md` and `DEVELOPER_HANDOFF.md`.
2.  Inspect the target directory and Python/Git environment.
3.  Report available disk space and Python version.
4.  Propose the exact Phase 0 file tree and dependency plan.
5.  Identify any conflicts or risks.
6.  Wait for approval before creating or changing project files.

The first milestone is a clean, reproducible Python project
foundation---not an unfinished dashboard or a premature ML model.
