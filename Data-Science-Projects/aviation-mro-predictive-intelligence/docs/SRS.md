# Software Requirements Specification (SRS)

## Aviation MRO Predictive Intelligence Platform

  Document field       Value
  -------------------- ----------------------------------------
  Version              1.1
  Status               Development baseline
  Product              Aviation MRO Predictive Intelligence
  Implementation       Python only
  Target runtime       Python 3.13.x
  UI                   Streamlit
  Primary database     SQLite
  Default deployment   Local Windows development
  Repository           `aviation-mro-predictive-intelligence`

------------------------------------------------------------------------

## 1. Purpose and product vision

Build an end-to-end, reproducible Python application that demonstrates
how data analytics, statistics, machine learning and business
intelligence can support aviation maintenance planning.

The application will combine: 1. Public prognostics / engine-health
datasets. 2. Public airline operational datasets for a separate
operational-performance analysis. 3. Clearly labelled synthetic MRO
business data for work orders, inventory, labour and financial
scenarios.

**Important:** Public engine-health benchmark data and public airline
performance data do not provide a joined, real-world airline MRO
history. The application must not imply that a flight delay was caused
by a predicted engine fault. Synthetic MRO data must be visibly labelled
as synthetic.

The product is a portfolio and decision-support demonstration, not
certified aviation maintenance software. Predictions must never be
represented as airworthiness determinations or maintenance release
instructions.

## 2. Business objectives

  -----------------------------------------------------------------------
  ID                      Objective               Measurable product
                                                  outcome
  ----------------------- ----------------------- -----------------------
  BO-01                   Understand engine       Sensor and degradation
                          degradation             analysis with
                                                  documented findings

  BO-02                   Estimate remaining      RUL models evaluated on
                          useful life             held-out engines

  BO-03                   Detect unusual sensor   Anomaly scores and
                          behaviour               reviewable examples

  BO-04                   Plan maintenance        Time-based workload
                          workload                forecasts and backtests

  BO-05                   Improve inventory       Demand, reorder point
                          visibility              and stockout-risk
                                                  analysis

  BO-06                   Understand maintenance  Transparent cost and
                          economics               scenario calculations

  BO-07                   Communicate findings    Interactive dashboard
                                                  and exportable reports

  BO-08                   Demonstrate             Reproducible pipelines,
                          professional data       tests, documentation
                          science                 and model governance
  -----------------------------------------------------------------------

## 3. Users and use cases

  -----------------------------------------------------------------------
  User                                Use cases
  ----------------------------------- -----------------------------------
  Executive / business stakeholder    Review fleet-level indicators,
                                      costs and scenario results

  Reliability analyst                 Explore sensor behaviour, RUL and
                                      anomaly results

  Maintenance planner                 Review historical work orders and
                                      workload forecasts

  Inventory analyst                   Review stock levels, demand and
                                      reorder recommendations

  Data analyst / scientist            Inspect data quality, EDA,
                                      statistical tests and model metrics

  Interviewer / reviewer              Reproduce the workflow and verify
                                      claims from source data
  -----------------------------------------------------------------------

## 4. Scope

### 4.1 In scope

-   Dataset discovery, download instructions and source registry.
-   CSV, TXT and Parquet ingestion.
-   Data profiling, validation and cleaning.
-   Exploratory and statistical analysis.
-   Engine-level RUL regression.
-   Sensor anomaly detection.
-   Maintenance workload and cost forecasting on synthetic MRO history.
-   Inventory analytics on synthetic parts transactions.
-   Scenario-based financial analysis.
-   SHAP-based model explanation where supported.
-   Streamlit dashboard.
-   SQLite metadata/business store and Parquet analytical storage.
-   Reproducible experiments, tests, reports and GitHub documentation.

### 4.2 Out of scope

-   Live aircraft or engine telemetry.
-   Direct integration with airline/MRO proprietary systems.
-   Flight-critical or airworthiness decisions.
-   Certified maintenance planning.
-   Automatic work-order approval.
-   Claiming that benchmark predictions represent a real airline fleet.
-   Claiming measured cost savings from simulated data.
-   Paid cloud infrastructure in the first release.
-   A JavaScript, Angular, React, .NET or separate API application.

## 5. Data input sources and source-of-truth policy

### 5.1 Source register

The source register must be maintained in `docs/DATA_SOURCES.md` and
mirrored in machine-readable form in `config/data_sources.yaml`. The
developer must record retrieval date, source URL, dataset version/file
name, checksum, citation, licence/terms review and any transformation.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ID          Source              Input / purpose         Access URL                                                                                                      Data character   Usage rules
  ----------- ------------------- ----------------------- --------------------------------------------------------------------------------------------------------------- ---------------- ------------------
  DS-01       NASA Prognostics    Primary engine sensor   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/   Simulated engine Cite NASA and
              Center of           and run-to-failure                                                                                                                      data             original dataset
              Excellence (PCoE)   benchmark for RUL                                                                                                                                        authors; review
              repository --                                                                                                                                                                repository terms
              Turbofan Engine                                                                                                                                                              and any
              Degradation                                                                                                                                                                  dataset-specific
              Simulation                                                                                                                                                                   restrictions
              (C-MAPSS)                                                                                                                                                                    before
                                                                                                                                                                                           redistribution

  DS-02       NASA PCoE direct    Download archive for    https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip               Simulated engine Use only after
              C-MAPSS archive     DS-01                                                                                                                                   data             verifying download
                                                                                                                                                                                           integrity and
                                                                                                                                                                                           included README;
                                                                                                                                                                                           retain original
                                                                                                                                                                                           archive

  DS-03       NASA PCoE Turbofan  Optional advanced       https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/   Simulated        Optional
              Engine Degradation  engine degradation                                                                                                                      run-to-failure   extension; inspect
              Simulation-2 (often dataset with more                                                                                                                       data             its own README,
              referred to as      realistic flight                                                                                                                                         schema, citation
              N-CMAPSS)           conditions                                                                                                                                               and terms before
                                                                                                                                                                                           implementation

  DS-04       BTS TranStats --    Independent airline     https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ                           Public reported  Select required
              Reporting Carrier   operations analysis:                                                                                                                    operational data fields and period;
              On-Time Performance scheduled/actual times,                                                                                                                                  follow BTS field
                                  delays, cancellations                                                                                                                                    definitions and
                                  and related fields                                                                                                                                       reporting notes;
                                                                                                                                                                                           do not infer
                                                                                                                                                                                           engine causality

  DS-05       BTS Airline         Reference page for      https://www.bts.gov/airline-data-downloads                                                                      Public           Check the selected
              Information         airline data products                                                                                                                   government data  product's
              downloads           and documentation                                                                                                                                        documentation and
                                                                                                                                                                                           terms

  DS-06       OurAirports open    Airport reference       https://ourairports.com/data/                                                                                   Public airport   Public domain
              data                dimensions for                                                                                                                          reference data   according to
                                  mapping/airport-level                                                                                                                                    source; source
                                  descriptive analysis                                                                                                                                     says data has no
                                                                                                                                                                                           accuracy/fitness
                                                                                                                                                                                           guarantee;
                                                                                                                                                                                           attribute where
                                                                                                                                                                                           practical

  DS-07       Project-generated   Work orders,            Generated by this project; no external download                                                                 Synthetic        Must be labelled
              synthetic MRO data  aircraft/engine                                                                                                                                          synthetic in
                                  relationships, parts,                                                                                                                                    filenames,
                                  inventory movements,                                                                                                                                     metadata, charts
                                  technicians, downtime                                                                                                                                    and README; fixed
                                  and costs                                                                                                                                                seed and
                                                                                                                                                                                           generation
                                                                                                                                                                                           configuration
                                                                                                                                                                                           required
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Source availability note:** NASA data links and repository content can
change. The developer must test the official source URL at
implementation time. If a source is unavailable, do not silently
substitute an unofficial mirror. Record the failure and request approval
for an alternate source. A NASA catalog record has at times shown
separate C-MAPSS simulator software as unavailable; this is distinct
from the PCoE benchmark data archive. Verify the exact asset being
downloaded.

### 5.2 Recommended first-release dataset plan

1.  **Mandatory:** DS-01 / DS-02 C-MAPSS. Start with FD001 for a small,
    understandable baseline, then evaluate FD002--FD004.
2.  **Mandatory:** DS-07 synthetic MRO tables for maintenance, inventory
    and financial modules.
3.  **Optional, separate analysis:** DS-04 BTS on-time performance, with
    its own data model and dashboard section.
4.  **Optional:** DS-06 airport reference data if airport-level BTS
    analysis is implemented.
5.  **Deferred:** DS-03 advanced NASA engine dataset until C-MAPSS
    pipeline, evaluation and documentation are complete.

### 5.3 C-MAPSS input format and expected schema

The classic C-MAPSS benchmark is commonly distributed as
whitespace-delimited text files, not ordinary CSV. The parser must
support whitespace-separated rows and must not assume a header row.

Expected columns:

  Position / canonical name     Meaning
  ----------------------------- -----------------------
  `unit_id`                     Engine identifier
  `cycle`                       Operating cycle
  `setting_1`                   Operational setting 1
  `setting_2`                   Operational setting 2
  `setting_3`                   Operational setting 3
  `sensor_01` ... `sensor_21`   Sensor channels

The parser should assign canonical names after verifying the number of
columns. Preserve raw source values and create a separate processed
representation.

Expected files may include: - `train_FD001.txt` ... `train_FD004.txt` -
`test_FD001.txt` ... `test_FD004.txt` - `RUL_FD001.txt` ...
`RUL_FD004.txt` - `readme.txt`

Actual archive contents must be inspected; missing or differently named
files must produce a clear validation report.

### 5.4 RUL target construction

For a run-to-failure training trajectory, a common uncapped target is:

`RUL(unit, cycle) = max_cycle_for_unit - cycle`

If a capped target is used, document the cap and treat it as a modelling
choice, not ground truth.

For the official test trajectories, use the supplied RUL label file and
the final observed cycle per test unit to calculate the target at the
final observation. Do not manufacture row-level test labels for earlier
cycles unless the methodology is explicitly justified.

### 5.5 BTS input handling

BTS TranStats provides selectable fields and time periods. The ingestion
process must save: - Original download file. - Selected year/month
range. - Selected field list. - BTS table name. - Download date. -
Documentation URL. - File checksum.

Use BTS for questions such as delay distribution by carrier, airport,
route or month, where selected fields support the analysis. Do not join
BTS delays to NASA engine units. Do not state or imply that delay causes
are engine failures unless a source explicitly supports that
relationship.

### 5.6 Synthetic MRO data specification

Create a deterministic generator under
`src/aviation_mro/ingestion/synthetic_generator.py`. The default seed
must be fixed and configurable.

Required logical tables:

**Aircraft** - `aircraft_id` (primary key) - `registration` (synthetic
identifier; never imitate a real registration) - `aircraft_type` -
`manufacturer` - `manufacture_year` - `operator_id` - `status`

**Engines** - `engine_id` (primary key) - `aircraft_id` (foreign key) -
`engine_model` - `installation_date` - `total_cycles` -
`total_flight_hours` - `status`

**Maintenance orders** - `work_order_id` (primary key) - `aircraft_id`,
`engine_id` - `maintenance_type`, `maintenance_category` - `opened_at`,
`scheduled_at`, `completed_at` - `labour_hours`, `labour_cost`,
`material_cost`, `total_cost` - `status` - `is_synthetic` (must be true)

**Technicians** - `technician_id` (primary key) - `skill_category` -
`experience_years` - `available_hours` - `employment_status`

**Inventory parts** - `part_id` (primary key) - `part_number`
(synthetic) - `description`, `category` - `current_stock`,
`minimum_stock` - `unit_cost`, `lead_time_days` - `supplier_id`

**Inventory transactions** - `transaction_id` (primary key) - `part_id`
(foreign key) - `transaction_date` - `transaction_type` - `quantity` -
`unit_cost`

**Downtime events** - `downtime_id` (primary key) - `aircraft_id`,
`engine_id` - `start_at`, `end_at` - `downtime_reason` -
`estimated_cost_per_hour` - `is_synthetic` (must be true)

Synthetic relationships must be internally consistent. Costs, dates,
quantities and lead times must follow documented generation assumptions.
The generator must produce a data dictionary and a generation manifest.

### 5.7 Data licensing, attribution and redistribution

-   Do not assume that every dataset on a NASA-hosted page has identical
    licensing or redistribution rights.
-   Review the dataset's own README, source page and any included terms.
-   BTS dataset access/terms must be checked for the selected data
    product.
-   OurAirports states that its data is public domain and also disclaims
    accuracy/fitness guarantees.
-   Never commit third-party raw data to Git without confirming
    redistribution permission.
-   Prefer scripts and instructions that let users download source data
    themselves.
-   Store citations and retrieval metadata alongside every imported
    dataset.

## 6. Functional requirements

### FR-01 Data source management and ingestion

-   Support local CSV, whitespace-delimited TXT and Parquet inputs.
-   Provide source-specific NASA C-MAPSS loader.
-   Provide optional BTS CSV loader.
-   Generate synthetic MRO data.
-   Validate file existence, format, size, columns and parseability.
-   Record ingestion metadata and SHA-256 checksum.
-   Preserve raw files unchanged.
-   Never silently replace an unavailable source with an unapproved
    mirror.

**Acceptance:** A valid source can be ingested reproducibly; invalid
inputs produce actionable errors and a quality report.

### FR-02 Data validation and cleaning

-   Missingness, duplicate and type analysis.
-   Numeric range and date checks.
-   Referential integrity checks for synthetic tables.
-   Outlier flagging without automatic deletion.
-   Explicit cleaning rules and before/after row counts.
-   Data quality report with issue severity.

**Acceptance:** Every transformation is traceable; raw data is
unchanged; records are not silently dropped.

### FR-03 Exploratory data analysis

-   Univariate summaries and distributions.
-   Bivariate and multivariate analysis.
-   Sensor correlations and trends.
-   Engine-level degradation curves.
-   BTS delay and cancellation analysis where applicable.
-   Export charts and summary tables.

### FR-04 Statistical analysis

-   Pearson and Spearman correlation.
-   Appropriate two-group tests (t-test or Mann--Whitney U).
-   Chi-square for categorical associations.
-   ANOVA only when assumptions are sufficiently met.
-   Confidence intervals, effect sizes and p-values.
-   Assumption checks and plain-language interpretation.

Statistical significance must not be equated with practical/business
significance.

### FR-05 Feature engineering

-   Cycle and engine-age features.
-   Lagged sensor values.
-   Rolling mean and standard deviation.
-   Sensor slopes/trends.
-   Operating-condition normalisation where justified.
-   Feature pipeline fit on training data only.
-   Past-only rolling windows for temporal features.

### FR-06 RUL prediction

Baseline and candidate regressors: - DummyRegressor baseline. -
LinearRegression or Ridge. - RandomForestRegressor. -
HistGradientBoostingRegressor or GradientBoostingRegressor. - XGBoost
only if dependency installation and value are justified.

Metrics: - MAE - RMSE - R² - Residual analysis - Per-engine performance

Validation: - Split by engine/unit, not random rows. - Keep a final
holdout untouched until model selection. - Use validation data for
tuning. - Document capped/uncapped RUL strategy. - No future-cycle
leakage.

### FR-07 Anomaly detection

-   IsolationForest baseline.
-   Optional LocalOutlierFactor or OneClassSVM.
-   Score and flag unusual observations.
-   Visualise sensor context.
-   Evaluate with labels only where defensible labels exist.
-   If labels do not exist, describe results as unsupervised anomaly
    candidates, not confirmed failures.

### FR-08 Maintenance workload forecasting

Use synthetic maintenance history: - Work-order counts by week/month. -
Labour-hour demand. - Maintenance expenditure. - Naive and
seasonal-naive baselines. - Exponential smoothing / ARIMA only where
data supports them. - Chronological backtesting. - MAE, RMSE and WAPE. -
Forecast horizon and uncertainty displayed.

### FR-09 Inventory analytics

-   Consumption and demand summaries.
-   Reorder point = average demand during lead time + safety stock.
-   Safety stock method and service-level assumptions documented.
-   Stockout-risk indicators.
-   Slow-moving inventory.
-   Inventory value and turnover.
-   Scenario comparison.

### FR-10 Financial analytics

-   Planned/unplanned cost categories.
-   Labour and material costs.
-   Downtime cost scenarios.
-   Cost per order and per synthetic engine.
-   Baseline vs scenario comparisons.
-   Benefit, investment and ROI calculations.
-   Explicit assumption inputs and sensitivity analysis.

`ROI (%) = ((Estimated Benefit - Investment Cost) / Investment Cost) × 100`

All scenario outputs must be labelled estimates, not realised savings.

### FR-11 Explainability

-   Feature importance.
-   SHAP global and local explanations where compatible.
-   Explain model version and feature values.
-   Avoid causal claims from SHAP or feature importance.

### FR-12 Streamlit dashboard

Required pages: 1. Executive Overview. 2. Data Explorer & Quality. 3.
Engine Health & RUL. 4. Anomaly Analysis. 5. Maintenance Forecasting. 6.
Inventory Intelligence. 7. Financial Scenarios. 8. Model Performance &
Explainability. 9. Data Sources & Model Registry.

Each page must show source/data type (NASA simulated, BTS public
operational, or project-generated synthetic), units, filters, data
freshness and empty/error states. Avoid hardcoded demo KPIs.

### FR-13 Reporting and exports

-   CSV exports for tables.
-   PNG chart exports.
-   HTML report where practical.
-   Optional PDF executive report.
-   Include generated timestamp, filters, source lineage and
    synthetic-data warning where relevant.

### FR-14 Experiment and model registry

Track model name/version, dataset fingerprint, feature set, training
date, parameters, metrics, random seed and artifact location. MLflow is
optional; a simple metadata registry is acceptable for the first
release.

## 7. Non-functional requirements

  -----------------------------------------------------------------------
  ID                                  Requirement
  ----------------------------------- -----------------------------------
  NFR-01                              Python 3.13.x compatibility; exact
                                      dependencies pinned or constrained

  NFR-02                              Local-first execution without paid
                                      services

  NFR-03                              Modular, typed, testable code

  NFR-04                              No secrets in source control

  NFR-05                              Raw data immutable

  NFR-06                              Reproducible random seeds and
                                      transformations

  NFR-07                              Clear logs and actionable
                                      exceptions

  NFR-08                              Cache expensive dashboard
                                      operations safely

  NFR-09                              No misleading charts or unsupported
                                      business claims

  NFR-10                              Accessible labels, units and
                                      colour-independent meaning
  -----------------------------------------------------------------------

## 8. Proposed architecture

``` text
Public sources / local files / synthetic generator
                       |
                       v
                Ingestion adapters
                       |
                       v
             Schema + quality validation
                       |
                       v
              Raw / processed storage
                 |             |
                 v             v
              Parquet        SQLite
                 |             |
                 +------+------+
                        |
                        v
              Analytics / feature pipelines
                        |
             +----------+-----------+
             |          |           |
             v          v           v
            EDA         ML       Business rules
             |          |           |
             +----------+-----------+
                        |
                        v
               Application services
                        |
                        v
                 Streamlit UI
                        |
                        v
               Reports / CSV exports
```

Streamlit is the presentation layer only. Business rules, model training
and data processing must live in reusable Python modules.

## 9. Repository structure

``` text
aviation-mro-predictive-intelligence/
├── config/
│   ├── settings.yaml
│   ├── data_sources.yaml
│   ├── model_config.yaml
│   └── business_rules.yaml
├── data/
│   ├── raw/                 # gitignored
│   ├── processed/           # gitignored by default
│   ├── synthetic/           # generated; large files gitignored
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
│   ├── config/
│   ├── ingestion/
│   ├── validation/
│   ├── preprocessing/
│   ├── analytics/
│   ├── features/
│   ├── models/rul/
│   ├── models/anomaly/
│   ├── models/forecasting/
│   ├── business/
│   ├── persistence/
│   └── services/
├── dashboard/
│   ├── app.py
│   ├── pages/
│   ├── components/
│   └── utils/
├── models/                  # artifacts gitignored unless tiny/intentional
├── reports/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── data_quality/
│   └── model_validation/
├── .github/workflows/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

## 10. Database and logical entities

SQLite is for metadata and synthetic business entities. Parquet is for
larger analytical tables.

Tables: - `data_sources` - `ingestion_runs` - `aircraft` - `engines` -
`maintenance_orders` - `technicians` - `inventory_parts` -
`inventory_transactions` - `downtime_events` - `engine_predictions` -
`anomaly_results` - `model_registry` - `experiments`

Use primary/foreign keys, indexes, schema migrations or versioned
initialisation, UTC timestamps internally, and explicit units. Do not
store every raw sensor row in SQLite if Parquet is more suitable.

## 11. KPI definitions

Each KPI must have a central definition, source, unit, time window, null
policy and test.

  -----------------------------------------------------------------------
  KPI                                 Definition
  ----------------------------------- -----------------------------------
  Total engines                       Distinct engine IDs in selected
                                      source/scope

  High-risk engines                   Count above configured project risk
                                      threshold

  Mean predicted RUL                  Mean of valid model predictions in
                                      cycles

  Maintenance expenditure             Sum of selected work-order costs

  Completion rate                     Completed work orders / eligible
                                      work orders

  Downtime hours                      Sum of valid downtime durations

  Inventory value                     Sum of stock quantity × unit cost

  Stockout risk                       Parts whose projected stock
                                      breaches configured requirement

  Technician utilisation              Assigned hours / available hours

  Inventory turnover                  Consumption value / average
                                      inventory value, with period
                                      defined
  -----------------------------------------------------------------------

Risk bands and business thresholds must be configurable and labelled as
illustrative project settings, not regulatory limits.

## 12. Testing and quality gates

-   Unit tests for calculations, transformations and feature
    engineering.
-   Integration tests for ingestion-to-storage and train-to-predict.
-   Schema and data-quality tests.
-   Reproducibility tests.
-   Leakage tests for engine/time splits.
-   Model metric calculation tests.
-   Streamlit smoke tests and key interaction tests.
-   Ruff lint/format checks.
-   Mypy configured-scope checks.
-   Dependency and licence review.

Release acceptance requires all critical tests passing, no unexplained
data loss, documented warnings, reproducible outputs and successful
clean-environment startup.

## 13. Delivery phases

  -----------------------------------------------------------------------
  Phase                   Deliverable             Exit criteria
  ----------------------- ----------------------- -----------------------
  0                       Repository and Python   Clean install, app
                          environment             skeleton, tests and
                                                  config

  1                       Data source registry    NASA parser, synthetic
                          and ingestion           generator, source
                                                  metadata and validation

  2                       EDA and statistics      Reproducible notebooks,
                                                  report and findings

  3                       RUL and anomaly models  Baseline comparison,
                                                  valid holdout, metrics
                                                  and explanations

  4                       Forecasting, inventory  Tested business
                          and finance             calculations and
                                                  backtests

  5                       Streamlit dashboard     All pages functional,
                                                  no hardcoded KPIs

  6                       Release and portfolio   Documentation, demo,
                                                  screenshots, tests and
                                                  GitHub readiness
  -----------------------------------------------------------------------

## 14. Definition of done

-   [ ] Public input sources documented and retrieval tested.
-   [ ] Data licensing/provenance reviewed and recorded.
-   [ ] NASA C-MAPSS parser tested against actual archive structure.
-   [ ] Synthetic MRO data generator reproducible and labelled.
-   [ ] Raw data preserved.
-   [ ] Data validation and EDA completed.
-   [ ] Statistical assumptions and findings documented.
-   [ ] RUL baseline and candidate models evaluated without leakage.
-   [ ] Anomaly results correctly described.
-   [ ] Forecasts backtested chronologically.
-   [ ] Inventory and financial formulas tested.
-   [ ] SHAP explanations available where supported.
-   [ ] All dashboard pages functional.
-   [ ] Exports contain provenance and timestamps.
-   [ ] Unit/integration tests pass.
-   [ ] README, data dictionary, source register and model card
    complete.
-   [ ] No confidential or improperly redistributed data committed.
-   [ ] No unsupported claims of actual airline savings or safety
    impact.

## 15. References

1.  NASA PCoE Prognostics Data Repository:
    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2.  NASA C-MAPSS benchmark archive:
    https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
3.  NASA C-MAPSS catalog record:
    https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data
4.  BTS TranStats On-Time field selection/download:
    https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ
5.  BTS airline downloads: https://www.bts.gov/airline-data-downloads
6.  OurAirports open data and terms: https://ourairports.com/data/

**Document control:** This SRS is the baseline for implementation. Any
change to scope, data source, modelling method or acceptance criteria
must be recorded in the project change log before implementation.
