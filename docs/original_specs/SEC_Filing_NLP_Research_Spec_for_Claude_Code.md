# SEC Filing NLP and Market-Impact Research

## Implementation Specification for Claude Code --- Google Colab Pro+

**Version:** 1.0\
**Research window:** 2000-01-01 through 2026-10-31\
**Minimum universe:** 500 US-listed stocks, subject to documented
historical data availability\
**Primary sources:** SEC filings (Forms 10-K, 10-Q, and 8-K) and
historical market-price data\
**Explicit exclusion:** Do not use the
`Brianferrell787/financial-news-multisource` dataset or any other
general financial-news corpus. This project studies SEC disclosures
only.

------------------------------------------------------------------------

## 0. Master instruction to Claude Code

Act as a senior quantitative researcher, financial NLP engineer, ML
engineer, and reproducible-research software developer. Build a
complete, well-documented Google Colab Jupyter notebook (`.ipynb`) that
can be run end to end on Google Colab Pro+.

The notebook must audit the available data before modeling; acquire
missing eligible SEC filings from official SEC sources where possible;
align filing information with market data without look-ahead bias;
benchmark at least five pretrained NLP checkpoints; tune a carefully
limited number of hyperparameters; compare before/after-tuning
performance; and produce interpretable visualizations, tables, saved
artifacts, and a final research report.

### Non-negotiable requirements

1.  **Use SEC Forms 10-K, 10-Q, and 8-K only for the primary textual
    analysis.**
2.  Do not load, download, merge, or use the previously considered
    Hugging Face financial-news dataset. No news corpus is in scope.
3.  Research period: **2000-01-01 to 2026-10-31**. Since October 2026 is
    not complete until 2026-10-31, explicitly report that the last month
    is partial if the run occurs earlier.
4.  Target at least **500 distinct US-listed stocks**. Do not silently
    claim 500 if fewer are successfully mapped. Report target, eligible
    universe, successful universe, and exclusions.
5.  Use official SEC sources for filings where possible, follow SEC
    fair-access rules and identify the declared User-Agent in
    configuration. Respect rate limits, retries, caching, and SEC access
    policies.
6.  Build a metadata-first pipeline. Do not download and retain every
    filing's full HTML at the outset. First audit metadata, coverage,
    and ticker/CIK mappings; then fetch and process the minimum
    necessary text.
7.  Keep raw data, derived data, splits, model checkpoints, metrics, and
    figures in separate, versioned directories. Use checkpoints and
    resumable stages.
8.  All model evaluation must use chronological, leakage-resistant
    splits. No random row-level split across time.
9.  Distinguish association, prediction, and causal evidence. Never
    claim that a filing caused a price move solely because a model
    predicts it.
10. Never fabricate missing dates, documents, model results, metrics, or
    successful downloads. Record errors and missingness explicitly.
11. Every major cell must have an explanatory Markdown cell: objective,
    method, assumptions, expected output, and interpretation.
12. The notebook must run from top to bottom with configuration switches
    for a quick smoke test, pilot, and full experiment.

------------------------------------------------------------------------

## 1. Research objectives and hypotheses

### Main research questions

**RQ1 --- Price association:** How are the content and tone of 10-K,
10-Q, and 8-K disclosures associated with subsequent stock returns,
abnormal returns, volatility, and drawdowns?

**RQ2 --- Informative content:** Which filing sections, facts, event
categories, risks, outlook statements, and numerical disclosures are
most associated with later market movements?

**RQ3 --- Direction:** Can SEC filing text help classify subsequent
movement as up, down, or sideways beyond simple historical-price and
market baselines?

**RQ4 --- Incremental information:** Does text add out-of-sample value
beyond filing metadata, sector, market returns, and price/volatility
features?

**RQ5 --- Heterogeneity:** Do relationships vary by form type, sector,
market regime, company size, and historical period?

**RQ6 --- Robustness:** Do results survive alternate event windows,
class thresholds, sector splits, time splits, and controls for
overlapping disclosures?

### Pre-register hypotheses before inspecting final test results

-   H1: SEC disclosure text and selected filing features provide
    incremental out-of-sample information beyond a simple price-only
    baseline.
-   H2: 8-K event disclosures and selected 10-Q/10-K sections have
    different predictive value.
-   H3: Specific event/risk categories and changes in disclosure
    language are more informative than document-wide sentiment alone.
-   H4: Results differ by sector and market regime.
-   H5: Any apparent gains shrink when using strict publication
    timestamps, a realistic trading lag, costs, and an untouched final
    test period.

The notebook must show results even when a hypothesis is not supported.
Do not tune the research design after repeatedly inspecting the final
holdout.

------------------------------------------------------------------------

## 2. SWOT analysis of the project

### Strengths

-   SEC filings are primary corporate disclosures with identifiable
    forms, companies, filing dates, and often acceptance timestamps.
-   10-K, 10-Q, and 8-K have complementary roles: annual context,
    quarterly updates, and current material events.
-   A 500-stock cross-section enables sector comparisons and tests
    beyond a handful of familiar companies.
-   Financial-domain pretrained models can identify tone and semantic
    features that simple keyword counts miss.
-   A disciplined time-based evaluation can produce more credible
    evidence than a random split.
-   Combining document features with market outcomes supports both
    interpretability and predictive testing.

### Weaknesses

-   The existing Hugging Face SEC dataset is not assumed to contain
    every required form or every filing from 2000--2026. The notebook
    must verify this.
-   Full filings are long, repetitive, and structurally noisy. A single
    standard BERT input cannot represent an entire 10-K.
-   Stock prices react to many simultaneous influences, not just the
    filing.
-   A 500-stock target creates substantial document processing and
    market-data volume.
-   Historical ticker/CIK mappings, delistings, corporate actions, and
    timestamp precision complicate joins.
-   Sentiment labels can be misleading: negative-sounding risk language
    may be boilerplate, while a bad result relative to expectations can
    be expressed neutrally.

### Opportunities

-   Build a reproducible, auditable SEC-to-market event dataset.
-   Compare sentiment, forward-looking statements, event extraction,
    section-level modeling, and embedding-based approaches.
-   Identify which filing forms and sections add incremental predictive
    information.
-   Test whether filing text can identify changes in uncertainty, risk,
    guidance, liquidity, leverage, litigation, and capital allocation.
-   Release a methods paper, documented code, aggregate tables, and a
    non-sensitive reproducibility package where licensing permits.

### Threats

-   Look-ahead bias from using filing acceptance dates, fiscal period
    dates, or downloaded timestamps incorrectly.
-   Survivorship bias if the universe contains only current S&P 500 or
    currently listed companies.
-   Label leakage through post-event returns, future price data, filing
    metadata, or labels encoded in source datasets.
-   Data availability or SEC rate-limit failures; missing older filings
    and incomplete text extraction.
-   Overfitting from testing many model configurations, event windows,
    labels, and feature sets.
-   False discoveries from thousands of event/category comparisons.
-   A model may learn company identity, sector, boilerplate, or time
    period instead of meaningful disclosure information.

### Mitigations required

-   Prefer point-in-time company universes and include delisted firms
    where reliable data is available.
-   Use the SEC acceptance/publication timestamp where available; if
    unavailable, use a conservative fallback and mark timestamp quality.
-   Use chronological validation and a locked final test.
-   Deduplicate filings and syndicated/amended copies.
-   Report coverage, failed retrievals, and data quality by year, form,
    and company.
-   Apply multiple-testing corrections for large families of statistical
    tests.
-   Treat weak or null findings as valid results; never optimize the
    notebook for a desired hypothesis.

------------------------------------------------------------------------

## 3. Data sources and data-completeness audit

### 3.1 Existing Hugging Face SEC dataset

Start by inspecting the dataset card and available configurations for:

`JanosAudran/financial-reports-sec`

The known configurations discussed for exploration include `large_lite`,
`small_full`, and `small_lite`. Inspect the live
repository/configuration metadata rather than assuming that any
configuration name or schema is unchanged.

Requirements: - Print dataset card/repository URL, configuration, split
names, features, row counts, and sample records. - Identify whether each
record is a sentence, paragraph, document, or filing. - Inspect fields
such as CIK, ticker(s), form, filing date, report date,
accession/document ID, section, text, and any return/label fields. -
Check form values. Do not assume that the corpus includes 10-Q or 8-K. -
Check earliest/latest filing dates, annual counts, counts by form,
unique companies, duplicate IDs, null values, and ticker mapping
quality. - If the dataset contains only 10-K records, clearly report
that 10-Q and 8-K are absent from this dataset and retrieve them
separately. - Never treat a dataset's stated date range as proof that
every company-year or every filing is present. - Do not use precomputed
return labels from the dataset as ground truth without understanding
their construction. Recompute outcomes from the chosen price source.

### 3.2 Official SEC sources for missing records

Use the SEC's official EDGAR resources as the preferred source for
missing filing metadata and documents. Relevant official
endpoints/documentation include:

-   SEC company submissions API:
    `https://data.sec.gov/submissions/CIK##########.json`
-   SEC company facts API:
    `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
-   SEC filing archive: `https://www.sec.gov/Archives/edgar/data/`
-   SEC filing search: `https://www.sec.gov/edgar/search/`
-   SEC developer guidance:
    `https://www.sec.gov/edgar/sec-api-documentation`

Implementation requirements: - Configure a truthful identifying
User-Agent with contact information supplied by the user in a notebook
configuration cell. Do not invent contact details. - Follow current SEC
fair-access guidance and rate limits. Implement a conservative
configurable request rate, exponential backoff for 429/5xx errors, cache
responses, and avoid redundant requests. - Use submissions metadata to
identify candidate 10-K, 10-Q, and 8-K accessions and filing dates. -
Retrieve filing HTML only when needed. Cache raw source responses and
their retrieval metadata. - Support amended filings (e.g. 10-K/A,
10-Q/A, 8-K/A) as a separately flagged group; do not silently merge them
with original filings. - Record source URL, accession number, CIK, form,
filing date, acceptance timestamp when available, retrieval timestamp,
HTTP status, parser status, and checksum. - If SEC access is
unavailable, pause gracefully and report the missing stage. Do not
bypass SEC restrictions.

### 3.3 Stock universe: minimum 500

Do not choose 500 tickers solely from today's S&P 500 list. That causes
survivorship bias and can exclude failed, acquired, or delisted
companies.

Preferred universe procedure: 1. Build a broad historical US
common-equity universe from a properly licensed point-in-time source
where available. 2. If point-in-time membership is unavailable,
transparently describe the approximation and its survivorship
limitations. 3. Map ticker symbols to CIKs and effective date ranges.
Preserve ticker changes and corporate actions. 4. Include only companies
with a defensible identifier mapping and usable market prices. 5. Aim
for at least 500 stocks across sectors, including historical/delisted
firms where feasible. 6. Report counts before and after each filter, and
export the exact universe list with rationale for exclusions.

Do not scrape or use a current list and then claim the universe is
survivorship-free.

### 3.4 Market data

SEC filings alone do not provide the future return series needed to
evaluate price impact. Acquire daily adjusted prices and, where
available, OHLCV and corporate-action data from a source whose
historical coverage, terms, ticker history, and delisted-stock support
are documented.

-   Use adjusted prices for return calculations where appropriate.
-   Preserve unadjusted prices and corporate-action fields if available.
-   Include a broad market benchmark and preferably sector benchmarks.
-   Record source, license/terms, date range, missingness, and whether
    delisted securities are represented.
-   Never silently mix price vendors without a provenance flag.
-   If a free source cannot provide reliable coverage for 500 stocks
    across 2000--2026, report the limitation and use a pilot or an
    explicitly licensed source rather than fabricating coverage.
-   Prevent adjusted-price data revisions or vendor-specific limitations
    from being misrepresented as point-in-time information.

### 3.5 Coverage audit outputs

Create these artifacts before model training:

-   `reports/data_coverage_summary.md`
-   `reports/coverage_by_year_form.csv`
-   `reports/coverage_by_company.csv`
-   `reports/missing_company_year_form.csv`
-   `reports/missing_market_price_periods.csv`
-   `reports/ticker_cik_mapping_issues.csv`
-   `reports/duplicate_and_amended_filings.csv`
-   `reports/data_quality_dashboard.html` or notebook charts
-   `data/manifests/filing_manifest.parquet`
-   `data/manifests/market_data_manifest.parquet`

The coverage report must explicitly answer: - Are 10-K, 10-Q, and 8-K
present? - For which years does each form have records? - How many
distinct CIKs and tickers are represented? - Which company-year-form
combinations are missing? - What portion of filings has an acceptance
timestamp? - How many filings failed to download or parse? - Which
years, forms, or companies have unusually low coverage? - Is the target
of 500 successfully mapped stocks met? - Does the actual usable dataset
cover January 2000--October 2026, or are there gaps?

A zero count is not the same as a known missing document: label audit
results carefully as observed coverage, suspected gaps, and verified
missing records.

------------------------------------------------------------------------

## 4. Filing parsing and text preparation

### 4.1 Filing forms

-   **10-K:** annual report.
-   **10-Q:** quarterly report.
-   **8-K:** current report, often tied to a material event and
    potentially containing exhibits.
-   Amended variants must be separately tagged.
-   Use the actual form type and accession metadata, not text heuristics
    alone.

### 4.2 Parse and retain document structure

Extract where possible: - Item/section headings - Section text - Tables
and numerical facts - Exhibit references and exhibit text where legally
accessible and relevant - Filing date and acceptance timestamp - Fiscal
period end - CIK, ticker mapping, accession number, form, amendment
flag - Text extraction quality and length

Prioritize relevant sections such as: - 10-K: Business, Risk Factors,
MD&A, financial statements, legal proceedings, market risk, and relevant
outlook discussion. - 10-Q: MD&A, risk-factor changes, financial
statements, liquidity, and material changes. - 8-K: item-level event
disclosures and relevant exhibits, including earnings releases when
filed as exhibits.

Do not treat every section as equally informative. Retain section labels
so the model can be evaluated by section.

### 4.3 Cleaning

-   Remove HTML navigation, repeated headers/footers, and extraction
    artifacts without deleting substantive content.
-   Preserve negation, financial numbers, percentages, units, dates, and
    legal qualifiers.
-   Deduplicate exact duplicates using accession number plus content
    checksum; detect near-duplicates separately.
-   Do not remove boilerplate before the baseline audit. Instead,
    measure boilerplate and test a documented boilerplate-reduction
    variant.
-   Preserve raw text and store cleaned text separately.
-   Track parser errors, empty text, and document length.

### 4.4 Chunking long documents

Standard BERT-style models cannot ingest entire long filings at once.
Use section-aware chunking: - Split by filing section first, then into
token-length chunks with a small overlap. - Respect each checkpoint's
tokenizer and maximum sequence length. - Keep a mapping from chunk to
filing, company, section, and character/token offsets. - Aggregate chunk
outputs to section-level and filing-level features using documented
rules (mean, max, distribution, and relevant event scores). - Do not
label every chunk with a filing-level future return as if all chunks
independently represent a distinct market event. - Use group-aware
evaluation so chunks from one filing cannot cross
training/validation/test boundaries. - Keep document-level aggregation
and evaluation at the event/filing level.

------------------------------------------------------------------------

## 5. Pretrained model benchmark: minimum five checkpoints

Inspect each model card, license, architecture, label mapping, maximum
sequence length, and intended use before inference. The notebook must
verify the exact model ID is accessible and compatible instead of
assuming it.

Recommended starting candidates:

1.  `ProsusAI/finbert` --- financial sentiment classification.
2.  `yiyanghkust/finbert-tone` --- financial communication tone.
3.  `yiyanghkust/finbert-pretrain` --- financial-domain pretrained
    encoder for downstream fine-tuning or feature extraction.
4.  `yiyanghkust/finbert-fls` --- forward-looking statement
    classification.
5.  `allenai/longformer-base-4096` --- longer-context baseline requiring
    a task-specific head.
6.  A verified financial event-classification checkpoint, such as
    `ritessshhh/FinDeBERTa`, **only after confirming its model card,
    label schema, license, and actual task compatibility**.
7.  A verified financial sentence-embedding model as a semantic
    representation baseline; choose a concrete checkpoint after checking
    its model card and license.
8.  Optional specialized ESG/risk classifier only if a suitable,
    accessible checkpoint and label schema are verified.

Hugging Face links: - `https://huggingface.co/ProsusAI/finbert` -
`https://huggingface.co/yiyanghkust/finbert-tone` -
`https://huggingface.co/yiyanghkust/finbert-pretrain` -
`https://huggingface.co/yiyanghkust/finbert-fls` -
`https://huggingface.co/allenai/longformer-base-4096` -
`https://huggingface.co/ritessshhh/FinDeBERTa`

Important: these candidates do **not** all perform the same task. Do not
compare their native label scores as though they were competing
classifiers on one common label set. Build a common supervised task and
fine-tune compatible encoder checkpoints on the same training/validation
labels. Evaluate specialized models on their native tasks separately and
label the comparisons accurately.

### Benchmark tracks

**Track A --- Frozen pretrained features / zero-shot baselines** - Run
inference without gradient updates where the model supports the task. -
Extract sentiment/event probabilities or pooled embeddings. - Establish
inexpensive baselines.

**Track B --- Fine-tuned common task** - Use compatible pretrained
encoder checkpoints with a classification head. - Fine-tune for a common
outcome such as three-class future return direction. - Use the exact
same event-level train/validation/test definitions. - Report which
models were excluded and why.

**Track C --- Specialized information extraction** - Forward-looking
statement classification, event classification, and optional ESG/risk
labels. - Report task-specific quality where a credible labeled
evaluation set exists. - If no human-verified labels exist, do not
present heuristic pseudo-label evaluation as ground truth.

### Label quality

Price-based labels can be constructed from returns, but they are not
ground-truth labels for semantic correctness. For event classification
and factual extraction, create a manually reviewed stratified sample
with an annotation guide, two annotators if possible, agreement
statistics, and adjudication. If human annotation is not feasible,
report this as a limitation.

------------------------------------------------------------------------

## 6. Outcome definitions and event alignment

### 6.1 Filing event timestamp

Use the SEC acceptance timestamp or another defensible
public-availability timestamp when available. The fiscal period end is
not the release timestamp. The downloaded timestamp is not the release
timestamp.

Rules: - If the filing became public after market close, the earliest
tradable response may begin on the next trading session. - If only a
date is known and time is unknown, apply a conservative
next-trading-session rule. - If timestamp quality is poor, flag the
event and run a sensitivity analysis. - Use exchange calendars to
determine sessions. - A filing can contain information already disclosed
in another filing or press release. Without external news, you cannot
know all prior public disclosures; flag this limitation and test
overlapping filings. - Deduplicate event records and do not treat
multiple chunks from one filing as independent events.

### 6.2 Outcomes

Compute multiple forward windows from the first eligible tradable price
after the public filing time, for example: - 1 trading day - 2 trading
days - 5 trading days - 20 trading days - 60 trading days, where
coverage supports it

For each window, compute: - Simple and log forward returns -
Benchmark-adjusted returns - Market-model abnormal returns where
estimation data is adequate - Realized volatility - Absolute return -
Drawdown or adverse excursion where useful

Avoid overlapping windows where they create dependence; use clustered
standard errors, event-level bootstrap, or non-overlapping sensitivity
samples as appropriate.

### 6.3 Direction labels

Define up/down/sideways thresholds using a documented,
training-period-only rule. Do not choose thresholds based on final-test
performance.

Prefer volatility-adjusted labels or a threshold that exceeds a
reasonable noise/cost band. Example concept: - Down: forward abnormal
return below a negative threshold. - Sideways: within the threshold
band. - Up: above the positive threshold.

The actual threshold must be configurable, justified, and fit using
training data only. Run threshold sensitivity analyses. Report class
distribution for each split and each horizon.

### 6.4 Baselines

At minimum compare: - Majority-class classifier - Simple
historical-price/momentum baseline - Metadata-only model (form type,
section, sector, time features available at event time) - Frozen
financial sentiment features + classical classifier - Fine-tuned NLP
model - Combined text + valid point-in-time market/metadata model

The key question is whether SEC text adds incremental out-of-sample
value beyond simpler baselines.

------------------------------------------------------------------------

## 7. Leakage-safe data splitting

Use a chronological, point-in-time design. Never randomly split chunks
or rows.

### Recommended split structure

Use a configurable expanding-window or rolling-window design. A
reasonable initial template is: - Training: 2000--2017 -
Validation/model selection: 2018--2020 - Development test / robustness:
2021--2022 - Locked final test: 2023--2026-10

These are proposed calendar blocks, not immutable truth. Inspect
coverage first and adjust only before final evaluation. The final test
must remain locked until model selection and thresholds are finalized.
If prior experimentation has already inspected this period, label it a
development holdout rather than an untouched final test.

Also report: - A walk-forward evaluation with multiple chronological
folds. - A company-held-out or sector-held-out stress test where sample
size permits. - A time split and a company split answer different
questions; do not confuse them.

### Leakage controls

-   All filing chunks and derived rows for one accession stay in the
    same split.
-   No future filing, future return, future-adjusted feature, or
    post-event statistic may enter predictors.
-   Fit scalers, imputers, vocabularies, feature selection, calibration,
    class weights, and thresholds using training data only.
-   Use validation only for model selection and hyperparameter tuning.
-   Use the locked test once for final reporting.
-   If a feature is computed over time, ensure it is available as of the
    event timestamp.
-   Avoid forward-filled company attributes that were not yet known at
    the historical date.
-   Keep duplicate and amended filing relationships visible; prevent
    near-duplicate content from leaking across splits.
-   Apply embargo/purging around overlapping event windows if they
    create shared outcome intervals.
-   Ensure target labels are not encoded in file paths, row ordering,
    filenames, source splits, or text added after the event.
-   Use robust pipeline tests to assert these rules.

Create a leakage audit table listing each feature, its source,
availability timestamp, and whether it can contain future information.

------------------------------------------------------------------------

## 8. Hyperparameter tuning under a Colab Pro+ compute budget

The notebook must be cost-conscious because other projects also need the
user's Colab resources.

### Three execution profiles

1.  **Smoke test:** 5--10 companies, a small number of filings, one
    epoch or inference-only, verify every stage and artifact.
2.  **Pilot:** 30--50 sector-balanced companies, short historical slice,
    limited candidate models and a tiny tuning search.
3.  **Full study:** target 500+ stocks, complete available period,
    selected models only.

Do not start the full run until smoke and pilot stages pass.

### Resource management

-   At runtime, detect GPU type, GPU memory, RAM, disk space, and
    available mixed-precision support.
-   Print the runtime configuration and estimated workload before
    downloading model weights.
-   Use mixed precision when stable and supported; use dynamic padding,
    tokenization caching, batched inference, gradient accumulation, and
    gradient checkpointing only when useful.
-   Use a smaller batch size automatically after out-of-memory errors,
    then retry a limited number of times.
-   Set deterministic seeds and record library versions, GPU type, and
    configuration.
-   Cache raw filings, cleaned text, tokenized chunks, model outputs,
    and metrics so reruns skip completed work.
-   Save checkpoints after each company batch, form, model, epoch, and
    tuning trial.
-   Store resumable progress manifests. A restarted Colab session must
    not restart completed work.
-   Make Drive persistence optional and explicit. Avoid writing enormous
    intermediate tensors to Drive.
-   Do not store full copies of the same corpus for each model.
-   Provide `MAX_COMPANIES`, `MAX_FILINGS_PER_COMPANY`, `MAX_CHUNKS`,
    `MAX_TRIALS`, `EPOCHS`, and `RUN_PROFILE` configuration settings.
-   Add dry-run estimates for token count, storage, runtime, and
    model-download size before the full experiment.
-   Do not promise a specific runtime or GPU availability.
-   No paid external APIs, proprietary hosted inference, or unrequested
    services.
-   Do not use an LLM API to label millions of examples.

### Hyperparameter tuning plan

Tune only a small, predefined grid on validation data: - Learning rate:
e.g. `1e-5`, `2e-5`, `3e-5` - Batch size: e.g. `8`, `16`, limited by GPU
memory - Epochs: `1`, `2`, `3` with early stopping - Weight decay:
e.g. `0.0`, `0.01` - Warmup ratio: e.g. `0.0`, `0.06` - Max sequence
length appropriate to each checkpoint - Chunk aggregation method as a
separate, small ablation

Start with a small search (for example, 6--12 trials per selected model,
not every possible combination). Use a fixed validation metric chosen in
advance, such as macro-F1 or validation PR-AUC where suitable. Do not
select models using the final test.

Use class weighting or focal loss only if justified by training-set
imbalance and validated against an ordinary cross-entropy baseline. Do
not automatically assume focal loss is better.

### Training-loss requirements

For every fine-tuned model, save: - Training loss by epoch/step -
Validation loss by epoch - Primary validation metric by epoch - Learning
rate by step where available - Best checkpoint and early-stopping
epoch - Training duration and peak memory if measurable

Visualize training and validation loss curves. Explain: -
**Overfitting:** training loss improves while validation loss worsens or
validation performance degrades. - **Underfitting:** both training and
validation performance remain poor, often with high losses. - **Healthy
learning:** both improve and validation performance stabilizes. -
**Unstable training:** spikes, divergence, or large train/validation
gaps.

Do not claim a global minimum was found. Neural-network optimization is
non-convex; the notebook finds a practical validation-selected
checkpoint, not a proof of a global optimum.

------------------------------------------------------------------------

## 9. Metrics and how to interpret them

Report metrics for every relevant model, before and after tuning, with
confidence intervals where practical.

### Classification metrics

-   **Precision:** among predicted positive/class instances, how many
    are correct?
-   **Recall:** among true positive/class instances, how many were
    found?
-   **F1:** harmonic mean of precision and recall; report per-class and
    macro-F1.
-   **Accuracy:** overall fraction correctly classified; can mislead
    with imbalanced classes.
-   **Balanced accuracy:** average class recall.
-   **ROC-AUC:** ranking discrimination; use one-vs-rest macro/weighted
    variants for multiclass classification.
-   **PR-AUC / Average Precision:** especially informative for rare
    classes; report per-class and macro/weighted summaries.
-   **Confusion matrix:** reveals which classes are confused.
-   **Matthews correlation coefficient (MCC):** useful balanced summary
    across classes.
-   **Cohen's kappa:** agreement beyond chance, with appropriate
    interpretation.
-   **Log loss / cross-entropy:** quality of predicted probabilities;
    lower is better.
-   **Brier score:** probability calibration error; lower is better.
-   **Expected calibration error (ECE):** calibration diagnostic, with
    binning caveats.
-   **Top-class confidence and abstention coverage:** optional
    reliability analysis.

For multiclass metrics, specify averaging (`macro`, `weighted`, `micro`,
or one-vs-rest) explicitly. Do not report one ambiguous "F1" or "AUC."

### Regression metrics

-   **R²:** proportion of target variance explained relative to a
    mean-prediction baseline; can be negative out of sample.
-   **MAE:** average absolute prediction error.
-   **RMSE:** penalizes larger errors more strongly.
-   **Spearman rank correlation:** evaluates rank association, useful
    for cross-sectional ranking.
-   **Pearson correlation:** linear association, not causation.
-   **Directional accuracy:** fraction of predicted signs that match
    realized signs, with the label/threshold defined.
-   **Information coefficient (IC):** cross-sectional rank correlation
    between signals and future returns, calculated per date and
    summarized over time.

R² is not an appropriate primary score for a three-class classification
task. Use it for a genuine continuous-return regression task only.

### Market/event-study metrics

-   Mean and median abnormal returns by event/category
-   Confidence intervals and bootstrap intervals
-   Event counts and coverage by category
-   Standard errors clustered by company and/or event date where
    suitable
-   Multiple-testing-adjusted p-values for large hypothesis families
-   Long-short spread, turnover, and cost-adjusted returns only if a
    clearly specified portfolio simulation is implemented
-   Sharpe ratio and maximum drawdown for a backtested signal, with
    caveats and costs

### Statistical uncertainty

-   Use time-aware or company-clustered resampling where appropriate.
-   Do not use ordinary random bootstrap as if events were independent
    when they are clustered or overlapping.
-   Report sample size and missingness beside every important metric.
-   Provide confidence intervals for key differences between baseline
    and tuned models.
-   Use a correction such as Benjamini--Hochberg false-discovery-rate
    control for large families of related tests.

------------------------------------------------------------------------

## 10. Visualizations --- required, explanatory, and not decorative

Use Matplotlib/Seaborn or Plotly as appropriate, but keep figures
readable and label all axes, units, date windows, and sample sizes. Use
a consistent visual style without misleading scales.

### Data quality

1.  Filing counts by year and form (10-K, 10-Q, 8-K).
2.  Unique companies by year and form.
3.  Heatmap of company-year-form coverage.
4.  Missingness heatmap and missing-record reasons.
5.  Filing timestamp-quality distribution.
6.  Distribution of document and chunk lengths.
7.  Failed download/parser counts over time.
8.  Sector and company coverage distribution.

### Event and market analysis

9.  Return distributions by form and horizon.
10. Event-study cumulative abnormal-return curves with confidence bands.
11. Abnormal returns around filing publication time, using an explicitly
    defined event clock.
12. Up/down/sideways class balance by period and form.
13. Filing section and event-category frequencies.
14. Sector-by-event-category return heatmaps.
15. Year/regime stability plots.
16. Results before and after controlling for market/sector benchmarks.
17. Similarity or redundancy plots for overlapping/amended filings, if
    implemented.

### Model performance

18. Training loss and validation loss curves.
19. Validation metric versus epoch.
20. Confusion matrices, normalized and raw.
21. Per-class precision, recall, and F1 plots.
22. ROC curves and PR curves where valid, including per-class or
    one-vs-rest curves.
23. Calibration/reliability diagrams and Brier score.
24. Baseline versus tuned metrics with confidence intervals.
25. Model ranking table with uncertainty and runtime.
26. Learning-rate/validation-score comparison across tuning trials.
27. Error analysis: false positives, false negatives, difficult
    examples, and representative filing excerpts with
    sensitive/irrelevant data minimized.
28. Performance by form, sector, year, document length, and timestamp
    quality.
29. Feature ablation plot: text only vs metadata only vs price baseline
    vs combined.
30. Predicted versus realized return plots for regression and residual
    diagnostics.

### Interpretation rules

-   Do not show ROC-AUC for a class if that class is absent from the
    test set; mark it unavailable.
-   Use PR curves for rare classes.
-   Always include class counts and the confusion matrix.
-   A visual pattern is not proof of causality.
-   Plot uncertainty, not just point estimates.
-   Do not show test-set metrics during tuning.
-   Ensure test-period charts are generated only after the selected
    model and analysis choices are frozen.

------------------------------------------------------------------------

## 11. Compare models before and after tuning

Create a single tidy results table, at minimum:

`model_id`, `checkpoint`, `task`, `form_scope`, `split`, `horizon`,
`tuning_status`, `best_hyperparameters`, `macro_f1`, `weighted_f1`,
`accuracy`, `balanced_accuracy`, `roc_auc_macro`, `pr_auc_macro`, `mcc`,
`log_loss`, `brier_score`, `mae`, `rmse`, `r2`, `ic`, `runtime_seconds`,
`peak_gpu_memory_mb`, `n_events`, `n_companies`.

Fields not applicable to a model/task should be null, not zero.

Deliver: - Untuned validation leaderboard. - Tuned validation
leaderboard. - Final locked-test leaderboard. - Before/after tuning
delta table. - Confidence intervals for key performance differences. -
Results by 10-K, 10-Q, and 8-K. - Results by prediction horizon. -
Results by sector and time period. - Model selection decision with
evidence and limitations. - A clear list of all models tried, excluded,
failed, or abandoned and why.

Do not claim tuning improved performance if the difference is within
uncertainty or only improves one metric while materially degrading
others. Use the validation metric preselected in the configuration to
choose the best checkpoint.

------------------------------------------------------------------------

## 12. Testing specific market-drift hypotheses

The project must evaluate whether the available SEC and market data
support the following hypotheses. It must not assume the effects exist.

### 12.1 Post-Earnings Announcement Drift (PEAD)

**Potentially testable, with caveats.** - Detect earnings-related 8-K
disclosures and related 10-Q/10-K filings. - Determine the first public
event timestamp and avoid double-counting the same earnings event across
an 8-K, 10-Q, and 10-K. - Define earnings surprise only if a reliable
point-in-time consensus/expectations dataset is available. SEC filings
alone generally do not provide analyst consensus expectations. - If no
consensus data exists, test a narrower disclosure/return continuation
hypothesis around earnings-related SEC events and explicitly do not call
it a full classic PEAD test. - Calculate post-event abnormal returns at
multiple horizons, with confidence intervals, controls, and robustness
checks.

### 12.2 Post-Dividend Announcement Drift

**Partially testable.** - Search 8-K item text and relevant exhibits for
dividend declarations and record/payment dates; cross-check relevant
10-Q/10-K disclosures. - Verify the actual public announcement timestamp
and avoid treating the ex-dividend date as the announcement date. -
Separate announcement effects from mechanical ex-dividend price
adjustment. - A full dividend-event study may require reliable
dividend-event and corporate-action data beyond filing text. Report
whether coverage is complete enough.

### 12.3 Analyst Revisions Drift

**Not directly testable from 10-K/10-Q/8-K alone.** - Analyst
estimate/recommendation revisions are external analyst actions, not a
complete standardized series in these filings. - SEC documents may
mention analyst expectations, but such mentions are not a substitute for
a point-in-time analyst revisions dataset. - Mark this hypothesis **out
of scope / insufficient data** unless a separate, properly licensed
analyst-estimates source is explicitly approved. Do not use news as a
proxy.

### 12.4 Short-Term Price Momentum

**Testable with market-price data, not an SEC-text hypothesis by
itself.** - Construct a transparent price-only momentum baseline using
only prices available before the prediction timestamp. - Evaluate
whether SEC text adds incremental value over this baseline. - Do not
claim the drift is caused by filings merely because the momentum signal
and text model both predict returns.

### 12.5 Insider Trading Drift

**Not adequately testable using only 10-K, 10-Q, and 8-K.** - Form 4 is
the primary SEC form for many insider transaction disclosures; it is
outside the primary three-form scope. - 10-K/10-Q/8-K may discuss
insider-related matters but do not provide a complete point-in-time
insider transaction dataset. - Mark as **not testable in the primary
scope**. An extension would need explicit authorization to ingest Form 4
data and define transaction/announcement timing.

### 12.6 Institutional Ownership Drift

**Not adequately testable using only 10-K, 10-Q, and 8-K.** -
Institutional holdings are typically analyzed using Form 13F and related
ownership filings, which are outside the primary scope and have
reporting delays. - Report this hypothesis as **not testable in the
primary scope**. A future extension would need a defined point-in-time
13F dataset and careful treatment of disclosure lags.

### 12.7 Spin-Off and Corporate Restructuring Drift

**Potentially testable, subject to event coverage.** - Detect spin-offs,
mergers, divestitures, restructuring, bankruptcy/reorganization, asset
sales, and major capital structure events in 8-K and relevant 10-K/10-Q
text. - Build a human-reviewed validation sample for event extraction. -
Ensure event dates correspond to first public disclosure and do not mix
announcement, completion, and ex-distribution dates. - Consider that
spin-offs may create a new ticker/security and require robust entity
mapping and corporate-action-aware prices. - Report event counts and
statistical power before interpreting results.

### Drift-study reporting standard

For every hypothesis, report: - Status: supported, unsupported,
inconclusive, or not testable in scope. - Exact operational
definition. - Required data and which fields are available. - Number of
unique events and companies. - Event dates and return windows. -
Benchmark and abnormal-return methodology. - Confidence intervals,
p-values, adjusted p-values, and robustness checks. - Data coverage gaps
and alternative explanations.

Do not label a result "evidence of drift" merely because one horizon has
an unadjusted p-value below 0.05. Require consistent definitions, a
credible event sample, correction for multiple testing, and
out-of-sample or confirmatory evidence.

------------------------------------------------------------------------

## 13. Feature attribution and economic interpretation

Use interpretable methods cautiously: - Feature ablation: remove text,
form, section, sentiment, event labels, and price baseline features
separately. - Permutation importance computed only on validation/test
features without refitting to the test. - SHAP or other attribution for
compatible models, with compute limits and clear explanation of what the
attribution means. - For transformer models, token attribution/attention
visualization may be used as diagnostics but must not be presented as
causal explanation. - Compare model findings with a simple, regularized
baseline such as logistic regression or gradient boosting on frozen
embeddings and structured features. - Group correlated features and
avoid over-interpreting unstable ranks. - Show example passages only
after verifying they are genuinely associated with the model prediction
and not accidentally post-event information.

The main interpretation target is **which document information is
predictive and robust**, not a claim that the model has discovered a
causal mechanism automatically.

------------------------------------------------------------------------

## 14. Suggested notebook architecture

Build one readable notebook with clearly numbered sections and reusable
functions. Do not hide the whole project inside one giant cell.

1.  Project overview, scope, hypotheses, and limitations.
2.  Runtime/configuration, GPU/RAM/disk diagnostics, seeds, and
    versions.
3.  SEC and market source documentation, terms, and rate-limit settings.
4.  Dataset card/configuration/schema inspection.
5.  Metadata-first SEC coverage audit.
6.  Universe creation and CIK/ticker mapping.
7.  Missing filings retrieval from SEC, with caching and resume.
8.  Filing parsing and quality checks.
9.  Market-price retrieval/loading and adjustment checks.
10. Event timestamp normalization and market-calendar alignment.
11. Return/abnormal-return target construction.
12. Deduplication and leakage audit.
13. Train/validation/test construction and split verification.
14. Exploratory analysis and required data visualizations.
15. Frozen pretrained model baselines.
16. Common-task label preparation and fine-tuning.
17. Small hyperparameter search and early stopping.
18. Loss curves, validation curves, and overfit/underfit diagnostics.
19. Final locked-test evaluation.
20. Event-study analyses for supported drift hypotheses.
21. Feature attribution and ablation studies.
22. Model leaderboard and final comparison.
23. Findings, limitations, and decision log.
24. Export artifacts and reproducibility manifest.

------------------------------------------------------------------------

## 15. Recommended project folder structure

``` text
sec-filing-nlp-study/
├── README.md
├── SEC_FILING_NLP_RESEARCH_SPEC.md
├── notebooks/
│   └── sec_filing_nlp_end_to_end.ipynb
├── config/
│   └── experiment_config.json
├── data/
│   ├── raw_sec/
│   ├── raw_market/
│   ├── processed_filings/
│   ├── chunks/
│   ├── labels/
│   └── manifests/
├── cache/
│   ├── http/
│   ├── tokenized/
│   └── model_outputs/
├── checkpoints/
│   ├── best/
│   └── resumable/
├── reports/
│   ├── data_coverage_summary.md
│   ├── results_summary.md
│   ├── coverage_by_year_form.csv
│   ├── model_leaderboard.csv
│   ├── tuning_trials.csv
│   ├── hypothesis_tests.csv
│   └── figures/
├── logs/
│   ├── retrieval.log
│   ├── parsing.log
│   └── training.log
└── manifests/
    └── run_manifest.json
```

Use Drive or another persistence location only after checking available
storage. Keep large data artifacts out of the notebook's embedded
outputs. Clear large cell outputs before saving the final notebook.

------------------------------------------------------------------------

## 16. Reproducibility, validation, and automated checks

Create explicit checks that fail loudly when critical assumptions are
violated:

-   Filing forms are in the expected set, with amendments flagged.
-   Filing timestamps parse correctly and have a timestamp-quality flag.
-   Company identifiers map consistently to the point-in-time security
    mapping.
-   No duplicate accession/filing events are silently counted twice.
-   No company/event group crosses data splits.
-   Every feature has an availability timestamp no later than the
    prediction cutoff.
-   Targets use only prices after the eligible event time.
-   Training/validation/test periods do not overlap.
-   Test data is never used for fitting transforms, tuning, thresholds,
    or early stopping.
-   There is no news data dependency anywhere in the notebook.
-   Metric calculations handle absent classes and return unavailable
    metrics as null.
-   Model label IDs are mapped correctly.
-   A checkpoint can be reloaded and reproduces stored predictions
    within tolerance.
-   A resumed run skips completed tasks without corrupting the manifest.
-   All result tables include split, horizon, sample size, model ID, and
    tuning status.

Include a small synthetic test that deliberately introduces a leakage
condition and demonstrates that the audit catches it.

------------------------------------------------------------------------

## 17. Final deliverables and final research report

The finished project must produce:

1.  **A complete `.ipynb` notebook** that runs in Colab.
2.  **A data coverage report** covering January 2000--October 2026,
    including exact observed dates and verified gaps.
3.  **A missing-data report** for 10-K, 10-Q, 8-K, market prices,
    company identifiers, and timestamps.
4.  **A model comparison report** showing each model before and after
    tuning.
5.  **Loss and validation curves** for every fine-tuned model.
6.  **Classification and regression metric tables** with proper
    averaging, uncertainty, and sample counts.
7.  **A locked-test leaderboard** and explanation of the selected
    model(s).
8.  **Drift hypothesis results** with
    supported/unsupported/inconclusive/not-testable status.
9.  **A leakage audit** and split manifest.
10. **An ablation and feature-importance report** explaining whether SEC
    text adds value.
11. **A run manifest** containing model IDs/revisions, package versions,
    data source versions, dates, random seeds, hardware, configuration,
    and completion status.
12. **A short executive summary** that includes negative/null findings,
    limitations, and what the evidence does and does not support.

### Final conclusions must answer

-   Did we successfully cover at least 500 stocks?
-   What fraction of the requested filing universe was obtained and
    parsed?
-   Which years/forms/company groups are missing or weakly covered?
-   Which models were benchmarked, which were tuned, and why?
-   Did tuning improve validation performance and did that improvement
    persist on the locked test?
-   Does filing text outperform price-only and metadata-only baselines?
-   Which form types, sections, and information categories are most
    predictive?
-   Which drift hypotheses are actually testable under this scope?
-   What are the largest sources of bias, uncertainty, and failure?
-   What is the smallest defensible claim supported by the evidence?

------------------------------------------------------------------------

## 18. Claude Code execution instructions

Before writing the notebook: 1. Inspect all model cards and dataset
schemas that are actually accessible. 2. Verify available dataset
configurations and the fields they contain. 3. Propose the pipeline
architecture and compute estimate. 4. Implement the data audit and smoke
test first. 5. Do not start a 500-stock full run until the audit
demonstrates that the required sources and storage are workable.

When implementing: - Write modular functions with type hints where
useful. - Include concise comments and explanatory Markdown, not
filler. - Make configuration values easy to change in one place. -
Handle retries, missing values, rate limits, partial failures, and Colab
disconnects. - Do not assume every model is compatible with every
task. - Do not use fake example outputs in the final report. - If a
source or model is unavailable, log the failure and offer a compatible,
documented alternative rather than silently skipping the requirement. -
Do not run an expensive full experiment automatically when the notebook
is first opened. Default to `RUN_PROFILE="smoke"` and require the user
to change the profile to `pilot` or `full`. - Keep the final notebook
understandable to a researcher who wants to learn what every stage does.

**Definition of done:** a reproducible, resumable, leakage-audited
notebook that honestly reports data coverage, benchmarks at least five
verified pretrained NLP checkpoints where task-compatible, tunes a
limited subset under the available Colab budget, compares baselines and
tuned models, visualizes diagnostics, and explicitly distinguishes
supported findings from hypotheses that cannot be tested with the
selected SEC forms.
