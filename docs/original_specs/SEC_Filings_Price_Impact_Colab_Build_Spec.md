# SEC Filings → Price Impact & Drift Study (Jan 2000 – Oct 2026)

**Build spec for Claude Code. Deliverable: ONE Google Colab notebook (`sec_filings_price_impact_2000_2026.ipynb`) that runs end to end on a Colab Pro+ plan.**

This file contains **requirements only, no code**. Claude Code writes all code.

---

## 0. Instructions to Claude Code (read first)

**Your job:** build a single, well-narrated, resumable Colab notebook that implements everything below. The owner is a student who wants to *understand* the study, so every stage needs a plain-English markdown cell (what / why / how to read the output) and every major plot needs a short "how to read this" note.

**Hard rules**
1. **NO news data.** Do not load, reference, or mention the Hugging Face news repository. Sources are **SEC EDGAR filings only** (10-K annual, 10-Q quarterly, 8-K current reports) plus market prices for the return side.
2. **Study window: 2000-01-01 → run date (late Oct 2026).** Events whose forward return window is not fully observed by the data cutoff must be excluded from labels (right-censoring).
3. **Minimum 500 stocks** in the full run. A `pilot` mode (about 40 stocks, 2016–2020) must exist and exercise every stage and every plot before the full run is attempted.
4. **Never fabricate results.** This spec contains no results. Every number in the final report must be computed by the notebook.
5. **Verify before you build.** Check current EDGAR endpoints and rate-limit rules, `yfinance` behaviour, and that every Hugging Face model ID in Section 9 exists. If an ID is missing or renamed, substitute the closest equivalent and log the substitution in the report. Model IDs 3–8 in Section 9 were listed from memory.
6. **Credit discipline is a first-class requirement** (Section 4). Cache everything to Google Drive, make every stage idempotent and resumable, and keep GPU work in a clearly separated part of the notebook.
7. **No secrets in the notebook.** The SEC `User-Agent` (name + email) is read from a Colab secret or an interactive prompt.
8. **Reproducibility:** fixed seeds, a saved config JSON with a run ID, and logged library versions and hardware.
9. **Test before delivering:** validate the notebook JSON, syntax-check every code cell, and (if network is available) run the CPU-only stages on a tiny slice (about 3 tickers, 2019–2020). State plainly in your hand-off what you could and could not run.
10. Do not reinstall `torch` or other heavy libraries Colab already ships. Install only the extras, version-pinned.

---

## 1. Frozen scope decisions

| Item | Decision |
|---|---|
| Text sources | 10-K (annual; include 10-K405 and 10-KT; exclude 10-KSB), 10-Q (exclude 10-QSB), 8-K (material item groups only, Section 6 S4) |
| Excluded | All news. Amendments (10-K/A, 10-Q/A, 8-K/A) are kept for the completeness audit only, not as events |
| Period | 2000-01 → 2026-10 |
| Universe | At least 500 US-listed stocks with continuous coverage; built from historical S&P 500 membership (current list plus the index-change history) so it is not just today's winners; stratified by sector; fixed seed |
| Prices | Adjusted daily prices (free sources, cached); benchmark SPY; sector ETFs optional; Fama-French daily factors from the Kenneth French data library |
| Task | Predict the abnormal price reaction to a filing: 3-class (Up / Sideways / Down) plus a regression head on abnormal return |
| Horizons | Reaction windows [0,+1], [0,+5], [0,+20] trading days; drift window [+2,+60] |
| Primary horizon | [0,+1] for headline results; others reported as secondary |
| Compute | Colab Pro+, credit-conscious (Section 4) |

---

## 2. SWOT of the idea (with the design response built into this spec)

### Strengths
- SEC filings are free, official and **timestamped to the second** (`acceptanceDateTime`), which allows clean event timing and leakage control.
- Three complementary filing types: periodic (10-K, 10-Q) and event-driven (8-K with structured item codes).
- 26 years covers the dot-com bust, the 2008 crisis, COVID, the 2022 rate shock and 2023–26, so findings can be tested across regimes.
- 500+ stocks gives roughly 50,000+ earnings-related events, which is real statistical power for event studies.
- Pretrained financial/SEC language models exist, so strong baselines need no pretraining.
- Ties directly to well-known research (post-earnings drift, tone dictionaries, "Lazy Prices" filing-change signals), so results are defensible and comparable.

### Weaknesses
- **Core scope has no analyst, insider or institutional data.** Three of the seven requested anomalies cannot be tested directly (see Section 12). The spec says so openly instead of faking proxies.
- Free price sources are unofficial and weak on delisted tickers, so **survivorship bias** is likely.
- Filing text is long and boilerplate-heavy; year-over-year 10-K text is nearly identical, which invites memorization by firm.
- Text parsing quality varies, especially pre-2009 plain-text filings.
- A large-cap-leaning universe tends to show weaker anomalies than small caps.
- Financial return prediction has a very low signal-to-noise ratio. Expect modest metrics.
- Fine-tuning several transformers on hundreds of thousands of documents is expensive on a credit plan.

### Opportunities
- **Filing-change features** (similarity to the firm's own previous filing) are a published, more promising signal than raw sentiment.
- Combine 8-K Item 2.02 tone with XBRL earnings surprise to test whether text adds value beyond the numbers.
- Item-level analysis ("which 8-K item types move prices") is novel enough for a strong course project or paper.
- Optional SEC-sourced extensions (Form 4 insiders, 13F institutions) can be switched on later without redesign.
- Negative or null results are reportable findings: "text adds no lift beyond tabular baselines" is a legitimate, credit-saving outcome.

### Threats
- **Leakage and look-ahead bias** can silently inflate results (overlapping return windows, same-day market shocks, firm memorization, XBRL restatements). Section 7 is the countermeasure.
- Anomaly decay after publication means a null result may be a true result, not a bug.
- Multiple testing (many signals × horizons × subsamples) produces false discoveries if uncorrected.
- SEC rate limits or blocks on shared Colab IPs; Colab disconnects; credit burn on long runs.
- Regime shift: the 2020–26 test period behaves differently from 2000–19 training.
- Vendor risk: `yfinance` is unofficial and can break.

---

## 3. Research questions → what the notebook must answer

| # | Question | Where answered |
|---|---|---|
| Q1 | How do 10-K, 10-Q and 8-K filings affect price movements across the timeline? | S6, S7, S14 (event studies by form, year, regime) |
| Q2 | Which types of filings or specific information move prices? | S14 (8-K item groups, 10-K/10-Q sections, topics, entities, filing-change features) |
| Q3 | Correlations: (a) filing ↔ filing, (b) filing ↔ price, with the news leg dropped | S14 |
| Q4 | Which specific filings push the stock up, down or sideways? | S12, S14 (classifier, SHAP, error analysis, top examples) |
| Q5 | In-depth NLP study and patterns | S14 (topics, dictionaries, readability, embeddings, attribution) |
| Q6 | Best-performing pretrained models; before vs after tuning; chosen models; end results | S9–S13, S16 |
| Q7 | Evidence for PEAD, dividend drift, analyst-revision drift, momentum, insider drift, institutional drift, spin-off/restructuring drift | S15 |
| Q8 | Is the split unbiased and leak-free? Over/underfitting? | S8, S13, Section 7, Section 11 |

---

## 4. Colab Pro+ credit plan (mandatory design constraints)

**Notebook layout: three parts, separable by runtime type.**
- **Part A (CPU runtime, high-RAM if needed):** S0–S8. Network-bound scraping and parsing. Low credit use, long wall-clock. Use background execution if the plan offers it.
- **Part B (GPU runtime):** S9–S11. Embeddings and fine-tuning only. Its first cell must verify Part A caches exist and **stop immediately if they don't**, so no GPU time is wasted.
- **Part C (CPU runtime):** S12–S16. Evaluation, analysis, visuals and report.

**Controls**
1. `RUN_MODE = pilot` by default; the user flips to `full` deliberately.
2. Every stage writes versioned Parquet/NPY artifacts to Drive, keyed by a config hash, and skips work if the artifact exists.
3. Optuna studies are stored in a SQLite file on Drive; fine-tuning checkpoints every epoch; scraping checkpoints every N filings. A disconnect must cost minutes, not hours.
4. A **benchmark cell** times a small batch on the current GPU and prints the projected duration for the full stage **before** it starts. The user can abort.
5. `MAX_GPU_HOURS` guard: a running tracker of GPU time that stops gracefully and checkpoints when the cap is hit.
6. Mixed precision (bf16 where supported, otherwise fp16); length-bucketed dynamic padding; inference mode and no gradients for embedding extraction.
7. Frozen-embedding strategy first (compute each encoder once, cache, tune cheap heads on the cache). Full fine-tuning only for the top 2 candidates, and only if the **text-lift gate** passes (Section 10).
8. Hyperparameter search runs on a stratified-by-time **subsample** with pruning and early stopping. Only the winning configuration is retrained on the full training data.
9. Store embeddings as float16. Store parsed text as zstd-compressed Parquet; truncate stored text to the configured per-section character budget. Do not keep raw HTML on Drive.
10. Disable external experiment trackers; log metrics to local CSV and save figures as PNG to Drive.
11. Print a cumulative "resources used" line (wall-clock and GPU hours) at the end of each stage. Do not hard-code credit-unit rates; tell the user to read the live Colab resource panel.

**Rough scale (planning estimates only; the notebook must replace them with measured values):**
- Filing counts for 500 stocks over 26 years are on the order of tens of thousands of 10-K/10-Q filings and tens of thousands to low hundreds of thousands of 8-Ks, depending on item filtering.
- SEC fair-access limits throughput, so full scraping takes **hours of wall-clock on a CPU runtime**, which is cheap in credits but needs resumability.
- Embedding cost scales with `chunks per document × documents × number of encoders`; the chunk caps in Section 5 keep this bounded.

---

## 5. Configuration (one config cell at the top; every value overridable)

| Key | Default (full) | Pilot |
|---|---|---|
| `RUN_MODE` | `full` | `pilot` |
| `STUDY_START` / `STUDY_END` | 2000-01-01 / run date | 2016-01-01 / 2020-12-31 |
| `N_STOCKS` | 500 (minimum; more allowed if coverage passes) | 40 |
| `SEED` | 42 | 42 |
| `SEC_MAX_RPS` | 6 (stay below SEC's published ceiling; verify current rule) | 4 |
| `SEC_USER_AGENT` | from Colab secret / prompt | same |
| `FORMS` | 10-K family, 10-Q, 8-K (material groups) | same |
| `TEXT_CHAR_BUDGET` | 10-K sections ≈100k chars, 10-Q ≈60k, 8-K ≈30k | same |
| `MAX_CHUNKS_PER_DOC` | 10-K 8, 10-Q 4, 8-K 2 (512 tokens each) | same |
| `HORIZONS` | [1, 5, 20] trading days | same |
| `DRIFT_WINDOW` | [+2, +60] | same |
| `SIDEWAYS_K` | 0.5 (band = k × firm volatility × √h) | same |
| `MIN_EST_OBS` | 120 days for the market-model fit | same |
| `DEV_END` / `TEST_START` | 2019-12-31 / 2020-01-01 | 2018-12-31 / 2019-01-01 |
| `EMBARGO_CAL_DAYS` | 90 | 90 |
| `OPTUNA_TRIALS_HEADS` / `_FT` | 40 / 8 | 6 / 2 |
| `FT_MAX_CHUNKS` | 2 | 2 |
| `MAX_GPU_HOURS` | 12 (user-adjustable) | 1 |
| `TEXT_LIFT_GATE` | 0.01 macro-F1 over tabular baseline, CI excludes 0 | same |
| `N_BOOTSTRAP` | 1000 (block bootstrap by month) | 200 |
| `FDR_Q` | 0.05 | 0.05 |
| `ENABLE_OPTIONAL_SEC_MODULES` | False (Form 4 insiders, 13F institutions) | False |
| `TEST_UNLOCK` | False until the manifest freeze cell has run | same |

---

## 6. Notebook stages

Each stage: purpose → inputs → outputs (cache path) → built-in assertions → part (A/B/C). Every stage opens with a plain-English markdown cell.

### S0 Setup (Part A, B, C)
Mount Drive; create the run folder tree (`data/`, `cache/`, `models/`, `figures/`, `reports/`); load config; seed everything; log versions and hardware; read the SEC User-Agent from a secret; define the checkpoint/resume helper and the GPU-time tracker.

### S1 Universe construction
Build the candidate pool from historical S&P 500 membership (current constituents plus index additions/removals; log that early-year change data is incomplete). Map tickers to CIKs through SEC's ticker/CIK files and EDGAR company records, including companies with changed or retired tickers. Keep candidates that have at least 5 years of overlapping prices and filings within the study window. Stratify by sector (SIC-based sector mapping from EDGAR) and by liquidity bucket (trailing average dollar volume), then sample to `N_STOCKS` with the fixed seed.
**Assertions:** at least 500 unique CIKs in full mode; no duplicate CIK; the report states how many candidates were dropped and why.
**Outputs:** `universe.parquet` (CIK, ticker history, sector, first/last filing date, price coverage).

### S2 Prices & benchmarks
Download adjusted daily prices and volume for the universe, SPY, optional sector ETFs, and Fama-French daily factors (Mkt-RF, SMB, HML, RMW, CMA if available, MOM). Cache per ticker; retry with backoff; log a per-ticker coverage table (first date, last date, gaps).
**Rules:** returns from adjusted close only; winsorize extreme daily returns at the 0.5/99.5 percentiles per year for modelling (raw kept for audits); no forward filling across long gaps.
**Outputs:** `prices.parquet`, `factors.parquet`, `price_coverage.parquet`.
**Known limitation to print:** delisted names are often missing from free sources (survivorship bias).

### S3 EDGAR filing index + completeness audit
For every CIK, pull the full filing history from EDGAR's submissions data (including the older paginated files, not just the recent block). Keep: accession number, form, filing date, **acceptance date-time**, report period, primary document, and **8-K item codes**. Build the *expected* filing list for 10-K, 10-Q and 8-K inside each firm's active window.
**Completeness audit (answers "what is missing"):** compare expected vs downloaded vs successfully parsed, by firm × year × form. Produce the coverage heatmap (V01) and a missing-data table with a reason code (not filed, not found, download failed, parse failed, filtered out). Expected pattern for sanity: about 1 10-K and 3 10-Qs per firm-year (a Q4 10-Q does not exist).
**Optional cross-check (off by default, to save time):** for 10-Ks up to 2020, sample-compare the downloaded set against the Hugging Face SEC dataset `JanosAudran/financial-reports-sec`. Keep it SEC-only; no news dataset.
**Outputs:** `filing_index.parquet`, `completeness_audit.parquet`.

### S4 Download, gap-fill and parse (the heavy CPU stage)
**What to fetch**
- 10-K family: sections Item 1A (Risk Factors) and Item 7 (MD&A); optionally Item 1 and 7A behind a flag.
- 10-Q: Part I Item 2 (MD&A) and Part II Item 1A (risk-factor updates).
- 8-K: item text, plus the press-release exhibit (EX-99.x) for earnings and results items. Using the complete-submission text file for 8-Ks may save requests; verify and choose the cheaper approach.
- **8-K item groups to keep** (the table maps modern item codes; Claude Code must also map the legacy numbering used before the August 2004 rule change, for example older "Item 5 Other Events", "Item 9 Regulation FD" and the short-lived "Item 12 Results of Operations" in 2003–04, and must detect earnings releases by exhibit content in those years):

| Group | Modern items (examples) |
|---|---|
| EARNINGS | 2.02 (and legacy equivalents) |
| DEAL_MA | 1.01, 1.02, 2.01 |
| RESTRUCTURING | 2.05, 2.06 |
| FINANCING | 2.03, 2.04, 3.02 |
| LISTING / DELISTING | 3.01 |
| AUDITOR / RESTATEMENT | 4.01, 4.02 |
| MANAGEMENT | 5.02 |
| OTHER_DISCLOSURE | 7.01, 8.01 (subtyped later by zero-shot: dividend, buyback, guidance, litigation, spin-off, etc.) |
| Dropped | 5.07 (vote results), 9.01 alone, pure exhibit-only filings |

**Gap-fill:** after the first pass, re-read the completeness audit and retry failed/missing filings (alternate document URL, alternate parser, then fallback to truncated full text). Log the final fill rate per form and year. This is the "take additional SEC info if pieces are missing" step; it must also flag firms/years that remain genuinely absent.
**Parsing:** HTML and plain-text parsers; section extraction by item headings with a fallback to truncated full text; strip tables of numbers and boilerplate (cover page, signatures, exhibit lists); **mask company names and tickers** in the text used for modelling. Log parse success rate by form and year (V02).
**Politeness:** throttled requests, descriptive User-Agent, exponential backoff, resumable queue, checkpoint every N filings.
**Outputs:** `filings_text.parquet` (accession, CIK, form, group, item codes, section, text, char counts, parse flags).

### S5 Structured numbers (XBRL) and filing-change features
- From SEC's company-facts data, pull reported EPS and key items (revenue, net income, shares). **Use only facts whose "filed" date is on or before the event date** (avoids restatement look-ahead). Compare each quarter with the prior-year quarter **as reported in the same filing** to sidestep stock-split mismatches.
- **Standardized Unexpected Earnings (SUE):** seasonal random-walk change in EPS divided by the standard deviation of recent changes. Available only from roughly 2009–2011 onward (XBRL phase-in); the report must state the coverage start date by firm size.
- **Earnings Announcement Return (EAR)** (the abnormal return over [-1,+1] around the earnings date) is the surprise proxy for **all years**, including 2000–2008.
- **Filing-change features ("Lazy Prices" style):** similarity between this filing's section and the same firm's previous-period section (TF-IDF cosine and, later, embedding cosine); change in length, in negative-word share, in risk-factor count. Computed point-in-time only.
**Outputs:** `fundamentals.parquet`, `change_features.parquet`.

### S6 Event dating, abnormal returns and labels
- **Event time rule:** use acceptance date-time (US Eastern). Accepted before the 16:00 close on a trading day → day 0 is that day; accepted after close, on a weekend or on a holiday → day 0 is the next trading day. Pre-market filings count as day 0.
- **Earnings announcement date:** earliest of {8-K earnings item, validated by earnings-release content} and {10-Q/10-K filing date} within the fiscal-period window. Mark which source set the date.
- **Same-day dedup:** several 8-Ks from one firm on one day become one event with multi-label item groups.
- **Abnormal return:** market-model benchmark (SPY) with beta fitted on trading days [-250,-30] before the event, requiring at least `MIN_EST_OBS` observations; robustness version using the four-factor (market, size, value, momentum) model. Compute CAR over [0,+1], [0,+5], [0,+20] and the drift window [+2,+60]; also [-5,-1] as a **placebo window**.
- **Labels (3-class):** Up if CAR > +τ, Down if CAR < −τ, else Sideways, where τ = `SIDEWAYS_K` × firm idiosyncratic daily volatility (measured on [-250,-30], point-in-time) × √h. Also store the winsorized CAR for the regression head. Print class balance by year, form and sector.
- **Right-censoring:** drop events whose forward window extends past the data cutoff.
- **Tabular controls attached at event time:** log size/liquidity proxy, trailing volatility, 1-month and 12-1-month returns, pre-event run-up, SUE/EAR (where available), form family, 8-K item group, sector, year.
**Outputs:** `events.parquet` (one row per event; the unit of observation is the **filing event, never a sentence**).

### S7 Exploratory analysis and data-quality visuals (Part A end)
Produce V01–V10 (Section 11) with "how to read" notes. Print the data-readiness verdict: filings present vs expected, per form and per year, and the missing-points summary the owner asked for.

### S8 Split design and leakage controls (Part A end)
Implement exactly the protocol in Section 7. Produce the split-timeline figure, the split-balance tables, adversarial validation and the automated leakage assertions. **The notebook must refuse to continue if any assertion fails.**

### S9 Baselines (CPU; Part A or C)
All models here use the identical split and metrics as the transformers:
- **B0** majority-class and class-prior predictors.
- **B1** momentum/tabular-only gradient-boosted trees (no text).
- **B2** Loughran-McDonald dictionary tone (negative, positive, uncertainty, litigious, constraining word shares) plus logistic regression.
- **B3** TF-IDF plus logistic regression (text-only, linear).
- **B4** tabular + dictionary tone + filing-change features with gradient-boosted trees.
Purpose: show whether deep NLP adds anything beyond cheap baselines.

### S10 Embeddings (Part B; Tier 1)
For each candidate encoder in Section 9 (roles "candidate"), embed up to `MAX_CHUNKS_PER_DOC` chunks per document, mean-pool to a document vector (also store chunk-level sentiment probabilities for sentiment models), and cache float16. Run the benchmark cell first. Run the **feature-extractor models** (zero-shot event typing, entity extraction) once and cache their outputs as features.
**Outputs:** `emb_<model>.npy` plus an index file; `zeroshot_events.parquet`; `entities.parquet`.

### S11 Hyperparameter tuning (Parts B/C)
- **Tier 1 (cheap):** for each candidate encoder's cached embeddings, train three head types (regularized logistic regression, gradient-boosted trees, small MLP) with Optuna over the Section 10 search spaces, evaluated on the **walk-forward validation folds**. Record **default-settings** performance first (the "before"), then tuned (the "after").
- **Text-lift gate:** compare the best text-using model with B1/B4 on validation. Proceed to Tier 2 only if the lift is at least `TEXT_LIFT_GATE` macro-F1 with a bootstrap CI that excludes 0. If it fails, skip Tier 2, say so in the report, and finish with the Tier 1 results (a legitimate, credit-saving outcome).
- **Tier 2 (expensive):** full fine-tuning of the **top 2** encoders from Tier 1 with Optuna (`OPTUNA_TRIALS_FT` trials, median pruning, early stopping) on a stratified time subsample; retrain only the best configuration on the full training data. Log train/validation loss per step and per epoch for the loss curves.
- **Optional Tier 3 (off by default):** an LLM-based sentiment/event model (for example the FinGPT family) as a zero-shot feature extractor. Do not fine-tune LLMs in this notebook.
- **Freeze before test:** write a manifest listing every candidate and its chosen hyperparameters, hash it, save it, and only then set `TEST_UNLOCK`.
**Outputs:** `optuna_*.db`, `tuning_results.parquet`, `models/` checkpoints, `manifest.json`.

### S12 Final evaluation (Part C)
After unlocking, evaluate **once** on the held-out test period (2020-01 → last fully observed event): every baseline, every tuned Tier-1 candidate, and the Tier-2 finalists, each in "default" and "tuned" form. Compute all metrics in Section 8 with block-bootstrap CIs. Break results out by form family, 8-K item group, year, sector, size bucket and horizon. Select the **champion on validation results only**; the test table is then reported, not used for selection.

### S13 Diagnostics: loss, learning and validation curves; over/underfitting (Part C)
See Sections 8, 10 and 11. Produce the loss curves, validation curves (metric vs one hyperparameter at a time, train and validation), learning curves (metric vs training-set size), the train-vs-validation gap table, and the automated verdict per model: **Underfit / Good fit / Overfit / Unstable**.

### S14 What moves prices; correlations; NLP pattern study (Part C)
**Information content by type:** per 8-K item group, per 10-K/10-Q section, per form: mean signed CAR, mean absolute CAR, squared-return ratio versus normal days, and abnormal volume, all with CIs and FDR-adjusted p-values. Include the placebo window.
**Correlations:**
- filing ↔ filing: tone and risk-language persistence across consecutive filings; 8-K earnings tone vs the following 10-Q tone; cross-form-type correlation heatmap; lead-lag at event time.
- filing ↔ price: Spearman rank correlation (information coefficient) between text signals and CAR, by form, item group, year, sector, size; partial correlations controlling for size, momentum, volatility and earnings surprise; mutual information as a non-linear check.
**Deep NLP patterns:**
- topic modelling on embeddings **fit on training data only**, then applied forward; topic prevalence over time (shocks such as 2008, 2020, rate-shock language); topic → CAR table with FDR control.
- dictionary categories, readability (Fog), length, number density, forward-looking-statement share, risk-factor count and their relationships to returns.
- explanation of the best text model: global feature importance (SHAP on the tree head), top terms per class, token-level attribution on a handful of example filings.
- error analysis: top confident wins and losses, by year and sector, with short text excerpts (keep excerpts short).
- 2-D embedding map fitted on a training subsample, coloured by label, form and year.
- "Which filings push Up / Down / Sideways": ranked lists of item groups, topics and phrases with effect sizes and CIs.

### S15 Anomaly evidence tests (Part C): Section 12.

### S16 Final report (Part C)
Auto-generate `RESULTS.md` and save all figures. Template in Section 13.

---

## 7. Split protocol and leakage controls (the core of "no bias, no leakage")

### 7.1 Unit and ordering
- The unit is the **filing event** (grouped by accession), never a sentence, chunk or section. All chunks and sections of a filing stay in one split.
- Splits are **by time**. A calendar date never appears in more than one split (same-day market shocks hit every stock at once).

### 7.2 Layout
| Block | Period | Use |
|---|---|---|
| Dev | 2000-01 → 2019-12 | training and tuning only |
| Walk-forward folds (expanding window) inside Dev | Fold 1: train 2000–2009, validate 2010–2011. Fold 2: train 2000–2011, validate 2012–2013. Fold 3: train 2000–2013, validate 2014–2015. Fold 4: train 2000–2015, validate 2016–2017. Fold 5: train 2000–2017, validate 2018–2019 | hyperparameter selection and model selection |
| Final test | 2020-01 → last fully observed event | untouched until `TEST_UNLOCK`; evaluated once |

- **Purge + embargo:** drop any training event whose forward return window (day 0 plus the longest horizon, 60 trading days) reaches into the validation or test period, and enforce a minimum 90-calendar-day gap. Apply the same rule at the Dev/Test boundary.
- **Final model:** retrain on all Dev data (with epochs and early-stopping points fixed from the folds), then score the test block once.
- Pilot mode uses the same logic with shorter dates.

### 7.3 Leakage and bias checklist (every item gets an automated assertion or report line)
| # | Risk | Control | Automated check |
|---|---|---|---|
| 1 | Overlapping forward return windows between train and validation/test | Purge + embargo (7.2) | Max train forward-window end is earlier than min validation day 0 |
| 2 | Same-date cross-sectional shocks | Date-disjoint splits; abnormal (not raw) returns | Date sets of splits are disjoint |
| 3 | Same filing, duplicate or near-duplicate text across splits | Split by accession; text-hash and near-duplicate detection; amendments not used as events | Zero shared accessions or hashes across splits |
| 4 | Firm memorization (boilerplate repeats each year) | Mask names/tickers; filing-change features; **firm-holdout robustness test** (group folds by CIK on Dev) | Report the gap between temporal and firm-holdout scores |
| 5 | Preprocessing fitted on future data (scalers, vocabularies, TF-IDF, topic model, UMAP, PCA, class weights, band thresholds, target encodings) | Every fit uses training indices of that fold only; objects saved per fold | Fit-index assertion per fold |
| 6 | Test peeking / tuning on test | Manifest freeze + `TEST_UNLOCK`; test evaluated in a single cell with timestamp | Manifest hash logged before test scores exist |
| 7 | Event-time errors (after-close filings treated as same-day) | Acceptance-timestamp rule (S6) | Share of events re-dated; placebo window check |
| 8 | Information already priced before the filing (10-Q after earnings 8-K) | Report earnings-date source; analyse 10-Q reaction separately; placebo CAR[-5,-1] | Placebo CAR vs text signal should be near zero |
| 9 | XBRL restatements / later-reported values | Only facts with filed date ≤ event date | Date comparison assertion |
| 10 | Survivorship in the universe and prices | Historical membership universe; coverage table; sensitivity on delisted-inclusive subset where available | Report coverage and caveat |
| 11 | Right-censored labels | Drop events lacking a full horizon | Assert max event date + horizon ≤ data cutoff |
| 12 | Target-aware feature selection or label thresholds computed on full data | Forbidden; all selection inside folds | Code review assertion |
| 13 | Data snooping across many tests | Pre-declared test list (Section 12), FDR correction | Number of tests logged and used in correction |
| 14 | Composition bias between splits | Compare split distributions by year, sector, size bucket, form, item group and class balance | Tables and chart V12; KS/χ² tests |
| 15 | Covariate shift between train and test | **Adversarial validation:** train a classifier to tell train from test; AUC near 0.5 means similar, clearly above means drift | Reported AUC and top drift features |

### 7.4 Leakage sanity experiments (reported as visuals)
- **Random-split vs temporal-split demo:** same model, same data; a random split should look better. Report the inflation as a lesson.
- **Permutation test:** shuffle labels within month; metrics should collapse to chance.
- **Placebo:** predict the pre-event window [-5,-1]; no skill expected.
- **Red-flag rule:** if test macro AUC exceeds about 0.65 or R² is clearly positive at the event level, stop and audit for leakage before celebrating.

---

## 8. Metrics catalog (all computed, none assumed)

**Classification (3-class and per-class one-vs-rest)**
- accuracy, **balanced accuracy**, per-class and macro precision, recall, F1 (macro-F1 is the primary selection metric), weighted F1
- Matthews correlation coefficient, Cohen's kappa
- ROC-AUC (one-vs-rest, macro and per class), PR-AUC (macro)
- log-loss, Brier score, calibration (reliability curve and expected calibration error)
- confusion matrices (counts and row-normalised)
- always shown next to the **majority-class baseline** and the chance level (about 0.33 macro-F1 for balanced 3-class)

**Regression head (abnormal return)**
- RMSE, MAE, **R²** in-sample and out-of-sample
- **out-of-sample R² versus the expanding historical-mean forecast** (Campbell-Thompson style)
- **Spearman rank IC**, IC by month, IC t-statistic, ICIR, directional hit rate

*Note for the report:* R² is not defined for the 3-class labels, so it is reported for the regression head only. In return prediction, out-of-sample R² near 0–1% and AUC only slightly above 0.5 are typical, not failures.

**Economic checks**
- long-short portfolios formed on predicted score deciles: mean return, Sharpe, turnover, and results under 10 and 20 bps cost assumptions

**Uncertainty**
- block-bootstrap (by month) confidence intervals for every headline metric; paired bootstrap or McNemar-style tests for model-vs-model and before-vs-after-tuning comparisons

---

## 9. Model roster

Roles: **candidate** = competes as a predictive encoder; **feature extractor** = produces features, evaluated by ablation (with vs without).

| ID | Hugging Face / GitHub | Role | Cost tier |
|---|---|---|---|
| M1 | `ProsusAI/finbert` (GitHub: ProsusAI/finBERT) | candidate | Tier 1, Tier 2 eligible |
| M2 | `yiyanghkust/finbert-tone` (siblings `finbert-esg`, `finbert-fls` optional) | candidate | Tier 1, Tier 2 eligible |
| M3 | `nlpaueb/sec-bert-base` (optionally `sec-bert-num`) | candidate | Tier 1, Tier 2 eligible |
| M4 | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | candidate (fast) | Tier 1, Tier 2 eligible |
| M5 | `BAAI/bge-base-en-v1.5` (general embedding model) | candidate | Tier 1 only |
| M6 | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | feature extractor (zero-shot event typing, such as earnings beat, guidance cut, M&A, litigation, dividend, buyback, spin-off, restructuring) | inference only |
| M7 | `dslim/bert-base-NER` | feature extractor (entities such as organisations, locations, money) | inference only |
| M8 | FinGPT-family sentiment adapters (GitHub: AI4Finance-Foundation/FinGPT) | optional feature extractor | **off by default** |

Rules: verify every ID exists (Section 0 rule 5); log substitutions; encoders cap at 512 tokens, so use chunk-and-pool; sentiment models also give three class probabilities as direct features.

**Ablations to report:** tabular only; text only; text + tabular; with vs without filing-change features; with vs without zero-shot event features; per form family; per 8-K item group.

---

## 10. Hyperparameter search spaces and selection rules

| Model type | Parameters tuned |
|---|---|
| Logistic regression head | regularization strength C (log scale), L1/L2 mix, class weights on/off |
| Gradient-boosted trees head | learning rate, number of trees (with early stopping), leaves, min child samples, feature and bagging fractions, L1/L2 regularization, max depth |
| MLP head | hidden sizes, dropout, weight decay, learning rate, batch size |
| Transformer fine-tuning | learning rate (1e-5 to 5e-5, log), batch size, epochs (1–3), warmup ratio, weight decay, max length (256 or 512), dropout, label smoothing, layer-wise learning-rate decay, pooling (CLS vs mean), chunk aggregation (mean vs max vs attention), class-weighting |

- **Optimizer:** Optuna TPE sampler with median pruner; the study is persisted to Drive.
- **Loss functions (document and justify in the report):** class-weighted cross-entropy for the 3-class head (optional focal-loss variant when imbalance is high); Huber loss on the winsorized abnormal return for the regression head; for a multi-task fine-tune use the weighted sum of both; for tree heads, multiclass log-loss.
- **Primary objective:** mean validation macro-F1 across walk-forward folds. Break ties by lower fold-to-fold standard deviation, then by lower cost. Prefer the simpler model if the difference is within one bootstrap standard error.
- **Text-lift gate** (Section 6, S11) decides whether to spend credits on Tier 2.
- **Before/after table:** for each model, default hyperparameters vs tuned, on both validation folds and test, with the change in macro-F1, AUC, MCC and IC and a significance flag.

---

## 11. Visualization catalog (static PNG saved to Drive; interactive versions optional; each with a "how to read" note)

| ID | Visual | Answers |
|---|---|---|
| V01 | Coverage heatmap: firms × years, per form (10-K, 10-Q, 8-K), filled/missing | Are all filings present? What is missing? |
| V02 | Parse success and fill rate by form and year | How reliable is the text? |
| V03 | Filings per year by form and 8-K item group | Volume over time |
| V04 | Universe composition: sector, size bucket, listing years | Representativeness |
| V05 | Text length, tokens and chunk counts per form | Compute and truncation impact |
| V06 | Return and CAR distributions by horizon (with winsorization marks) | Label sanity |
| V07 | Class balance by year, form, sector, item group | Bias check |
| V08 | Event-time diagram: timing rule outcomes (pre-market, intraday, after-close) | Timing validity |
| V09 | Average CAR paths around events (day −10 to +60) by form and item group | Q1/Q2 |
| V10 | Placebo window results | Leakage and timing check |
| V11 | Split timeline: dev folds, purge/embargo bands, test block | Split design |
| V12 | Split balance charts (year, sector, size, form, class) + adversarial-validation ROC and top drift features | Is the split unbiased? |
| V13 | Random-split vs temporal-split performance comparison | Leakage magnitude |
| V14 | Permutation-test distribution vs actual score | Pipeline integrity |
| V15 | **Loss curves**: train vs validation loss per epoch/step (per fold, per model, best vs default config) | Training behaviour |
| V16 | **Validation curves**: metric vs single hyperparameter (train and validation) | Hyperparameter sensitivity |
| V17 | **Learning curves**: metric vs training-set size | Under- vs overfitting |
| V18 | Optuna: optimization history, parameter importance, parallel-coordinates | Tuning insight |
| V19 | Before vs after tuning bars with CIs per model | Q6 |
| V20 | Model leaderboard (validation and test) with CIs and cost | Which models are best/worst |
| V21 | Confusion matrices (counts and normalized) for champion and baselines | Error structure |
| V22 | ROC and PR curves per class | Discrimination |
| V23 | Calibration (reliability) curves | Probability quality |
| V24 | Metrics by form, item group, year, sector, size bucket (small-multiple panels) | Where the signal lives |
| V25 | Rolling out-of-sample IC by month | Stability over time |
| V26 | Decile long-short cumulative returns | Economic relevance |
| V27 | Train-vs-validation gap dashboard and fit-verdict table | Overfit/underfit |
| V28 | Correlation heatmaps: filing↔filing (tone persistence, cross-form) and filing↔price | Q3 |
| V29 | Information-content bars: mean absolute CAR, squared-return ratio, abnormal volume per item group/section | Q2 |
| V30 | Topic prevalence over time + topic→CAR chart | Q5 |
| V31 | SHAP summary and top terms per class; token attribution examples | Q4/Q5 |
| V32 | 2-D embedding map (colour by label, form, year) | Pattern structure |
| V33 | Anomaly dashboards: CAR by signal quantile, calendar-time cumulative alpha, subperiod bars (one panel set per anomaly) | Q7 |
| V34 | Final summary dashboard (one page) | Executive view |

---

## 12. Anomaly evidence tests (the "true evidence" section)

**Method for all tests:** event-time CARs with standard errors clustered by firm and by date; calendar-time portfolio alphas using the four-factor model; Newey-West errors; subperiods **2000–2009, 2010–2019, 2020–2026**; size/liquidity splits; sensitivity to 10 and 20 bps costs; Benjamini-Hochberg FDR across the pre-declared test list; report effect size with confidence interval, not just p-values; flag cells with fewer than 30 events as "insufficient power".

**Verdict rubric (printed per anomaly):**
- **Supported:** FDR-adjusted p < 0.05, same sign in at least 2 of 3 subperiods, and not erased by transaction costs.
- **Weak / mixed:** significant in some cuts only, or sign unstable.
- **Not supported:** no significant effect after clustering and FDR.
- **Not testable with this data:** the required inputs are not in 10-K/10-Q/8-K.

| Anomaly | Testable in core scope? | How it is tested |
|---|---|---|
| **Post-Earnings Announcement Drift (PEAD)** | Yes | Event = earnings announcement (S6 rule). Surprise signals: SUE (2009–11 onward), EAR (all years), earnings-text tone change. Quintiles/deciles of surprise vs CAR over [+2,+60]; calendar-time long-short with four-factor alpha; subperiod and size splits; check whether text tone adds value beyond SUE/EAR (double sorts and regression) |
| **Post-Dividend Announcement Drift** | Yes (partially) | Detect dividend declarations in 8-K text (items 8.01/2.02/7.01) via zero-shot plus pattern checks; classify initiation, increase, decrease, omission, special using extracted amounts and dividend history. CAR [0,+1] and [+2,+60] per class. Likely few initiations/omissions in a large-cap universe, so report counts and power honestly |
| **Analyst Revisions Drift** | **No** | Analyst forecast data is not in SEC filings. **Proxy only, labelled as a proxy:** "management guidance revision drift" (guidance raised/lowered/reaffirmed in earnings releases). Never present it as analyst-revision evidence |
| **Short-Term Price Momentum** | Yes (prices only) | 1-month reversal, 2–12 and 12–1 month momentum with monthly cross-sectional sorts inside the universe; also test how prior returns condition filing reactions (interaction) and whether text signals add beyond momentum |
| **Insider Trading Drift** | **No in core scope** | Needs Forms 3/4/5. Optional module (`ENABLE_OPTIONAL_SEC_MODULES`): SEC's bulk insider-transactions data sets; open-market purchases vs sales, cluster buying, CAR [+1,+60] and calendar-time alpha. If disabled, the report must say "not tested" |
| **Institutional Ownership Drift** | **No in core scope** | Needs Form 13F. Optional module: SEC's bulk 13F data sets (structured data from about 2013 only); changes in institutional ownership. **Use the filing date, not the period-end date, as the information date** (13Fs are filed with a lag). If disabled, "not tested" |
| **Spin-Off and Corporate Restructuring Drift** | Yes (partially) | Events: 8-K items 2.05 and 2.06 (clean structured codes, mainly from late 2004), 2.01, plus text-detected spin-off / separation / split-off announcements in 8.01, 7.01, 1.01. CAR [0,+1], [+2,+60] and for spin-offs also [+2,+250] where the window is observable. Sample sizes may be small; report counts and power |

---

## 13. Final report template (`RESULTS.md`, generated by S16 from computed values)

1. **Data readiness:** universe size and composition; filings expected vs obtained vs parsed by form and year; remaining missing points with reasons; known biases (survivorship, large-cap tilt).
2. **Design:** event timing, abnormal-return method, label definition, class balance.
3. **Split and leakage audit:** diagram, balance tables, adversarial-validation result, all assertion outcomes, leakage experiments (random-split inflation, permutation, placebo).
4. **Models evaluated:** the full list, which were frozen into the manifest, which were dropped by the text-lift gate and why.
5. **Tuning:** search spaces used, trials run, before-vs-after table with significance.
6. **Results:** validation and test leaderboards with CIs; champion and why; ablations; metrics by form, item group, year, sector, size.
7. **Fit diagnostics:** loss, validation and learning curves with the verdict per model.
8. **What moves prices:** item-group and section tables, topics, entities, filing-change effects, correlations.
9. **Anomaly verdict table** (Section 12) with effect sizes, CIs, FDR-adjusted p-values and subperiod consistency.
10. **Limitations and honest conclusions:** what the data can and cannot support; items not tested; resource usage.

---

## 14. Acceptance checklist (the notebook must satisfy all)

- [ ] Runs top-to-bottom in `pilot` mode with every figure V01–V34 produced.
- [ ] Full mode processes at least 500 stocks and prints the exact count.
- [ ] No news data anywhere; no reference to the Hugging Face news repository.
- [ ] All stages resumable; killing the runtime mid-stage loses at most one checkpoint interval.
- [ ] GPU part refuses to start if Part A caches are missing.
- [ ] All Section 7.3 assertions pass or the notebook stops with a clear message.
- [ ] Test set is evaluated exactly once, after the manifest freeze.
- [ ] Before/after tuning table exists for every candidate.
- [ ] Anomalies that cannot be tested are labelled "not testable" or "proxy", never silently skipped or faked.
- [ ] Every number in `RESULTS.md` comes from executed cells; the red-flag rule (Section 7.4) is evaluated and printed.
- [ ] The cumulative resource-use summary is printed at the end of each part.

---

## 15. Build and hand-off instructions for Claude Code

1. Generate the notebook programmatically as a valid `.ipynb`, with a table of contents cell, then Part A / Part B / Part C section banners that clearly say which runtime type to select.
2. Order: config cell, install cell (extras only, pinned), setup, then stages S1 → S16. Keep functions small and documented; put plain-English markdown before each stage and "how to read" notes after each major plot.
3. Use assertions liberally (Section 7.3). Use progress bars that do not flood output, and cap printed rows.
4. Provide a short companion `README` in the notebook's first markdown cell: how to start in pilot mode, how to switch runtime between parts, how to resume after a disconnect, where outputs live on Drive.
5. In your hand-off message list: what you verified (endpoints, model IDs, notebook validity), what you could not run (anything needing Colab GPU or the live SEC at scale), any model-ID substitutions, and any deviations from this spec with reasons.
