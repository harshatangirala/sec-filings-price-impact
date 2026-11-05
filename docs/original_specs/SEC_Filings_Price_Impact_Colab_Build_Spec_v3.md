# SEC Filings → Price Impact: Architecture-Search Study, v3

**Build spec for Claude Code. Deliverable: ONE Google Colab notebook (`sec_filings_price_impact_v3.ipynb`) that runs end to end on Colab Pro+.**

This document contains requirements only, **no code**. Claude Code writes all code. v3 replaces v2. It keeps v2's leakage-safe event-study core and adds an **architecture search**: a disciplined tournament over pretrained encoders, financial LLMs, retrieval-augmented generation (RAG) designs, time-series foundation models and tabular learners. The aim is the most accurate architecture that still holds up out of sample, without fooling ourselves.

> **Universe size and study window are deliberately left as parameters (Section 3).** The owner will set them in the next stage. Every rule in this spec is written relative to those parameters.

---

## 0. Instructions to Claude Code (read first)

**Your job:** build one well-narrated, resumable, leakage-audited Colab notebook that implements this spec. The owner is a Master's student in applied AI who wants to *understand* every stage. Each stage opens with a plain-English markdown cell (objective, method, assumptions, expected output, how to read it). Every major plot gets a short "how to read this" note.

**Hard rules**
1. **SEC filings only (10-K, 10-Q, 8-K) as text. No news, social-media, transcript or analyst text of any kind.** Do not load the Hugging Face news repository, FNSPID, FinMultiTime's news modality, StockNet tweets or any other news corpus. Market prices are used only to build outcomes and price features. Some pretrained models were trained by their authors on news-style text; that is allowed, but note it in the limitations.
2. **Universe and window are config parameters** (`N_STOCKS`, `STUDY_START`, `STUDY_END`). The notebook refuses a `full` run while either is unset. `smoke` and `pilot` have safe defaults.
3. **Never fabricate** dates, documents, downloads, model results or metrics. Record errors and missingness explicitly. This spec contains no results; every number in the report must be computed by the notebook.
4. **Association is not causation.** Report prediction and association. Never claim a filing *caused* a move.
5. **Point-in-time discipline everywhere.** That covers features, retrieval indexes, nearest-neighbour outcome look-ups, preprocessing fits and pretrained-model knowledge (Section 8). At decision time t, nothing may use information that was not public before t.
6. **Verify before building.** Re-check every model and dataset ID in Section 9 (existence, license, gating, size, context length, pretraining-data dates). Pin the exact revision (commit hash) of everything used. Log every substitution. Never silently swap a model.
7. **Never select documents or events based on later price moves.** For example, do not "download filings around big moves". That is selection on the outcome and makes every result invalid.
8. **Default profile is `smoke`.** Never start an expensive run automatically.
9. **Credit discipline is a first-class requirement** (Section 17). Cache everything, make stages idempotent and resumable, separate CPU and GPU parts, estimate before spending, and gate every expensive step.
10. **No secrets in the notebook.** The SEC User-Agent (name + contact email) and any Hugging Face token come from Colab secrets or a prompt.
11. **No paid or hosted APIs.** All models run locally in Colab. LLM-as-judge uses a local open model, calibrated against human labels.
12. **SEC fair-access policy:** declared User-Agent, throttled requests, backoff on 429/5xx. **No proxy rotation or other evasion.** If blocked, pause and report.
13. **Reproducibility:** fixed seeds, saved config with a run ID, pinned package versions and model revisions, logged hardware and GPU type.
14. **Test before delivering:** validate the notebook JSON, syntax-check every code cell, run the CPU stages on a tiny slice if network allows, and state plainly what you could not run.

---

## 1. What changed from v2

| Change | Why |
|---|---|
| **Architecture search** (Section 11): staged tournament with successive halving, then stacking | Finds the most accurate *combination* instead of hand-picking one model |
| **Six-tower design**: market/fundamental, lexical, encoder, RAG-LLM extraction, analogue retrieval, time-series foundation model; fused by a stacked meta-learner | Each tower captures a different kind of signal; stacking keeps only what adds out-of-sample value |
| **RAG tournament** over 13 designs (Section 10), scored on retrieval quality, extraction accuracy and downstream price-prediction lift | RAG is how long filings (often tens of thousands of tokens) become usable evidence for models with short context windows |
| **Expanded model roster** (Section 9). Every requested model was checked; some don't exist publicly, aren't SEC models, or need news; each gets a verdict | Prevents wasted credits and silent mistakes (for example, "SecBERT" and "SecRoBERTa" are *cybersecurity* models, not SEC-filing models) |
| **Look-ahead control for pretrained models** (Section 8): knowledge-cutoff ledger, vintage-matched point-in-time models, anonymization, post-cutoff clean test, memorization test | LLMs trained after an event may "know" what happened next; this is the biggest hidden risk in LLM-based return prediction |
| **Full metric catalogue** (Section 14), including generation, similarity, retrieval and RAG metrics, each mapped to the sub-task it can legitimately measure | BLEU or ROUGE cannot measure price-prediction accuracy; each metric is used where it is meaningful |
| **Selection-bias control** for the many models compared: Probability of Backtest Overfitting, Model Confidence Set, deflated performance | Comparing dozens of configurations inflates the best score by chance |
| **Teacher-student distillation**: the LLM extracts from a sample, and a small encoder learns to reproduce it across the full corpus | Gives LLM-quality features at a fraction of the GPU cost |
| **Unified gold set**: one annotation pass yields detector, retrieval, extraction and summary references | Cuts the owner's manual labelling effort |
| **Selective prediction** (accuracy versus coverage) and meta-labelling | An honest way to raise accuracy on the cases the model is confident about |
| Universe and timeline turned into parameters; split dates derived by rule | Owner sets them later |

---

## 2. Objective: what "highest accuracy" means here

1. **Primary task:** 3-class direction (Up / Sideways / Down) of the **abnormal** return at the pre-registered primary horizon (default h = 1 trading session after the base price).
2. **Primary selection metric:** mean **macro-F1** across walk-forward validation folds. Plain accuracy is misleading when classes are unbalanced.
   - **Guardrails:** log-loss no worse than the class-prior baseline, and expected calibration error below a set threshold.
   - **Tie-breaks:** lower fold-to-fold variance, then lower compute cost.
3. **Secondary tasks**, each with its own leaderboard:
   - abnormal-return regression (rank information coefficient and out-of-sample R²)
   - volatility-ratio regression
   - standardized absolute move
   - longer horizons
4. **"Best architecture"** is the configuration with the best validation score that also:
   - (a) survives the development holdout
   - (b) belongs to the Model Confidence Set
   - (c) has an acceptable Probability of Backtest Overfitting
   - (d) is then confirmed **once** on the locked test

   If the locked test does not confirm it, the report says so. The test is never re-run with a different choice.
5. **Realistic expectations, stated in the notebook:** direction prediction from public filings in liquid stocks is a low signal-to-noise problem. Small gains over strong baselines are the normal outcome. Volatility and absolute-move targets usually show more skill than direction. A very high score is a **red flag for leakage** (Section 12.4), not a success.
6. **Selective prediction** is reported alongside, never instead:
   - Accuracy and macro-F1 at pre-registered coverage levels (100%, 50%, 25%, 10% most confident predictions).
   - A coverage-versus-accuracy curve.
   - A meta-labelling model that decides whether to trust the primary prediction.
   Higher accuracy at lower coverage is a legitimate result only when the coverage levels were fixed in advance.

---

## 3. Scope and parameters

| Item | Decision |
|---|---|
| Text sources | 10-K (incl. 10-K405, 10-KT; exclude 10-KSB), 10-Q (exclude 10-QSB), 8-K (material item groups, S5). Amendments are flagged and kept for the audit, never used as events |
| Universe | `N_STOCKS` = **TBD** (≥ 500 recommended for statistical power). Built from historical index membership, stratified by sector and liquidity; survivorship limits disclosed |
| Window | `STUDY_START` / `STUDY_END` = **TBD**. Right-censor events whose forward window is incomplete at the data cutoff |
| Prices | Adjusted daily OHLCV from a documented source; SPY benchmark; Fama-French daily factors; provenance and delisted coverage recorded |
| Unit of observation | The filing event (accession), never a chunk or sentence |
| Horizons | h ∈ {1, 2, 5, 20, 60} trading sessions after the base price; post-reaction drift window = sessions 3–60 |
| Compute | Colab Pro+, credit-conscious; LLM work restricted to sampled subsets plus distillation |

**Split rule (derived from the parameters, not hard-coded):**
- **Walk-forward folds:** expanding window, minimum `MIN_TRAIN_YEARS` (default 8) of training data, validation blocks of `VAL_BLOCK_YEARS` (default 2), covering everything before the development holdout.
- **Development holdout:** the `DEV_YEARS` (default 3) immediately before the locked test.
- **Locked test:** from `LOCKED_TEST_START` to the last fully observed event. Default: the last ~15% of the window, rounded to a calendar-year start.
- **Post-cutoff clean test:** the part of the locked test after the latest knowledge cutoff of any pretrained model in the final architecture (Section 8). It is computed automatically.
- **Warnings:** the notebook warns if the locked test has fewer than 12 months or fewer than a configured number of events, or if the clean test is empty.
- **Purge and embargo** apply at every boundary (Section 12).

---

## 4. Research questions and pre-registered hypotheses

| # | Research question |
|---|---|
| RQ1 | How are filing content and tone associated with subsequent returns, abnormal returns, volatility and drawdowns? |
| RQ2 | Which sections, 8-K items, event types, risks, outlook statements and numbers are most associated with later moves (up / down / sideways)? |
| RQ3 | Which architecture gives the highest out-of-sample accuracy, and does it beat strong non-text baselines? |
| RQ4 | Which towers add incremental value: encoders, RAG-LLM features, analogue retrieval, time-series foundation models? |
| RQ5 | Which RAG design retrieves and extracts best, and does better retrieval translate into better price prediction? |
| RQ6 | How large is pretrained-model look-ahead bias? Compare point-in-time with standard models, anonymized with raw inputs, and pre- with post-cutoff periods |
| RQ7 | Do results vary by form, sector, regime, size and period, and survive alternative windows, thresholds and trading lags? |
| RQ8 | Correlations: filing ↔ filing and filing ↔ price |
| RQ9 | Evidence for each drift effect (PEAD, dividend, analyst revision, momentum, insider, institutional, spin-off/restructuring) |
| RQ10 | Are splits unbiased and leak-free; are models under- or over-fit? |

**Hypotheses** (frozen in `preregistration.json`, hashed and timestamped before any development-holdout or test scoring):
- **H1:** Filing text adds incremental out-of-sample information beyond a price-plus-metadata baseline.
- **H2:** 8-K event disclosures and selected 10-Q/10-K sections differ in predictive value.
- **H3:** Specific event and risk categories and *changes* in disclosure language are more informative than document-wide sentiment.
- **H4:** Retrieval-augmented structured extraction adds value beyond pooled encoder embeddings.
- **H5:** A stacked ensemble of towers beats the best single tower out of sample.
- **H6:** Point-in-time (vintage-matched) models perform close to standard models, which would mean look-ahead bias is small; a large gap would mean contamination.
- **H7:** Results differ by sector and regime.
- **H8:** Gains shrink under strict timestamps, a realistic trading lag, costs, multiple-testing correction and the untouched final test.

The pre-registration also fixes: the primary metric and horizon, the label rule and threshold `k`, the coverage levels for selective prediction, the question bank for RAG (Section 10.2), the test list for drift effects, and split rules. Every later deviation goes into the decision log.

---

## 5. SWOT (v3)

**Strengths**
- Official, timestamped source text with structured 8-K item codes and XBRL numbers.
- A multi-decade, cross-sectional event panel.
- A rich roster of domain models, including **point-in-time vintages** (ChronoBERT, FinText Chronos) that few student projects use.
- The architecture search is disciplined and protected against selection bias.
- Distillation makes LLM-grade features affordable.

**Weaknesses**
- Low signal-to-noise in returns.
- No analyst, insider or institutional data in core scope.
- Free price data is weak on delisted stocks.
- Many LLM checkpoints are old or small-context (for example FinMA is LLaMA-1-era).
- Human gold labels are limited by one person's time.
- Daily data cannot resolve intraday reactions.

**Opportunities**
- Semantic "what changed" signals.
- Analogue retrieval (what happened after similar past disclosures).
- Volatility prediction, where text typically helps most.
- Measuring look-ahead bias is itself a contribution.
- Optional licensed-data hooks can unlock the three untestable drift effects.
- A clean null result is still a valid, defensible finding.

**Threats**
- Leakage: intraday contamination, overlapping windows, retrieval of future documents, pretrained knowledge of outcomes, XBRL restatements.
- Selection bias from testing many configurations.
- LLM hallucination in extraction.
- GPU cost overruns.
- SEC access limits.
- Anomaly decay and regime shifts.
- Licence restrictions (several models are non-commercial or gated; fine for coursework, record them).

---

## 6. Architecture overview

```text
                ┌──────────────────────── SEC EDGAR (10-K / 10-Q / 8-K) ────────────────────────┐
                │  acceptance timestamps · item codes · sections · exhibits · XBRL facts       │
                └──────────────┬─────────────────────────────────────────────────┬──────────────┘
                               │ parsed, anonymized, section-aware chunks          │ prices, factors
     ┌───────────────┬─────────┴───────┬────────────────────┬──────────────────┐  │
     ▼               ▼                 ▼                    ▼                  ▼  ▼
 T2 Lexical     T3 Encoders      T4 RAG-LLM extraction  T5 Analogue retrieval  T1 Market/fundamental
 (dictionary,   (domain + point- (question bank → hybrid (point-in-time kNN    (price, vol, SUE/EAR,
  TF-IDF, BM25,  in-time encoders, retrieval → rerank →  over past events with  XBRL, sector, timing)
  change and     chunk → section  local LLM → JSON       realized outcomes)         │
  novelty)       vectors, change  fields; distilled to       │                 T6 Time-series FM
     │           vectors, fine-   a small student)           │                 (point-in-time Chronos
     │           tuned heads)          │                     │                  vintages, Chronos-2,
     │               │                 │                     │                  TimesFM; PatchTST/TFT)
     ▼               ▼                 ▼                     ▼                       ▼
  ───────────── base learners per tower (LightGBM / CatBoost / ElasticNet / MLP / fine-tuned heads) ─────────────
                                               │  time-respecting out-of-fold predictions
                                               ▼
                    Stacked meta-learner (regularized logistic or shallow GBDT) + calibration
                                               │
                                               ▼
              Direction probabilities · return / volatility forecasts · confidence (meta-label)
```

**Design principles**
- **Point in time:** every tower may use only information available before the event's decision time.
- **Exact time order:** out-of-fold predictions for stacking come from walk-forward folds, so the meta-learner never sees predictions made with future data.
- **Every tower must earn its place:** a tower is kept in the final architecture only if removing it lowers validation performance beyond noise (ablation with bootstrap confidence intervals).
- **One end-to-end challenger is also tested:** a multimodal Temporal Fusion Transformer with text embeddings as covariates, so the stacked design is compared against an integrated one.

---

## 7. Notebook stages

Each stage lists its purpose, cached outputs, built-in assertions and the notebook part it belongs to (Parts are defined in Section 17). Each stage logs its resource use.

### Part A: data and baselines (CPU)

**S0. Setup and pre-registration.**
- Mount Drive (optional) and create the folder tree.
- Load config; seed everything; log versions, hardware and model revisions.
- Read secrets.
- Define checkpoint and resume helpers, the GPU-time tracker, the dry-run estimator and the cache-key scheme (model revision + prompt hash + passage hash).
- Write `preregistration.json`, hash it and refuse to pass S12 without it.

**S1. Hugging Face SEC dataset audit.** Audit the live card and schema of `JanosAudran/financial-reports-sec` (configs, forms present, date range, counts by year and form, CIKs, duplicates, nulls). Use it as an audit and cross-check source only. Never use its return labels. Output: `hf_dataset_audit.md`.

**S2. Universe construction.**
- Build from historical index membership; map ticker → CIK through SEC data, preserving ticker changes.
- Eligibility: minimum overlap of prices and filings.
- Stratified sample to `N_STOCKS`.
- Report target, eligible, successful and excluded counts with reasons; output `ticker_cik_mapping_issues.csv`.

**S3. Prices and benchmarks.**
- Download adjusted OHLCV, SPY and Fama-French daily factors; cache per ticker; log coverage and the delisted flag.
- Preprocessing: winsorize for modelling (raw values kept); no long forward-fills.
- Outputs: `prices.parquet`, `factors.parquet`, `missing_market_price_periods.csv`.

**S4. EDGAR filing index and completeness audit.**
- Pull the full submissions history: accession, form, filing date, **acceptance date-time**, period, primary document, 8-K item codes.
- Label coverage as observed / suspected gap / verified missing.
- Outputs: `filing_manifest.parquet`, coverage tables, coverage heatmap, `data_coverage_summary.md`.

**S5. Download, gap-fill and parse.** This stage adapts steps 2–19 of the owner's 40-step pipeline (Appendix A).
- **Routing:** route documents by format (HTML, plain text, XBRL). Optical character recognition is only a fallback for the rare image-only exhibit; EDGAR filings are electronic text.
- **Sections:**
  - 10-K: Items 1A and 7 (optional 1 and 7A)
  - 10-Q: Part I Item 2 and Part II Item 1A
  - 8-K: item text plus press-release exhibits (EX-99.x)
- **Layout clean-up:** strip page numbers, repeated headers and footers, cover pages, signatures and exhibit lists. Rebuild reading order. **Separate tables from narrative**: tables go to a table store (serialized text plus parsed numeric cells) for numeric features and table-aware retrieval.
- **Normalization:**
  - Unicode clean-up and broken ligatures.
  - Normalize numbers and currency to floats with units (thousands or millions as stated).
  - Resolve relative dates ("last quarter", "fiscal 2023") to ISO ranges using the filing's period of report.
  - Expand a curated finance acronym list (light touch; keep the original text too).
  - Sentence segmentation that survives abbreviations and decimals.
- **Anonymization layer** (Section 8.3): mask company names, tickers, subsidiaries and executive names, and optionally shift exact dates to relative form. Keep both raw and masked text. Models are evaluated on the masked text by default.
- **Gap-fill and dedup:** retry missing filings with alternate URLs and parsers; dedup by accession plus checksum; detect near-duplicates.
- **8-K item groups:** map modern and legacy item numbering:

  | Group | Items |
  |---|---|
  | EARNINGS | 2.02 and legacy equivalents |
  | DEAL_MA | 1.01, 1.02, 2.01 |
  | RESTRUCTURING | 2.05, 2.06 |
  | FINANCING | 2.03, 2.04, 3.02 |
  | LISTING | 3.01 |
  | AUDITOR/RESTATEMENT | 4.01, 4.02 |
  | MANAGEMENT | 5.02 |
  | CYBER | 1.05, from Dec 2023 |
  | OTHER_DISCLOSURE | 7.01, 8.01, subtyped later |
  | Dropped | 5.07, 9.01 alone |

- **Outputs:** `filings_text.parquet`, `filings_tables.parquet`, parse-quality table.

**S6. Structured numbers and change features.**
- **XBRL facts:** use only those with filed date ≤ event date.
- **Earnings surprise:**
  - SUE (seasonal random walk) where XBRL allows; EAR for all years. State plainly that neither is a consensus surprise.
  - **Deterministic numeric cross-check:** text-extracted numbers versus XBRL, with mismatches flagged (adapts step 32).
- **Lazy-Prices-style change features:** similarity to the same firm's prior same-section text (TF-IDF cosine now, embedding cosine in S11). Plus changes in length, negative-word share and risk-factor count.
- **Novelty features:** similarity of the 10-Q/10-K MD&A to the same quarter's earnings 8-K, and the earnings-cluster ID.
- **Availability ledger:** every feature gets an availability timestamp.

**S7. Event timing, targets, labels.**
- **Timing buckets (US Eastern, exchange calendar):**

  | Bucket | Rule | Day 0 | Base price |
  |---|---|---|---|
  | PRE | accepted on a trading day before 09:30 | that day | prior close |
  | INTRA | accepted on a trading day 09:30–16:00 | that day | day-0 close; reaction starts next session |
  | POST | accepted on a trading day at or after 16:00 | next trading day | acceptance-day close |
  | NONTRADING | accepted on a weekend or holiday | next trading day | last close before acceptance |

  - **Primary sample:** PRE + POST + NONTRADING. INTRA is a flagged separate stratum.
  - **Timestamp quality:** A = exact acceptance time; B = date only, treated as POST; C = uncertain.
- **Abnormal returns:** market model versus SPY with beta estimated on sessions −250 to −30 (minimum 120 observations); four-factor robustness check. Also realized volatility, drawdown and adverse excursion.
- **Placebo window:** sessions −5 to −1.
- **Targets:**
  - 3-class direction
  - winsorized cumulative abnormal return
  - volatility ratio: log of realized volatility over sessions 1–20 divided by sessions −60 to −5
  - standardized absolute move
- **Labels:**
  - Up if CAR > +k·σ·√h; Down if CAR < −k·σ·√h; otherwise Sideways. σ is point-in-time idiosyncratic volatility.
  - `k` is pre-registered (default 0.5); sensitivity at 0.25 and 0.75.
  - Print class balance by block, form, year, sector and horizon.
- **Sample weights:** weight by label uniqueness (down-weight events whose windows overlap heavily for the same firm); unweighted results are also reported.
- **Earnings date and dedup:** the earnings date is the earliest of {validated 8-K earnings release, 10-Q/10-K}. Same-firm same-day 8-Ks merge into one multi-label event. Earnings clusters are never counted as independent events.
- **Censoring:** drop events without a full horizon.
- **Tabular controls:** size proxy, trailing volatility, 1-month and 12-1-month momentum, pre-event run-up, SUE/EAR, form, item group, sector, year, bucket, timestamp quality, market-volatility regime.

**S8. Exploratory analysis and data-readiness verdict.** Coverage, label sanity and event-study plots. Print the readiness verdict.

**S9. Split design, leakage audit, self-tests.**
- Implement Section 12, including the feature-availability table, retrieval-index time audit and pretrained-model cutoff ledger.
- **Planted-leak self-test:** inject a future-return feature and a future-document retrieval; both must be caught.
- The notebook stops if any assertion fails.

**S10. Tower T1 + T2 baselines (the ladder).** Identical splits and metrics for every rung:
1. majority class / class prior
2. metadata only
3. price only
4. price + metadata
5. \+ Loughran-McDonald dictionary features
6. \+ TF-IDF / BM25-weighted logistic regression
7. \+ change and novelty features
8. \+ SUE/EAR

Tabular learners: LightGBM, CatBoost and ElasticNet-logistic as core; XGBoost, Random Forest and SVM/SVR as challengers. Every later tower must beat rung 8 to be called "text value-add".

**S11. Gold set export and import** (Section 15). Export the unified annotation sample and template. The notebook must still run without labels; detector-dependent results are then marked *provisional*.

### Part B: encoders and time-series models (GPU)

**S12. Tower T3, encoder embeddings (screening, frozen).**
- **Chunking:** section-aware hierarchical chunks; small child chunks (~256–512 tokens) linked to larger parent sections (adapts steps 21–23). Each chunk carries metadata: accession, CIK, form, section, period, acceptance time.
- **Context limits:** respect each model's maximum context. Long-context models (ModernBERT 8k, Qwen3-Embedding 32k) get whole sections where affordable.
- **Pooling:** mean, max and **attention pooling over chunk vectors** (a small learned pooling head) to produce section and document vectors.
- **Change vectors:** current section vector minus the prior same-section vector, with cosine distance as an extra feature.
- **Point-in-time variant:** **vintage-matched ChronoBERT**. Embed an event filed in year Y with the vintage whose cutoff is 31 Dec of year Y−1.
- **Caching:** store float16 caches. Run the timed benchmark and dry-run first.

**S13. Tower T6, time-series features.**
- **Zero-shot forecasts** of next-session and 20-session return quantiles and realized volatility from the pre-event price window:
  - **Point-in-time FinText Chronos vintages:** use the vintage trained only on data up to year Y−1.
  - **General Chronos-2 / Chronos-Bolt.**
  - **TimesFM** (only where its training-data cutoff precedes the evaluation window, Section 8).
- **Trainable challengers** on our training folds only: PatchTST, LSTM/GRU, N-BEATS.
- **Hybrid challenger:** a Temporal Fusion Transformer with text embeddings and event metadata as static/known covariates (the end-to-end multimodal option).
- **Summary features:** forecast mean, quantile spread and forecast volatility.

**S14. Tower T3 tuning and fine-tuning.**
- **Tier 1:** tuned heads on cached embeddings (Optuna on walk-forward folds), "default" and "tuned" both recorded.
- **Gate G3:** best text model beats rung 8 by `TEXT_LIFT_GATE`.
- **Tier 2:** fine-tune the top 2–3 encoders over the small fixed grid (Section 13), with multi-task heads (direction + |move| + volatility).
- **Tier 3 (optional, gated):**
  - **Domain-adaptive continued pretraining** (masked-language modelling) of the best encoder, on **training-period filings only**, then re-fine-tune.
  - **QLoRA classification fine-tune** of the best 7–8B LLM using retrieved evidence as input.

### Part C: RAG and LLM tower (GPU, heaviest; sampled)

**S15. Indexes.**
- **Sparse:** BM25 over chunks.
- **Dense:** one dense index per embedding model in the RAG tournament (local FAISS or LanceDB files, no servers).
- **Table index** from S5.
- **Entity graph** for GraphRAG (S16, built only for its subsample).
- **Time partitioning:** all indexes are partitioned by acceptance time, and every query carries a time filter (Section 12).

**S16. RAG tournament** (Section 10) on the RAG dev set. Retrieval-only variants are scored first (cheap); the top variants then go through LLM extraction; heavy variants (GraphRAG, Agentic, Self-RAG) run only on a small slice. Choose the winning RAG design plus the winning generator LLM.

**S17. Teacher extraction and distillation.**
- The winning RAG + LLM (teacher) extracts the question-bank fields as schema-validated JSON for a stratified **teacher sample** (size set by budget, for example a few thousand filings across eras, forms and sectors).
- **Guardrails** (adapts steps 32–35):
  - schema validation with retry
  - deterministic numeric cross-check against XBRL
  - faithfulness check: every field must cite an evidence chunk that supports it, or the field is set to null
  - citations as accession + section + character offsets
- **Students** (FinBERT/ModernBERT-class encoders) are trained per field on teacher outputs from training-period filings only, then applied to the full corpus.
- **Report:** student-versus-teacher agreement and student-versus-human accuracy on the gold set. Features come from the student everywhere, plus teacher features where available as an ablation.

**S18. Tower T5, analogue retrieval** (retrieval-augmented prediction).
- **Query:** the embedding (and extracted-field vector) of the new filing.
- **Index:** past events whose **outcome window had fully closed before the query's decision time**. Exclude the same earnings cluster.
- **Features:**
  - similarity-weighted mean abnormal return of the k neighbours
  - neighbour label distribution
  - mean similarity
  - neighbour count
  - same-firm and cross-firm variants
- `k` and the similarity temperature are tuned on validation folds only.

**S19. Generation outputs and their evaluation** (Section 14.3):
- "What changed in this filing" summaries
- evidence-grounded explanations of the top model's predictions
- RAG answers to the question bank

Scored against the gold references and with a local LLM judge calibrated on human ratings. These outputs are for interpretation; they are **never** used as features unless they pass the same point-in-time rules.

### Part D: fusion, evaluation, analysis (CPU)

**S20. Architecture search and stacking** (Section 11): successive-halving results, the Caruana-style greedy ensemble and the stacked meta-learner on time-respecting out-of-fold predictions; calibration; meta-labelling; selective prediction.

**S21. Development holdout, freeze, locked test.**
- Score every finalist, default and tuned, on the development holdout.
- Write the Model Confidence Set and Probability of Backtest Overfitting results.
- **Gate G4:** freeze the manifest (architecture, hyperparameters, model revisions) and hash it.
- Retrain on all pre-test data with purge.
- Score the **locked test once**; report the **post-cutoff clean test** separately.
- Add trading-lag (next open) and cost sensitivity.

**S22. Diagnostics.** Loss, validation and learning curves; fit verdicts (Section 13.3); calibration; residuals. Neural optimisation is non-convex, so describe results as "validation-selected", never "global optimum".

**S23. What moves prices; correlations; NLP patterns.**
- **By filing type:** information content by item group, section and form (mean signed and absolute CAR, volume and volatility effects, with confidence intervals and FDR-adjusted p-values, plus the placebo window).
- **Correlations:** filing ↔ filing (tone persistence, earnings 8-K versus the following 10-Q, lead-lag) and filing ↔ price (Spearman by group, partial correlations, mutual information).
- **Interpretation tools:**
  - topic models fitted on training data only
  - SHAP on tree learners
  - stacking weights per tower
  - per-field effects of RAG-extracted events, for example "guidance lowered" → mean CAR
  - token attribution for examples (diagnostic, not causal)
  - error analysis
  - ranked "what pushes Up / Down / Sideways" lists with effect sizes

**S24. Drift and anomaly tests** (Section 16).

**S25. Final report and exports** (Section 19).

---

## 8. Look-ahead control for pretrained models

A model whose pretraining data extends past an event may have read text written *after* that event. In effect, it may have read the answer. This matters most for LLMs and time-series foundation models and less for small encoders, but it is measured for all of them.

### 8.1 Knowledge-cutoff ledger
The notebook keeps a table with these fields for every model:
- model ID and revision
- stated pretraining-data end date (or "unknown")
- fine-tuning data and its dates
- source of the information
- risk tier: point-in-time / pre-window / overlapping / unknown

Unknown counts as overlapping.

### 8.2 Controls
1. **Vintage matching** (strongest control): ChronoBERT vintages (`manelalab/chrono-bert-v1-YYYY1231`, one per year 1999–2024) for text, and FinText Chronos vintages (`FinText/Chronos_*_YYYY_US`) for prices. An event in year Y uses the vintage ending Y−1. Where vintages stop (2024), later events use the last vintage, which is still point-in-time for them.
2. **Post-cutoff clean test:** the locked-test subset after every final-architecture model's cutoff. This is the headline result whenever any non-point-in-time LLM or time-series model is in the final architecture.
3. **Anonymization:** mask firm identity (names, tickers, subsidiaries, executives) and absolute dates by default. Research on GPT sentiment found that anonymized inputs did *better* in-sample, pointing to "distraction" by firm knowledge; anonymization therefore helps both bias and accuracy.
4. **Memorization probes:**
   - (a) **Perplexity test:** compare each LM's perplexity on held-out filings dated before versus after its cutoff (lower pre-cutoff perplexity beyond the trend suggests memorization).
   - (b) **Lookahead-propensity test:** estimate how likely the model is to recall the realized outcome from a date-only query, then test whether prediction accuracy rises with that propensity. A positive interaction indicates contamination.
5. **Head-to-head:** vintage-matched ChronoBERT versus standard FinBERT/SEC-BERT under identical heads (tests H6). Published evidence on returns found look-ahead bias modest for encoder embeddings; the study tests that here rather than assuming it.

### 8.3 Anonymization specification
- **Replacements:** names, tickers and known aliases from the SEC company record → "[COMPANY]"; subsidiaries detected by NER → "[SUBSIDIARY]"; executives → "[PERSON]".
- **Dates:** absolute dates are converted to period-relative form where feasible.
- **Masking quality:** report the masking recall on a small audited sample.
- **Ablation:** raw versus masked inputs on the validation folds.

---

## 9. Model roster: every requested model, verified, with a verdict

**Verdicts:**
- **Core:** enters the tournament by default.
- **Challenger:** enters screening; advances only if it ranks well.
- **Feature:** produces features, not a predictor on its own.
- **Optional:** off by default; budget-gated.
- **Excluded:** with the reason given.

Re-verify everything at build time (rule 6).

### 9.1 Encoders (Tower T3; also students and rerankers)

| Requested | Checkpoint (verified) | Verdict | Notes |
|---|---|---|---|
| FinBERT | `ProsusAI/finbert` | Core | Financial sentiment; also the "FinBERT (TRC2)" family (further pretrained on Reuters news by its authors) |
| FinBERT-Tone | `yiyanghkust/finbert-tone` | Core | Analyst-report tone |
| FinBERT-pretrain | `yiyanghkust/finbert-pretrain` | Core | Masked-language model incl. 10-K/10-Q text; no head; license unstated |
| SEC-BERT | `nlpaueb/sec-bert-base` (+ `-num`, `-shape` variants) | Core | ~260k 10-Ks 1993–2019; CC-BY-SA-4.0. **This is the SEC model** |
| SecBERT / SecRoBERTa | `jackaduma/SecBERT`, `jackaduma/SecRoBERTa` | **Excluded** | **Cybersecurity** models (trained on APTnotes, CASIE etc.), not SEC filings. Name confusion |
| Financial-RoBERTa | `soleimanian/financial-roberta-large-sentiment` | Challenger | RoBERTa-large financial sentiment (~355M; slower) |
| FLANG-BERT | `SALT-NLP/FLANG-BERT` | Challenger | Finance-adapted masked-language model (fill-mask); needs a head |
| FLANG-RoBERTa | `SALT-NLP/FLANG-Roberta` | Challenger | Same; FLANG-ELECTRA also exists (optional) |
| DeBERTa-v3-Finance | `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` | Challenger | DeBERTa-v3 fine-tuned on financial-news sentiment (model only; no news data used here) |
| Sigma-Financial-Sentiment | `Sigma/financial-sentiment-analysis` | Challenger (low priority) | Small community fine-tune; documentation thin |
| FinBERT-ESG | `yiyanghkust/finbert-esg` | Feature | E / S / G / None on ~2k annotated sentences; a 9-category variant may exist (verify) |
| FinBERT-FLS | `yiyanghkust/finbert-fls` | Feature | Specific / non-specific / not forward-looking; MD&A sentences |
| BBER-Finance (BB-BERT) | none found | **Excluded** | No public checkpoint (Bloomberg-internal) |
| MiniLM-Finance | no canonical model | Replaced | Use `sentence-transformers/all-MiniLM-L6-v2` as a fast baseline and fine-tune our own finance version on our labels |
| distilBERT-Finance | no canonical model | Replaced | Fine-tune `distilbert-base-uncased` ourselves (honest "distilBERT-Finance"); also a candidate distillation student |
| (added) ChronoBERT | `manelalab/chrono-bert-v1-YYYY1231` | **Core** | Point-in-time vintages 1999–2024; 1,024-token context; key for look-ahead control |
| (added) ModernBERT | `answerdotai/ModernBERT-base` / `-large` | Core | 8,192-token native context; Apache-2.0; candidate student |
| (added) FinModernBERT-embed | `BatuhanECB/FinModernBERT-embed-large-v1` | Challenger (embedding) | ModernBERT-large adapted on finance incl. SEC text; trained at 512 tokens |
| (added) Qwen3-Embedding | `Qwen/Qwen3-Embedding-0.6B` | Core (retrieval) | 32k context, instruction-aware, flexible dimension; Apache-2.0 |
| (added) BGE | `BAAI/bge-base-en-v1.5`, `bge-large-en-v1.5` | Core (retrieval) | Strong general embedders |
| (added) MPNet | `sentence-transformers/all-mpnet-base-v2` | Challenger | Balanced baseline |
| (added) Reranker | `BAAI/bge-reranker-v2-m3` or `bge-reranker-base` (verify) | Core (RAG) | Cross-encoder reranking (adapts step 30) |
| (added) Zero-shot NLI | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | Feature | Zero-shot event typing on lead passages; ~512-token limit |
| (added) DistilRoBERTa | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | Challenger | Fast, Apache-2.0 |
| (optional) E5-Mistral-7B | `intfloat/e5-mistral-7b-instruct` | Optional | Heavy 7B embedder; subsample only |

### 9.2 Decoder LLMs (Tower T4 generators and judges; optional QLoRA classifier)

| Requested | Checkpoint (verified) | Verdict | Notes |
|---|---|---|---|
| FinGPT | `FinGPT/fingpt-mt_llama3-8b_lora` (multi-task), `FinGPT/fingpt-sentiment_llama2-13b_lora` | Challenger | LoRA adapters on Llama bases; also forecaster adapters (Dow30) |
| InvestLM | `yixuantt/InvestLM-awq` | Optional | LLaMA-65B, 4-bit AWQ; needs an ~80 GB GPU; skip on 40 GB |
| FinMA | `ChanceFocus/finma-7b-full` / `finma-7b-nlp` | Challenger (low) | PIXIU project, LLaMA-based, gated; short context and an older base limit its RAG use |
| FinLlama | no public weights found | **Excluded** unless weights are located | Paper (Llama-2-7B sentiment fine-tune) only |
| AdaptLLM-Finance | `AdaptLLM/finance-chat` (Llama-2-7B), `instruction-pretrain/finance-Llama3-8B` | Core | Finance continued-pretraining; prefer the Llama-3 version |
| BloombergGPT | not released | **Excluded** | No public weights |
| Qwen-2.5-Coder-Finance | no canonical model | **Excluded** | Coder models are irrelevant; use `Qwen/Qwen2.5-7B-Instruct` as general baseline instead |
| Llama-2-Banking-FineTune | no canonical model (Bitext banking models are customer-service chatbots) | **Excluded** | Wrong task |
| Sujet-Finance-Instruct | `sujet-ai/Sujet-Finance-8B-v0.1` (Llama-3-8B LoRA; Apache-2.0) | Challenger | Finance QA / topic / sentiment |
| DianJin-R1 | `DianJin/DianJin-R1-7B` (Qwen2.5-7B; MIT) | Optional (low) | Chinese financial/compliance reasoning focus |
| Fin-R1 | `SUFE-AIFLM-Lab/Fin-R1` (Qwen2.5-7B; Apache-2.0) | Challenger | Bilingual financial reasoning; long reasoning outputs are costly, so cap tokens |
| (added) General baselines | `meta-llama/Llama-3.1-8B-Instruct` (gated), `Qwen/Qwen2.5-7B-Instruct` | Core | Strong instruction following and long context, needed to judge whether "finance-tuned" really helps |
| (added) Point-in-time LLM | `manelalab/chrono-gpt-instruct-v1-YYYY1231` | Feature | Look-ahead-free perplexity "surprise" features and zero-shot probes; small model |
| (added) Self-RAG | `selfrag/selfrag_llama2_7b` (MIT) | Optional | Only for the Self-RAG variant |
| Command R+ | `CohereForAI/c4ai-command-r-plus` | **Excluded** | 104B and non-commercial; far beyond budget |

**Serving:** local offline batch inference (vLLM if compatible, otherwise Transformers), bf16 or 4-bit quantization, **constrained JSON decoding** for extraction, and a response cache keyed by (model revision, prompt hash, passage hash) (adapts step 38).

### 9.3 Time-series, multimodal and hybrid (Tower T6)

| Requested | Checkpoint / approach (verified) | Verdict | Notes |
|---|---|---|---|
| FinText-TSFM / "Chronos-Finance" | `FinText/Chronos_{Mini,Small}_YYYY_US` | **Core** | Chronos pretrained on US excess-return series up to year YYYY; point-in-time vintages; Apache-2.0 |
| TimesFM-Finance | `pfnet/timesfm-1.0-200m-fin` | Optional | Financial fine-tune of TimesFM; CC BY-NC-SA; training-data dates not on the card → treat as overlapping; post-cutoff evaluation only |
| (general) Chronos-2 / Chronos-Bolt | `amazon/chronos-2`, `amazon/chronos-bolt-*` | Challenger | Covariate-aware (Chronos-2); training-data composition only partly documented → overlapping tier |
| (general) TimesFM | `google/timesfm-2.x` (verify current ID) | Challenger | Same caveat |
| Open-FinLLM Multimodal | `TheFinAI/FinLLaVA` (image + text) | **Excluded** | Filings carry little image content. Chart-image modelling belongs in the owner's separate vision project; license unstated |
| PRAGMA | Revolut banking-event foundation model (2026) | **Excluded** | Built for bank transaction sequences (credit, fraud), not equities or filings; no public weights found |
| FinMultiTime Transformer | no such model; FinMultiTime is a *dataset* (news + tables + charts + prices) | **Excluded as named** | Its benchmark idea (small Transformer / TimesNet over aligned modalities) is replicated by our TFT hybrid **without news** |
| FNSPID Multimodal Transformer | FNSPID is a *news* + price dataset | **Excluded** | Conflicts with the SEC-only rule |
| (added) PatchTST, TFT, LSTM/GRU, N-BEATS | trained from scratch on our folds | Challenger | TFT with text covariates = the end-to-end hybrid. Informer optional. LOB-based GRU not applicable (no order-book data) |

### 9.4 Tabular learners (T1 and stacking)

| Model | Verdict |
|---|---|
| LightGBM, CatBoost, ElasticNet / regularized logistic regression | Core |
| XGBoost, Random Forest, SVM/SVR (on reduced features), MLP | Challenger |

---

## 10. RAG design and tournament

### 10.1 Why RAG here
Filings are long. A 10-K MD&A plus risk factors often exceeds tens of thousands of tokens, while most encoders read 512–1,024 tokens. RAG turns a filing into **targeted evidence** for a fixed question bank, and a local LLM converts that evidence into **structured, auditable features**. RAG does not predict prices by itself. It feeds Tower T4, and its quality is judged by (a) retrieval metrics, (b) extraction accuracy and (c) the **price-prediction lift** of its features. (c) is decisive.

### 10.2 Question bank (pre-registered, ~20 items)
Each question has a JSON schema with these fields:
- `present` (bool)
- `direction` (+ / − / 0)
- `magnitude` (number + unit, or null)
- `certainty` (stated / hedged)
- `horizon` (current / forward-looking)
- `evidence` (chunk IDs + character offsets)

Topics:
- earnings beat/miss language
- guidance initiated / raised / lowered / withdrawn / reaffirmed
- revenue and margin drivers
- impairment or write-down
- restructuring or layoffs
- going-concern or liquidity stress
- covenant issues and new financing
- M&A
- dividend change and type
- buyback
- litigation or regulatory action
- auditor change or restatement
- management change
- customer or supplier concentration
- segment changes
- new or removed risk factors
- cybersecurity incident
- supply chain
- demand outlook

### 10.3 Variants

| Variant | Implementation here | Cost tier | Default |
|---|---|---|---|
| **Naive RAG** | Dense top-k chunks (single embedder) → LLM | Low | Run; baseline |
| **Simple RAG with Memory** | Adapted as **firm memory**: the retriever also returns the firm's prior extracted facts (prior guidance, prior risk factors), point-in-time, so the LLM reports *changes* | Low | Run |
| **Hybrid RAG** | BM25 + dense, fused with reciprocal-rank fusion; metadata filters (form, section, period) | Low | Run |
| **Advanced RAG** | Hybrid + parent-child chunks + cross-encoder reranking + query rewriting | Medium | Run |
| **HyDE** | Generate a hypothetical answer passage per question, embed it, retrieve. The question bank is fixed, so hypothetical passages are generated **once per question**, not per filing | Low | Run |
| **Corrective RAG (CRAG)** | A retrieval evaluator scores evidence; if weak, widen scope (other sections, exhibits, tables, prior filing) and retry. **No web-search fallback** (would break the SEC-only and point-in-time rules) | Medium | Run on dev set |
| **Self-RAG** | `selfrag_llama2_7b` with reflection tokens, or reflection by prompting the chosen LLM | High | Small slice only |
| **GraphRAG** | Per-firm entity-relation graph (company–segment–subsidiary–metric–event) built by LLM / NER on the slice; graph-neighbourhood retrieval (a lighter LightRAG-style graph is acceptable) | Very high | Small slice only |
| **Multi-Hop RAG** | Chained retrieval across documents: earnings 8-K → same-quarter 10-Q MD&A → prior 10-K risk factors, strictly point-in-time | Medium | Run on dev set |
| **Adaptive RAG** | A router (small classifier) chooses no retrieval / single-step / multi-hop per question and document length | Medium | Run on dev set |
| **Agentic RAG** | Tool-using agent with tools: section retriever, table / XBRL lookup, calculator (deterministic), prior-filing diff; step budget capped | Very high | Small slice only |
| **Multimodal RAG** | Adapted as **text + table** retrieval (serialized tables and parsed cells from S5); images are rare in filings and skipped | Medium | Run on dev set |
| **Modular RAG** | The framework itself: retriever, reranker, router, generator, verifier as swappable modules. Every variant above is a configuration of it | n/a | Always |

**Retrieval embedders compared inside the tournament:** Qwen3-Embedding-0.6B, BGE-base/large, FinModernBERT-embed, all-MiniLM-L6-v2 and all-mpnet-base-v2 (E5-Mistral-7B optional). FinMTEB research found that general-benchmark rankings poorly predict finance performance and that bag-of-words can beat dense models on some finance tasks, so BM25 stays in every comparison.

**Generators compared:** the 9.2 Core plus Challenger LLMs, screened on the dev set with the cheapest strong RAG variant first.

### 10.4 Tournament protocol
1. **RAG dev set:** a stratified sample of filings (eras × forms × sectors), drawn from training-period years only, and overlapping the gold set (Section 15).
2. **Round 1 (retrieval only, no LLM):** all retrieval configurations scored on the gold evidence passages (Section 14.4). Keep the top 4.
3. **Round 2 (end to end):** top retrieval configurations × top 3 generators, scored on extraction accuracy, faithfulness and context metrics. Keep the top 2.
4. **Round 3 (heavy variants, small slice):** Self-RAG, GraphRAG and Agentic are compared against the round-2 winners on the same slice. A heavy variant advances only if it beats them beyond the bootstrap CI **and** its cost per filing fits the budget.
5. **Round 4 (downstream):** features from the top 2 pipelines (teacher sample → students) are added to the stack. The pipeline with the larger validation lift wins.
6. Log the cost per filing (GPU seconds, tokens) for every variant; plot accuracy versus cost.

---

## 11. Architecture-search protocol

1. **Search space:**
   - tower subsets
   - per-tower base learners
   - encoder choice and pooling
   - RAG pipeline
   - time-series model
   - fusion method: stacking with logistic regression or shallow GBDT; Caruana greedy ensemble; end-to-end TFT hybrid
   - label weighting
   - anonymized versus raw input
2. **Successive halving under a fixed budget:**
   - **Stage 0, screening:** cheap proxies (frozen features + logistic head) on a time-stratified subsample of training folds.
   - **Stage 1:** tuned heads on full folds for the top third.
   - **Stage 2:** fine-tuning or LLM work for the top 2–3 per family.
   - **Stage 3:** ensemble selection.

   The budget per stage is set in config and logged.
3. **Out-of-fold predictions** for every surviving base learner come from walk-forward folds only (each fold's prediction is made by a model trained strictly earlier). The meta-learner is trained on those OOF predictions, in time order.
4. **Ensemble methods:** stacked meta-learner, Caruana greedy forward selection with replacement, simple probability averaging across seeds and folds. Pick by validation macro-F1 with the log-loss guardrail.
5. **Calibration:** temperature or isotonic scaling fitted on the most recent validation fold only, then frozen.
6. **Meta-labelling and selective prediction:** a secondary classifier predicts whether the primary prediction will be correct. Report metrics at the pre-registered coverage levels.
7. **Selection-bias controls** (all reported):
   - **Model Confidence Set** (Hansen-Lunde-Nason) over finalists on the development holdout.
   - **Probability of Backtest Overfitting** via combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado, Zhu) over all configurations tried.
   - **Deflated metrics:** adjust the best validation score for the number of configurations tried. For the long-short portfolio, use the Deflated Sharpe Ratio.
   - **Paired tests** (block bootstrap; Diebold-Mariano-style on log-loss) for "A beats B".
   - Log the full count of configurations tried.
8. **Ablation waterfall:** starting from the price + metadata baseline, add towers one at a time in pre-registered order and record the marginal gain with CIs. Then do leave-one-tower-out on the final ensemble.

---

## 12. Splits and leakage controls

### 12.1 Layout
Walk-forward folds → development holdout → locked test (Section 3), plus the post-cutoff clean test. Purge every training event whose forward window (base + 60 sessions) reaches the next block, and embargo `EMBARGO_CAL_DAYS` (default 90). Stress tests: firm-held-out and sector-held-out folds on the pre-holdout period. Retrain on all pre-test data only after the G4 freeze. If the locked block is ever inspected early, relabel it a development holdout.

### 12.2 Leakage checklist (each item has an assertion or report line)

| # | Risk | Control |
|---|---|---|
| 1 | Overlapping forward windows across blocks | Purge + embargo |
| 2 | Same-date shocks across splits | Date-disjoint splits; abnormal returns |
| 3 | Same filing or near-duplicate across splits | Split by accession; text hashes; amendments not events |
| 4 | Firm memorization | Anonymization; change features; firm-held-out test |
| 5 | Preprocessing fitted on future data (scalers, vocabularies, TF-IDF, BM25 statistics, topic models, PCA/UMAP, imputers, calibration, thresholds, class weights) | Fit on each fold's training indices only |
| 6 | Test peeking | Pre-registration; G4 manifest hash; `TEST_UNLOCK`; single scoring cell |
| 7 | Intraday contamination | Timing buckets; INTRA excluded from the primary sample |
| 8 | Event-time errors | Acceptance-timestamp rule; quality flag |
| 9 | Already-priced information | Earnings clusters; novelty; placebo window |
| 10 | XBRL restatements | Filed-date ≤ event-date rule |
| 11 | Survivorship | Historical membership; coverage reporting |
| 12 | Right-censored labels | Drop incomplete horizons |
| 13 | **Retrieval of future documents** | Every index query carries a time filter: acceptance < decision time; assertion on every retrieved chunk |
| 14 | **Analogue look-up of unrealized outcomes** | Neighbour outcome windows must close before the decision time; same cluster excluded |
| 15 | **Pretrained-model look-ahead** | Cutoff ledger; vintage matching; anonymization; post-cutoff clean test; memorization probes |
| 16 | **Distillation leakage** | Students trained only on teacher outputs for training-period filings; teacher prompts contain no outcomes |
| 17 | **Domain-adaptive pretraining leakage** | Continued pretraining only on training-period filings |
| 18 | **Time-series model leakage** | Point-in-time vintages; models trained by us are trained on training folds only |
| 19 | **Stacking leakage** | Meta-learner trained only on walk-forward OOF predictions |
| 20 | Outcome-based selection of documents | Forbidden (rule 7); assertion that event selection uses no price fields |
| 21 | Data snooping | Configuration counter; PBO; Model Confidence Set; FDR |
| 22 | Composition bias and covariate shift | Balance tables; adversarial validation |
| 23 | Labels hidden in file names, order, paths | Audit |
| 24 | Audit itself broken | **Planted-leak self-tests:** future-return feature, future-document retrieval, unrealized-neighbour outcome; all must be caught |

### 12.3 Feature-availability table
The table lists, for every feature: feature, tower, source, availability timestamp, and whether it can contain future information. The assertion is that every feature's availability timestamp is before its event's decision time.

### 12.4 Sanity experiments and red flags
- **Random versus temporal split:** run the same pipeline under a random split and report the inflation.
- **Permutation test:** shuffle labels within month; metrics should drop to chance.
- **Placebo prediction:** predicting sessions −5 to −1 should show no skill.
- **Red-flag rule:** if locked-test macro AUC exceeds ~0.65, macro-F1 far exceeds the baselines, or event-level R² is clearly positive at short horizons, **stop and audit before reporting**.

---

## 13. Tuning and fit diagnosis

### 13.1 Search spaces
| Model type | Tuned |
|---|---|
| Logistic / ElasticNet | C (log scale), L1 ratio, class weights |
| LightGBM / CatBoost / XGBoost | learning rate, trees (early stopping), leaves / depth, min child samples, feature and bagging fractions, L1/L2 |
| MLP / attention-pooling head | hidden sizes, dropout, weight decay, learning rate, batch size |
| Encoder fine-tuning (small fixed grid, 6–12 trials per model) | learning rate {1e-5, 2e-5, 3e-5}; batch size {8, 16}; epochs {1–3} with early stopping; weight decay {0, 0.01}; warmup {0, 0.06}; max length per checkpoint; multi-task loss weights {balanced, direction-heavy} |
| QLoRA LLM classifier (optional) | rank {8, 16}, alpha {16, 32}, learning rate {1e-4, 2e-4}, 1 epoch, evidence length cap |
| RAG | top-k {5, 10, 20}, chunk size {256, 512}, fusion weights, reranker on/off, rewrite on/off |
| Analogue retrieval | k {10, 25, 50, 100}, similarity temperature, same-firm weight |
| Time-series models | context length, patch length (PatchTST), hidden size, dropout, learning rate |

- **Heads:** Optuna (TPE sampler, median pruner, persisted study).
- **Fine-tuning:** a small fixed grid.
- **Objective:** mean walk-forward macro-F1 (tie-breaks: lower std, lower cost; prefer the simpler model within one bootstrap standard error).
- **Before/after:** record "default" and "tuned" results for every model.

### 13.2 Losses (document and justify)
- **Classification:** class-weighted cross-entropy; focal loss only if it beats CE on validation.
- **Regression:** Huber loss on winsorized targets.
- **Multi-task:** a weighted sum of the above.
- **Probabilistic forecasts:** quantile (pinball) loss for time-series models.
- **Tree heads:** multiclass log-loss.

Save train and validation loss per step and epoch, the learning-rate schedule, the best checkpoint, run time and peak memory.

### 13.3 Automated fit verdict (configurable thresholds)
| Verdict | Rule |
|---|---|
| Overfit | train − validation macro-F1 gap > 0.10, or validation loss rises 2 consecutive epochs while train loss falls |
| Underfit | train macro-F1 within 0.02 of the majority baseline |
| Unstable | fold-to-fold macro-F1 std > 0.03, or loss spikes > 50% |
| Good fit | none of the above |
In this domain, "underfit" may simply mean weak signal, not a too-small model.

---

## 14. Metric catalogue: every requested metric, mapped to where it is valid

**Principle:** price-prediction quality is measured only by classification, regression and finance metrics. Generation and similarity metrics measure the quality of generated text (summaries, explanations, RAG answers). Retrieval metrics measure evidence finding. A great BLEU score says nothing about predicting prices, and the report must never imply it does.

### 14.1 Prediction: classification (direction, detectors, extraction fields)
| Metric | Used for |
|---|---|
| Accuracy, balanced accuracy | Direction, detectors |
| Precision, recall, F1: per class, **macro** (primary), weighted | Direction, detectors, extraction fields |
| ROC-AUC (one-vs-rest macro and per class), PR-AUC | Direction |
| Cohen's kappa | Direction; also inter-annotator agreement on the gold set |
| Matthews correlation, log-loss, Brier, expected calibration error, reliability curves | Direction |
| Confusion matrices (counts and normalized) | Direction, detectors |
| Accuracy / F1 at fixed coverage; coverage-accuracy curve | Selective prediction |
| Precision@top-decile | Trading-relevant precision |

"Sequence labelling" in this project = token or span tagging (NER for anonymization and GraphRAG, evidence-span extraction), scored with span-level precision, recall and F1 (exact and partial match). Always state the averaging; mark a metric **null** (not zero) when a class is absent.

### 14.2 Prediction: regression, time series and finance
| Metric | Used for |
|---|---|
| R² (in- and out-of-sample; can be negative), OOS R² versus historical mean, MAE, RMSE | Return, volatility, |move| heads |
| Spearman / Pearson, rank IC per month, IC t-stat, ICIR, directional accuracy | Return ranking |
| QLIKE and MSE on realized variance | Volatility forecasts |
| Pinball / quantile loss, CRPS, MASE | Time-series model forecasts |
| Decile long-short return, Sharpe, **Deflated Sharpe**, turnover, max drawdown, under 10/20 bps costs and a one-session lag | Economic relevance |

### 14.3 Generation and similarity (summaries, explanations, RAG answers)
| Metric | Valid use here | Caveat |
|---|---|---|
| **ROUGE-N (1, 2), ROUGE-L** | "What changed" summaries versus human reference summaries | Rewards word overlap, not correctness |
| **BLEU** | Same references; also RAG short answers | Designed for translation; weak for summaries; reported for completeness |
| **NIST** | Same | A BLEU variant weighting informative n-grams; largely redundant with BLEU |
| **METEOR** | Same | Adds stemming and synonyms; somewhat better correlated with humans |
| **chrF / chrF++** | Same | Character-level; robust to number and morphology variants |
| **BERTScore** | Summaries and answers versus references | Semantic similarity; report the model used (e.g. a DeBERTa / RoBERTa-large backbone) and use baseline rescaling |
| **BLEURT** | Same | Learned quality metric; use a distilled BLEURT-20 checkpoint for cost (verify port) |
| **MoverScore** | Same | Earth-mover distance over contextual embeddings |
| **Perplexity** | (1) domain fit: each LM's perplexity on held-out *training-period* filings (lower = better fit; helps choose which LM to adapt); (2) **novelty feature**: filing perplexity under a point-in-time ChronoGPT vintage (surprising text → possibly bigger move); (3) memorization probe (Section 8.2) | Comparable only between models with the same tokenizer, or use bits-per-byte |
| **Numeric fidelity** (added) | Share of numbers in generated text that match the source or XBRL | Generation metrics miss wrong numbers, the most damaging error in finance |

References: human-written summaries from the gold set (Section 15). With fewer than ~50 references, report CIs and treat the comparison as indicative.

### 14.4 Information retrieval and RAG
| Metric | Valid use here |
|---|---|
| **MRR**, **MAP**, nDCG@k, Recall@k, Precision@k, Hit@k | Retrieval of gold evidence passages per question (gold qrels from Section 15) |
| **Context relevancy / context precision / context recall** | Share of retrieved context that is relevant, and share of needed evidence retrieved |
| **Faithfulness** (groundedness) | Share of generated claims or fields supported by the retrieved evidence. Scored by a local judge LLM (different from the generator), calibrated against ~100 human judgements, with agreement (kappa) reported; plus the deterministic numeric cross-check |
| Answer relevancy, answer correctness (field-level exact match; numeric match within tolerance) | RAG answers and extraction |
| Citation accuracy | Cited offsets actually contain the supporting text |
| Cost per filing, latency | Efficiency |

### 14.5 Uncertainty and multiple testing
- **Confidence intervals:** block bootstrap by month, clustered by firm where relevant, for every key metric.
- **Paired comparisons:** paired tests for model-vs-model and default-vs-tuned.
- **Multiple testing:** Benjamini-Hochberg FDR across test families; Model Confidence Set and PBO for architecture selection.
- **Context:** sample size and missingness are reported beside every important metric.

### 14.6 Results table schema (tidy; null where not applicable, never zero)
Columns:
- **Identity:** `run_id`, `config_id`, `tower`, `model_id`, `checkpoint_revision`, `rag_variant`, `generator_llm`, `fusion_method`
- **Task and slice:** `task`, `target_type`, `horizon`, `form_scope`, `timing_bucket`, `block`, `coverage_level`, `anonymized`, `point_in_time_model`, `tuning_status`, `best_hyperparameters`
- **Metric:** `metric_name`, `value`, `ci_low`, `ci_high`
- **Sample and cost:** `n_events`, `n_companies`, `gpu_seconds`, `tokens`, `peak_gpu_memory_mb`

(Long format, one metric per row, so generation, retrieval and prediction metrics live in one table.)

---

## 15. Unified gold set and human labelling

Labelling is the scarcest resource, so one annotation pass serves every need:

1. The notebook samples **gold filings** from the training period, stratified by era, form, sector and item group. Minimum viable: 100. Recommended: 200–300.
2. **For each filing, the annotator:**
   - (a) answers the question bank (fields + evidence spans), which yields extraction gold and retrieval qrels
   - (b) confirms or rejects detector flags: earnings release, dividend type, spin-off, guidance revision, M&A
   - (c) for ~60 filings, writes a 3–5 sentence "what changed" summary, which provides the generation references
   - (d) rates ~100 generated answers for faithfulness, which calibrates the judge LLM
3. **Optional LLM pre-fill** speeds the work. The annotator edits rather than writes, with anchoring risk noted. A random 20% are labelled blind (no pre-fill) to measure anchoring.
4. **Agreement:** two annotators are preferred (Cohen's kappa + adjudication). With one annotator, state the limitation.
5. **Detector threshold:** a detector with precision below `DETECTOR_MIN_PRECISION` (0.80) is *provisional*, and every result that depends on it carries the flag.
6. **No price-based validation:** price labels are never used to validate semantic correctness.
7. **No reuse in scoring:** gold-set filings are excluded from validation and test scoring of predictive models (or reported both ways).

---

## 16. Drift and anomaly evidence tests

**Method:**
- event-time abnormal returns with standard errors clustered by firm and date
- calendar-time portfolio alphas (four-factor) with Newey-West errors
- subperiods (three roughly equal eras of the window)
- size splits
- 10 and 20 bps costs and a one-session lag
- Benjamini-Hochberg FDR over the pre-registered test list
- cells with fewer than 30 events flagged "insufficient power"

**Status vocabulary:** supported, unsupported, inconclusive, not testable in scope. *Supported* requires an FDR-adjusted p < 0.05, the same sign in at least 2 of 3 subperiods, and survival under costs and the lag.

| Effect | Testable? | Design |
|---|---|---|
| **PEAD** | Yes (qualified) | Surprise = time-series SUE (XBRL era) and EAR (all years) + RAG-extracted beat/miss and guidance fields. CAR over sessions 3–60 by surprise quantile; calendar-time long-short; does text add value beyond SUE/EAR? Not a consensus-based surprise. |
| **Post-dividend announcement drift** | Partial | Validated detector + RAG fields (type, amount change); announcement date, never ex-date; report counts and power. |
| **Analyst revisions drift** | **No** | Not in SEC filings. A labelled *management-guidance-revision proxy* is reported separately if its detector validates. External-import hook for licensed analyst data. |
| **Short-term momentum** | Yes | 1-month reversal, 2–12 and 12–1 momentum; does momentum condition filing reactions; does text add value beyond momentum? |
| **Insider trading drift** | No (core) | Needs Forms 3/4/5. Optional module or external import. |
| **Institutional ownership drift** | No (core) | Needs 13F (filing date as information date). Optional module or external import. |
| **Spin-off / restructuring drift** | Partial | 8-K 2.05 / 2.06 / 2.01 + validated spin-off detection. Keep announcement / completion / ex-distribution dates distinct; map new tickers or restrict to the parent. |

**Optional modules (default off):** SEC bulk insider and 13F data; external-import hooks for user-supplied licensed files (documented column schemas). If off, the report says "not tested".

---

## 17. Compute and credit plan

### 17.1 Profiles
| Profile | Stocks | Window | Scope |
|---|---|---|---|
| `smoke` (default) | 8 | ~5 recent years | Every stage end to end with tiny settings: 1 encoder, 1 small LLM on ~20 filings, 2 RAG variants, 1 time-series model |
| `pilot` | ~40, sector-balanced | ~10 years | Screening of all Core / Challenger models on subsamples; RAG rounds 1–2; teacher sample small |
| `full` | `N_STOCKS` (TBD) | `STUDY_START`–`STUDY_END` (TBD) | Full tournament under budgets and gates |

### 17.2 Notebook parts
- **Part A (CPU):** S0–S11.
- **Part B (GPU, mid):** S12–S14.
- **Part C (GPU, high VRAM preferred):** S15–S19.
- **Part D (CPU):** S20–S25.

Each part checks that its inputs exist and stops otherwise.

### 17.3 Budgets and controls
1. **Config budgets:** `MAX_GPU_HOURS_B`, `MAX_GPU_HOURS_C`, `LLM_TEACHER_FILINGS`, `RAG_DEV_FILINGS`, `HEAVY_RAG_SLICE`, `MAX_TOKENS_PER_CALL`, `MAX_AGENT_STEPS`. Graceful stop with checkpoint when a budget is hit.
2. **Dry run before every expensive stage:** tokens, chunks, storage, model-download size and runtime from a timed benchmark batch. The user can abort.
3. **Cheapest first:** frozen embeddings before fine-tuning; retrieval-only RAG scoring before any LLM call; teacher LLM on samples, students for the full corpus.
4. **Precision and memory:** mixed precision, length-bucketed batching, 4-bit or AWQ for 7B+ models, offline batched generation, constrained decoding, an OOM auto-retry that halves the batch size.
5. **Caches:** caches keyed by config and content hashes; float16 embeddings; zstd-compressed text; no duplicate corpus copies; local `/content` disk for heavy temporary files, Drive for checkpoints and final artifacts.
6. **Resource summary:** the notebook prints a resource summary at the end of each part. It never hard-codes credit prices and points to the live Colab resource panel.

### 17.4 Gates
| Gate | Before | Passes when |
|---|---|---|
| G1 | smoke → pilot | all stages and artifact types ran; no assertion failures |
| G2 | pilot → full | coverage meets minimums; leakage assertions pass and planted leaks are caught; `N_STOCKS` and window set; dry-run fits budgets; user opts in |
| G3 | encoder screening → fine-tuning | best text model beats baseline rung 8 by `TEXT_LIFT_GATE` (0.01 macro-F1, CI excluding 0) |
| G3b | RAG round 2 → teacher extraction | best RAG pipeline reaches the minimum extraction F1 and faithfulness on the gold set (configurable, e.g. faithfulness ≥ 0.85) |
| G3c | heavy RAG / QLoRA / continued pretraining | the cheaper alternative's ceiling has been reached (no gain in the last stage) and the budget remains |
| G4 | locked test | manifest frozen and hashed; pre-registration verified |

---

## 18. Visualization catalog
(Static PNGs saved to `reports/figures/`, each with a "how to read" note. Axes, units, windows and sample sizes labelled.)

**Data and labels**
- V01 coverage heatmap
- V02 parse and fill rates
- V03 filing volume by form and item group
- V04 universe composition
- V05 text and token lengths
- V06 timestamp quality and timing-bucket mix
- V07 target distributions
- V08 class balance by slice
- V09 failures over time
- V10 event-time diagram

**Event study**
- V11 average CAR paths by form and item group
- V12 placebo window
- V13 sector × event heatmap
- V14 regime stability
- V15 controls-adjusted results
- V16 redundancy and dedup

**Splits and leakage**
- V17 split timeline incl. purge, embargo, dev holdout, locked and clean tests
- V18 split balance + adversarial-validation ROC
- V19 random versus temporal split
- V20 permutation test
- V21 planted-leak results
- V22 feature-availability audit summary

**Look-ahead**
- V23 knowledge-cutoff timeline of all models versus the study window
- V24 vintage-matched versus standard model performance by year
- V25 perplexity before / after cutoff (memorization probe)
- V26 lookahead-propensity interaction plot
- V27 anonymized versus raw

**Training and tuning**
- V28 loss curves (train / validation, default and tuned)
- V29 validation curves
- V30 learning curves
- V31 tuning history and parameter importance
- V32 before versus after tuning with CIs

**Architecture search**
- V33 successive-halving bracket chart
- V34 leaderboard by tower and model, with cost
- V35 ablation waterfall
- V36 leave-one-tower-out
- V37 stacking weights
- V38 Model Confidence Set membership
- V39 PBO distribution (logit of out-of-sample rank)
- V40 deflated-score table

**RAG**
- V41 retrieval metrics (MRR, MAP, nDCG, Recall@k) by variant and embedder
- V42 extraction F1 and faithfulness by variant × generator
- V43 accuracy versus cost frontier
- V44 judge-versus-human agreement
- V45 teacher versus student agreement per field
- V46 downstream lift per RAG pipeline

**Generation**
- V47 ROUGE / BLEU / METEOR / chrF / BERTScore / BLEURT / MoverScore panel with CIs
- V48 numeric-fidelity errors
- V49 perplexity domain-fit by LM

**Prediction quality**
- V50 confusion matrices
- V51 per-class P/R/F1
- V52 ROC and PR curves
- V53 calibration
- V54 coverage-accuracy curve (selective prediction)
- V55 performance by form, item group, bucket, timestamp quality, document length, sector, year, size
- V56 predicted versus realized + residual diagnostics (regression heads)
- V57 rolling IC
- V58 decile long-short incl. lag and costs
- V59 fit-verdict dashboard

**Insights**
- V60 information content by item group and section
- V61 correlation heatmaps (filing ↔ filing, filing ↔ price)
- V62 topic prevalence and topic → return
- V63 SHAP summary and top terms
- V64 RAG-field effect sizes ("guidance lowered → CAR")
- V65 embedding map
- V66 analogue-retrieval examples

**Drift**
- V67 detector validation
- V68 drift-effect dashboards
- V69 one-page executive dashboard

---

## 19. Final report (`results_summary.md`, generated from computed values only)

1. **Data readiness:** stocks (target, eligible, successful), filings expected / obtained / parsed by form and year, missing points, Hugging Face audit, known biases.
2. **Design:** timing buckets, abnormal returns, labels, class balance, pre-registration hash.
3. **Leakage audit:** assertions, planted leaks, sanity experiments, feature-availability table.
4. **Look-ahead audit:** cutoff ledger, vintage comparisons, anonymization effect, memorization probes; size of the clean test.
5. **Model roster:** every model tried, excluded (with reason), failed or gated out.
6. **RAG tournament:** results per round, winner, cost, judge calibration.
7. **Architecture search:** configurations tried (count), successive-halving results, final architecture diagram, stacking weights, ablations, Model Confidence Set, PBO, deflated scores.
8. **Tuning:** before / after table with significance.
9. **Results:** validation, development-holdout, locked-test and post-cutoff clean-test leaderboards with CIs, for all metric families. Selective-prediction results at pre-registered coverage. Sensitivity to lag, costs, `k`, INTRA and timestamp quality.
10. **Fit diagnostics:** curves and verdicts.
11. **What moves prices:** item groups, sections, RAG fields, topics, change and novelty, correlations.
12. **Drift verdicts:** status, counts, effects, CIs, adjusted p-values, detector flags.
13. **Generation quality:** metric panel with the caveat that it does not measure prediction.
14. **Limitations and resource use.**
15. **Decision log:** every choice made after seeing development-holdout results.
16. **Executive summary**, which must answer:
    - Which architecture is best, and by how much does it beat price-only and metadata-only baselines?
    - Which towers and models actually helped?
    - Did tuning help, and did it hold on the locked test?
    - How big is look-ahead bias?
    - Which RAG design won, and did better retrieval mean better prediction?
    - What is the probability the selected architecture is overfit?
    - **What is the smallest defensible claim the evidence supports?**

---

## 20. Folder structure

```text
sec-filing-nlp-study/
├── notebooks/sec_filings_price_impact_v3.ipynb
├── config/  experiment_config.json · preregistration.json · model_cutoff_ledger.csv · question_bank.json
├── data/    raw_sec/ · raw_market/ · processed_filings/ · tables/ · chunks/ · labels/ · manifests/
├── gold/    gold_sample.parquet · annotation_guide.md · labels_import/ · agreement_report.csv
├── indexes/ bm25/ · dense/<embedder>/ · tables/ · graph/
├── cache/   http/ · tokenized/ · embeddings/ · llm_responses/ · ts_forecasts/
├── checkpoints/ best/ · resumable/ · students/
├── reports/ data_coverage_summary.md · hf_dataset_audit.md · results_summary.md · decision_log.md
│            coverage_*.csv · missing_*.csv · ticker_cik_mapping_issues.csv · duplicate_and_amended_filings.csv
│            leakage_audit.csv · feature_availability.csv · lookahead_audit.csv · detector_validation.csv
│            rag_tournament.csv · architecture_search.csv · results_long.csv · hypothesis_tests.csv · figures/
├── logs/    retrieval.log · parsing.log · training.log · llm.log
└── manifests/ run_manifest.json · frozen_architecture.json
```

---

## 21. Acceptance checklist

**Run behaviour**
- [ ] `smoke` (default) runs top to bottom and produces every artifact type and plot.
- [ ] `full` refuses to run until `N_STOCKS`, `STUDY_START` and `STUDY_END` are set.

**Data and model sources**
- [ ] No news, social or transcript data anywhere.
- [ ] Every model ID was re-verified and pinned; exclusions are logged with reasons; the SEC-BERT vs SecBERT confusion is handled.

**Leakage and look-ahead**
- [ ] All Section 12 assertions pass; all three planted-leak self-tests are caught.
- [ ] Retrieval and analogue look-ups are time-filtered.
- [ ] The cutoff ledger is complete.
- [ ] The post-cutoff clean test is computed and reported.

**Pre-registration and test discipline**
- [ ] The pre-registration hash exists before development-holdout scoring.
- [ ] The manifest hash exists before the locked test.
- [ ] The locked test was scored exactly once.

**Reporting**
- [ ] Every requested metric is computed where valid; inapplicable metrics show null with a reason.
- [ ] Generation metrics are never presented as prediction accuracy.
- [ ] The RAG tournament, distillation agreement, Model Confidence Set, PBO and deflated scores are reported.
- [ ] Untestable drift effects are labelled; detector-based results carry validation flags.
- [ ] Every number in the report comes from executed cells; the red-flag rule is evaluated.

**Resilience and cost**
- [ ] Resumable after disconnects; budgets and gates are enforced; a resource summary prints at the end of each part.

---

## 22. Build and hand-off instructions for Claude Code

1. **Notebook skeleton:** generate a valid `.ipynb` with a table of contents, Part A–D banners (each stating which runtime to select), a config cell, a pinned-extras install cell, then S0 → S25.
2. **Helper code:** keep functions small; long helper code may live in a `src/` module the notebook writes to Drive on first run and imports, so the notebook stays readable.
3. **README cell:** put a short README in the first markdown cell: start in `smoke`, switch runtimes between parts, resume after a disconnect, label the gold set, set `N_STOCKS` and the window.
4. **Build order:** inspect live model cards and dataset schemas first; build the audit, labels, splits and baselines; then add towers one by one, each smoke-tested before the next.
5. **Unavailable sources:** if a model or source is unavailable or gated without access, log it, use the documented alternative, and never silently skip a requirement.
6. **Hand-off message:** list what you verified, what you could not run (GPU and LLM parts, live SEC at scale), substitutions, and deviations from this spec with reasons.

**Definition of done:** a reproducible, resumable, leakage- and look-ahead-audited notebook that:
- searches across encoders, RAG designs, LLMs, time-series models and tabular learners within a Colab budget
- selects an architecture with protection against selection bias
- confirms it once on an untouched test
- reports every metric where it is valid
- clearly separates supported findings from what SEC filings alone cannot test

---

## Appendix A: The owner's 40-step pipeline, mapped to this study

| Steps | Original idea | Decision here |
|---|---|---|
| 1 | GPU clusters + Kubernetes | **Replaced** by Colab Pro+ runtimes (CPU / GPU parts) |
| 2 | API connectors (EDGAR, terminals, data lakes) | **Adapted:** EDGAR submissions, archives and XBRL APIs only |
| 3 | Proxy rotation to avoid blocks | **Rejected:** violates SEC fair-access policy. Use a declared User-Agent, throttling and backoff |
| 4 | Download based on market triggers | **Rejected:** selecting documents by price moves is outcome leakage. Download by schedule or index for the whole universe |
| 5 | Route by format (PDF / HTML / XML) | Adopted (HTML, text, XBRL) |
| 6 | Index metadata at ingestion | Adopted, plus acceptance timestamp and item codes |
| 7 | Immutable raw store | Adopted in light form: content-hashed raw cache + manifest |
| 8 | Classify document types | Adopted: form type from EDGAR, plus section and exhibit classification |
| 9–10 | Layout CV models, OCR | **Fallback only:** EDGAR filings are electronic HTML/text; OCR only for rare image-only exhibits |
| 11 | Strip headers, page numbers, boilerplate | Adopted. Boilerplate is *measured* before removal, since boilerplate change can itself be a signal |
| 12 | Isolate tables | Adopted: table store + numeric features + table-aware RAG |
| 13 | Reading-order reassembly | Adopted |
| 14–19 | Unicode clean-up, term mapping, number normalization, relative dates, acronyms, sentence segmentation | Adopted (S5), keeping raw text alongside |
| 20 | Domain tokenizer | Adopted implicitly: each model uses its own tokenizer; domain vocabularies (SEC-BERT, FinBERT) compared |
| 21–23 | Semantic chunking, parent-child, metadata in chunks | Adopted (S12, S15) |
| 24 | Fine-tuned financial embeddings | Adopted: embedder tournament; optional contrastive fine-tune on training-period pairs |
| 25 | Vector database | Adapted: local FAISS / LanceDB files (no server) |
| 26 | NER | Adopted for anonymization and GraphRAG |
| 27 | Relationship graphs | Adopted for GraphRAG (small slice) |
| 28 | FinBERT sentiment | Adopted (Tower T3 / features) |
| 29 | Hybrid BM25 + dense | Adopted (Hybrid / Advanced RAG) |
| 30 | Cross-encoder reranking | Adopted |
| 31 | LLM generation with context | Adopted, as structured extraction (JSON) plus summaries |
| 32 | Deterministic math cross-check | Adopted: numbers versus XBRL |
| 33 | Strict output schemas | Adopted: constrained JSON decoding + validation |
| 34 | Hallucination checks | Adopted: faithfulness gate; unsupported fields set to null |
| 35 | Citations to page / PDF coordinates | Adapted: accession + section + character offsets (filings are not paginated PDFs) |
| 36 | PII masking | Adapted: identity anonymization for look-ahead control; executive names are public, so masking serves bias control, not privacy |
| 37 | Async API endpoints | **Dropped** (batch research notebook) |
| 38 | Semantic cache | Adapted: deterministic response cache keyed by model, prompt and passage hashes |
| 39 | Frontend dashboard | Optional, off by default: a small Gradio "filing explorer" at the end of Part D |
| 40 | Feedback and drift monitoring | Adapted: offline drift report (adversarial validation, rolling IC, performance by year) |

## Appendix B: Owner's second model list, mapped

| Listed | Decision |
|---|---|
| FinBERT (Prosus) / FinBERT (TRC2) | Core (same Prosus family) |
| FinBERT-FLS / FLS-Classifier | Feature (`yiyanghkust/finbert-fls`) |
| SEC-BERT | Core (`nlpaueb/sec-bert-base`) |
| BB-BERT | Excluded: not public |
| Llama 3 8B, Mistral 7B | Core / challenger general LLM baselines (Llama-3.1-8B-Instruct; Mistral-7B-Instruct optional) |
| FinMA, FinGPT | Challengers (9.2) |
| BloombergGPT | Excluded: not released |
| Command R+ | Excluded: 104B, non-commercial |
| all-MiniLM-L6-v2, bge-large-en-v1.5, mpnet-base-v2 | Retrieval embedder tournament |
| E5-Mistral-7B-Instruct | Optional heavy embedder |
| XGBoost, CatBoost, LightGBM, Random Forest, SVR, ElasticNet | Tabular learners (9.4) |
| PatchTST, Informer, TFT, LSTM, GRU, N-BEATS | Time-series challengers; TFT = hybrid. GRU on order-book data not applicable |
| AlphaFinder | Not verified as a public model; excluded unless located |
| StockNet | Excluded: needs social-media text |
