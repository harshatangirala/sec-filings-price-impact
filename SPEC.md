# SEC Filings → Price Impact: Build-Ready Spec, v5

**For: Claude Code in VS Code.** This file is the single source of truth. Put it in the repository root as `SPEC.md`. It contains requirements only, **no code**.

**Deliverable:** a Python repository (package + configs + tests + two notebooks).
- **Local (VS Code):** run every CPU stage and every test on your own machine. That costs no Colab credits.
- **Colab Pro+ (GPU):** run only the GPU stages, through a runner notebook.
- **End-to-end option:** the whole pipeline can also run on Colab from the same runner.

**What v5 is:** v4's research design, corrected by a second independent review (the "professor review"). It is narrowed into a **core study** that must be completed and an **extension pack** that runs only when gates pass, and given the engineering detail Claude Code needs: repository layout, configuration, data contracts, stage interfaces, tests and a milestone plan.

---

## PART I: RESEARCH DESIGN

## 0. Read this first (Claude Code)

### 0.1 How to use this document
1. **Session start:**
   - Read the whole spec.
   - Create `CLAUDE.md` in the repo root: a one-page digest of Section 0.2 (hard rules), Section 13 (environment), Section 14 (repo layout) and the current milestone.
   - Keep `CLAUDE.md` updated at the end of every milestone.
2. **Milestones:** build in the order of Section 22. Each milestone has a "done when" list. Do not start the next milestone until it is satisfied and committed to git.
3. **Ambiguity:**
   - Prefer the more conservative choice for leakage and point-in-time issues.
   - Record the decision in `docs/decision_log.md`.
   - Never silently deviate from this spec.
4. **Spec changes:** if a rule is impossible to implement (an endpoint changed, a model is gone), log it, use the documented fallback, and tell the owner in the milestone summary.

### 0.2 Hard rules
1. **SEC filings only (10-K, 10-Q, 8-K) as text.** No news, social-media, transcript or analyst text. Never load the Hugging Face news repository, FNSPID, FinMultiTime news or StockNet tweets. Market prices are used only for outcomes and price features. Pretrained models whose authors used news-style text are allowed, with their provenance recorded.
2. **Three tracks stay separate** (Section 2): **A** event study (association), **B** prospective prediction, **C** content validity. They share cleaned data but not timestamps, targets or availability assumptions.
3. **Four timestamps per event** (Section 5.1). Features must be available **no later than** `decision_time`. Retrieval of *prior* documents uses strict `public_available_time < decision_time` of the current event, and always excludes the current filing and any later filing.
4. **`RUN_PROFILE = smoke` is the default.** `pilot` and `full` need an explicit change. `secprice preflight` must pass before any `full` run.
5. **Never fabricate** data, dates, results or metrics. The report contains only computed values.
6. **Association is not causation; prediction is not profitability.**
7. **The earnings announcement return (EAR) is never a feature for its own event or its earnings cluster** (Section 5.6). **SUE** uses only the event's own filing/exhibit or *earlier* public accessions. A later 10-Q/10-K of the same cluster can never feed an earnings-8-K event.
8. **8-Ks are event-driven:** a company-year without an 8-K is not missing data.
9. **Never choose documents or events using later price moves.**
10. **High scores trigger an audit, not termination.** A pre-event window is a diagnostic, not a required zero.
11. **Model provenance decides claims.** Only models with documented cutoffs before the evaluated events may appear on the **strict point-in-time leaderboard**. Unknown-cutoff models are **retrospective only** and can never support a "clean test" claim (Section 6.4).
12. **Verify before use.** Every model, dataset, endpoint and library is checked at build time; exact revisions are pinned; substitutions are logged.
13. **SEC fair access:**
    - declared User-Agent (name + email) read from an environment variable or Colab secret, never hard-coded
    - ≤ `sec.max_rps` (default 6; SEC's ceiling is 10)
    - exponential backoff on 429 / 5xx
    - no proxy rotation or evasion
    - pause and report if blocked
14. **No paid or hosted model APIs.** All models run locally or on Colab.
15. **Reproducibility:**
    - seeds fixed
    - config hash and git commit stamped on every artifact
    - an environment manifest
    - pinned dependencies (do not reinstall Colab's own torch without reason)
16. **Tests are part of the deliverable** (Section 18). Network-dependent tests are marked and skipped offline; unit tests run on fixtures.

---

## 1. What changed from v4

| Change | Why |
|---|---|
| **Default parameters set:** `N_STOCKS = 500` unique firms, `STUDY_START = 2000-01-01`, `STUDY_END = 2026-10-31` (usable cutoff = last observed price/document date at runtime), market warm-up from 1998 | The original intent; still overridable |
| **Four timestamps:** acceptance, public availability, feature-ready, decision | SEC states filings typically appear 1–3 minutes after the EDGAR timestamp, sometimes longer; acceptance ≠ readable time |
| **Intraday tradable target fixed:** every T_trade entry is at the **open of the first session starting after `decision_time`** | A same-day close fill is not guaranteed with daily data |
| **SUE timing made cluster-aware:** earnings-8-K events use EPS parsed from the release exhibit; XBRL from a later 10-Q is forbidden for that event and used only for quality checks or labelled ex-post sensitivity | Closes a subtle leak the review identified |
| **Gold set split** into gold-dev / gold-val / gold-test | Selection and final reporting must not share labels |
| **Strict point-in-time vs retrospective leaderboards**; no clean-test claim for unknown cutoffs | Honest temporal claims |
| **Common-cohort vs maximum-available comparisons** | Lift must not come from a different event set |
| **Narrowed core** (4 encoders, 3 retrieval setups, 1 LLM, lexical and structured baselines); everything else becomes a **gated extension pack** | A solo, credit-limited project must finish its core |
| **Generation metrics moved to an optional sub-study**; core content metrics are field P/R/F1, numeric fidelity, citation accuracy, faithfulness, retrieval metrics | Matches what the core study claims |
| **Concrete backtest:** signal score, ranking rule, minimum names, overlapping-position policy, execution per bucket, borrow assumptions | Removes ambiguity |
| **"Economically trivial" pre-defined in basis points** | Inference rule fixed before results |
| **Engineering spec:** repository, config, data contracts, CLI, artifacts, tests, milestones; CPU locally in VS Code, GPU on Colab | Makes the build executable end to end and saves credits |

Appendix A has the item-by-item reconciliation of the professor review. Appendix B summarises the earlier review's reconciliation (still in force).

---

## 2. Research question, tracks and hypotheses

**Main contribution (core):** *Do SEC filing text and evidence-grounded event features add reliable out-of-time information beyond point-in-time market, filing-metadata, accounting and filing-change baselines?*

**Tracks**
- **Track A (event study):** what happens around disclosures: abnormal returns, volume and volatility by form, section and 8-K category, with clean-event and pre-trend sensitivity, plus the drift tests. Association only.
- **Track B (prediction):** what could have been predicted at `decision_time`: the common-cohort ablation ladder, walk-forward tuning, a development holdout, a locked out-of-time test, strict versus retrospective leaderboards and one concrete backtest.
- **Track C (content validity):** are sections, item mappings, extracted events and numbers correct? Measured against human labels on held-out gold-test data, never against price outcomes.

**Core hypotheses** (pre-registered in `configs/preregistration.yaml`; its hash is recorded before any development-holdout scoring):
- **H1:** SEC text adds out-of-time predictive value beyond price, metadata and accounting baselines.
- **H2:** Changes in filing language add more value than absolute sentiment alone.
- **H3:** Validated structured event extraction adds value beyond frozen embeddings and lexical features.
- **H4:** Strict point-in-time model vintages perform differently from retrospective static checkpoints; the size of the gap estimates look-ahead sensitivity.

**Extension hypotheses** (tested only if extensions run):
- **H5:** A stacked multi-tower ensemble beats the best single model.
- **H6:** Richer RAG designs beat hybrid + rerank.
- **H7:** Analogue retrieval or time-series models add value.
- **H8:** Gains shrink under strict timestamps, costs, clean-event filtering and the locked test. This is evaluated for the core too.

**Primary task:** 3-class direction (Up / Sideways / Down) of **T_info** abnormal return at h = 1. **Primary metric:** mean walk-forward macro-F1, with guardrails (log-loss no worse than the class prior; calibration error below threshold). Secondary metrics are listed in Section 9.

**Expectations, stated in the report:**
- Low signal-to-noise is normal; small lifts are realistic, and a credible null result is a valid result.
- Volatility and absolute-move targets usually show more skill than direction.

---

## 3. Parameters and defaults

| Parameter | Default | Notes |
|---|---|---|
| `run_profile` | `smoke` | `pilot` and `full` by explicit edit |
| `universe.n_stocks` | 500 | Unique firms across the study. Report firms per year separately; never claim 500 per year unless verified |
| `universe.source` | historical S&P 500 membership | Candidate public source: the `fja05680/sp500` repo (MIT; history from 1996; built from Wikipedia changes, which its author calls incomplete; early years may miss symbols). Verify it; a licensed point-in-time list can replace it |
| `study.start` / `study.end` | 2000-01-01 / 2026-10-31 | Usable cutoff = min(last observed price date, last document date). If the run is before 31 Oct 2026, mark the month partial |
| `market.warmup_start` | 1998-01-01 | ≥ 250 sessions before the first event, for beta and volatility |
| `text.predecessor_lookback` | 2 years before `study.start` | Predecessor filings loaded only to compute change features for early events; they are never evaluated |
| `timing.public_lag_minutes` | 5 (sensitivity 0, 15, 60) | public_available = acceptance + lag |
| `timing.decision_latency_minutes` | 30 (sensitivity 10, 120) | decision = public_available + latency. One value for all models keeps the cohort and targets identical |
| `targets.horizons` | [1, 2, 5, 20, 60] | Sessions |
| `targets.primary` | T_info, h = 1, 3-class | Pre-registered |
| `labels.k` | 0.5 (sensitivity 0.25, 0.75) | Sideways band = ±k·σ·√h |
| `contam_window_sessions` | max horizon in the analysis | Clean-event filter |
| `splits.min_train_years` / `val_block_years` / `dev_years` | 8 / 2 / 3 | Walk-forward |
| `splits.locked_test_start` | 2023-01-01 | After SEC-BERT's 2019 cutoff and the 2020–22 regime; overridable |
| `splits.embargo_days` | 90 | Calendar days |
| `trivial_effect_bps` | 10 for h ≤ 2; 25 for h = 5; 50 for h ≥ 20 | Pre-registered "economically trivial" thresholds |
| `fdr.q` | 0.05 | Benjamini-Hochberg within pre-declared families |
| `bootstrap.n` | 1000 (smoke 100) | Block bootstrap by month |
| `gates.text_lift` | 0.01 macro-F1, CI excludes 0 | Gate G3 |
| `gates.detector_min_precision` | 0.80 | Below this, a detector is provisional |
| `audit.auc_trigger` / `audit.f1_lift_trigger` | 0.65 / 0.10 | Review triggers |
| `budget.*` | per-part GPU hours, storage GB, maximum error rate, LLM sample sizes | Section 13.3 |
| `extensions.*` | all `false` | Section 7.3 |

**Profiles**
- `smoke`: 5 firms, 2021–2023, ~10 filings per firm-form, one encoder, LLM on ≤ 20 filings, fixtures where possible; must finish on a laptop CPU, except the optional GPU smoke.
- `pilot`: ~40 sector-balanced firms, 2012 onward, the full core pipeline with small samples.
- `full`: the defaults above.

---

## 4. SWOT (v5, condensed)

**Strengths**
- Auditable primary-source event panel.
- Separate research tracks.
- Point-in-time accounting.
- A strong common-cohort baseline ladder.
- Vintage models for temporal provenance.
- Reproducible stage artifacts and tests.

**Weaknesses**
- Daily bars only.
- The S&P-based universe is survivorship-aware, not survivorship-free.
- Free prices are weak on delistings.
- Selected sections only.
- Single annotator unless a second is found.
- SUE coverage is limited before the XBRL era.

**Opportunities**
- Filing-change and novelty signals.
- Validated structured extraction.
- Measuring temporal provenance (look-ahead) effects.
- A reusable dataset.
- An honest null or positive result suitable for a capstone or workshop paper.
- University data (CRSP / Compustat / IBES) could later remove the survivorship and consensus limits.

**Threats**
- Timestamp mis-specification.
- Pretrained-model memory.
- LLM extraction errors.
- Multiple testing.
- Price-data gaps.
- Colab variability.
- SEC throttling.
- Over-building before the core works. Mitigated by milestones and gates.

---

## 5. Event clock, event types and targets

### 5.1 Four timestamps (all stored in US Eastern with timezone; trading calendar = NYSE, including early-close days)
| Field | Definition |
|---|---|
| `acceptance_time` | EDGAR acceptance date-time. **Verify timezone semantics** by comparing the submissions API value with the `ACCEPTANCE-DATETIME` in the filing's SGML header on a sample (unit test); store the raw string too |
| `public_available_time` | Exact if observed; otherwise `acceptance_time + public_lag_minutes`. Quality flag records which |
| `feature_ready_time` | `public_available_time + decision_latency_minutes` (assumed processing time) |
| `decision_time` | `feature_ready_time`. The prediction is made here. The current filing's own text is a legitimate input at this time |

- **Timestamp quality:** A = exact acceptance time; B = date only (assume after the close: acceptance 17:30 ET); C = derived or uncertain.
- **Same-day ordering:** `acceptance_time`, then accession number.
- **Session bucket** (from `acceptance_time` against the actual session open/close, including early closes): PRE / INTRA / POST / NONTRADING / UNKNOWN.

### 5.2 Event construction and types
- **Periodic events:** 10-K (+405 / KT), 10-Q. Amendments are kept for the audit, not used as events.
- **8-K groups:** same-firm, same-day 8-Ks merge into one multi-label event, keeping every item code and category.
- **Earnings types:**
  - `earnings_announcement_verified`: 8-K Item 2.02 (or a legacy equivalent) whose exhibit content is confirmed as a results release, with usable timing
  - `periodic_filing_only`: a 10-Q/10-K with no earlier verified release
  - `earnings_date_ambiguous`: conflicting or weak sources

  PEAD analysis uses the verified type only.
- **`cluster_id`:** links a verified release with its later 10-Q/10-K. A cluster is never treated as independent events in inference.
- **`event_id`:** a stable hash of (CIK, earliest accession in the group).

### 5.3 Track A clock
- **Day 0:** the session in which `public_available_time` falls if that is before the close; otherwise the next session.
- **Windows:** CAR[−1,+1], [0,+1], [0,+5], [+2,+20], [+2,+60]; pre-event [−20,−2] and [−5,−1].
- **INTRA:** day-0 returns are labelled a *mixed pre/post interval*.
- **Reporting:** results are reported by bucket.

### 5.4 Track B targets
| Target | Base | Exit | Eligibility |
|---|---|---|---|
| **T_info** (information response; primary) | Last regular-session close **before** `acceptance_time` | Close of the h-th session whose close is after `public_available_time` | PRE, POST, NONTRADING only. **INTRA excluded.** The pre-publication window (base → acceptance) contains no regular session; its length is logged |
| **T_trade** (tradable) | **Open of the first session whose open is at or after `decision_time`** (all buckets) | Close of the h-th session counting the entry session as 1 (h = 1 = entry-day open → close) | All buckets. Same economic holding interval for every bucket |

Sensitivity: entry one session later; `decision_latency` at 10 and 120 minutes; `public_lag` at 0, 15 and 60 minutes.

### 5.5 Returns, abnormal returns, labels
- **Series:** raw close, split-adjusted and total-return series kept separately. Outcomes use total return; dividend analyses also report raw price.
- **Abnormal returns:**
  - market model against SPY, beta on sessions −250 to −30 before the base (≥ 120 observations)
  - robustness: **Carhart 4-factor** (market, SMB, HML, momentum) primary; **FF5 + momentum** secondary (Kenneth French daily factors)
- **3-class label:**
  - Up if AR > +k·σ·√h; Down if AR < −k·σ·√h; otherwise Sideways
  - σ = point-in-time idiosyncratic daily volatility from the estimation window
- **Regression targets:**
  - AR
  - volatility ratio: log of realized vol over sessions 1–20 divided by sessions −60 to −5
  - standardized |AR|
- **Clipping:** ML clip bounds are estimated on training folds only; un-clipped test metrics are also reported. Track A CARs are never silently winsorized (a winsorized version is a labelled robustness check).
- **Censoring:** per horizon, drop events whose exit date is after the usable cutoff.
- **Weights:** label-uniqueness weights for same-firm overlapping windows; unweighted results also reported.

### 5.6 Earnings surprise (SUE) and EAR
- **SUE formula:** (EPS_q − EPS_{q−4}) / σ(ΔEPS), with σ from the prior 8 seasonal differences (minimum 4), each value as **originally reported** (the earliest accession that reported that period).
- **EPS source by event type:**
  - **Earnings-8-K events:** EPS_q parsed from the release exhibit (EX-99.x tables; diluted GAAP EPS preferred, basic as fallback, flagged). History EPS_{q−k} comes from XBRL company facts or prior releases with `public_available_time < decision_time`.
  - **10-Q/10-K events:** EPS_q from the event's own XBRL (quarterly derived from year-to-date by within-year differencing; units checked).
  - **Never** a later accession from the same cluster.
- **QA:** the parsed release EPS is compared with the later XBRL value (QA metric only). Coverage of `SUE_release` and `SUE_xbrl` is reported by year and size.
- **EAR:** AR[−1,+1] around a verified release.
  - It is a **Track A outcome** and a Track A drift sort variable (drift measured from +2).
  - As a Track B feature, it is allowed only for a **later, different cluster** whose `decision_time` is after the EAR window closes.
  - The feature registry enforces this, and a planted-leak test checks it.
- **Naming:** "SUE-conditioned post-earnings drift", never a consensus-based PEAD replication.

### 5.7 Overlaps and clean events
- **Flags:** an event is flagged if another qualifying same-firm event falls within `contam_window_sessions` after its base (defined from dates only).
- **Analyses:** two versions, **all-event** (clustered errors) and **clean-event**. Report events removed and whether conclusions change.

---

## 6. Splits, leakage control and provenance

### 6.1 Layout
- **Blocks:** walk-forward expanding folds (validation blocks of 2 years) → development holdout (2020–2022 by default) → locked test (2023-01-01 → usable cutoff).
- **Purge:** any training event whose T_info or T_trade window reaches the next block.
- **Embargo:** 90 days.
- **Stress tests:** firm-held-out and sector-held-out folds before the holdout.
- **Lockbox:**
  - the locked test is scored **once**, after the G4 manifest freeze (hash), by `secprice unlock-test`
  - retraining on all pre-test data happens before scoring
  - if the lockbox is opened early, it is relabelled a development holdout

### 6.2 Cohorts (every comparison declares one)
- **Common cohort (primary):** events eligible for *every* compared model (including SUE availability where accounting features are in the comparison).
- **Maximum-available (secondary):** each model on all events it can score. Clearly labelled as not like-for-like; sample differences reported.

### 6.3 Leakage checklist (each has an assertion in `splits/leakage` and a line in `leakage_audit.csv`)
Same 24 items as v4, updated:
1. Overlapping windows across blocks → purge + embargo.
2. Same-date shocks → date-disjoint splits.
3. Duplicate or near-duplicate filings → split by accession; hashes.
4. Firm memorization → anonymization; firm-held-out test.
5. Preprocessing fitted on future data (scalers, vocabularies, TF-IDF / BM25 statistics, topics, imputers, calibration, thresholds, clip bounds) → fitted on training indices only.
6. Test peeking → pre-registration hash; G4 freeze; lockbox command.
7. Intraday / pre-publication contamination → T_info eligibility; logged windows.
8. Timestamp errors → four-timestamp model; quality flags; sensitivity.
9. EAR misuse → registry rule + planted leak.
10. Earnings-date fabrication → three types.
11. **SUE from a later cluster accession** → registry rule + planted leak.
12. XBRL ordering → accession / `public_available_time` ordering.
13. Same-firm overlaps → flags + clean cohort.
14. Survivorship → universe table + wording.
15. Right-censoring → per horizon.
16. **Retrieval leakage** → the historical index returns only documents with `public_available_time < decision_time` and never the current filing; current-document evidence travels on a separate path.
17. Analogue outcomes (extension) → neighbour windows closed before `decision_time`.
18. Pretrained-model look-ahead → provenance rules (6.4).
19. Distillation / continued pretraining → training-period data only.
20. Stacking → walk-forward out-of-fold predictions only.
21. Outcome-based selection → assertion that event construction reads no price fields.
22. Data snooping → configuration counter; FDR; MCS / PBO as diagnostics.
23. Covariate shift → adversarial validation (an AUC near 0.5 does **not** prove a split is unbiased).
24. Broken audit → **planted-leak self-tests**: future-return feature, same-event EAR, later-cluster SUE, future-document retrieval. All must be caught.

### 6.4 Provenance leaderboards
- **Ledger** (`configs/models.yaml` + `reports/model_provenance_ledger.csv`): model ID, revision, card URL, license, training corpus, stated cutoff (or `unknown`), fine-tune data, task and labels.
- **Strict point-in-time leaderboard:** includes a model only if its documented cutoff precedes **every** event in the evaluated block:
  - vintage-matched ChronoBERT (an event in year Y uses vintage Y−1)
  - models trained from scratch on our folds (TF-IDF, GBDT, heads)
  - lexical / structured baselines
  - SEC-BERT only for blocks after 2019-12-31
- **Retrospective leaderboard:** all models, including unknown or later cutoffs.
- **Clean-test subset:** for a known-cutoff model, the locked-test events after its cutoff. **Unknown-cutoff models get no clean-test claim.**
- **Per-fold label:** every fold is marked strict or retrospective for each model.
- **H4:** vintage versus static comparison on identical rows and heads.

### 6.5 Review triggers and sanity experiments
- **Review triggers:** locked AUC > `audit.auc_trigger`, F1 lift > `audit.f1_lift_trigger`, or R² unusually high against the expanding-mean baseline.
  - **Effect:** open the audit checklist (chronology, target overlap, source fields, timestamps, duplicates, class balance, baselines, retrieval filters, provenance); outcomes go to the decision log.
  - **Not a stop:** they never terminate the run, and a positive R² alone is not evidence of leakage.
- **Sanity experiments:**
  - **Random-split demo:** pedagogical only.
  - **Permutation test:** within date blocks, class-stratified.
  - **Pre-event windows:** pre-trend diagnostics, interpreted rather than forced to zero.

---

## 7. Models

### 7.1 Core roster (default on)
| Role | Model (verify and pin) | Why |
|---|---|---|
| SEC-domain encoder | `nlpaueb/sec-bert-base` | 10-Ks 1993–2019; CC-BY-SA-4.0; strict after 2019 |
| Financial-tone encoder | `yiyanghkust/finbert-tone` | Pretrained on 10-K/10-Q, earnings calls and analyst reports |
| Point-in-time encoder | `manelalab/chrono-bert-v1-YYYY1231` (1999–2024 vintages; 1,024-token context) | Strict comparator for H4; verify each vintage used |
| Modern embedder | `BAAI/bge-base-en-v1.5` (512 tokens) | General baseline; also the dense retriever |
| Reranker | `BAAI/bge-reranker-base` (or v2-m3, verify) | Hybrid + rerank |
| Extraction LLM | `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, not gated) | One local generator; cutoff not precisely documented, so retrospective only |
| Distillation student | `answerdotai/ModernBERT-base` or the best core encoder | Applies extraction to the full corpus cheaply |
| NER (limited) | `dslim/bert-base-NER` | PER / ORG only, for anonymization; **never money** |
| Tabular learners | LightGBM, CatBoost, ElasticNet-logistic | All ladder rungs |
| Lexical | Loughran-McDonald dictionary (Notre Dame SRAF; academic use; verify terms), TF-IDF, BM25 | Baselines |

### 7.2 Encoder protocol
- **One model in memory at a time:** load → embed → persist float16 + index → unload.
- **Chunking:** section-aware chunks within the context limit; mean / max / attention pooling to section and document vectors; change vectors versus the prior same section.
- **Comparison:** all candidates use the **same event rows, labels, splits, head families, tuning budget and metric**.
- **Native outputs:** sentiment and tone scores are features, scored natively only against human labels.
- **Fine-tuning:** top 2 encoders by validation, and only if G3 passes (fixed grid, Section 9.4).

### 7.3 Extension pack (default off; each needs its flag + gate G3c + budget)
| Flag | Contents |
|---|---|
| `ext_encoders` | FinBERT (Prosus), FinBERT-pretrain, FLANG-BERT / RoBERTa, Financial-RoBERTa-large, DeBERTa-v3 finance, DistilRoBERTa finance, ModernBERT-large, FinModernBERT-embed, Qwen3-Embedding-0.6B, MPNet / MiniLM, `sec-bert-num` / `shape`, Longformer (only if chunking underperforms) |
| `ext_feature_models` | FinBERT-FLS, FinBERT-ESG, zero-shot DeBERTa-v3-large (only on ambiguous / `OTHER_OR_UNMAPPED` cases) |
| `ext_llms` | One challenger if core extraction misses G3b: `meta-llama/Llama-3.1-8B-Instruct` or `instruction-pretrain/finance-Llama3-8B`; further: Sujet-Finance-8B, Fin-R1, FinGPT adapters, FinMA (low). InvestLM only with an 80 GB GPU |
| `ext_rag` | HyDE, firm-memory RAG, CRAG (no web fallback), multi-hop, adaptive router, table-aware "multimodal" RAG; small-slice only: Self-RAG, GraphRAG, Agentic |
| `ext_analogue` | Point-in-time analogue retrieval (closed-window neighbours) |
| `ext_timeseries` | FinText Chronos vintages (strict); Chronos-2 / Bolt, TimesFM (retrospective); PatchTST, LSTM, N-BEATS, TFT hybrid |
| `ext_stacking` | Multi-tower stacking, Caruana ensemble, successive-halving architecture search |
| `ext_heavy` | QLoRA classifier, continued masked-language pretraining on training-period filings |
| `ext_generation_metrics` | Summary sub-study with ROUGE-N/L, BLEU, NIST, METEOR, chrF/chrF++, BERTScore, BLEURT, MoverScore, perplexity (needs human reference summaries) |
| `ext_data_modules` | Form 4, 13F, user-supplied licensed files (analyst, CRSP delistings) |
| `ext_app` | Gradio filing explorer |

**Excluded (unchanged from v3/v4):** BloombergGPT, BB-BERT, FinLlama (no public weights found), SecBERT / SecRoBERTa (cybersecurity), Qwen-Coder-Finance, Llama-2-Banking, Command R+, FinLLaVA, PRAGMA, FinMultiTime / FNSPID (datasets with news), StockNet (social media), FinDeBERTa (headline model).

---

## 8. Core RAG and extraction (Tracks C → B)

**Question bank** (`configs/question_bank_v1.yaml`, pre-registered; ~12 core fields):
- guidance direction (initiated / raised / lowered / withdrawn / reaffirmed)
- results versus prior-year tone (beat / miss language)
- impairment or write-down
- restructuring or layoffs
- liquidity or going-concern stress
- new financing or covenant issue
- M&A or disposition
- dividend change and type
- buyback
- litigation or regulatory action
- management change
- new or removed risk factors

Each field has a schema:
- `present`
- `direction`
- `magnitude` + unit (or null)
- `certainty`
- `evidence` (chunk IDs + character offsets)

**Retrieval setups (core tournament):**
1. BM25
2. Hybrid BM25 + bge-base dense, reciprocal-rank fusion, metadata filters
3. Hybrid + cross-encoder rerank

**Two evidence paths:**
- **Current-document path:** chunks of the filing being processed, allowed at `decision_time`.
- **Historical path:** the firm's prior filings with `public_available_time < decision_time`, used only for change questions (for example, "new risk factor").

**Pipeline:**
1. retrieve (top-k)
2. Qwen2.5-7B-Instruct with constrained JSON decoding
3. guardrails:
   - schema validation and retry
   - numeric cross-check against XBRL / source
   - faithfulness gate: an unsupported field becomes null
   - citation offsets verified
4. response cache keyed by model revision, prompt hash and passage hash

**Selection protocol:**
1. Retrieval-only metrics on **gold-val**.
2. End-to-end extraction F1 and faithfulness on **gold-val**.
3. Winner frozen (G3b).
4. Final content-validity numbers reported on **gold-test** only.

**Distillation:**
- **Teacher:** the winner extracts for a stratified teacher sample (`budget.llm_teacher_filings`, e.g. 3,000) from **training-period** filings.
- **Student:** a multi-task student is trained on those outputs and applied to all events.
- **Report:** student versus teacher (agreement) and student versus human (gold-test).

---

## 9. Evaluation (Track B)

### 9.1 Ablation ladder (same folds; common cohort primary)
| Rung | Features |
|---|---|
| L0 | Majority / class prior |
| L1 | Price only (prior returns, volatility, volume / liquidity, market regime) |
| L2 | Metadata only (form, 8-K categories, section availability, industry, year, bucket, timestamp quality) |
| L3 | Accounting only (point-in-time SUE + XBRL facts) |
| L4 | Lexical: Loughran-McDonald + TF-IDF + filing-change + novelty |
| L5 | Frozen embeddings only (per core encoder) |
| L6 | L1 + L2 + L3 + L4 (best valid structured set) |
| L7 | L6 + embeddings |
| L8 | L7 + distilled RAG-extraction features; + fine-tuned encoder if G3 passes |

The core evidence is the **lift of L7 / L8 over L6** with CIs, on strict and retrospective leaderboards. H2 compares change features against absolute sentiment inside L4/L7; H3 compares L8 against L7.

### 9.2 Metrics (core)
**Classification**
- Accuracy, balanced accuracy.
- Per-class and **macro** P / R / F1; weighted F1.
- ROC-AUC (one-vs-rest macro and per class), PR-AUC.
- Cohen's kappa, Matthews correlation.
- Log-loss, Brier, expected calibration error and reliability curves.
- Confusion matrices.
- Accuracy and F1 at pre-registered coverage (100 / 50 / 25 / 10%), and the coverage curve.
- **Null with a reason** when a metric is inapplicable.

**Regression**
- R² and **out-of-sample R² versus the expanding historical mean**, MAE, RMSE.
- Spearman, monthly rank IC (t-statistic, ICIR), directional accuracy.
- QLIKE for volatility.

**Content (Track C, gold-test)**
- Field-level P / R / F1.
- Numeric exact match and tolerance match (source and XBRL).
- Citation-offset accuracy.
- Faithfulness: local judge (not the generator), calibrated on ~100 human ratings, kappa reported.
- Retrieval Recall@k, nDCG@k, MRR, MAP on held-out qrels.
- Section-parse accuracy; 8-K mapping accuracy; detector P / R / F1.
- Cost per filing and latency.

**Uncertainty**
- Block bootstrap by month (clustered by firm where relevant).
- Paired tests for model comparisons.
- BH-FDR within families.
- Counts (events, firms, dates, overlaps) beside every result.
- Power language:
  - n < 30 → **warning**
  - 30–99 → **exploratory**
  - a wide CI → **inconclusive**

**Selection diagnostics**
- Model Confidence Set and PBO (combinatorially symmetric cross-validation) as **diagnostics only**. They never replace or rescue walk-forward and locked-test evidence.
- Each diagnostic documents which loss streams and configurations enter it.

### 9.3 Fit diagnosis
- **Outputs:** loss curves (train / validation per step and epoch), validation curves, learning curves.
- **Verdicts:**
  - **Overfit:** gap > 0.10, or validation loss rising 2 epochs while train loss falls.
  - **Underfit:** within 0.02 of majority (this may mean weak signal).
  - **Unstable:** fold std > 0.03.
- **Wording:** results are "validation-selected", never a global optimum.

### 9.4 Tuning
- **Heads:** Optuna (TPE sampler, median pruner, SQLite study). Search spaces as in v4 §14.1, fixed before the holdout is unlocked.
- **Fine-tuning grid (6–12 trials):** learning rate {1e-5, 2e-5, 3e-5} × batch {8, 16} × epochs ≤ 3 with early stopping; weight decay {0, 0.01}.
- **Losses:**
  - class-weighted cross-entropy (focal loss only if it wins on validation)
  - Huber for regression
  - weighted sum for multi-task
- **Before/after:** default and tuned results are reported for every model.

---

## 10. Track A and drift tests

### 10.1 Event-study analyses
- **Analyses:**
  - CAR, abnormal volume and volatility response by form, section, 8-K category (taxonomy versioned, with availability by year) and extracted field
  - all-event and clean-event versions
  - pre-trend windows
  - by bucket
  - correlations (filing ↔ filing tone persistence, release versus later 10-Q, filing ↔ price Spearman / partial)
  - topics fitted on training years only
- **Errors:** clustered by firm and date.
- **Factor alphas:** Carhart primary.

### 10.2 Drift verdicts
**Labels:**
- `supported`
- `weak/mixed`
- `not supported`
- `inconclusive/low power`
- `not testable in scope`

**Supported** requires:
- consistent definitions
- adequate coverage
- an FDR-adjusted p < 0.05 **and** a CI excluding the pre-registered trivial threshold
- the same sign in ≥ 2 of 3 subperiods
- clean-event robustness
- survival under 10 / 20 bps costs and a one-session lag

**Failing one test never proves an anomaly is absent.**

| Effect | Status | Design |
|---|---|---|
| SUE-conditioned post-earnings drift | Testable (qualified, exploratory) | Verified releases only. Primary sort `SUE_release`; sensitivity `SUE_xbrl` (labelled ex-post, not tradable) and EAR (observed before +2). Drift CAR[+2,+60] |
| Short-term momentum | Testable (benchmark) | 1-month reversal, 2–12 and 12–1. Does text add value beyond it? Never attributed to filings |
| Post-dividend announcement drift | Partial, only if validated counts give adequate power | Announcement ≠ ex-date; raw versus total return |
| Spin-off / restructuring drift | Partial, same condition | 8-K 2.05 / 2.06 / 2.01 + validated spin-off detection; announcement / completion / distribution dates distinct; parent-only unless new tickers are mapped |
| Analyst revisions drift | **Not testable** | A management-guidance proxy is a different construct, named as such |
| Insider trading drift | **Not testable (core)** | Form 4 extension |
| Institutional ownership drift | **Not testable (core)** | 13F extension |

---

## 11. Backtest (Track B economic check)

1. **Signal:** s = P(Up) − P(Down) from the frozen model (regression variant: predicted AR). Computed at `decision_time`.
2. **Thresholds:** long if s ≥ the training-period 80th percentile of s; short if ≤ the 20th percentile. Percentiles come from training folds only.
3. **Execution:** enter at the **open of the first session at or after `decision_time`** (same rule for every bucket); exit at the close of session h (pre-registered h, default 5); sensitivity: entry one session later.
4. **Portfolio:**
   - calendar-time daily portfolio of open positions
   - equal weight within each leg
   - maximum 5% per name
   - one position per firm (a newer event replaces the older one)
   - a leg with fewer than `min_names_per_side` (default 10) positions holds cash for that leg
5. **Costs:**
   - 10 and 20 bps per side
   - hypothetical borrow cost 50 bps per year on shorts (configurable)
   - results reported **long-only versus SPY** and **long-short (hypothetical)**
   - no claim of short availability or capacity
6. **Corporate events:**
   - halts → hold to the next price
   - delistings / mergers → delisting proceeds if the source has them, otherwise the last price, flagged
   - spin-offs → parent total-return series, flagged
7. **Metrics:** return, volatility, Sharpe, **Deflated Sharpe**, Carhart alpha, turnover, maximum drawdown, exposure counts.
8. **Wording:** an economic-relevance check, not an investable strategy, given unknowns about borrow, capacity and delisting data.

---

## 12. Gold set (Track C)

Sampled from **training-period** filings, stratified by form, era (2000–08 / 2009–19 / 2020+, using training years only), category and sector; rare categories deliberately oversampled.

| Split | Size (recommended) | Use |
|---|---|---|
| gold-dev | ~50 filings | Schema and prompt iteration, debugging |
| gold-val | ~100 filings + ~200 passages | RAG / LLM / detector selection and thresholds |
| gold-test | ~100 filings + ~200 passages | **Untouched** final content-validity numbers |

**Annotation tasks**
- **Filings:** question-bank fields with evidence spans (extraction gold + retrieval qrels); section boundaries; 8-K category check.
- **Passages:** does this passage support the assigned label? (detectors, categories, zero-shot extension).
- **Faithfulness:** ~100 judge-calibration ratings drawn from gold-val outputs.

**Second rater**
- A second annotator independently labels a stratified ≥ 30% of gold-test, with positives and negatives for every important category; Cohen's kappa + adjudication reported.
- Without a second rater, all content-validity findings are labelled **single-rater, provisional**.

**Pre-fill and blind labelling**
- LLM pre-fill is allowed for gold-dev and gold-val only.
- **gold-test is labelled blind** (no pre-fill), and anchoring risk is noted.

**Tooling**
- The notebook / CLI exports CSV/Excel templates + an annotation guide and imports completed files with validation.
- The pipeline runs without labels; dependent results are then marked *exploratory / unvalidated*.

**Rule:** price outcomes never validate semantic labels.

---

## PART II: ENGINEERING SPEC

## 13. Execution environment

### 13.1 Where things run
| Work | Where | Why |
|---|---|---|
| Development, unit tests, smoke run | Local VS Code (CPU) | Free; fast iteration |
| EDGAR download and parse, prices, events, features (no GPU), baselines, Track A, report | **Local** (recommended) or Colab CPU | No credits used; SEC throttling makes this wall-clock bound, not compute bound |
| Encoder embeddings and fine-tuning, LLM extraction, student training | **Colab Pro+ GPU** via `notebooks/colab_gpu_runner.ipynb` | Only GPU work spends credits |

### 13.2 Data movement
- **Two roots:** artifacts live under `DATA_ROOT` (environment variable). Local: a project data folder. Colab: a Drive folder (`/content/drive/MyDrive/sec-price-study`).
- **Sync:** local → Drive with Google Drive for desktop, or a provided `secprice sync` command that copies only the artifacts the GPU stages need (sections, events, splits, configs). GPU outputs come back the same way.
- **Hot temporary data:** on Colab, heavy temporary data goes to `/content` local disk and is copied to Drive at checkpoints.

### 13.3 Budgets and preflight
- **`secprice preflight`** prints, for the chosen profile:
  - firms, filings and sections
  - estimated chunks and tokens
  - embedding storage (documents × chunks × hidden size × bytes, plus about 10% index overhead)
  - model download sizes
  - LLM prompt count and tokens
  - disk free (local and Drive)
  - projected GPU time from a timed micro-benchmark
- **Refusal:** it refuses `full` if `budget.max_storage_gb` or `budget.max_gpu_hours_{b,c}` would be exceeded.
- **Monitoring:**
  - `budget.max_error_rate`: a stage stops cleanly if exceeded
  - wall time and GPU-active time are logged per stage; neither is a credit measurement
- **Colab rules:**
  - restart the runtime when switching CPU and GPU
  - every stage reloads from `DATA_ROOT`
  - never rely on in-memory state or background execution
- **GPU efficiency:**
  - mixed precision
  - length bucketing
  - 4-bit or AWQ for the 7B LLM, if needed
  - vLLM batch inference where compatible, otherwise Transformers
  - OOM retry that halves the batch

### 13.4 Python environment
- Python 3.11 (match Colab's version at build time).
- `pyproject.toml` with pinned core dependencies and extras `[gpu]`, `[llm]`, `[dev]`.
- On Colab, install only missing extras; do not reinstall torch.
- Lock file committed.
- `secprice env` writes `manifests/environment_manifest.json`.

### 13.5 Candidate libraries (verify; choose and document)
| Need | Candidate | Notes |
|---|---|---|
| EDGAR access | Own thin client on the official endpoints (submissions JSON, archives, companyfacts), **or** `edgartools` (MIT; identity required; rate-limit aware) | If a library is used, wrap it so our rate limiter, User-Agent and cache stay in control |
| 10-K item cross-check | `eloukas/edgar-corpus` (HF; 10-K items, 1993–2020; ~40 GB) | Optional QA of our 10-K section parser on a sample; never a substitute for our own point-in-time pipeline |
| Hugging Face SEC audit | `JanosAudran/financial-reports-sec` | Sentence-level 10-K data with return labels; audit / cross-check only; never use its labels |
| Trading calendar | `pandas_market_calendars` (NYSE, early closes) | Session boundaries |
| Prices | Pluggable `PriceProvider`: yfinance default (explicit `auto_adjust` / `actions`; record version); CSV import for licensed data (e.g. CRSP via university WRDS) | yfinance lacks most delisted history; the coverage table must show it |
| Factors | Kenneth French data library | Daily FF5 + momentum |
| ML | scikit-learn, LightGBM, CatBoost, Optuna | |
| NLP | transformers, sentence-transformers, rank-bm25 or bm25s, FAISS (CPU / GPU) | |
| LLM | vLLM (if compatible) or transformers + bitsandbytes; a JSON-constrained decoding library | |
| Stats | statsmodels (Newey-West, clustered errors), arch (MCS, if used) | |
| Storage | pyarrow / Parquet (zstd), DuckDB for queries | |

---

## 14. Repository layout

```text
sec-filings-price-impact/
├── SPEC.md                     # this file
├── CLAUDE.md                   # digest maintained by Claude Code
├── README.md                   # quick start: local smoke, Colab GPU, profiles, labelling
├── pyproject.toml / lock file
├── configs/
│   ├── default.yaml            # all parameters (Section 3)
│   ├── profiles/{smoke,pilot,full}.yaml
│   ├── preregistration.yaml    # hypotheses, primary metric/target, k, coverage levels, FDR families, trivial bps
│   ├── models.yaml             # roster + provenance ledger fields + pinned revisions
│   ├── taxonomy_8k_v1.yaml     # raw/legacy item → category, effective dates
│   ├── question_bank_v1.yaml
│   └── lexicons/               # Loughran-McDonald (downloaded, versioned)
├── src/secprice/
│   ├── cli.py                  # commands (Section 17.2)
│   ├── config.py               # load, merge profile, validate, hash
│   ├── storage.py              # DATA_ROOT, artifact paths, manifests, _SUCCESS markers, hashing
│   ├── runner.py               # stage registry, dependency resolution, skip-if-done
│   ├── logging_utils.py
│   ├── sec/                    # client (rate limit, UA, cache), submissions, archives, companyfacts, headers
│   ├── universe/               # membership, ticker↔CIK mapping, eligibility, universe table
│   ├── market/                 # calendar, price providers, corporate actions, factors
│   ├── audit/                  # coverage + reason codes, HF dataset audit
│   ├── parse/                  # html/text, sections, exhibits, tables, normalize, anonymize, taxonomy, eps_release
│   ├── events/                 # timestamps, grouping, earnings types, clusters, overlaps
│   ├── targets/                # returns, abnormal returns, labels, censoring
│   ├── features/               # registry, price, metadata, accounting (SUE), lexical, change, novelty, embeddings
│   ├── splits/                 # walk-forward, purge/embargo, cohorts, leakage assertions, planted leaks, provenance
│   ├── models/                 # ladder, heads, fine-tune, tuning, calibration, selective prediction
│   ├── rag/                    # chunking, bm25, dense, hybrid, rerank, extract, guardrails, distill
│   ├── gold/                   # sampling, templates, import, agreement
│   ├── evaluation/             # metrics (cls/reg/ir/content), bootstrap, fdr, mcs, pbo, results table
│   ├── track_a/                # event study, drift, correlations, topics
│   ├── backtest/
│   ├── viz/                    # one function per figure ID
│   ├── report/                 # results_summary.md builder
│   └── extensions/             # ext_* modules (Section 7.3), each behind its flag
├── tests/
│   ├── unit/                   # Section 18.1
│   ├── integration/            # offline smoke on fixtures
│   └── fixtures/               # tiny synthetic + recorded SEC/price responses
├── notebooks/
│   ├── colab_gpu_runner.ipynb  # mount Drive, clone/pull repo, install extras, run GPU stages by name
│   └── study_walkthrough.ipynb # reads artifacts; explains each stage with plots (learning notebook)
└── docs/
    ├── decision_log.md
    ├── annotation_guide.md
    └── stage_cards/            # one markdown per stage: purpose, inputs, outputs, assertions, how to read
```

---

## 15. Configuration

- **Files:** one YAML (`default.yaml`) plus profile overlays. Validated at load (types, ranges, required secrets present only when needed).
- **Config hash:** a hash of the resolved config, excluding paths and secrets, stamps every artifact.
- **Required sections:**
  - `run_profile`, `seed`, `paths`
  - `sec` (`max_rps`, `user_agent_env`, cache)
  - `universe`, `study`, `market`, `timing`, `text` (sections, char budgets, chunk sizes)
  - `taxonomy_version`, `targets`, `labels`, `splits`
  - `models` (core list + revisions), `rag`, `distill`
  - `tuning`, `gates`, `audit`, `bootstrap`, `fdr`, `trivial_effect_bps`
  - `backtest`, `budget`, `extensions`
- **Pre-registration:** `preregistration.yaml` is hashed by `secprice prereg-freeze` (writes `manifests/prereg_hash.txt`). Stages that score the development holdout refuse to run without it.

---

## 16. Data contracts (Parquet; one row definition each)

Every artifact also writes a sidecar manifest (`*.manifest.json`):
- stage
- config hash
- git commit
- input artifact hashes
- row count
- start and end times
- warnings

| Artifact | One row = | Key columns |
|---|---|---|
| `universe.parquet` | firm | cik, tickers_history, name, sic, ff12_industry, index_start, index_end, price_start, price_end, delisted_flag, eligible, exclusion_reason |
| `filing_manifest.parquet` | observed filing | accession, cik, form, is_amendment, filing_date, acceptance_raw, acceptance_time, ts_quality, period_of_report, primary_doc, items_raw, items_source, url, audit_reason_code, parse_status, checksum |
| `sections.parquet` | filing × section | accession, section_id, text_raw_ref, text_masked_ref, n_chars, n_tokens_est, parse_method, boilerplate_share, mask_counts |
| `exhibits.parquet` | filing × exhibit | accession, exhibit_type, text_ref, is_results_release |
| `xbrl_facts.parquet` | fact × accession | cik, accession, taxonomy, tag, unit, value, period_start, period_end, fy, fp, form, filed, public_available_time |
| `eps_release.parquet` | release | accession, eps_diluted, eps_basic, period_end, parse_confidence, xbrl_match_flag (QA) |
| `prices.parquet` | firm × session | cik, ticker, date, open, high, low, close_raw, close_split_adj, total_return_index, volume, source, source_version |
| `corporate_actions.parquet` | action | cik, date, type, value |
| `factors.parquet` | session | date, mkt_rf, smb, hml, rmw, cma, mom, rf |
| `events.parquet` | event | event_id, cik, accessions, forms, categories, event_type, earnings_type, cluster_id, acceptance_time, public_available_time, decision_time, ts_quality, session_bucket, trackA_day0, tinfo_base_date, tinfo_eligible, ttrade_entry_date, contam_minutes, overlap_flag_h{1..60} |
| `targets.parquet` | event × horizon × target_def | event_id, horizon, target_def, raw_ret, ar_mm, ar_carhart, sigma, label3, vol_ratio, abs_std, censored |
| `features/<family>.parquet` + `feature_registry.csv` | event | event_id + features; the registry gives name, family, source, availability rule, max availability time, strict/retrospective |
| `splits.parquet` | event × scheme | event_id, scheme, fold, role (train / val / purged / dev / locked), cohort flags |
| `embeddings/<model>/<rev>/` | chunk / doc | float16 matrix + index parquet (event_id, accession, section, chunk_id, offsets) |
| `extractions.parquet` | event × field | event_id, field, present, direction, magnitude, unit, certainty, evidence, source (teacher / student), confidence |
| `predictions.parquet` | run × event | run_id, model_id, rung, fold / block, p_up, p_side, p_down, yhat_ar, yhat_vol |
| `results_long.parquet` | metric value | run_id, track, rung, model_id, revision, leaderboard (strict / retro), cohort (common / max), target_def, horizon, block, event_set (all / clean), coverage, metric, value, ci_low, ci_high, n_events, n_firms, n_dates, n_overlap |
| `gold/*.parquet` | item | gold_split, item_id, annotator, labels, spans, timestamps |

**Rules**
- Text is stored zstd-compressed, by reference.
- Raw HTML is not retained beyond the cache window, but source URL and checksum are.
- Nulls mean "not applicable or unavailable", never zero.

---

## 17. Stage pipeline

### 17.1 Stages
Each stage declares its inputs, outputs, assertions and part. Each writes a stage card in `docs/stage_cards/`.

| ID | Stage | Part | Outputs | Key assertions |
|---|---|---|---|---|
| S00 | env / config / prereg check | CPU | environment manifest, config hash | config valid; prereg present |
| S01 | HF dataset audit | CPU | `hf_dataset_audit.md` | no labels used |
| S02 | Universe | CPU | `universe.parquet`, universe table | ≥ n_stocks unique firms (full) or failure report |
| S03 | Calendar, prices, actions, factors | CPU | prices, actions, factors, coverage CSV | split/dividend unit tests pass; explicit downloader parameters recorded |
| S04 | EDGAR index + coverage audit | CPU | `filing_manifest`, coverage CSVs, reason codes, `data_coverage_summary.md` | no 8-K "expected"; timestamp-quality table |
| S05 | Download + parse + taxonomy + anonymize | CPU | sections, exhibits, tables, parse QA | taxonomy version recorded; `OTHER_OR_UNMAPPED` share reported |
| S06 | XBRL + release EPS + SUE | CPU | xbrl_facts, eps_release, SUE features | no later-cluster source; ordering by public time |
| S07 | Events + clocks | CPU | `events.parquet` | four timestamps; buckets; earnings types; overlaps |
| S08 | Targets + labels | CPU | `targets.parquet` | INTRA not T_info-eligible; per-horizon censoring |
| S09 | Structured + lexical features | CPU | features (price, metadata, accounting, lexical, change, novelty), registry | availability ≤ decision_time for all |
| S10 | Splits + cohorts + leakage audit | CPU | `splits.parquet`, `leakage_audit.csv` | all assertions; 4 planted leaks caught |
| S11 | Baselines L0–L4, L6 | CPU | predictions, results | common cohort recorded |
| S12 | Gold export / import | CPU | templates; gold splits | gold-test blind; split integrity |
| S13 | Preflight (GPU) | GPU | workload report | budgets OK |
| S14 | Core embeddings (one model at a time) + vintage ChronoBERT | GPU | embeddings | pooling + change vectors; revisions pinned |
| S15 | L5 / L7 heads + tuning; fine-tune top 2 if G3 | GPU / CPU | predictions, curves | finalist caps; before/after |
| S16 | Core RAG: indexes, retrieval eval, extraction (gold-val), freeze | GPU | indexes, rag_tournament.csv | historical path time-filtered; current path separate |
| S17 | Teacher sample + student distillation → L8 | GPU | extractions, student model | training-period only |
| S18 | Development holdout + diagnostics (MCS / PBO) + G4 freeze | CPU | frozen_architecture.json + hash | no test access |
| S19 | `unlock-test`: retrain + locked test + clean test | CPU / GPU | locked results | single use; logged |
| S20 | Track A event study + correlations + topics | CPU | track_a results | clean / all; pre-trends |
| S21 | Drift tests | CPU | hypothesis_tests.csv | FDR families; verdict rubric |
| S22 | Backtest | CPU | backtest.csv | entry-rule assertions |
| S23 | Track C final (gold-test) | CPU | content_validity.csv | gold-test touched once |
| S24 | Figures + report | CPU | figures, `results_summary.md` | computed values only |
| X* | Extensions | per flag | per module | gate G3c |

### 17.2 CLI commands (implemented in `cli.py`)
- **`secprice run`:** `--profile`, `--stages` (single, list or range), `--force` (rerun even if done).
- **`secprice status`:** done / pending / stale per stage.
- **`secprice preflight`**
- **`secprice prereg-freeze`**
- **`secprice freeze`** (G4)
- **`secprice unlock-test`:** requires the G4 hash; refuses a second run unless `--acknowledge-reuse`, which relabels the block as development.
- **`secprice gold export|import|agreement`**
- **`secprice sync push|pull`**
- **`secprice report`**
- **`secprice env`**

### 17.3 Artifact rules
- **Layout:** `DATA_ROOT/<profile>/<stage>/<config_hash>/...` with a `_SUCCESS` marker.
- **Skipping:** a stage is skipped if `_SUCCESS` exists and its input hashes match; otherwise it is stale.
- **Atomic writes:** write to temp, then rename.
- **Checkpointing:** downloads and embeddings checkpoint in batches, and resume without duplicates.

---

## 18. Testing

### 18.1 Unit tests (must pass before G1)
1. **Session buckets:** across DST changes, holidays, early closes, and the 09:30 / 16:00 boundaries.
2. **Timestamps:**
   - acceptance-time timezone parsing against an SGML header fixture
   - public / decision computation
   - quality-B rule
3. **T_info:** base and exit per bucket; INTRA excluded.
4. **T_trade:** entry = first open at or after `decision_time` for every bucket (including INTRA and late PRE); h = 1 = open → close.
5. **Returns:** around known split and dividend fixtures; raw versus total return.
6. **Abnormal returns:** market model and Carhart on synthetic data with a known beta.
7. **Labels:** k-band labelling; per-horizon censoring.
8. **SUE:**
   - YTD → quarterly conversion
   - as-first-reported selection
   - **release-EPS path for 8-K events**
   - **rejection of later-cluster XBRL**
9. **EAR:** a same-cluster feature is rejected; a later-cluster feature is accepted only after the window closes.
10. **Overlaps:** flags and clean cohort.
11. **8-K taxonomy:** legacy → modern mapping by date; catch-all.
12. **Coverage:** reason codes; no 8-K expectations.
13. **Retrieval:** the historical index never returns current or future documents.
14. **Splits:** purge / embargo boundaries; date-disjointness; accession grouping.
15. **Leakage self-tests:** all four planted leaks are caught.
16. **Cohorts:** the common cohort is identical across compared models.
17. **Metrics:** each metric matches a hand-computed fixture; nulls for inapplicable cases.
18. **Bootstrap:** block structure respected.
19. **Resilience:** a resume does not duplicate rows; checkpoint reload reproduces predictions.
20. **Config:** validation and hash stability.

### 18.2 Integration
- **Offline smoke** on fixtures, S00 → S12 and S20 → S24 (CPU), with a GPU-free fallback (TF-IDF in place of embeddings) so the full chain runs locally in minutes.
- **Online smoke:** `smoke` profile against the live SEC and price source (marked `network`).

### 18.3 Optional CI
A GitHub Actions workflow runs unit and offline integration tests on push (CPU only).

---

## 19. Logging, manifests, reproducibility

- **Logs:** structured logs per stage (`logs/<stage>.log`), plus `errors.log` with error-rate tracking.
- **`run_manifest.json`:** profile, config hash, git commit, stage timings, wall time and GPU-active time, model revisions, data source versions.
- **`frozen_architecture.json`:** the G4 freeze, with chosen models, hyperparameters, thresholds and their hash.
- **Seeds:** set for Python, NumPy, torch and Optuna. Note where GPU nondeterminism may remain.

---

## 20. Figures

### 20.1 Core figures (`viz/`; each with a "how to read" caption in the stage card and walkthrough notebook)

**Data and coverage**
- F01 coverage heatmap (observed versus parsed) by year × form
- F02 audit reason codes over time
- F03 8-K category availability by year (taxonomy version)
- F04 universe: unique firms versus firms per year, delisted coverage
- F05 timestamp quality and session buckets by year
- F06 pre-publication contamination-window distribution and public-lag sensitivity

**Events and targets**
- F07 event-clock diagram (Track A, T_info, T_trade)
- F08 earnings types over time
- F09 overlap counts and clean share
- F10 target distributions and class balance

**Track A**
- F11 CAR paths by form and category (all versus clean)
- F12 by bucket
- F13 pre-trend diagnostics
- F14 abnormal volume and volatility response
- F15 correlation heatmaps
- F16 SUE-drift quintile CAR[+2,+60]

**Track B**
- F17 split timeline with purge / embargo, holdout, lockbox and per-fold provenance
- F18 ladder L0–L8 lift with CIs (common cohort; strict and retrospective)
- F19 vintage versus static (H4)
- F20 loss curves
- F21 validation curves
- F22 learning curves
- F23 before / after tuning
- F24 confusion matrices
- F25 ROC / PR curves
- F26 calibration
- F27 coverage–accuracy curve
- F28 rolling IC
- F29 T_info versus T_trade
- F30 backtest (cumulative, costs, lag)
- F31 performance by form, category, bucket, timestamp quality, size, sector, year

**Track C**
- F32 retrieval metrics by setup
- F33 extraction F1 / faithfulness (gold-test)
- F34 numeric fidelity
- F35 judge versus human
- F36 student versus teacher versus human

**Sanity**
- F37 permutation null
- F38 random-split demo
- F39 adversarial-validation ROC (with interpretation note)
- F40 MCS / PBO diagnostics

### 20.2 Extension figures
Produced by each extension module: encoder tournament, RAG variants frontier, time-series models, stacking weights, generation-metric panel.

---

## 21. Final report (`reports/results_summary.md`, computed values only)

1. **Data integrity:** universe naming and table; coverage and reason codes; 8-K category availability; timestamp quality; Hugging Face audit; limitations (survivorship-aware, large-cap tilt, selected sections).
2. **Design:** four timestamps, targets, labels, overlaps, censoring; pre-registration hash.
3. **Leakage and provenance audit:** assertions, planted leaks, sanity experiments, review-trigger outcomes, ledger, per-fold labels.
4. **Track A:** results (all / clean), pre-trends, correlations, SUE drift and other verdicts with power.
5. **Track B:**
   - ladder (common and maximum cohorts)
   - strict and retrospective leaderboards
   - H1–H4 results with CIs
   - tuning before / after
   - holdout and locked-test results (clean subsets for known-cutoff models)
   - selective prediction
   - T_info versus T_trade
   - backtest
6. **Track C:** gold-test content validity, single- or double-rater status.
7. **Fit diagnostics.**
8. **Extensions run** (if any) and their gates.
9. **Limitations, resource use, decision log.**
10. **Executive summary** ending with **the smallest defensible claim**.

---

## 22. Milestone plan for Claude Code

Each milestone ends with: tests green, stage cards written, `CLAUDE.md` updated, a git commit and a short summary to the owner (what works, what's stubbed, decisions logged).

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold: pyproject, package, config loader + validation + hash, storage / manifests, runner + CLI skeleton, logging, test harness, `CLAUDE.md`, README | `secprice status` and `secprice run --profile smoke --stages S00` work; config tests pass |
| M1 | SEC client (rate limiter, UA from env, cache, backoff) + submissions + headers; universe (S02); EDGAR index + coverage audit (S04); HF audit (S01) | Smoke universe and manifest built; coverage summary readable; acceptance-time timezone test passes; recorded fixtures for offline tests |
| M2 | Calendar, prices (provider interface), corporate actions, factors (S03) | Split / dividend / early-close tests pass; coverage CSV shows delisted gaps |
| M3 | Download, parse, sections, exhibits, tables, normalize, anonymize, taxonomy (S05) | Parse QA table; section-extraction spot check on fixtures; taxonomy tests pass |
| M4 | XBRL facts, release-EPS parser, SUE (S06) | SUE tests pass, including the later-cluster rejection; coverage by year reported |
| M5 | Events, four timestamps, earnings types, clusters, overlaps (S07); targets / labels / censoring (S08) | All timing and target unit tests pass; F05–F10 render |
| M6 | Features + registry (S09); splits, cohorts, leakage audit, planted leaks (S10) | All four planted leaks caught; availability assertion passes |
| M7 | Metrics module, bootstrap, FDR, results table; baselines L0–L4, L6 (S11); core figures for coverage and Track B baselines | **G1:** offline integration smoke S00 → S11 passes end to end on CPU |
| M8 | Track A event study, correlations, topics, SUE drift, momentum, other drift tests with verdict rubric (S20–S21) | F11–F16; hypothesis_tests.csv with verdict labels |
| M9 | Gold-set tooling (S12) + annotation guide | Export / import round-trip; gold splits enforced; blind gold-test |
| M10 | Colab GPU runner; preflight (S13); core embeddings + vintage ChronoBERT (S14); heads, tuning, G3, fine-tune (S15) | GPU smoke runs on Colab; L5 / L7 results; curves F20–F23 |
| M11 | Core RAG + extraction + guardrails (S16); teacher / student distillation → L8 (S17) | Retrieval and extraction metrics on gold-val; G3b decision logged |
| M12 | Holdout, diagnostics, G4 freeze, unlock-test, locked and clean tests (S18–S19); backtest (S22); Track C final (S23) | Lockbox single-use test passes; strict / retrospective leaderboards produced |
| M13 | Figures + report builder (S24); walkthrough notebook | `results_summary.md` generated from artifacts; walkthrough runs on smoke artifacts |
| M14 | **G2:** pilot run end to end; review with owner | Pilot report; budget projection for full |
| M15+ | Full run; then extensions one by one (each needs flag + G3c + budget) | Per extension |

**Gates**
| Gate | Before | Passes when |
|---|---|---|
| G1 | anything on GPU | unit tests + offline smoke integration pass |
| G2 | full run | pilot completed; preflight within budget; owner opts in |
| G3 | fine-tuning | text lift over L6 ≥ 0.01 macro-F1 on validation, CI excludes 0 |
| G3b | teacher extraction at scale | gold-val extraction F1 and faithfulness thresholds met |
| G3c | any extension | core complete; budget remains; a pre-stated question the extension answers |
| G4 | locked test | frozen manifest hashed; prereg hash present |

---

## 23. Acceptance checklist

**Data**
- [ ] Unique-firm and per-year counts; universe named precisely; survivorship-aware wording.
- [ ] Coverage observed versus parsed, with reason codes; no 8-K expectation; 8-K taxonomy versioned with a catch-all and availability by year.
- [ ] Timestamp-quality table; acceptance timezone verified; four timestamps per event.
- [ ] Explicit price-source parameters; raw / split / total-return series separate; corporate-action tests.
- [ ] Usable cutoff printed; per-horizon censoring; the partial final month flagged.

**Timing and leakage**
- [ ] INTRA excluded from T_info; T_trade entry at the first open at or after decision time for all buckets.
- [ ] No same-cluster EAR; no later-cluster SUE; release-EPS path implemented.
- [ ] Historical retrieval time-filtered; current document on a separate path.
- [ ] All leakage assertions pass; four planted leaks caught.

**Modelling**
- [ ] Common-cohort and maximum-available comparisons both reported.
- [ ] Strict versus retrospective leaderboards; no clean-test claims for unknown cutoffs.
- [ ] Core roster only unless extensions were gated in.
- [ ] Default and tuned results; curves and verdicts.
- [ ] The lockbox scored once.

**Content**
- [ ] Gold-dev / val / test separation; gold-test blind; second-rater status stated.

**Inference**
- [ ] Verdict rubric with trivial thresholds; FDR families; power language.
- [ ] MCS / PBO used as diagnostics only.

**Backtest**
- [ ] Section 11 rules; long-only and hypothetical long-short; borrow caveat.

**Engineering**
- [ ] Smoke is the default; preflight enforces budgets; stages resumable.
- [ ] Tests green; offline smoke runs locally.
- [ ] GPU runner works on Colab.
- [ ] No news data anywhere; the report contains computed values only.

---

## Appendix A: Professor review reconciliation (v4 → v5)

| Review item | Decision | Where |
|---|---|---|
| P0.1 Set parameters to 500 firms, Jan 2000 – Oct 2026, warm-up, predecessor filings | **Accepted** (overridable defaults) | §3 |
| P0.2 Intraday T_trade fill at the same close is unsafe | **Accepted:** all buckets enter at the first open at or after decision time; h defined identically | §5.4 |
| P0.3 Acceptance vs public vs feature-ready vs decision time | **Accepted:** four timestamps; separate current / historical evidence paths; "≤ decision_time" for features, strict "<" for prior documents | §5.1, §8, §6.3 #16 |
| P0.4 Gold-set reuse | **Accepted:** gold-dev / val / test; final numbers on gold-test only; blind gold-test | §12 |
| P0.5 Unknown cutoffs | **Accepted:** strict versus retrospective leaderboards; no clean-test claims for unknown cutoffs | §6.4 |
| P0.6 SUE cluster-aware | **Accepted and extended:** v5 also notes that XBRL is generally not available at an earnings-8-K's decision time, so 8-K events need release-exhibit EPS parsing; XBRL SUE for those events is ex-post QA or sensitivity only | §5.6 |
| P0.7 Common cohort | **Accepted:** common (primary) and maximum-available (secondary) | §6.2, §9.1 |
| P1.8 PBO / MCS as diagnostics | Accepted | §9.2 |
| P1.9 Annotation reliability | Accepted: second rater on ≥ 30% of gold-test; single-rater label otherwise; no pre-fill on gold-test | §12 |
| P1.10 Narrow the default architecture | **Accepted with modification:** core roster as the review proposed (SEC-BERT, FinBERT-tone, ChronoBERT, BGE; BM25 / hybrid / hybrid + rerank; one LLM). The owner's broader roster stays implemented-on-demand as a gated extension pack rather than deleted | §7 |
| P1.11 Simplify generation metrics | Accepted: core content metrics redefined; the full generation panel moves to `ext_generation_metrics` (the owner's requested metrics remain available) | §9.2, §7.3 |
| P1.12 Concrete backtest | Accepted: signal score, thresholds from training folds, minimum names, overlap policy, execution, borrow, corporate events | §11 |
| P1.13 Drift inference rule | Accepted: trivial thresholds in bps pre-registered; failing a test never proves absence | §3, §10.2 |
| Keep / Add / Remove lists | Applied throughout | — |
| Implementation order | Adopted as the milestone plan | §22 |
| External checks (SEC lag, HF dataset scope, ChronoBERT, FinText) | Verified and recorded | Appendix C |

## Appendix B: Earlier review reconciliation (v3 → v4), still in force
- EAR is never a same-event feature.
- Verified versus periodic earnings types.
- 8-K is event-driven; reason codes.
- Smoke default.
- Survivorship wording.
- NER is never used for money.
- Review triggers instead of stops; pre-trends interpreted.
- Provenance ledger.
- Overlap controls.
- Versioned 8-K taxonomy.
- Explicit price handling.
- Ablation hierarchy.
- Native tasks scored separately.
- Selective zero-shot use.
- Carhart / FF5 naming.
- Correct reading of adversarial validation.
- Power language.

## Appendix C: Verification notes (re-check at build time)
- **SEC timing and access:** the SEC webmaster FAQ states a 10 requests/second maximum and that filings typically appear on sec.gov 1–3 minutes after the EDGAR timestamp, possibly longer under load; the lag is not guaranteed. This underpins `timing.public_lag_minutes`.
- **`JanosAudran/financial-reports-sec`:** sentence-level 10-K records with 1d / 5d / 30d return labels; configs `large_full` / `large_lite` / `small_full` / `small_lite`; Apache-2.0. Audit only.
- **`eloukas/edgar-corpus`:** annual-report items 1–15 (incl. 1A, 7, 7A), 1993–2020, ~40.7 GB; optional QA source for 10-K section parsing.
- **`edgartools`:** MIT; requires an identity; rate-limit aware; section extraction (risk factors, MD&A), 8-K items, XBRL / company facts. Wrap it if used.
- **`fja05680/sp500`:** historical S&P 500 components since 1996 (MIT). The author notes that Wikipedia-based changes are incomplete, early years may be missing symbols, and free sources lack delisted price history.
- **Model facts** (verified in the v3 pass):
  - SEC-BERT: 1993–2019 10-K corpus
  - ChronoBERT: vintages 1999–2024 (`manelalab/chrono-bert-v1-YYYY1231`)
  - FinText Chronos: dated vintages; confirm each year exists
  - SecBERT / SecRoBERTa: cybersecurity models
  - no public weights found for BloombergGPT / FinLlama / BB-BERT
- **To verify on a sample at build time (not assumed):** the timezone semantics of `acceptanceDateTime` in the submissions JSON versus the SGML header; availability of 8-K `items` metadata for pre-2004 filings (fall back to parsing headers or text); yfinance defaults in the installed version.
