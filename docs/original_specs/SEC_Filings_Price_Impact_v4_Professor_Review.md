# Professor-Style Review of SEC Filings → Price Impact v4

**Review type:** research-design and implementation-specification review  
**Document reviewed:** `SEC Filings → Price Impact: Three-Track Quant Research Spec, v4`  
**Important limitation:** this is a review of the specification, not an audit of a generated or executed notebook. Code behavior and results still need to be inspected after Claude Code produces the `.ipynb`.

## Executive verdict

**v4 is a material improvement over v3 and is now a strong research blueprint.** The most important fixes are the separation of event-study, prospective-prediction, and content-validation tracks; the dual information-response/tradable targets; the EAR safeguard; the distinction between verified earnings announcements and periodic filings; the 8-K completeness logic; and the explicit unit tests and audit trail.

However, **v4 is still too broad for the main contribution of a solo project, and several important timing and evaluation decisions should be resolved before implementation.** I would approve it for a smoke test and data audit, but not yet for the expensive full architecture tournament.

A defensible paper could emerge from this work. Its contribution should be framed around whether SEC filing text provides robust incremental information beyond point-in-time accounting, price, metadata, and filing-change baselines. The paper should not be framed as “we tried the largest number of models” or “we built the most complex architecture.”

## What v4 improves over v3

| Area | v4 status | Assessment |
|---|---|---|
| Research aims | Three separate tracks: event study, prediction, content validity | Excellent. Association, forecasting and extraction correctness are different claims and need different targets. |
| EAR | Same-event EAR removed as a predictor; lagged use restricted | Essential correction. This fixes a serious potential leakage route in v3. |
| Event dates | Verified earnings announcements separated from periodic filing dates | Essential. A 10-Q date should not automatically be labelled an earnings-announcement date. |
| 8-K audit | No expected-annual-8-K rule; retains observed item codes and reason codes | Correct and important. 8-K filing is event-driven. |
| Intraday contamination | Two targets and session buckets, intraday excluded from T_info | Much better, although the T_trade intraday execution convention still needs correction. |
| Historical data | Raw, split-adjusted and total-return series; corporate-action tests | Strong improvement, provided the source actually supplies adequate history, including delistings. |
| Point-in-time features | Accession/time ordering and availability ledger | Good, but public-availability time must be separated from EDGAR acceptance time. |
| Model provenance | Vintage ledger and strict versus retrospective labels | Strong conceptual improvement. Unknown cutoffs still need a firmer rule for the clean test. |
| Model comparability | Native tasks separated from the common direction-prediction task | Correct. Sentiment F1 is not stock-direction F1. |
| Human evaluation | Gold filings plus passage-level event validation | Good addition, but one shared gold set must not be used for both selection and final reporting. |
| Resource discipline | Smoke-only default, preflight, caps, one encoder at a time | Excellent in principle. The overall full design still has a very large cumulative compute footprint. |
| Statistical claims | Review triggers, power language, confidence intervals and FDR | More responsible than hard-coded “high score means leakage” rules. |

## SWOT analysis

### Strengths

1. **Primary-source data:** SEC accession metadata, acceptance timestamps, filing types, item codes and XBRL allow an auditable disclosure-event panel.
2. **A clear separation of scientific questions:** event-study associations, predictive information and content validity are no longer conflated.
3. **Strong leakage awareness:** the spec covers feature availability, overlapping outcomes, point-in-time accounting, retrieval, analogues, stacking, pretrained-model cutoffs and planted-leak tests.
4. **Meaningful baselines:** price-only, metadata-only, XBRL, dictionary, TF-IDF and filing-change models create a credible ladder for incremental-value testing.
5. **A credible null result is allowed:** the text-lift gate avoids assuming deep NLP must win.
6. **Reproducibility:** revision pinning, saved manifests, preregistration, resumable caches, unit tests, resource usage and computed-only reports are appropriate research practice.
7. **Scope discipline around other data:** the specification excludes news/social/transcripts/analyst text while preserving the explicit possibility that some pretrained model weights may have used such data, with provenance disclosed.

### Weaknesses

1. **Overengineering:** many encoders, multiple local LLMs, time-series foundation models, a TFT challenger, analogue retrieval, distillation, many RAG variants, stacking, PBO/MCS and 70+ visualizations create a very large search space.
2. **Potentially weak point-in-time clean test:** generic checkpoints may have unknown training cutoffs. If an unknown-cutoff model enters the final architecture, a “clean test after the latest model cutoff” cannot honestly be dated without further evidence.
3. **Execution fidelity:** the proposed intraday T_trade entry at day-0 close is not safely tradable under daily-bar assumptions for all intraday filing times.
4. **Gold-set reuse risk:** the RAG tournament uses overlapping development and gold sets; final content-quality scores need a held-out human-evaluation slice not used for prompt, retriever or model selection.
5. **Human annotation reliability:** one annotator is acceptable for a student exploratory study, but less persuasive for publication. A second independent annotator on a stratified subset should be treated as a priority, not merely a possibility.
6. **Coverage and point-in-time universe:** historical index membership and free prices may not capture delisted names accurately. The “survivorship-aware, not survivorship-free” wording is appropriate and must remain.
7. **Hypothesis sprawl:** the many forms, categories, horizons, sectors, subperiods and architectures invite multiple-testing problems even with FDR and selection-bias methods.

### Opportunities

1. Test whether **filing-change/novelty features** beat absolute document sentiment.
2. Test whether text adds value **after conditioning on point-in-time XBRL and price baselines**.
3. Study which 8-K event categories and selected 10-K/10-Q sections contain incremental information.
4. Compare retrospective encoders with chronological model vintages and quantify the sensitivity to pretrained-model provenance.
5. Validate a small structured extraction schema with source-grounded evidence and numerical cross-checks.
6. Produce a reusable, documented SEC filing-to-market event panel and an honest null or positive result.

### Threats

1. **Acceptance/public-availability mismatch:** SEC notes filings may become available on sec.gov one to three minutes after the EDGAR timestamp, sometimes longer. Treating acceptance time as identical to model-readable public time can make the historical decision clock too optimistic.
2. **Extended-hours reaction:** daily bars cannot reveal all reactions between the last close and regular-session open. This is measured/limited, not eliminated.
3. **Pretrained-model memory:** a checkpoint can have seen the test filing or later information; an unknown training cutoff is not proof of safety.
4. **Survivorship and price-source quality:** lack of delisted returns can materially bias the universe and strategy results.
5. **LLM extraction errors:** plausible JSON may still be wrong, unsupported, or numerically inconsistent.
6. **Backtest overfitting:** a tournament with many choices can select a winner by chance, even with a locked test and multiple-testing controls.
7. **Colab variability:** runtime/GPU/VRAM/storage are not guaranteed; a specification this large may exceed practical budgets.

## Priority changes before any expensive run

### P0 — Must resolve before pilot results are trusted

#### 1. Set parameters to the already requested study

Earlier project instructions specify a target of at least 500 unique stocks and January 2000 through October 2026. v4 still leaves `N_STOCKS`, `STUDY_START`, and `STUDY_END` as TBD.

Recommended defaults:
- `N_STOCKS = 500` (minimum unique companies across the study; report unique firms and firms represented each year separately).
- `STUDY_START = 2000-01-01`.
- `STUDY_END = 2026-10-31`, but the actual usable cutoff must be the last observed price/document date at runtime. If the run takes place before October 31, explicitly report the month as partial and apply per-horizon censoring.
- `MARKET_WARMUP_START`: at least 250 trading sessions before the first target event, plus any additional history needed for the volatility and beta estimators. For January 2000 events, retrieve price history into 1998/1999 where available.
- Load predecessor filing text as needed to compute point-in-time filing-change features at the start of the sample, while keeping evaluated events within the registered window.

Do not claim 500 companies in every year unless that is actually verified; “500 unique firms across the study” is a different claim.

#### 2. Fix the intraday tradable target

Section 6.4 makes `T_trade` for INTRA events start at the same day’s close. With daily data, one cannot guarantee a fill at the closing price after an event accepted at an arbitrary intraday time. Closing-auction cutoffs and execution latency matter.

Recommended rule:
- Keep INTRA excluded from `T_info`.
- For an intraday event, either (a) use intraday bars and a documented latency/execution assumption, or (b) make the daily-data backtest enter at the next session’s open. The latter is the safer default.
- Make the `T_trade` endpoint/horizon definition explicit so h=1 means the same economic holding interval for PRE, POST, NONTRADING and INTRA events.

#### 3. Distinguish SEC acceptance from public availability and decision time

The specification sets decision time to the acceptance timestamp but also says only information public before that time may be used, while the current filing itself must be the model input. Resolve the language and implementation with four explicit timestamps:

- `acceptance_time` — EDGAR acceptance metadata.
- `first_public_available_time` — exact if observed; otherwise conservative documented proxy/lag and quality flag.
- `feature_ready_time` — when parsing/inference could have completed under the assumed research deployment setup.
- `decision_time` — the point at which the prediction is assumed to be made.

The current filing can be passed directly as the contemporaneous input only once it is assumed public and readable. Historical-memory/RAG indexes must exclude the current filing and all later documents; current-document evidence can be a separately supplied context. Define the legal feature rule as “available no later than decision time” while keeping strict `< decision_time` for prior-document retrieval where appropriate.

#### 4. Split the gold set into selection and final evaluation portions

The RAG tournament uses a development sample that overlaps the gold set. If the same annotations decide the winner and produce the reported final extraction/retrieval scores, those scores are selected upward.

Recommended split, all sampled from training-period filings:
- gold-development: schema/prompt iteration and early debugging;
- gold-validation: RAG/LLM model selection;
- gold-test: untouched final content-validity reporting.

Keep a separate untouched market-prediction locked test as well. Do not use price outcomes as semantic labels. Stratify by form, era and event category; deliberately sample rare categories; report per-label counts and uncertainty. Human references used for summaries should not overlap prompt/model selection if the report calls their score a held-out result.

#### 5. Give “unknown model cutoff” a strict operational consequence

V4 says unknown cutoffs count as overlapping, which is the right conservative label. But if a final architecture includes an unknown-cutoff LLM, the latest model cutoff is unknown and a clean test after it cannot be established.

Require two result tracks:
- **Strict point-in-time leaderboard:** only models/vintages with defensible training cutoffs before the event period, with fold-specific provenance.
- **Retrospective frozen-model leaderboard:** static models with later or unknown pretraining cutoffs; useful as a practical model comparison, but not described as a fully point-in-time historical simulation.

Unknown-cutoff models must not qualify for the strict clean-test claim. A post-cutoff clean sample must be defined only when the cutoff is known. Keep the ChronoBERT/FinText vintage experiment as the clean temporal comparator and verify each vintage actually exists and its stated training cutoff is documented.

#### 6. Make SUE feature availability event-cluster-aware

V4 correctly prohibits using same-event EAR as a feature and defines an explicit SUE calculation. Also ensure that when an earnings 8-K is the event, an EPS value from a later 10-Q/10-K in the same earnings cluster cannot leak into that 8-K prediction. SUE may use the event’s own filing/exhibit or earlier public accessions, never a later periodic report. Enforce this with accession and actual-public-availability ordering.

#### 7. Define reproducible ablation/sample eligibility

The L0–L9 ladder is good, but every rung must use the same event cohort for the primary comparison. If an accounting feature such as SUE exists only for a subset, choose and report both:
- a **common-cohort comparison** where every model is scored on the same rows;
- a **maximum-available-data comparison** where the sample may differ, clearly labelled as non-apples-to-apples.

This avoids claiming a model lift caused by NLP when it actually comes from a different set of events.

### P1 — Important for paper quality

#### 8. Treat PBO/MCS as diagnostics, not a substitute for chronology

Keep walk-forward selection and the locked out-of-time test as the principal generalization evidence. PBO/CSCV and the Model Confidence Set can supplement the analysis, but should not replace temporal validation or be used to rehabilitate a poor locked-test result. Explain exactly which loss/return streams and configurations enter each diagnostic.

#### 9. Strengthen annotation reliability

The current gold programme (100–300 filings and 300–500 validation passages) is a reasonable start for a student project, but category-specific precision/recall estimates can be imprecise, especially for rare events. A second annotator should independently label a stratified subset, ideally including a meaningful sample of positive and negative cases per important category. Report agreement and adjudication. If only one annotator is possible, label content-validity findings as single-rater and provisional.

Avoid showing annotators LLM-generated labels on the final blind subset. If pre-fill is used to reduce effort, randomly reserve an independently annotated blind portion and report that pre-fill may cause anchoring.

#### 10. Narrow the default research architecture

Keep v4’s gating, but reduce what is mandatory in the main paper:

**Core model comparison:**
- price/metadata + point-in-time accounting baseline;
- Loughran–McDonald/dictionary features;
- TF-IDF or BM25 baseline;
- SEC-BERT;
- FinBERT-Tone or FinBERT-pretrain;
- ChronoBERT vintage model;
- one modern encoder/embedding baseline.

**Core RAG comparison:**
- BM25;
- hybrid sparse+dense retrieval;
- hybrid plus reranking;
- one verified local LLM for extraction.

**Optional extension gates:**
- one second LLM only if extraction accuracy is inadequate;
- analogue retrieval only after the core baseline works;
- one time-series foundation model only as a baseline if it answers a clearly defined incremental question;
- no default GraphRAG, Agentic RAG, Self-RAG, QLoRA, continued pretraining or large TFT tournament.

V4’s architecture search may remain in an appendix/extension plan, but not every branch needs to run for the paper to be complete. “Every tower earns its place” should mean components can be skipped if evidence or budgets do not justify them.

#### 11. Simplify generation metrics

For SEC extraction, prioritize:
- event-field precision/recall/F1 against human labels;
- numeric exact match / tolerance against source and XBRL;
- evidence citation/offset accuracy;
- faithfulness judged by a calibrated local evaluator and human audit;
- retrieval Recall@k / nDCG / MRR on held-out qrels;
- cost and latency.

ROUGE/BLEU/BERTScore-type metrics are only relevant if high-quality human summary references are a genuine sub-question. They are not a substitute for numeric fidelity or evidence correctness; a huge generation metric panel is not needed in the core paper.

#### 12. Make the economic backtest more concrete

The current specification sets event entry/exit, equal weights, transaction costs and factor-adjusted alphas. Add:
- an exact signal score (e.g. expected abnormal return or P(Up)−P(Down));
- the cross-sectional ranking/decile rule and minimum names per side;
- the precise rebalance and overlapping-position policy;
- execution price and timestamp eligibility for every event bucket;
- borrow/short availability assumptions and short borrow costs, or explicitly constrain the result to a hypothetical long-short signal;
- treatment of halts, mergers, spin-offs and delisting proceeds;
- a clear distinction between event-level forecast metrics and a portfolio backtest.

Do not treat a 10/20 bps cost sensitivity as proof the strategy is investable if borrow, capacity, price-data quality and delistings remain unknown.

#### 13. Make the “supported drift” criterion an inference rule, not a universal truth

The proposed rule—FDR-adjusted p<0.05, confidence interval outside trivial effects, same sign in at least two of three subperiods, clean-event robustness, and cost/lag robustness—is conservative and defensible as a pre-registered rubric. Define “economically trivial” in units before looking at results. Do not assert that an anomaly is universally absent merely because it fails one chosen test; use the spec’s `inconclusive` verdict when power/coverage is insufficient.

## What to keep, add, and remove

### Keep
- The three-track structure.
- Dual event-reaction and tradable targets, once execution rules are corrected.
- SEC-only input text with market data used for outcomes/features.
- Full observed filing manifest and explicit reason codes.
- Point-in-time XBRL and filing-change features.
- Strong baseline ladder, identical-cohort comparisons, walk-forward validation and locked test.
- EAR safeguard, verified-earnings event types, overlap analyses and censoring.
- Model provenance ledger, selective prediction, calibration and explainable uncertainty.
- Smoke-only default and resource preflight.

### Add
- Defaults reflecting the already established 500-stock, Jan 2000–Oct 2026 intent.
- Market warm-up history before Jan 2000 for event beta/volatility estimation.
- Separate acceptance, public-availability, feature-ready and decision timestamps.
- Distinct historical retrieval versus current-document evidence paths.
- Held-out RAG/content-validity test subset, separate from dev/selection.
- Strict point-in-time versus retrospective leaderboards; no clean-test claim for unknown cutoffs.
- A clear common-cohort ablation table.
- Execution feasibility and short-borrow caveats in the backtest.
- Second-rater agreement if publishing content-validity claims.

### Remove or defer
- Running all RAG variants by default; keep only a small tournament in the main experiment.
- Multiple 7B/8B/13B/65B LLMs; keep one local generator and one challenger only if needed.
- Default time-series foundation model and TFT tournament; these can become a separate follow-up study.
- Default GraphRAG, Self-RAG, Agentic RAG, QLoRA and continued pretraining.
- Redundant BLEU/ROUGE metrics if summaries are not a primary research objective.
- Treating all seven drift topics as equally important. Keep SUE-conditioned earnings drift as qualified exploratory analysis; keep dividend and restructuring only where verified counts/power are sufficient; explicitly label analyst revisions, insider and institutional effects as not testable in core scope.
- Any implication that 500 historical index constituents provide a fully representative or survivorship-free US equity universe.

## Recommended publishable core design

### Main contribution

“Do SEC filing text and evidence-grounded event features add reliable out-of-time information beyond point-in-time market, filing metadata, accounting and filing-change baselines?”

### Main experiments

1. **Dataset and event integrity:** audited coverage, mapping, timing, parsing, overlap and censoring.
2. **Event-study track:** abnormal-return/volatility association by form, section and 8-K category, with clean-event and pre-trend sensitivity.
3. **Predictive track:** common-cohort L0–L8 ladder, with walk-forward tuning and a locked temporal test.
4. **Temporal provenance experiment:** strict chronological model vintage versus retrospective static model, with limits stated.
5. **Small content-validation experiment:** one event schema and a compact RAG tournament with held-out human evaluation.
6. **Economic check:** one explicitly defined event-driven portfolio with realistic entry, exit, costs and caveats.

### Primary hypotheses

- H1: SEC text adds out-of-time predictive value beyond price, metadata and accounting baselines.
- H2: changes in filing language add more value than absolute sentiment alone.
- H3: validated structured event extraction adds value beyond frozen embeddings and lexical features.
- H4: strict point-in-time model vintages yield different performance from retrospectively trained static checkpoints.

Keep the larger tower tournament as optional extensions, not conditions for the paper’s completion.

## Publishability assessment

- **Portfolio / capstone project:** strong potential if the data and code pipeline actually work and results are reported honestly.
- **Student research workshop / conference:** plausible after the P0 changes, a literature review, human validation and a credible held-out test.
- **Peer-reviewed journal:** possible, but not guaranteed. A publishable contribution requires novelty relative to existing work on filing-language changes, filing uncertainty, financial NLP and temporal model leakage; clean experiments and evidence matter more than the number of models.

The defensible claim should be the smallest claim supported by the data. A null result—e.g. that RAG/LLMs do not improve beyond filing-change and XBRL baselines—can be publishable if it is methodologically credible and clearly distinguished from a failed or incomplete implementation.

## Implementation order

1. Resolve parameters and target-clock semantics.
2. Implement SEC/market coverage audit and report actual missingness.
3. Implement event timestamps, targets, SUE and EAR guards with unit tests.
4. Implement splits, planted-leak tests and baselines.
5. Run the small encoder benchmark and freeze the common cohort.
6. Create and split the human gold set.
7. Add the compact RAG experiment only if the data and baseline stages pass.
8. Freeze the development choices, run locked test once, and produce the paper tables/figures.

## External checks worth preserving in the build instructions

- SEC webmaster guidance states the maximum access rate is 10 requests/second and notes that filings may become available on sec.gov about 1–3 minutes after the EDGAR system timestamp (sometimes longer): https://www.sec.gov/files/about/webmaster-faq.htm
- The Hugging Face SEC financial-reports dataset is described as annual 10-Ks from 1993–2020; its `large_full` form field is documented as always `10-K`. It is an audit/cross-check dataset, not the complete 10-K/10-Q/8-K historical corpus: https://huggingface.co/datasets/JanosAudran/financial-reports-sec
- ChronoBERT collection lists year-ended model vintages from 1999 through 2024: https://huggingface.co/collections/manelalab/chronobert
- FinText Chronos US collections include dated model vintages; verify actual availability and cutoff per checkpoint rather than assuming each year through the study end exists: https://huggingface.co/collections/FinText/chronos-mini-us

## Final recommendation

**Proceed with v4 for smoke-mode implementation and data auditing. Do not yet launch the full tournament.** First implement the P0 items, especially the parameter defaults, intraday T_trade execution rule, public-availability clock, independent gold-set evaluation, unknown-cutoff policy, and same-cohort comparisons. Then let pilot results decide which high-cost components deserve to remain in the main paper.
