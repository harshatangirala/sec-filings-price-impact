# Quant review of the SEC-filings price-impact project

Reviewer stance: chief quant officer reviewing a research project for promotion to a production research programme. Date: 3 November 2026. Scope: every stage from data to code to results. Everything below is grounded in the repository code, the stored stage outputs, and numbers re-computed for this review (`evidence/review_facts.py`). Where something is an estimate or a judgement, it says so.

## How to read this folder

| File | Contents |
|---|---|
| `00_EXECUTIVE_SUMMARY.md` | Verdict, scorecard, the ten moves that matter, what not to do |
| `01_SWOT.md` | The full SWOT with evidence for every line |
| `02_data_and_universe.md` | Sources, universe, prices, corporate actions, timestamps, parsing, alternative data |
| `03_preprocessing_features_text.md` | Cleaning, masking, chunking, price, accounting, lexical and embedding features |
| `04_events_targets_labels.md` | Event definition, clocks, targets, abnormal-return model, labels, horizons |
| `05_modeling_training_selection.md` | Models, losses, calibration, tuning, ladder design, selection rule, RAG, fine-tuning |
| `06_validation_splits_inference.md` | Splits, purge and embargo, power, bootstrap, multiple testing, lockbox |
| `07_backtest_portfolio_execution.md` | Backtest engine, portfolio construction, costs, capacity, risk, monetisation routes |
| `08_engineering_mlops_governance.md` | Code, tests, CI, determinism, data versioning, process controls, live operation |
| `09_roadmap_and_experiments.md` | Phased plan, experiment cards with acceptance and kill criteria, budget tiers |
| `10_research_agenda_alpha_ideas.md` | Ideas that could turn this into a world-class programme, ranked |
| `11_appendix_issue_register_evidence.md` | File-and-line issue register, evidence table, reproducibility notes |
| `12_SECOND_PASS_FINDINGS.md` | Empirical second pass: three corrections to this review, six new weaknesses (accounting rule, calendar features, direction decay, magnitude baseline), confirmed strengths |
| `evidence/` | The script and CSVs behind the new numbers |

## Verdict in one paragraph

The project is a strong piece of research infrastructure and an honest null result. Its point-in-time discipline, leakage testing, and self-correction are better than most published work in this area, and the pipeline is reproducible end to end. It is not yet a world-class quant project, for four reasons that are about the research design and not the code. First, **power**: 150 surviving large caps and 8,199 locked-test events cannot detect a directional information coefficient below about 0.036 to 0.042 (measured design effects, file 12), and an IC of 0.02 to 0.03 would still be worth roughly 1.0 to 1.5 gross information ratio at our breadth, so "no detectable direction skill" does not mean "no tradable skill". Second, **the information is missing**: no consensus estimates, no transcripts, no guidance numbers, and the text pipeline deletes numeric lines, so the content that decides direction on an earnings 8-K is largely absent. Third, **the trade window starts after much of the reaction**: with daily bars the first executable price is the next open, and by our own estimate roughly half of the event-window return variance occurs before it. Fourth, **the economic test is a sanity check, not a strategy**: it uses a 150-name book, flat costs, no risk model, and the long book sits in cash two days in three. Fixing those four, not tuning models, is where the value is.

**Second-pass update.** A follow-up with measured checks (file 12) changes three statements in this folder and adds six weaknesses. Direction skill is not zero: rank IC is +0.018 on 2008-2019 walk-forward and about -0.008 since 2020, so the question is decay and sign instability. The magnitude signal is volatility persistence, since one trailing-volatility feature scores 0.16 to 0.26 against L6's 0.21 to 0.28. Measured design effects are small, so the minimum detectable IC is about 0.036 to 0.042, not 0.05 or higher. New findings: the accounting availability rule counts `sue_*` columns so pre-2009 events pass with no accounting values, and `meta_year` is a model feature. Price and event data pass integrity checks.

## Scorecard (reviewer's judgement)

| Area | Grade | Why |
|---|---|---|
| Point-in-time discipline and leakage control | A- | Four timestamps, row-level availability assertions (0 violations), planted-leak tests (4 of 4), three real leaks found and fixed. Lacks an independent replication |
| Data breadth and coverage | C- | 150 survivors, no delistings, free prices, 60k events. Underpowered for the effect sizes that matter |
| Data quality and provenance | B | Pinned sources and revisions, exact acceptance times for 96% of events. Prices are Yahoo-derived and unaudited; no corporate-action or delisting audit |
| Event and target design | B | Executable open-to-close target with interval-matched benchmark is right. Entry at the open discards the overnight reaction; intraday beta and volatility are mismatched; one label scheme for all event types |
| Feature engineering | C+ | Clean point-in-time features, but duplicated and collinear columns, raw levels with regime drift, 42% missing accounting, no cross-sectional normalisation, crude earnings surprise |
| Text and NLP | C | Frozen general encoders on mean-pooled whole-filing chunks, numbers dropped, masking incomplete, extraction never scored, fine-tuning never run. The null is a null for this design, not for text |
| Modelling and selection | B- | Sensible heads and calibration, but a macro-F1 objective that does not match a trading decision, tiny tuning budgets, tuning on reported folds, 20% class flips across machines |
| Validation design | B+ | Purged, embargoed walk-forward with uniqueness weights and a hashed lockbox. Coarse folds, no nested tuning, stress folds never scored |
| Statistical inference | C+ | Paired month-block bootstrap is good. No firm clustering in classification tests, no power analysis, no multiplicity control across 237 configurations, PBO degenerate |
| Backtest and portfolio | D+ | Event-driven but toy: equal weights, 150 names, flat 10 bps, no risk model, no capacity, shorts hypothetical |
| Engineering and reproducibility | B+ | Content-hashed resumable stages, config and code hashes, lock file, 115 tests. No CI, no container, no data versioning |
| Research process and governance | B | Decision log, pre-registration file, honest corrections. Registration is internal, lock opened 2.6 minutes after freeze, selection between finalists inside noise, post-lock exploration |

## The ten moves that matter, in order

| # | Move | Why it ranks here | Effort | Section |
|---|---|---|---|---|
| 1 | **Start a forward shadow test now**: signed, timestamped predictions on every new filing, scored after the fact | The locked test is spent. A live forward sample is the only clean new test and it accrues while you build everything else | 2 weeks | 06, 08 |
| 2 | **Fix the alpha-over-window flaw in the pipeline** (`track_a/event_study.py:56`) and add calendar-time tests as a stage | It produced a convincing false result that passed every check. Every long-window number in S20 and S21 is affected | 1 week | 04, 11 |
| 3 | **Report IC, rank IC, decile spreads and ICIR next to F1**, with power for each | Quants and PMs read these. Direction rank IC is +0.018 on 2008-2019 walk-forward and about -0.008 since 2020; magnitude rank IC is 0.20 to 0.28, which a single volatility feature nearly matches (file 12) | 1 week | 05, 06 |
| 4 | **Widen the universe 6x and add delisted firms** (point-in-time constituents and delisting returns) | Power and survivorship, the two limits on every conclusion | 4 to 8 weeks | 02 |
| 5 | **Add expectations data**: analyst consensus, guidance, earnings-call transcripts | Direction on earnings is surprise versus expectation. We have no expectation | 4 to 6 weeks | 02, 10 |
| 6 | **Buy or build intraday and extended-hours bars** and re-define the entry clock | Measures and trades the first minutes; removes the open-entry handicap | 4 to 8 weeks | 04, 07 |
| 7 | **Event-type models, rank and regression objectives, cross-sectional normalisation** | One pooled 3-class model across earnings, officer changes and Reg FD is the wrong shape | 3 to 4 weeks | 05 |
| 8 | **Nested walk-forward tuning, seed-bagged deterministic heads, firm-and-month clustered intervals** | Removes selection optimism and the 20% class-flip noise | 2 to 3 weeks | 05, 06 |
| 9 | **Rebuild the text stack**: keep numerics, passage-level change against the prior filing, target-aware fine-tuning, scored extraction, complete masking | The text test has not yet been run with a text model that could win | 6 to 10 weeks | 03, 05 |
| 10 | **Rebuild the backtest as a portfolio**: risk model, neutralisation, liquidity-aware costs, capacity, and test the options route for the magnitude signal | Turns "is there skill" into "is there money" | 6 to 8 weeks | 07 |

## What not to do

1. **Do not tune more hyperparameters on the current setup.** The median tuning gain is zero and the finalists are inside noise. Effort here is wasted.
2. **Do not re-score the locked 2023 to 2026 period** and treat it as clean. It is spent (decision D-076, `lockbox_log.json`).
3. **Do not add another encoder to the same pipeline and re-run L7 minus L6.** More encoders on mean-pooled whole-filing vectors tests the same hypothesis again.
4. **Do not trust any long-window abnormal return from the current stage outputs** until item 2 is fixed.
5. **Do not present the magnitude signal as alpha** until it beats a simple event-type-times-volatility baseline. A single trailing-volatility feature has rank IC 0.16 to 0.26 against absolute abnormal return, L6 has 0.21 (development) and 0.28 (locked), and a volatility-times-event-type baseline has 0.21 and 0.22 (file 12).

## What this project is already good for

A reference implementation of point-in-time event research, a worked example of a convincing false result and how it was found, and a credible null with honest limits. Publishing that, with the corrections in this folder, is worth doing. Turning it into a programme that finds alpha needs the ten moves above.
