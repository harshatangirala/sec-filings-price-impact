# Roadmap, priorities and experiment cards

Effort is in person-weeks of one competent quant engineer unless noted. Data costs are shown as tiers because prices and licences change and must be quoted by vendors: **$** (free or under a few thousand dollars a year), **$$** (low five figures), **$$$** (above that). Verify every figure before committing.

## 1. Priority scoring

Scores are the reviewer's judgement on a 1 to 5 scale: value if it works (V), probability it works (P), inverse effort (E, where 5 is cheap). Priority = V x P x E.

| ID | Initiative | V | P | E | Score | Phase |
|---|---|---|---|---|---|---|
| I-01 | Live shadow test and prediction ledger | 5 | 5 | 4 | 100 | 0 |
| I-02 | Fix alpha-over-window in the pipeline and add calendar-time stage | 4 | 5 | 5 | 100 | 0 |
| I-03 | IC, ICIR, decile spread and power reporting | 4 | 5 | 5 | 100 | 0 |
| I-04 | Report stored sensitivity targets (latency, label threshold) | 3 | 5 | 5 | 75 | 0 |
| I-05 | CI, golden-set regression and deterministic mode | 3 | 5 | 4 | 60 | 0 |
| I-06 | Two-way clustered bootstrap and fold dispersion | 3 | 5 | 4 | 60 | 0 |
| I-07 | Nested walk-forward tuning with larger budgets | 3 | 4 | 3 | 36 | 0 to 1 |
| I-08 | Magnitude baseline and incremental IC | 4 | 4 | 5 | 80 | 0 |
| I-09 | Survivor-free, broad universe with delisting returns | 5 | 4 | 2 | 40 | 1 |
| I-10 | Consensus estimates, guidance, revisions | 5 | 3 | 2 | 30 | 1 |
| I-11 | Earnings-call transcripts | 4 | 3 | 2 | 24 | 1 |
| I-12 | Cross-sectional normalisation and feature pruning | 3 | 4 | 4 | 48 | 1 |
| I-13 | Event-type heads and rank or regression objectives | 4 | 3 | 3 | 36 | 1 |
| I-14 | 500 hand-labelled filings and scored extraction | 4 | 4 | 3 | 48 | 1 |
| I-15 | Intraday and extended-hours clock | 5 | 3 | 2 | 30 | 2 |
| I-16 | Text rebuild: numerics, passage-level change, attention pooling, fine-tuning | 4 | 3 | 2 | 24 | 2 |
| I-17 | Complete masking and a firm-identification probe | 2 | 4 | 4 | 32 | 1 |
| I-18 | Portfolio-grade backtest with costs, risk and capacity | 5 | 4 | 2 | 40 | 2 |
| I-19 | Options test of the magnitude signal | 4 | 2 | 2 | 16 | 2 |
| I-20 | Peer and network spillovers | 3 | 3 | 2 | 18 | 3 |
| I-21 | Insider, ownership and short-interest features | 3 | 3 | 3 | 27 | 3 |
| I-22 | Firm-history sequence model | 2 | 2 | 2 | 8 | 3 |
| I-23 | Monitoring, model cards and source register | 3 | 5 | 3 | 45 | 1 to 2 |
| I-24 | External registration and independent re-implementation | 4 | 5 | 3 | 60 | 0 to 1 |
| I-25 | Production ingestion | 3 | 4 | 2 | 24 | 3 |

## 2. Phases

### Phase 0: make the current result unassailable (weeks 0 to 3)

| Week | Work | Output | Owner |
|---|---|---|---|
| 0 to 1 | I-02: replace the alpha term in `track_a/event_study.py:56` and `targets/build.py` Carhart helper with alpha-free and calendar-time alternatives; add regression tests that fail on the old code; mark old stage outputs as superseded | New S20 and S21 outputs; test names recorded in D-080 | Quant engineer |
| 0 to 1 | I-03 and I-08: IC, rank IC, ICIR, decile spread, magnitude baseline; power table per period | `results_ic.csv`, report section | Quant |
| 0 to 1 | I-04: re-run L6 and L7 on `T_trade@lat10`, `@lat120`, `T_info@pub0/15/60`, labels k = 0.25 and 0.75 | Sensitivity table | Quant |
| 1 to 2 | I-01: ingestion, frozen-model scoring, ledger, monthly scoring script; pre-state success thresholds | Ledger repository, README of the test | Engineer |
| 1 to 2 | I-05: GitHub Actions, golden-set regression, deterministic mode and seed bagging | Green CI badge | Engineer |
| 2 to 3 | I-06 and score stress folds: two-way bootstrap, firm- and sector-held-out scores | Updated paired lifts | Quant |
| 2 to 3 | I-24: external registration of Phase 1 hypotheses; independent recomputation of headline tables | Registration link; recompute log | Second analyst |
| 0 to 1 | Second-pass fixes N1 and N2: real-value accounting availability, drop `meta_year`, per-rung subsets (EXP-N1, EXP-N2) | Rebuilt L3, L6 and L7 comparisons | Quant |
| 1 to 3 | N3 and N4: structural-break and rolling-window diagnosis of the direction IC; direct magnitude regression against the volatility baseline (EXP-N3, EXP-N4) | Decay diagnosis; magnitude benchmark | Quant |
| 1 | N6: locate raw minute-bar files and audit coverage (EXP-N6) | Intraday data inventory | Engineer |

**Gate 0 (end of week 3):** all headline numbers survive the recomputation, and conclusions are restated with firm-and-month intervals. If the direction rank IC lower bound is still below 0 and the minimum detectable IC exceeds 0.03, proceed to Phase 1 with the explicit aim of power.

### Phase 1: data foundation (weeks 3 to 12)

| Weeks | Work | Output |
|---|---|---|
| 3 to 8 | I-09: survivor-free, broad universe; point-in-time identifiers; second price source and reconciliation (EXP-D2) | `universe_v2`, price audit |
| 4 to 9 | I-10 and I-11: consensus, guidance and transcripts with as-first-reported timestamps; vendor timestamp audit | Expectation features; surprise v2 |
| 4 to 8 | I-14: label 500 filings (templates exist); score the existing extraction; gate fields | Gold set, F1 by field |
| 6 to 10 | I-12 and I-13: cross-sectional normalisation, pruning, event-type heads, rank and regression objectives | Model v2 on the old universe first (to isolate effects) |
| 8 to 12 | I-07, I-17, I-23: nested tuning, masking probe, model cards and source register | Governance pack |

**Gate 1 (week 12):** on the new universe and with expectation data, evaluate L6-v2 on walk-forward and the development period (not the spent locked period). Go if direction rank IC reaches 0.015 with a lower bound above 0, or magnitude incremental IC reaches 0.03 with a lower bound above 0. No-go otherwise: drop the directional alpha goal and keep magnitude and risk applications.

### Phase 2: new information and a real portfolio (months 3 to 6)

| Work | Output |
|---|---|
| I-15 intraday and extended-hours clock; re-estimate T_trade with latency distributions | `T_trade_v2`; first-reaction capture |
| I-16 text rebuild (numerics, passage-level change, attention pooling, fine-tuning with nested tuning) | Text tournament with a fair test of G3 |
| I-18 portfolio-grade backtest (risk model, costs, capacity, delays) | Net IR with intervals; capacity at half-alpha |
| I-19 options test of the magnitude signal, if options data are available | Net P&L after option spreads |

**Gate 2 (month 6):** require net IR above 0.5 with a bootstrap lower bound above 0 at 2x costs in the walk-forward and development periods, and a shadow-test rank IC with a non-negative lower bound. Otherwise, publish and stop alpha work.

### Phase 3: scale and operate (months 6 to 12)

I-20 peer spillovers, I-21 insider and ownership features, I-22 sequence models, I-25 production ingestion, a paper-trading period with execution simulation, and a second public paper with the full results.

## 3. Dependency graph (text)

```
I-02 --> I-03 --> I-08 -----------------------------> Gate 0
I-01 (ledger) ---------------------------------------> Gate 2 (shadow results accumulate)
I-05 --> I-24
I-09 (universe) --> I-10 (consensus) --> I-13 (event-type, rank) --> Gate 1
I-09 --> I-15 (intraday) --> I-18 (portfolio) --> Gate 2
I-14 (labels) --> I-16 (text rebuild) --> Gate 2
I-17 (masking) --> I-16
I-18 --> I-19 (options)
```

## 4. Resourcing and budget tiers

| Item | People-weeks | Data tier | Compute |
|---|---|---|---|
| Phase 0 | 6 to 8 | $ | CPU |
| Phase 1 | 20 to 28 | $$ to $$$ (universe, estimates, transcripts) | CPU, small GPU |
| Phase 2 | 24 to 32 | $$ (intraday), $$ to $$$ (options) | Larger GPU for fine-tuning |
| Phase 3 | 20 to 30 | $$ | Moderate |

Roles: one quant researcher (owner of inference and models), one data and infrastructure engineer, a part-time second analyst for recomputation and review, and an annotator for 500 filings (about 40 to 60 hours).

## 5. Decisions the owner must make

1. **Scope of the alpha goal.** Direction alpha, magnitude and risk applications, or publishing a methods paper only. The Gate 1 evidence decides.
2. **Data budget.** Which tier for survivor-free data, consensus and intraday.
3. **Rule on SEC-only inputs.** `CLAUDE.md` hard rule 1 forbids news and social text. Consensus, transcripts and options are not in that list but widen scope. State a revised scope rule in D-080.
4. **Registration venue** for Phase 1 hypotheses.
5. **Shadow-test start date and the success threshold,** fixed before the first prediction.
6. **Whether to keep the repository public** during Phase 1; if so, decide a publication delay for live ledgers.

## 6. Experiment card template

Use for every new experiment.

```
ID:            EXP-XX
Question:      one sentence
Hypothesis:    directional, with the sign and size you expect
Data:          source, period, universe, as-first-reported check
Design:        features, model, split, tuning, seeds
Primary metric: one number, with the minimum effect that matters
Power:         minimum detectable effect, effective sample
Multiplicity:  family and correction
Accept if:     numeric rule with interval bound
Kill if:       numeric rule
Registered:    link and date
Result:        filled after the run
Decision:      go, stop, or revise, with reason
```

## 7. Index of experiments

| ID | Topic | File |
|---|---|---|
| EXP-D1 to D4 | Latency sensitivity, price-data error, survivorship bias, accounting gaps | 02 |
| EXP-F1 to F6 | Normalisation, numerics, passage change, firm identity, tone baseline, readability | 03 |
| EXP-T1 to T6 | First-reaction share, rank targets, label threshold, benchmark, INTRA entry, horizons | 04 |
| EXP-M1 to M8 | Direction IC, magnitude baseline, seed bagging, tuning optimism, event types, rank objectives, conformal, sequence | 05 |
| EXP-V1 to V7 | Effective sample, clustering, nesting, regimes, stress folds, reality check, shadow | 06 |
| EXP-P1 to P6 | Deployment, options, capacity, costs, sizing | 07 |
| EXP-E1 to E4 | Reproducibility, mutation tests, recomputation, replay | 08 |

## 8. Overall program stop rules

1. If after Gate 1 the direction rank IC upper bound is below 0.02 on the broad, survivor-free universe with expectation data, stop the direction program.
2. If the magnitude signal does not beat the event-type-times-volatility baseline by 0.03 incremental rank IC, do not pursue options or sizing products.
3. If the shadow test after 12 months has a rank IC lower bound below 0 and the point estimate below 0.01, publish the null and close the programme.
