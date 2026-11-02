# Appendix: issue register, evidence and reproducibility

## 1. Issue register

Severity: **H** changes conclusions, **M** costs accuracy or time, **L** hygiene. Status: **Open** unless stated. Line numbers refer to the repository at commit `c1ad3e6`.

| ID | Sev | Area | Location | Issue | Evidence | Proposed fix | Link |
|---|---|---|---|---|---|---|---|
| I-001 | H | Inference | `track_a/event_study.py:56` | CAR subtracts `alpha x (b - a + 1)` with alpha from days -250 to -30; over 59 days this injects pre-event alpha | D-079: Q5 minus Q1 is -2.66% with it, +0.12% without; calendar-time alpha +0.25% (t = 0.46) | Alpha-free and calendar-time stage; regression test that fails on the old code | 04, 09 I-02 |
| I-002 | H | Inference | `track_a/event_study.py:59-76` | Carhart CAR subtracts the intercept `coef[0]` per day | Same artifact in `car4_*` columns | Drop the intercept or estimate after the window | same |
| I-003 | H | Power | Locked test, 8,199 events, 43 months | Minimum detectable IC at t = 3 about 0.033 to 0.048 | `evidence/power_ic.csv` | Widen universe; report MDE with every null | 06 |
| I-004 | H | Target | `targets/build.py:147-149`, `events/timestamps.py:71-85` | Entry at the next open discards the first reaction; INTRA events enter the next day | Mean absolute T_trade / T_info = 0.67 (PRE), 0.72 (POST) | Intraday clock (I-15); non-event baseline (EXP-T1) | 04 |
| I-005 | H | Data | `universe/build.py`, `meta.json` | Survivor-only universe, no delisting returns | 150 firms listed in 2026 | Survivor-free universe | 02 |
| I-006 | H | Features | `parse/html.py`, `default.yaml:76`, `features/text.py:80` | Numeric lines deleted before text models | Config flag `drop_numeric_lines: true` | Typed numeric extraction | 03 |
| I-007 | H | Features | `features/accounting.py:76-101` | No consensus or guidance; seasonal-random-walk SUE; coverage 31% of events | `review_facts.json` | Expectation data | 02, 10 |
| I-008 | H | Backtest | `backtest/engine.py:76-79, 85-89` | Cash drag, equal weights, flat costs, no risk model, no capacity | Long book averages 8.6 names, 68% cash days on locked test | Portfolio-grade rebuild | 07 |
| I-009 | M | Model | `evaluation/metrics.py:68-78` | `argmax p/prior` forces balanced classes; objective is macro-F1 | Direction rank IC about 0 | Rank and regression objectives | 05 |
| I-010 | M | Model | `models/heads.py:110-112, 128` | Temperature scaling and early stopping use the same inner slice | Code | Separate slices | 05 |
| I-011 | M | Model | `models/heads.py:172` | Elastic net `saga` with `max_iter=400, tol=1e-3`; convergence recorded but not enforced | `info['converged']` | Enforce or raise iterations | 05 |
| I-012 | M | Tuning | `models/ladder.py:294-297`, `default.yaml:155-160` | Tuned on 3 of 6 reported folds; 30,000-row subsample; 6 to 15 trials | `tuning.csv` median gain 0.000 | Nested tuning, larger budgets | 05, 06 |
| I-013 | M | Determinism | LightGBM heads | Different machines give 19.8% different classes | `reproducibility.csv` | Deterministic mode, thread fix, seed bags | 05, 08 |
| I-014 | M | Selection | `stages_model.py:872-` | L6 versus L7 chosen across two runs; margin 0.0010 below noise 0.0045 | `meta.json`, report 5a | One run, tie-break rule | 05 |
| I-015 | M | Features | `features/price.py:48` | `px_rev_1m` duplicates `px_ret_20` | Spearman 1.000 | Remove or redefine | 03 |
| I-016 | M | Features | Registry | 18 pairs above abs(rho) 0.95; 2 zero-variance columns | `review_facts.json` | Prune; registry test | 03 |
| I-017 | M | Features | `ladder.py:160-193` | Raw levels without per-date normalisation; adversarial AUC 1.000 | `adversarial.csv` | Ranks and ratios | 03 |
| I-018 | M | Targets | `targets/build.py:164-166, 189` | Daily beta and sigma applied to an open-to-close return; estimation window includes prior events | Code | Intraday beta; exclude event windows | 04 |
| I-019 | M | Targets | `targets/build.py:68-75` | Fixed k and one threshold for all event types; Sideways share varies 26% to 46% across contexts | `sample_returns.csv` | Continuous primary target | 04 |
| I-020 | M | Text | `models/embeddings.py:143-161` | Mean and max pooling over all chunks; PCA to 64 | Code | Attention pooling, fine-tuning | 03, 05 |
| I-021 | M | Masking | `parse/anonymize.py` | Products, subsidiaries, executives unmasked; NER masking not run | Code | Extend masking, add probe | 03 |
| I-022 | M | Validation | `walkforward.py:66-82` | Stress folds built, never scored | `leakage_audit.csv` item 4 | Score them | 06 |
| I-023 | M | Inference | `evaluation/bootstrap.py` | Month-block only; no firm clustering; intervals pool folds | Code | Two-way bootstrap; fold dispersion | 06 |
| I-024 | M | Inference | `mcs_pbo.json` | PBO = 0.0 over 93 mixed configurations; permutation null with 10 draws | `mcs_pbo.json`, `permutation.csv` | Better-designed PBO set; 200 permutations | 06 |
| I-025 | M | Process | `lockbox_log.json`, decision log | Lock scored 2.6 minutes after freeze without the planned review | D-078 note | Freeze checklist and cooling-off | 08 |
| I-026 | M | Process | `preregistration.yaml` | Registration is internal; no external timestamp | Repo history | External registry | 06, 08 |
| I-027 | M | Backtest | `backtest_spec.json`, D-032 | Backtest model is a refit on h = 5 labels, not the frozen model | Spec | Use the frozen model | 07 |
| I-028 | M | Data | `market/prices.py:123-151` | Split-adjusted bars with an index built from dividends; no second source | Code | Reconcile with a second vendor | 02 |
| I-029 | M | Data | `default.yaml:79-87` | Ten fact tags and three EPS tags only | Config | Add tags | 03 |
| I-030 | M | Reporting | `build_report` pipeline outputs | Stored S20 and S21 long-window numbers carry I-001 | `results_summary.md` has a correction notice | Supersede stage outputs | 04, 08 |
| I-031 | M | Targets | `targets/build.py:150-159` | Sensitivity targets computed but never reported | `target_defs_stored` | Report | 02, 04 |
| I-032 | L | Engineering | repo root | No CI, no container, no data versioning | Directory listing | See 08 | 08 |
| I-033 | L | Engineering | `analysis/*.py` | Windows paths as defaults | Code | Environment variable only | 08 |
| I-034 | L | Engineering | `stages_model.py` and others | Large modules (1,000+ lines) | `wc -l` | Split | 08 |
| I-035 | L | Dashboard | Streamlit Cloud | Blank page after wake-up when last tested | Test on 2026-11-01 | Health check, logs | 08 |
| I-036 | L | Docs | `README.md`, reports | Earlier text described drift reversal as a finding | Corrected in D-079 | Keep correction log | 08 |

## 2. Evidence table

All numbers computed from stored stage outputs. "Source" names the file.

| Quantity | Value | Source |
|---|---|---|
| Events | 60,261 (8-K 45,193; 10-Q 11,271; 10-K 3,797) | `meta.json` |
| Labelled events (T_trade h = 1) | 59,698 | `facts.json` |
| Firms | 150 | `universe.csv` |
| Events per firm: min, median, max | 86, 386, 967 | `review_facts.json` |
| Events per year 2019 to 2025 | 2,387; 2,683; 2,506; 2,266; 2,299; 2,236; 2,279 | `review_facts.json` |
| Features | 131 columns in `features.parquet` (128 in the registry used for modelling) | `features.parquet` |
| Missing by family | accounting 42.1%; change 17.5%; reaction 6.8%; lexical 1.3%; metadata 1.1%; novelty 0.6%; price 0.6% | `review_facts.json` |
| SUE availability | 31.1% of events | `review_facts.json` |
| Accounting features available | 66.5% of events | `review_facts.json` |
| Duplicate and collinear features | `px_rev_1m` equals `px_ret_20`; 18 pairs above 0.95 | `review_facts.json` |
| Class shares | Down 29.4%, Sideways 40.5%, Up 30.0% | `meta.json` |
| T_trade sd over mean sigma | 1.43 | `review_facts.json` |
| Share of T_trade beyond 0.5 sigma | 59.5% (normal with sd = sigma: 61.7%) | `review_facts.json` |
| Mean absolute AR, earnings versus others | 2.37% versus 1.36% | `review_facts.json` |
| Mean absolute T_trade over T_info | PRE 0.67; POST 0.72 | `evidence/reaction_before_entry.csv` |
| Locked macro-F1 L6, L7 | 0.384, 0.383 | `locked_results.csv` |
| Direction rank IC, locked | L6 -0.0077; L7 -0.0021 | `evidence/rank_ic.csv` |
| Direction rank IC, development | L6 -0.0087; L7 +0.0033 | same |
| Monthly direction IC t-value, locked | L6 0.07; L7 -0.38 | same |
| Magnitude rank IC (P(move) versus abs(AR)), locked | L6 0.279; L7 0.280; L1 -0.021 | same |
| Magnitude rank IC, development | L6 0.205; L7 0.214; L1 0.018 | same |
| Earnings-subset direction IC, locked | L6 -0.031; L7 -0.036 | same |
| Up-class recall, locked L6 | 543 / 2,520 = 21.5% | `locked_confusion.csv` |
| Detectable IC at t = 3 | 0.033 (independent) to 0.048 (n_eff 3,834); 0.018 to 0.026 for walk-forward | `evidence/power_ic.csv` |
| Fundamental-law gross IR (breadth 2,379) | IC 0.01: 0.49; 0.02: 0.98; 0.03: 1.46; 0.05: 2.44 | `review_facts.json` |
| LightGBM cross-machine agreement | 80.2% same class; macro-F1 0.3898 versus 0.3943 | `reproducibility.csv` |
| Tuning gain across 32 combinations | median 0.000; mean +0.0034; 44% improved | `tuning.csv` |
| Optuna trials | LightGBM 15, CatBoost 6, elastic net 8 (embedding stage 8 and 4) | `default.yaml` |
| Configurations tried in S11 | 237 | `meta.json` |
| Backtest, locked, no delay, 10 bps | Long 1.3%; long minus SPY -5.6%; long-short -2.0% a year | `backtest.csv` |
| Long book names and cash days, locked | 8.6 names; 68% of days in cash | `backtest.csv` |
| Stored sensitivity targets | `T_info@pub0/15/60`, `T_trade@lat10/120`, `T_trade_lag1` | `evidence/review_facts.json` |
| Drift, pre-specified versus alpha-free | -2.66% versus +0.12% [-0.80, +0.88]; calendar-time +0.25% (t = 0.46) | `paper/tables/drift_by_measure.csv`, `calendar_time.csv` |

## 3. How the new numbers were computed

`evidence/review_facts.py` reads the stored stage outputs (`S07`, `S08`, `S09`, `S18`, `S19`, `S20`) and writes:

| File | Contents |
|---|---|
| `evidence/review_facts.json` | Feature quality, target geometry, event mix, breadth and the fundamental-law table |
| `evidence/reaction_before_entry.csv` | Mean absolute T_trade and T_info by session |
| `evidence/rank_ic.csv` | Rank IC of direction and magnitude by model and period; monthly IC and t-values |
| `evidence/power_ic.csv` | Detectable IC at t = 2 and 3 for illustrative effective samples |
| `evidence/event_mix.csv` | Event counts by form and earnings type |

Run: `RUN_ROOT=<finished run> python docs/review/evidence/review_facts.py`. The script refits nothing.

## 4. Limitations of this review

1. **Stored outputs only.** I did not re-run the pipeline, so I cannot rule out problems that appear only in intermediate stages I did not read (for example the EPS release parser and the 8-K taxonomy).
2. **The first-reaction share is an estimate.** It uses mean absolute abnormal returns and an independence assumption, and includes ordinary overnight variance. EXP-T1 gives the clean number.
3. **Power calculations use illustrative effective sample sizes.** The true intraclass correlation was not estimated (EXP-V1).
4. **The fundamental-law calculation is simplified.** It ignores costs, event correlation and non-independence of signals. It shows scale only.
5. **Grades and probabilities are judgements,** not measurements.
6. **Vendors, prices and licences were not checked.** Treat every data suggestion as a proposal to be priced.
7. **Code lines are from commit `c1ad3e6`** and will move.

## 5. Glossary

| Term | Meaning |
|---|---|
| IC, rank IC | Correlation, or Spearman correlation, between a signal and the subsequent return |
| ICIR | Mean of period ICs divided by their standard deviation |
| IR | Information ratio: active return divided by active risk |
| Breadth | Number of independent bets per year |
| Fundamental law | IR is approximately IC times the square root of breadth |
| MDE | Minimum detectable effect at a stated significance and power |
| CPCV | Combinatorial purged cross-validation |
| PIT | Point in time: only information available at the decision time |
| SPA, reality check | Tests that a selected model beats a set of alternatives after accounting for the search |
| PBO | Probability of backtest overfitting |
| Calendar-time portfolio | A daily portfolio of stocks within a window after events, regressed on factors; needs no event-level estimation |
