# SWOT analysis

Each line states a claim, the evidence, and the consequence. IDs are referenced from the other files. Severity: **H** changes conclusions or blocks progress, **M** costs accuracy or time, **L** hygiene.

## Strengths (S)

| ID | Strength | Evidence | Why it matters |
|---|---|---|---|
| S1 | Four-timestamp point-in-time design with row-level availability assertions | `events/timestamps.py`; `meta.json` shows 0 availability violations over 60,261 events; 57,817 events have exact acceptance times (quality A: SGML header, or the JSON value under the verified correction rule) | The single most common failure in event research is look-ahead. This is handled by construction, not by convention |
| S2 | Found and corrected a hidden timestamp convention | `timestamps.py` header: JSON feed mixes two conventions (about 70% double-shifted, 30% true UTC) found by checking 8,789 headers | A 4 to 5 hour error on 30% of filings would have placed many events on the wrong trading day |
| S3 | Leakage engineering that tests itself | 24-item checklist; 4 planted leaks all caught (`planted_leaks.json`); three real leaks found (same-day 8-K clock, 10-Q inside SUE, stale balance sheet) | A leakage audit that has never failed has never been tested. This one has been |
| S4 | Executable target with an interval-matched benchmark | `targets/build.py:187-190`: open-to-close stock return minus beta times SPY open-to-close; Carhart deliberately null for it | Avoids subtracting a close-to-close factor return from an open-to-close return, a common mistake |
| S5 | Walk-forward with purge, embargo and uniqueness weights | `splits/walkforward.py:50-60`, `targets/build.py:95-110`; 0 purged-overlap leaks (`leakage_audit.csv` item 1) | Overlapping labels inflate effective sample size. This is handled |
| S6 | Single-use lockbox with a hashed freeze | `lockbox_log.json`, `frozen_architecture.json` (g4_hash) | Gives one clean out-of-sample read, and a verifiable record that it was used once |
| S7 | Honest chance level | Shuffled-own-prediction chance 0.33 versus majority baseline 0.19 (`evaluation/context.py:31-49`) | Prevents the most common way a three-class score is oversold |
| S8 | Paired, month-block inference | `evaluation/bootstrap.py` | Respects within-month dependence and compares models on identical events |
| S9 | Provenance-aware model governance | Strict versus retrospective leaderboards, per-fold labels (`splits/provenance.py`), ChronoBERT vintages with pinned revisions | Separates results that can support a clean claim from ones that cannot |
| S10 | Reproducible, resumable pipeline | 25 content-hashed stages, `_SUCCESS` markers, config hash `1712ef1b...`, `requirements.lock`, 115 tests (114 pass, 1 skip) | A re-run skips finished work; a reviewer can reproduce any table |
| S11 | Decision log and self-correction culture | D-001 to D-079, including D-071, D-072, D-077 (leaks and a stale-data bug) and D-079 (the drift artifact) | Governance people value the record of being wrong and fixing it |
| S12 | Transparent delivery | Public repo, dashboard, report, deck, paper, replication parquet files | Anyone can audit it |
| S13 | Price and event data pass integrity checks | 0 OHLC violations, 99.8% firm completeness, 0 duplicate events, mild date and firm concentration, no feature with an implausibly strong univariate link to the target (file 12, section 3) | The inputs are not obviously broken, which makes the modelling findings easier to trust |

## Weaknesses (W)

| ID | Sev | Weakness | Evidence | Consequence |
|---|---|---|---|---|
| W1 | H | **Underpowered for the effects that matter.** Locked test n = 8,199 events over 43 months | Measured minimum detectable IC at t = 3 is 0.036 to 0.038 (locked) and 0.036 to 0.042 (development); month and firm block design effects 1.04 to 1.46 (`evidence/design_effect_ic.csv`). Direction AUC interval [0.477, 0.510] | "No detectable direction skill" is compatible with an IC of 0.02 to 0.03, worth about 1.0 to 1.5 gross IR at 2,379 events a year (fundamental law, `review_facts.json`) |
| W2 | H | **Survivor-only, free-data universe** | `universe.csv`: 150 firms listed in 2026; 353 of 1,094 index tickers had usable data; no delisting returns | Biases long-horizon and backtest results, and removes the firms where bad news is most informative |
| W3 | H | **Entry at the open discards the first reaction** | For PRE and POST events mean absolute T_trade is 0.67 and 0.72 of T_info (`evidence/reaction_before_entry.csv`); under independence that is about half the variance before the entry (includes ordinary overnight variance) | The tradable window is the afterglow. Direction is weakest exactly where information is richest |
| W4 | H | **No expectations data** | No consensus, guidance, or transcript inputs; SUE is a seasonal random walk (`features/accounting.py:76-101`), available for 31% of events; accounting features missing for 33% of events and 42% of accounting cells | On an earnings 8-K, direction is surprise versus expectation. The model cannot see the expectation |
| W5 | H | **Text pipeline is weak by design** | Frozen encoders; mean and max pooling over all chunks (`models/embeddings.py:143-161`); `drop_numeric_lines: true` (`default.yaml:76`, `features/text.py:80`) removes the numbers; PCA to 64; no target-aware training; SEC-BERT and FinBERT dropped | The text null applies to this design. A model that reads the guidance sentence and the numbers was never tested |
| W6 | M | **One pooled model for heterogeneous events** | 8-K earnings, officer changes, Reg FD, agreements, 10-Q and 10-K all share one 3-class head; earnings events have mean absolute AR 2.37% versus 1.36% for others | Skill is dominated by event-type metadata; per-type signal is diluted |
| W7 | M | **Objective and metrics do not match a trading decision** | Macro-F1 with `argmax p/prior` (`metrics.py:68-78`) forces balanced classes; direction rank IC +0.018 on 2008-2019 walk-forward but about -0.008 since 2020, with feature signs that flip across eras (file 12) | A model tuned for macro-F1 is not tuned for ranking or expected return |
| W8 | M | **Tuning is small, on reported folds, and nondeterministic** | 15, 6, 8 trials (`default.yaml:155`); tuned on the last three of six reported folds on a 30,000-row subsample (`ladder.py:294-297`); LightGBM reproduces only 80% of predicted classes across machines (`reproducibility.csv`) | Selection optimism in validation; finalist choice (L6 versus L7) inside noise |
| W9 | M | **Inference gaps** | Month-block bootstrap only, 43 months on the locked test; 237 configurations tried (`meta.json`); PBO = 0.0 is degenerate; permutation null uses 10 draws | Intervals ignore firm dependence; no multiplicity control on the primary selection |
| W10 | H | **Backtest is a sanity check** | `backtest/engine.py`: equal weight, cap 5%, flat 10 bps per side, 50 bps borrow, `min_names_per_side` 10 so the long book averages 9 names and sits in cash on 68% of locked-test days | Results say little about a real portfolio; capacity, impact, borrow, risk and neutralisation are absent |
| W11 | M | **Feature hygiene** | `px_rev_1m` identical to `px_ret_20` (`price.py:48`); 18 feature pairs with abs(rho) > 0.95; 2 zero-variance columns; raw level features (`acc_log_assets`, `acc_roa`) top the adversarial-validation drivers (AUC 1.000) | Wasted capacity, unstable linear fits, regime drift in levels |
| W12 | M | **Labels and abnormal returns are coarse** | Beta from daily close-to-close applied to an open-to-close return (`build.py:189`); sigma from daily residuals; estimation window [-250,-30] includes earlier events; three classes discard magnitude | Noise in the target; thresholds balanced by offsetting errors (T_trade sd is 1.43 times mean sigma, yet 59.5% exceed 0.5 sigma versus 61.7% under a normal) |
| W13 | H | **Long-window abnormal returns embed pre-event alpha** | `track_a/event_study.py:56` subtracts alpha x window length; produced a false -2.66% reversal that passed every check (D-079) | All S20 and S21 long-window outputs are affected; the pipeline still stores them |
| W14 | M | **Firm-identity leakage not fully controlled** | `parse/anonymize.py` masks names and tickers; single-word names only exact Title or UPPER case; no product, subsidiary or executive masking; NER masking optional and not run; firm-held-out stress folds built but never scored | Embeddings can recognise firms from product names; the control is unverified |
| W15 | M | **Process controls are weaker than they look** | Pre-registration file is internal (no third-party timestamp); lock opened 2.6 minutes after freeze without the planned review; L6 versus L7 chosen by comparing scores from two runs; several post-lock analyses | Reviewers will discount it; fixable with an external registry and rules |
| W16 | L | **Engineering gaps** | No CI workflow, no container, no data versioning (DVC or lakeFS), no prediction ledger, dashboard blank on Streamlit Cloud when last checked | Slows collaboration and weakens the audit trail |
| W17 | H | **Accounting availability rule is vacuous before 2009** | `ladder.py:72` counts `sue_*` columns; 65% of 2000-2008 events pass with 0% real accounting values (file 12, N1) | L3 and part of L6 are trained and tested on missing data; the weak L3 score is partly a coverage artifact |
| W18 | M | **Calendar-year feature in a walk-forward model** | `meta_year`, `meta_month`, `meta_dow` (file 12, N2) | Lets trees memorise regimes and cannot extrapolate; adds to covariate shift |
| W19 | H | **Direction skill decays and feature signs flip** | Rank IC +0.018 on 2008-2019, about -0.008 since 2020; size, liquidity, tone and reversal effects change sign (file 12, N3) | Whole-history fits carry signs that were wrong after 2020; live use needs regime awareness |
| W20 | M | **Magnitude skill is volatility persistence** | A single volatility feature scores 0.16 to 0.26 against absolute AR; L6 scores 0.21 to 0.28; the price-only classifier scores about 0 (file 12, N4) | The 3-class classifier is an inefficient magnitude estimator; no evidence of alpha beyond volatility |

## Opportunities (O)

| ID | Opportunity | Basis | Potential |
|---|---|---|---|
| O1 | Widen to a Russell-3000-scale universe with point-in-time constituents and delisting returns | Breadth rises about 6x (2,379 to about 14,300 events a year); fundamental-law gross IR scales with the square root of breadth | Fixes W1 and W2 together; small caps are where filings should matter most |
| O2 | Intraday and extended-hours bars | Reaction starts the minute of acceptance; PRE and POST events trade in extended hours | Removes W3; enables latency-realistic tests |
| O3 | Expectations data: consensus estimates, guidance, revisions, transcripts | Surprise is defined against an expectation | Directly addresses W4; likely the largest direction-signal gain |
| O4 | Monetise magnitude through options | Magnitude rank IC 0.28 on the locked test (`rank_ic.csv`); top-decile predicted movers have 2.9% mean absolute AR versus 1.4% in the bottom decile | A route that does not need direction, if the signal beats the implied move. Needs options data and an honest baseline |
| O5 | Modern text modelling under strict point-in-time rules | Passage-level change against the prior filing, numeric-aware extraction, target-aware fine-tuning, contrastive training on filing pairs, chronologically consistent LLMs | Gives text a fair test |
| O6 | Ranking and regression objectives, event-type models, cross-sectional normalisation | Standard practice in cross-sectional alpha research | Aligns training with the decision |
| O7 | Peer and network spillovers | Industry, supplier-customer and text-similarity peers react to a firm's filing | New signal from the same filings |
| O8 | Live forward testing with a signed prediction ledger | Locked test is spent; forward data accrue daily | Clean, cheap, and persuasive evidence |
| O9 | Production-grade ingestion | EDGAR real-time feeds, incremental parsing, monitoring | Moves from research to operations |
| O10 | Publication and external validation | SSRN draft exists; registered-report format possible | Credibility, and a way to attract feedback |
| O11 | Existing minute-bar data may be recoverable | 908,255 bars tagged `local:1min` (file 12, N6) | Could make the intraday clock affordable |

## Threats (T)

| ID | Threat | Basis | Mitigation |
|---|---|---|---|
| T1 | **Crowding and speed**: machine readers trade 8-Ks within seconds | Our decision time is acceptance + 35 minutes by assumption; published text anomalies decay after publication (McLean and Pontiff 2016) | Do not compete on speed. Target slower, harder content (relative change, cross-firm, multi-document) and longer horizons |
| T2 | **Data licensing and fragility** | Yahoo terms and `yfinance` breakage; Loughran-McDonald licence; Hugging Face dataset terms | Move to licensed data for anything beyond research; keep frozen caches (`yf_frozen.parquet`) |
| T3 | **Researcher degrees of freedom** | 237 configurations, 79 decisions, post-lock analyses; the drift artifact shows plausible-but-wrong results survive | Pre-register externally, adopt a hurdle (t > 3), log every run, require shadow-period confirmation |
| T4 | **Regime change** | Price-only model collapses in 2020 to 2022 (macro-F1 0.305); adversarial AUC 1.000; changing retail and options activity after 2020 | Rolling and regime-aware evaluation; features normalised cross-sectionally; stress periods reported |
| T5 | **Model and vendor drift** | Hugging Face revisions, LLM deprecation, library version changes affect predictions (LightGBM 80% identical across machines) | Pin and mirror weights; deterministic training; seed bagging |
| T6 | **Execution reality** | Opening-auction liquidity, short borrow, spread and impact on mid and small caps; earnings-day clustering limits capacity | Liquidity-aware cost and capacity model before any claim of profitability |
| T7 | **Look-ahead in new data sources** | Vendor timestamps for estimates, transcripts and revisions are often stamped late or restated | Vendor-timestamp audits and as-first-reported snapshots for every new source |
| T8 | **Public record of earlier errors** | Earlier report, dashboard and deck stated the drift reversal as a puzzle | Corrections are published (D-079); keep a visible change log |

## SWOT cross-reading: what combinations imply

| Combination | Implication |
|---|---|
| S1 to S6 with W1 | The infrastructure is good enough to support a much larger study. Spend on data breadth, not on more safeguards |
| W3 with O2 | Intraday data is not a luxury. It changes what question we can ask |
| W4 with O3 | Direction needs an expectation. No text model will recover what the text does not contain |
| W7 with O6 | Change the objective before changing the model |
| W10 with O4 and T6 | Decide the monetisation route first (stock long-short, options, or sizing overlay), then build the backtest for that route |
| W13 with T3 | One undetected artifact is evidence that more may exist. Add an independent re-implementation of the main numbers |
| T1 with O5 | The defensible niche is content that fast readers do not trade, not speed |
