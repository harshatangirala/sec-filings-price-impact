# Second-pass findings (3 November 2026)

The first review (files 00 to 11) was mostly a read-through of the code plus a handful of measurements. This pass ran data-quality, leakage, stability and effective-sample checks on the stored outputs. It changes the first review in three places, adds six new weaknesses, and confirms several strengths. Scripts and CSVs: `evidence/deep_checks.py`, `deep_checks2.py`, `deep_checks3.py`. Nothing was refitted and the locked test was not re-scored for selection.

## 1. Corrections to the first review

| # | First review said | The data say | Effect |
|---|---|---|---|
| C1 | Direction rank IC is about zero | It is **+0.018 on 2008-2019 walk-forward** (month-block standard error 0.006, about 3 s.e.) and **-0.008 on 2020-2026**. By year, 10 of 12 years from 2008 to 2019 are positive, and 2022 is significantly negative (-0.042, interval excludes 0). See section 3 | The direction question is about decay and instability, not absence. The "direction null" in the paper is a post-2020 result |
| C2 | Magnitude skill "is mostly event metadata an options market already knows"; L1 at -0.02 "points to metadata" | Event type times session alone has rank IC 0.037. **A single trailing-volatility feature has 0.16 to 0.26** against absolute abnormal return. L6 has 0.205 (development) and 0.279 (locked). A volatility-times-event-type baseline has 0.213 and 0.224. Incremental skill of L6 over that baseline is about -0.01 (development) to +0.06 (locked) | The magnitude signal is mostly volatility persistence, not metadata. It is not evidence of alpha |
| C3 | Minimum detectable IC at t = 3 is 0.033 to 0.048 or worse (illustrative effective samples of 2,500 to 3,800) | Measured design effects are small: month-block 1.04 to 1.30, firm-block 1.15 to 1.46. Effective samples are about 6,300 to 7,100 (locked) and 5,100 to 7,100 (development). **Minimum detectable IC at t = 3 is about 0.036 to 0.038 (locked), 0.036 to 0.042 (development), 0.019 to 0.020 (walk-forward)** | The earlier numbers were too pessimistic. The conclusion that a locked-test null cannot exclude an IC of 0.02 to 0.03 still holds |

Measured design effects (`evidence/design_effect_ic.csv`):

| Period | n | Rank IC (L6) | s.e. if independent | s.e. month-block | s.e. firm-block | Design effect (month, firm) | n_eff (month, firm) | MDE at t = 3 (month, firm) |
|---|---|---|---|---|---|---|---|---|
| Development | 7,453 | -0.0087 | 0.0116 | 0.0118 | 0.0140 | 1.04, 1.46 | 7,142, 5,091 | 0.036, 0.042 |
| Locked | 8,199 | -0.0077 | 0.0110 | 0.0126 | 0.0118 | 1.30, 1.15 | 6,320, 7,136 | 0.038, 0.036 |
| Walk-forward | 28,533 | +0.0183 | 0.0059 | 0.0062 | 0.0067 | 1.09, 1.29 | 26,079, 22,126 | 0.019, 0.020 |

## 2. New weaknesses

### N1. The "has accounting" rule is nearly vacuous, and accounting data do not exist before 2009 (High)

| Evidence | Detail |
|---|---|
| Rule | `models/ladder.py:72` treats an event as having accounting data if any column starting with `acc_` **or `sue_`** is non-null (`FAMILY_PREFIX["accounting"] = ("acc_", "sue_")`). `sue_n_diffs` is non-null even when no surprise is computed |
| Effect | 65% of 2000-2008 events pass the rule, yet **0% have any actual accounting value** (`accounting_rule_vs_content.csv`) |
| Coverage by year | Actual `acc_*` values: 2009 22%, 2010 76%, 2011 98%, 2012 and later about 100%. SUE coverage: 2005 8%, 2009 18%, 2011 36%, 2012 39% (`feature_coverage_by_year.csv`) |
| Consequences | (a) The "common cohort" does not exclude pre-2009 events, so L3 (accounting) is trained on a decade with no accounting data and its weak score (0.345 to 0.348) is partly a coverage artifact. (b) L6 and L7 carry accounting columns that are missing for the first 9 to 10 training years and present afterwards, a time-dependent missingness pattern the trees can use as a date proxy. (c) The first validation folds (2008 to 2011) test models on rows with little or no accounting data |
| Fix | Define availability by actual `acc_*` values only; report each rung on the subset where its inputs exist; add a missingness-by-year test to the feature registry |
| Experiment EXP-N1 | Rebuild L3 and L6 restricted to events with real accounting values (2009 onward, 2011 for full coverage) and compare macro-F1 and rank IC with the current numbers. If L3 improves by more than 0.01, the "accounting adds least" finding in the paper is a coverage artifact |

### N2. Calendar features inside a walk-forward model (Medium)

`meta_year`, `meta_month` and `meta_dow` are in the feature set (`calendar_features`). A year feature lets trees memorise regimes, and it cannot extrapolate: every test year after the last training year falls into the same leaf as the final training year. `meta_year` has a univariate rank correlation of -0.022 with direction and +0.047 with magnitude. It is likely a contributor to the adversarial AUC of 1.000. Fix: drop `meta_year`; keep month and day of week only if their effect is tested out of sample. Experiment EXP-N2: L6 without calendar features, compared on all folds, including adversarial AUC.

### N3. Direction signal that decays and flips sign (High)

Rank IC of `p_up - p_down` with the abnormal return, L6, by calendar year (`direction_ic_by_year.csv`):

| Year | IC | 95% interval | Period |
|---|---|---|---|
| 2008 | +0.007 | [-0.034, +0.057] | walk-forward |
| 2009 | +0.034 | [-0.003, +0.071] | |
| 2010 | +0.000 | [-0.040, +0.043] | |
| 2011 | +0.055 | [+0.021, +0.097] | |
| 2012 | +0.016 | [-0.032, +0.058] | |
| 2013 | -0.011 | [-0.049, +0.028] | |
| 2014 | +0.035 | [-0.007, +0.078] | |
| 2015 | +0.013 | [-0.020, +0.057] | |
| 2016 | -0.021 | [-0.054, +0.024] | |
| 2017 | +0.034 | [-0.002, +0.078] | |
| 2018 | +0.023 | [-0.020, +0.063] | |
| 2019 | +0.045 | [+0.010, +0.084] | |
| 2020 | +0.003 | [-0.036, +0.044] | development |
| 2021 | +0.006 | [-0.029, +0.044] | |
| 2022 | -0.042 | [-0.081, -0.002] | |
| 2023 | -0.011 | [-0.054, +0.036] | locked |
| 2024 | +0.010 | [-0.042, +0.046] | |
| 2025 | -0.015 | [-0.065, +0.022] | |
| 2026 | -0.019 | [-0.081, +0.041] | |

Pooled by era: 2008-2012 +0.021 (n = 11,429); 2013-2019 +0.016 (n = 17,104); 2020-2026 -0.008 (n = 15,652).

What carries it. Univariate rank correlations of single features with direction are all small (largest 0.025 across all events, so no feature is suspiciously strong) and **change sign across eras** (`feature_direction_rho_by_era.csv`, `direction_rho_by_type_era.csv`):

| Feature | 2008-2019 | 2020-2026 | Reading |
|---|---|---|---|
| Dollar volume (`px_logdv_20`) | -0.028 | -0.001 | Size and liquidity effect disappears |
| Log assets | -0.024 | -0.001 | Same |
| Risk-factor sentences removed (`rf_removed_share`) | -0.023 | +0.015 | Sign flips |
| Litigious tone (`main_lm_litigious`) | -0.020 | +0.016 | Sign flips |
| 5-day return, earnings releases only | +0.016 | **-0.052** | Momentum becomes reversal |
| Last earnings reaction, earnings releases | -0.042 | -0.003 | Disappears |
| 12-1 momentum, earnings releases | -0.040 | +0.008 | Disappears |
| 5-day return, all events | +0.000 | -0.031 | New post-2020 reversal |

Interpretation. The pre-2020 skill looks like a combination of weak, era-specific cross-sectional effects (a size and liquidity premium on event days, and tone and risk-factor change signs) that did not persist. Candidate explanations, to be tested and not assumed: crowding or alpha decay, a post-2020 regime change in retail and after-hours trading, tuning optimism (only the last three folds, 2014 to 2019, were tuned, but 2008 to 2013 show a similar IC), or noise. Consequences: a model fitted to the whole history carries signs that were wrong after 2020; any live use needs regime-aware or sign-constrained features.

Experiments: EXP-N3a structural-break test on the IC series (Bai-Perron or a simple CUSUM); EXP-N3b rolling and recency-weighted training compared with expanding; EXP-N3c event-type-specific IC by era; EXP-N3d sign-constrained or monotone models; EXP-N3e re-run on a wider universe to see whether the effect persists in small caps.

### N4. Magnitude is a volatility story, and the classifier estimates it poorly (Medium)

Rank IC of each score with absolute abnormal return (`magnitude_baselines.csv`):

| Score | Development | Locked |
|---|---|---|
| L6, 1 - P(Sideways) | 0.205 | 0.279 |
| L7, 1 - P(Sideways) | 0.214 | 0.280 |
| L1 (price only), 1 - P(Sideways) | 0.018 | -0.021 |
| Single feature `px_idio_vol` | 0.243 | 0.246 |
| Single feature `px_vol_60` | 0.261 | 0.189 |
| Single feature `px_vol_20` | 0.258 | 0.158 |
| Event type times session mean absolute AR (training years) | 0.037 | 0.037 |
| That group mean times `px_idio_vol` | 0.213 | 0.224 |

The price-only classifier L1 contains all of these volatility features and still scores about zero. Its 1 - P(Sideways) is a poor magnitude estimate: strong elastic-net regularisation (C = 0.0064), balanced class weights, prior correction and temperature scaling together flatten it. Implication: the 3-class classifier is an inefficient way to estimate magnitude. Fit magnitude directly (regression on |AR| over sigma, or a quantile model) and compare with the volatility baseline before any monetisation work (EXP-N4).

### N5. Elastic-net L6 is the best walk-forward model and its direction IC is the pooled +0.018 (Medium)

The pooled walk-forward direction IC quoted above is for the tuned L6 elastic net. The frozen L7 model (LightGBM plus BGE) has direction rank IC +0.003 on development and -0.002 on the locked test. Whether the pre-2020 IC appears under LightGBM was not computed for all years. Experiment EXP-N5: yearly IC for L7 and for LightGBM L6 from the S15 predictions.

### N6. The price source tag says 1-minute bars (Opportunity to check)

`bars_by_source`: 908,255 bars are tagged `local:1min`, 92,668 `yfinance_frozen`, 23,250 `local:yahoo_ext`. The tag suggests that the local daily bars were aggregated from 1-minute data in the earlier TriFrame project. If those raw 1-minute bars still exist for part of the sample, the intraday work in file 04 may be much cheaper than buying data. Check: locate the raw minute files, their date coverage and their adjustment status (EXP-N6).

## 3. Strengths confirmed by the checks

| Check | Result | Reading |
|---|---|---|
| OHLC consistency | 0 bars with open outside high-low; 0 with high below low; 0 non-positive prices | Clean structure |
| Missing and stale data | 206 zero-volume bars of 1,024,405; 9 runs of 5 or more identical closes; firm completeness 99.8% (median and 95th percentile), 7 firms below 97% (lowest 83.7%) | Good coverage |
| Large moves | 69 daily moves above 40% (426 above 25%), 58 from the local source and 11 from the frozen Yahoo cache. A spot check of the largest finds plausible real events (REGN March 2003, GNW November 2008, GME February 2021); one (DECK 1998, a $0.11 close in the warm-up period) looks like a data artifact outside any event window | Not verified against a second source (EXP-D2 stands) |
| Event integrity | 0 duplicate event ids; 130 events (0.2%) with a weekend decision time, entered on the next open; 59,698 labelled; 518 dropped for fewer than 120 estimation days, 31 for missing prices, 14 censored | Sound |
| Concentration | Top 20 decision dates hold 1.5% of events, top 100 hold 6.1%, top 10 firms 12.0%; 6,615 distinct decision dates | Little date or firm concentration |
| Beta and volatility | Beta quantiles 1%/5%/50%/95%/99%: 0.07, 0.36, 0.98, 1.88, 2.45; 0.6% negative; sigma median 1.49% | Sensible |
| Extreme returns | 46 abnormal returns above 20% in absolute value; 1 above 50% (GME, February 2021) | Few outliers |
| Label overlap | Mean uniqueness weight 0.950; effective events by uniqueness 57,252 of 60,261 | Overlap is mild |
| Leakage screen | Largest univariate rank correlation of any feature with direction is 0.025; with magnitude 0.317 (idiosyncratic volatility), as expected | No feature looks too good |
| Target stability by year | Mean absolute AR 1.2% to 1.4% in calm years, 2.2% to 2.5% in 2008, 2009 and 2020; class shares drift (Down 20% in 2001, 35% in 2008, 35% in 2026) | Labels are volatility-scaled, but class shares still move |
| Timestamp quality by year | 2000 and 2001: no events with exact times (74% and 71% quality B, rest C); 2002: 64% A; all later years 100% A | Early-era events are placed after the close by assumption. They are training-only |

## 4. Updated priorities

| Change | Why |
|---|---|
| **Move up:** N1 and N2 fixes (days, not weeks) | They change what L3, L6 and L7 mean and cost almost nothing to fix |
| **Move up:** diagnose the direction decay (EXP-N3) before building any strategy | The only evidence of directional skill is pre-2020 and fragile |
| **Add:** fit magnitude directly and test against the volatility baseline (EXP-N4) | Replaces the metadata story with a clean test |
| **Add:** check for existing 1-minute data (EXP-N6) | Could remove the largest data cost |
| **Unchanged:** forward shadow test, alpha-term fix, broader survivor-free universe, expectation data | Still the highest-value items |

## 5. New issue register entries

| ID | Sev | Location | Issue | Fix |
|---|---|---|---|---|
| I-037 | H | `models/ladder.py:72`, `FAMILY_PREFIX` | Accounting availability counts `sue_*` columns, so 65% of 2000-2008 events "have accounting" with 0% real values | Use real `acc_*` values; per-rung subsets; missingness test |
| I-038 | M | feature set | `meta_year`, `meta_month`, `meta_dow` in a walk-forward model | Drop `meta_year`; test month and day of week |
| I-039 | H | model behaviour | Direction rank IC +0.018 (2008-2019) then -0.008 (2020-2026); feature signs flip across eras | EXP-N3 series |
| I-040 | M | `heads.py`, classification of magnitude | L1 magnitude IC about 0 despite volatility features with univariate rank IC 0.2 to 0.3 | Fit magnitude directly; volatility baseline |
| I-041 | L | data | `local:1min` source tag: possible intraday data | Locate and audit |
| I-042 | L | data | 2000-2002 events have no exact acceptance times (quality B or C) | Report results from 2003 onward as sensitivity |

## 6. Limits of this second pass

1. The yearly IC intervals use 200 bootstrap draws on events within a year (no firm or month clustering), so they are somewhat too narrow; the pooled design effects in section 1 show the typical inflation is modest.
2. Univariate correlations ignore interactions, so the "carries it" table lists candidates, not a decomposition. A SHAP or permutation analysis on the stored model would be needed for attribution, and the models were not stored.
3. The accounting-availability finding comes from code and coverage tables. Re-estimating L3 and L6 on proper subsets (EXP-N1) is needed to size the effect.
4. The price-quality checks are internal-consistency checks, not comparisons with a second vendor.
5. Direction IC in the walk-forward period comes from the tuned elastic net; selection optimism cannot be fully ruled out (nested tuning is EXP-M3 and EXP-V3).

## 7. Files

| File | Contents |
|---|---|
| `evidence/deep_checks.py`, `deep_checks.json`, `deep_checks.log` | Price quality, events, targets, feature coverage, univariate screen, design effects |
| `evidence/deep_checks2.py`, `deep_checks2.json` | Accounting rule check, magnitude baselines, direction IC by year |
| `evidence/deep_checks3.py` | Feature direction correlations by era and event type |
| `evidence/design_effect_ic.csv` | Measured design effects and MDE |
| `evidence/direction_ic_by_year.csv`, `magnitude_baselines.csv`, `accounting_rule_vs_content.csv`, `feature_coverage_by_year.csv`, `feature_univariate_screen.csv`, `feature_direction_rho_by_era.csv`, `direction_rho_by_type_era.csv` | Supporting tables |
| `evidence/price_large_moves.csv`, `largest_abnormal_returns.csv`, `timestamp_quality_by_year.csv`, `target_by_year.csv` | Data-quality tables |
