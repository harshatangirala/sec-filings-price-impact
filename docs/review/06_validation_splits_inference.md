# Validation, splits, testing and inference

## 1. What we built

| Element | Logic | Code | Setting |
|---|---|---|---|
| Walk-forward folds | Expanding window; first validation block needs 8 training years; 2-year blocks 2008 to 2019 | `splits/walkforward.py:27-39` | 6 folds, 28,533 events |
| Development holdout | 2020 to 2022; used to choose the final model | `walkforward.py:38` | 7,455 events |
| Locked test | January 2023 to July 2026; scored once after a hashed freeze | `walkforward.py:39`, `stages_model.py:1012-1061` | 8,213 events |
| Purge | Training events whose target window reaches the test start are dropped | `walkforward.py:52-53` | Purge horizon 60 sessions |
| Embargo | 90 days before each test start | `walkforward.py:54` | `embargo_days: 90` |
| Stress folds | Firm-held-out (5 folds) and sector-held-out (12) on the pre-2020 period; random split demo | `walkforward.py:66-82` | Built, never scored |
| Inference | Month-block bootstrap (1,000 draws in the pipeline; 2,000 in the paper); paired lifts; gate G3 | `evaluation/bootstrap.py` | |
| Diagnostics | Permutation null (10), adversarial validation, MCS, PBO, learning curve, fit verdicts | `stages_model.py:315-372`, `evaluation/mcs_pbo.py` | |
| Lockbox | Hash of frozen architecture; single-use log | `stages_model.py:1064`, `lockbox_log.json` | Scored 2026-10-25T23:59:58Z |

## 2. Strengths

1. **Purge, embargo and uniqueness weighting** address overlapping labels; the audit confirms zero shared dates between train and validation across folds (`leakage_audit.csv` items 1 and 2).
2. **Time ordering everywhere**: inner validation for early stopping is time-ordered; preprocessing fits on training rows only.
3. **A real lockbox** with a hash and a reuse guard.
4. **Paired comparisons** on identical events and identical resampled months.
5. **Chance and permutation checks** show the pipeline does not score above chance with shuffled labels (observed 0.389 versus null mean 0.251).

## 3. Weaknesses, with evidence

### 3.1 Power (W1)

Power is the central limit on every conclusion.

| Period | Events | Months | Detectable IC at t = 2 | at t = 3 | Notes |
|---|---|---|---|---|---|
| Locked | 8,199 | 43 | 0.022 (independent) to 0.032 (n_eff 3,834) | 0.033 to 0.048 | `evidence/power_ic.csv` |
| Development | 7,453 | 36 | 0.023 to 0.035 | 0.035 to 0.052 | |
| Walk-forward | 28,533 | 144 | 0.012 to 0.018 | 0.018 to 0.026 | |

The clustering assumptions in the table are illustrative, since the true effective sample depends on within-month correlation, which was not estimated. Macro-F1 intervals tell the same story: paired lifts on the locked test have half-widths of about 0.01.

What a direction IC of 0.02 to 0.03 is worth: with 2,379 events a year (2019 to 2025 average), gross IR is roughly IC times the square root of breadth, which gives 0.98 at IC 0.02 and 1.46 at IC 0.03 (`review_facts.json`). This is a simplified fundamental-law calculation that ignores costs, correlation across events and non-independence, and is shown only to demonstrate that effects below our detection limit can be economically large.

### 3.2 Inference

| Finding | Evidence | Consequence |
|---|---|---|
| Bootstrap resamples months only | `bootstrap.py` | Firms recur across months. A firm-and-month (two-way) or wild cluster bootstrap would widen intervals |
| Intervals pool events across folds | `macro_f1_ci` | Fold-to-fold variation in skill is hidden. Report the distribution across folds as well |
| Month blocks are short for slow dependence | 43 months in the locked test | Block length one month; volatility clusters last longer |
| No multiplicity control in prediction hypotheses | `fdr.py` is used for Track A only | H1, H2, H4, H5, H8 are read separately |
| Selection over 237 configurations | `meta.json` | The maximum validation score is optimistic by an unmeasured amount |
| PBO = 0.0 | `mcs_pbo.json`: 93 configurations mix rungs of very different content | PBO is uninformative here, as the report notes |
| MCS on development log-loss only, 36 blocks | `mcs_pbo.json` | Low power |
| Permutation null with 10 draws | `permutation.csv` | The p-value floor is 1/11 |
| Macro-F1 is non-smooth | Metric | Wide intervals; comparisons on log-loss, IC or Brier are more powerful |

### 3.3 Split design

| Finding | Evidence | Consequence |
|---|---|---|
| Hyperparameters tuned on folds that are then reported | `ladder.py:294-297` | Validation scores are optimistic; no nested loop |
| Six two-year folds | `default.yaml:116-125` | Coarse; one fold covers 2008 to 2009 only |
| Expanding window only | `walkforward.py` | No rolling-window or recency-weighted comparison; regime changes are untested |
| The development period is also selection data | `s18` | It cannot also be a clean estimate of performance |
| Stress folds not scored | `leakage_audit.csv` item 4 "built, not scored" | Firm memorisation and sector transfer are unmeasured |
| Embargo of 90 days with 250-day lookbacks | `price.py` | Features at the start of a test block use 250-day history that overlaps training periods. That is legitimate for features, but the embargo may be too short to break serial dependence in volatility |
| Lock scored 2.6 minutes after freeze | `docs/decision_log.md` D-078 note | The planned review step was skipped |
| Pre-registration is internal | `preregistration.yaml` hash in `meta.json` | No external timestamp; the repo was private until the end |
| Post-lock analyses | Paper VI.B to VI.E, VII.C to VII.E | They are exploratory and labelled so |

### 3.4 Cross-sectional structure

Events cluster on earnings days and in earnings seasons, and firms report in blocks. Standard errors that ignore date clustering overstate precision for cross-sectional comparisons. The Track A tests already cluster by firm and date; the classification tests do not.

## 4. World-class target state

| Element | Target |
|---|---|
| Test sets | A final locked period plus a **live forward test** with signed, timestamped predictions |
| Folds | Annual or half-year folds; anchored and rolling comparisons; combinatorial purged cross-validation (CPCV) for a distribution of out-of-sample paths |
| Tuning | Nested; inner folds time-ordered; outer folds report only |
| Inference | Two-way (firm and month) clustered bootstrap; fold-level dispersion; block lengths tied to measured autocorrelation; reality check or SPA test for the selected model against all candidates |
| Multiplicity | A hypothesis family list fixed in advance; false-discovery or Bonferroni control within it; a t-hurdle above 3 for new signals (Harvey, Liu and Zhu 2016) |
| Power | Every claim reports minimum detectable effect and the effective number of independent observations |
| Stress | Firm-held-out, sector-held-out, crisis-period and volatility-regime tests scored and reported |
| Registration | External timestamp (OSF, AEA or a notarised hash) before data access; a written freeze checklist including the review step |
| Independent check | A second analyst re-implements the headline numbers from raw inputs |

## 5. Additions, ranked

| Rank | Addition | Effort | Notes |
|---|---|---|---|
| 1 | **Live shadow test with a prediction ledger** | 2 weeks | See 08 for design. Starts the clock on clean data |
| 2 | **Power and minimum-detectable-effect table in every report** | 2 days | Prevents "no effect" language where only "no effect above X" is warranted |
| 3 | **Two-way clustered bootstrap** for all classification and IC tests | 1 week | |
| 4 | **Nested walk-forward** | 1 to 2 weeks | See 05 |
| 5 | **Score the stress folds** (firm, sector, crisis years) | 1 week | The code exists |
| 6 | **CPCV and path-level distributions** | 2 weeks | Gives a distribution of Sharpe-like statistics, not one path |
| 7 | **Reality check or SPA test** on the selection of L7 over alternatives | 1 week | Addresses 237 configurations |
| 8 | **Rolling-window and recency-weighted alternatives** | 1 week | Tests regime stability |
| 9 | **External pre-registration and a freeze checklist** | 2 days | Process |
| 10 | **Independent re-implementation of headline numbers** | 1 to 2 weeks | Would have caught the alpha artifact earlier |
| 11 | **Annual folds** (2008 to 2025, one per year) | 1 week | Finer regime view |

## 6. Experiments

| ID | Question | Design | Accept or kill |
|---|---|---|---|
| EXP-V1 | What is the true effective sample? | Estimate within-month and within-firm intraclass correlation of the loss; compute n_eff | Replace the illustrative n_eff column |
| EXP-V2 | Do conclusions survive firm-and-month clustering? | Re-run the paper's paired lifts with a two-way bootstrap | If any interval changes sign or touches zero that did not before, restate |
| EXP-V3 | How optimistic is validation? | Nested versus non-nested walk-forward macro-F1 and IC | Report the gap |
| EXP-V4 | Does skill depend on the regime? | Score L6 by VIX tercile, earnings season versus off-season, and year | Report; if skill vanishes in any regime, state it |
| EXP-V5 | Does the model transfer to unseen firms and sectors? | Score the built stress folds | Pass if firm-held-out macro-F1 is within 0.01 of the main result |
| EXP-V6 | Does selection beat the field after multiplicity? | White reality check or Hansen SPA of the frozen model against all stored configurations | Treat the selection as unproven if the p-value exceeds 0.10 |
| EXP-V7 | What is the shadow-period result? | Predictions committed before outcomes; scored monthly | Define the success threshold before starting (for example rank IC at or above 0.02 with a lower bound above 0 after 12 months) |
