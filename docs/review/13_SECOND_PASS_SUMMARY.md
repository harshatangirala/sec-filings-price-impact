# 13. Second-pass summary (the plain-language version of file 12)

Written 5 November 2026. File 12 holds the full evidence, tables and file-and-line references. This file is the short version that was reported back after the second end-to-end check, saved here so it travels with the repository.

## What was done

A second, empirical pass over the project. It corrected three statements from the first review and found six new weaknesses. Files 00, 01, 03, 05, 06, 07, 08, 09, 10 and 11 were patched so earlier statements match the measurements. Evidence scripts and CSVs are in `evidence/` (`deep_checks.py`, `deep_checks2.py`, `deep_checks3.py`). The work is committed locally as ad94a86 and was not pushed when this file was written.

## New weaknesses

1. **The accounting availability rule is nearly empty.**
   - `src/secprice/models/ladder.py:72` counts `sue_*` columns as accounting data. 65% of 2000-2008 events therefore "have accounting data" while 0% have any real value.
   - XBRL-derived values exist from 2009 (22% coverage), 2010 (76%) and 2011 (98%).
   - The weak L3 accounting score is partly a coverage artifact, and L6 and L7 learn a missing-data pattern tied to time.
2. **A calendar-year feature is in the model.** `meta_year` (also `meta_month`, `meta_dow`) lets trees memorise regimes and cannot extrapolate to later years.
3. **The direction signal decays and flips sign.**
   - Rank IC (rank correlation between predicted direction and the realised abnormal return) is **+0.018 for 2008-2019, about 3 standard errors**, and **-0.008 since 2020**.
   - Ten of twelve years before 2020 are positive; 2022 is significantly negative.
   - Feature signs flip across eras. Example: the 5-day return on earnings releases goes from +0.016 to -0.052.
   - The question is decay and instability, not absence of skill.
4. **The classifier is a poor magnitude estimator.** The price-only model L1 scores about 0 on move size even though its own volatility features score 0.2 to 0.3 on their own.
5. **Possible intraday data.** 908,255 price bars are tagged `local:1min`. If the raw minute files still exist, the intraday work could be much cheaper than buying data.
6. **L7 and LightGBM yearly IC were never checked** (N5), so there is no year-by-year view for the heavier models.

## Corrections to the first review

- **Direction skill is not zero.** See item 3 above. The first review said "about zero".
- **The magnitude signal is not "mostly metadata".** A single trailing-volatility feature scores 0.16 to 0.26 against absolute return. L6 scores 0.21 (development) and 0.28 (locked). A volatility-times-event-type baseline scores 0.21 to 0.22. L6's extra skill over that baseline is about -0.01 to +0.06, so it is not evidence of alpha.
- **The power numbers were too pessimistic.** Measured design effects are 1.04 to 1.46, so the minimum detectable IC at t = 3 is about **0.036 to 0.042** on the locked and development periods. A locked-test null still cannot exclude an IC of 0.02 to 0.03.

## Strengths confirmed

- **Price data:** no OHLC violations, 99.8% firm completeness, 206 zero-volume bars.
- **Events:** no duplicate events; little concentration by date or firm.
- **Leakage:** no feature has a suspiciously strong univariate link to the target; the largest direction correlation is 0.025.

## Cheapest high-value fixes (in order)

1. Fix the accounting rule and drop `meta_year`. A matter of days.
2. Diagnose the direction decay before building any strategy.
3. Fit magnitude directly and test it against the volatility baseline.
4. Check whether the raw minute-bar files exist.

## Limits of this pass

- The yearly IC intervals ignore month and firm clustering, so they are a bit too narrow.
- The feature table lists candidates, not a decomposition. The fitted models were not stored, so skill could not be attributed to individual features directly.
- The accounting finding comes from the code and coverage tables. Re-fitting L3 and L6 on the correct subsets is needed to size the effect.

## What this means for the public paper

Not yet edited. Two statements in `paper/` are affected:

- "Accounting adds least" is probably partly the coverage artifact in item 1.
- The direction "null" is really a post-2020 result; before 2020 the walk-forward IC is small but positive.

A patch to the paper and a rebuild of the PDF is the natural next step, followed by re-uploading to SSRN.

## Where things are

- Full evidence and issue register: `docs/review/12_SECOND_PASS_FINDINGS.md`, `11_appendix_issue_register_evidence.md` (I-037 to I-042).
- Experiment cards for the fixes: `09_roadmap_and_experiments.md` (EXP-N1 to EXP-N6).
- Everything else about the project that is not in a file: `docs/HANDOFF_CONTEXT.md`.
