# sec-filings-price-impact

SEC filing text (10-K, 10-Q, 8-K) → price impact, built to `SPEC.md` (v5) and `SPEC_ADDENDUM.md`.
Three tracks: **A** event study (association), **B** prospective prediction (primary target T_trade h=1, 3-class),
**C** content validity (human gold labels).

## Quick start

**Colab (recommended for the full study).** Open `notebooks/SEC_Filings_Price_Impact_v5_EndToEnd.ipynb` in Colab, add the
secret `SEC_USER_AGENT` = `Your Name your.email@example.com`, check that `MyDrive/sec-price-study/inputs` holds the inputs,
and *Run all*. Profiles: `smoke` (default) → `pilot` → `full` (G2: pilot done + `owner_opt_in_full`).

**Local (CPU stages and tests, no credits).**
```bash
uv venv .venv --python 3.12 && uv pip install -e ".[dev,gpu]"
pytest                                   # unit + offline integration tests (gate G1); network tests are opt-in
export SEC_USER_AGENT="Your Name you@example.com" DATA_ROOT=/path/to/sec-price-study
secprice run --profile smoke --stages S00-S12 S20 S21
secprice prereg-freeze --profile smoke && secprice run --profile smoke --stages S13-S18 S22 S24
```

## Profiles
| Profile | Firms | Years | Purpose |
|---|---|---|---|
| smoke | 5 fixed (AAPL, MSFT, JNJ, XOM, JPM) | 2021-2023 | the chain on real data in under an hour on CPU |
| pilot | ~40 sector-balanced | 2012-2026 | full core pipeline, small samples; required before full (G2) |
| full | 500 | 2000-01 to 2026-07 | the study |

## Run order on Colab (Pro+)
Inputs already sit in `MyDrive/sec-price-study/inputs` (979 OHLCV files, frozen yfinance gap-fill, LM dictionary, HF census,
factor / S&P 500 / SIC caches). The owner waived the pilot gate (decision log D-058): G2 = opt-in (already set in the notebook)
+ the preflight budget check, which the notebook prints after S04 and S05 enforces before the download. Stage hashes do not
depend on the machine, so any stage already on Drive is skipped.

| Step | Runtime | Settings in the first code cell | Stages |
|---|---|---|---|
| 1. full Part 1 | CPU High-RAM | `RUN_PROFILE="full"`, `PART="cpu1"` | S00-S12, S20-S21: **already run** (artifacts on Drive), so this step only verifies and skips; from scratch ~15 h EDGAR at 6 req/s (resumable) + ~4 h S06-S11 |
| 2. full Part 2 | A100 | `PART="gpu"` | S13-S17: GPU preflight, embeddings, ChronoBERT vintages, L5/L7, G3, fine-tuning, RAG, distillation (~3-6 h) |
| 3. full Part 3 | CPU | `PART="cpu2"` | S18 dev holdout + MCS/PBO + G4 freeze, S22 backtest, S24 report (S19 stays sealed) |
| 4. lockbox | CPU/GPU | `OWNER_OVERRIDES["lockbox"]["unlock_locked_test"]=True` after reviewing S18 | S19, scored once; then S22 and S24 again |

Restart the runtime when switching between CPU and GPU. Fine-tuning runs only if G3 passes unless
`gates.owner_override_g3_finetune=True` (results then labelled EXPLORATORY). Track C (S23) needs the gold annotations
(`secprice gold export|import|agreement`).

## S&P 500 vs non-S&P 500 vs combined
The 500 firms are S&P 500 members (in the study window, pinned fja05680 list) first, then a seeded, sector-stratified fill of
non-members. Every result is reported for each bucket and combined (decision log D-056): composition, the combined models scored
per bucket (walk-forward, development holdout, locked test), bucket-only models against the pooled model on identical rows,
Track A CARs and CAR paths, SUE drift, and the backtest restricted to each bucket's names. The universe is **survivor-only**
(current listings): results are biased towards survivors (D-057).

## Pilot Part 1 results (superseded by the full run; predates D-066 to D-073)
40 sector-balanced S&P 500 firms, 2012-01 to 2026-07: 11,370 filings parsed (SGML acceptance header for every one), 10,105
events, all timestamp quality A; release-EPS parser agrees with XBRL on 79.8% of usable-confidence releases (45 probable full-year values withheld, D-052); leakage audit
passes and catches all 4 planted leaks.

| Track B, T_trade h=1, 3-class, walk-forward (2 folds, n = 2,871) | macro-F1 [95% block-bootstrap CI] |
|---|---|
| L1 price only (CatBoost, tuned) | 0.362 [0.342, 0.383] |
| L2 metadata (elastic net) | 0.396 [0.380, 0.412] |
| L4 text: lexical + change + novelty + TF-IDF (LightGBM, tuned) | 0.391 [0.373, 0.412] |
| L6 all CPU features (LightGBM, tuned, validation-selected) | 0.396 [0.378, 0.415] |
| Permutation null (10 within-month label shuffles) | mean 0.322; none reached 0.396 (p ≤ 0.09) |

Validation-selected, not a global optimum (the elastic-net certificate shows only that its convex loss reached the same
optimum from 3 starts). L6 does not beat L2 metadata alone (0.396 vs 0.396); training F1 0.85 against validation 0.40 flags
overfitting; adversarial AUC ≈ 1 reflects drift in filing vocabulary (covariate shift), not leakage (D-051). Track A: SUE
post-earnings drift Q5-Q1 = -2.5% [-4.7%, -0.6%], p = 0.10 (weak/mixed; opposite sign to classic PEAD in this small sample); text
beyond momentum not supported. Association is not causation; prediction is not profitability.

## Mini study: complete run and report (26 Oct 2026)
The 150-stock study is finished end to end (CPU part locally, GPU part on Colab, locked test scored once). Headline results: locked-test
macro-F1 0.384 for the best structured model against a real chance level of about 0.33; text adds nothing (L7 - L6 = -0.001 [-0.012, +0.009]);
the models know WHETHER a stock moves (AUC 0.635) but not WHICH WAY (AUC 0.494 [0.475, 0.509]); SUE drift -2.6% (a reversal, survivor-only
caveat). The full report (architecture, data, pre-processing, models, RAG, evaluation design, 48 figures, known issues) is
`reports/results_report.html` / `.pdf` / `results_summary.md` (decision D-078). Rebuild only the report from finished stages with S24
(`force=["S24"]`; it never re-runs a model or the locked test).

## Mini study: 150 S&P 500 stocks (26 Oct 2026) - how it was run
`N_STOCKS = 150` in the notebook's first cell (folder `sec-price-study-mini`). Same pipeline and filings as the 500-firm run for
those firms (decision log D-076): 150 S&P members drawn seeded and sector-stratified, 60,261 events, S&P-only (no non-S&P bucket).
The CPU part is already run and packed as `colab_upload_mini/full.zip` (1.3 GB); Colab only runs the GPU part (about a third of the
work: roughly 5-8 h in fast mode, 7-11 h in full mode, unmeasured).

| CPU-part result (walk-forward, T_trade h=1, n = 28,533) | |
|---|---|
| L6 all CPU features | macro-F1 0.390 [0.383, 0.397] (best head: elastic net 0.392); shuffled-label null 0.251 (max 0.273) |
| L2 metadata / L1 price / L4 text / L3 accounting | 0.379 / 0.362 / 0.374 / 0.347 (same ranking and level as the 500-firm run) |
| SUE-conditioned drift Q5-Q1 | -2.6% [-3.5%, -1.7%], n=3,825: a reversal, survivor-only caveat applies |
| Checks | events, timestamps and targets identical to the 500-firm run for the same firms (60,261 events; 1.3M target rows); 0 availability violations; 4/4 planted leaks caught; accounting features match a brute-force oracle in 1,500 of 1,500 sampled events |

## Superseded numbers (D-077)
The 500-firm and pilot results below were computed before a point-in-time accounting bug was found (total assets, leverage, ROA,
revenue, latest EPS, book-to-market and market cap used last year's value in about half of all events). Results that use those
features (L3, L6, bucket scores, size controls) are superseded until stages S09-S24 are re-run (about 4.5 h locally); events,
timestamps, targets, Track A abnormal returns, SUE and release-EPS drift, and the L1/L2/L4 baselines are unaffected. The 150-firm
mini study above was run on the corrected code.

## Full profile, Part 1 (local CPU, 23-24 Oct 2026; accounting-dependent rows superseded, see D-077)
Run on the final code, real EDGAR data, 500 firms (S&P 500 members first, then non-members), Jan 2000 - Jul 2026; artifacts on
Drive at `MyDrive/sec-price-study/full` (Colab skips these stages and starts at Part 2).

| Stage | Result |
|---|---|
| S05 download + parse | 202,607 filings (149,992 8-K, 39,373 10-Q, 13,242 10-K); 201,728 parsed, 879 without a target section, 0 errors; 385,387 sections; HF 10-K QA median token Jaccard: Item 1 0.95, 1A 1.00, 7 0.83, 7A 0.96 |
| S06 accounting | 975,196 XBRL facts; 47,967 earnings releases, 34,661 with usable EPS, 82.7% agreement with XBRL (n=26,075); SUE for 59,490 filings |
| S07 events | 190,947 events (141,300 8-K, 37,135 10-Q, 12,512 10-K); 96% timestamp quality A (every year from 2003: 100%); usable cutoff 2026-07-31 |
| S09 / S10 | 130 features; 0 availability and 0 SUE-rule violations; leakage audit passes; 4/4 planted leaks caught; 6 walk-forward blocks 2008-2019, development holdout 2020-2022, locked test 2023 - Jul 2026 (sealed) |
| S11 baselines (walk-forward, T_trade h=1, n=89,376) | L6 (LightGBM, tuned) macro-F1 0.401 [0.396, 0.405] vs majority 0.189 and a permutation null of 0.265 (max of 10: 0.276); L2 metadata 0.377 [0.371, 0.384]; L1 price 0.368; L4 text 0.376; L3 accounting 0.356; L6 without filing-change features 0.401; elastic-net convexity certificate spread 1e-9 |
| S&P 500 vs non-S&P 500 | 353 / 147 firms, 137,541 / 53,406 events; non-members are smaller (median log market cap 21.4 vs 23.5) and more volatile; combined L6 scores 0.402 [0.397, 0.407] on S&P firms and 0.396 [0.389, 0.403] on the others (no detectable difference); pooling helps the non-S&P firms (pooled minus non-S&P-only model +0.008 [+0.000, +0.016], p=0.05) |
| S20 / S21 Track A | CAR[-1,+1] +0.10% (t 4.6) and abnormal log volume +0.31 (t 45.8) around disclosures; SUE-conditioned drift Q5-Q1 -3.0% [-3.7%, -2.3%] (n=11,885), verdict 'supported' but a REVERSAL, likely inflated by the survivor-only universe (non-S&P -4.7% vs S&P -2.4%); text beyond momentum weak/mixed |
| Report | `full/reports/results_report.html` (+ .pdf, .md): 35 figures; GPU-dependent sections (L5/L7/L8, H1, dev holdout, locked test, backtest, Track C) say 'not produced' until Part 2/3 run |

The full run found and fixed defects that smaller runs could not show (decision log D-068 to D-073), including two point-in-time
leaks caught by the pipeline's own assertions (same-day 8-K groups, same-day 10-Q in a release's SUE) and a drift-verdict bug.
The universe covers about 32% of the tickers that were S&P 500 members in 2000-2026 (survivor-only).

## Early-era check (2000-2004, real filings)
Six firms, 361 filings, 317 events, run on real EDGAR data before any full run (decision log D-052/D-053). It found and fixed two
defects (Exhibit 13 MD&A under "FINANCIAL REVIEW" titles; full-year EPS read as a quarter). Expect for 2000-2008: quality B/C
timestamps (no usable SGML acceptance time before ~2003), no SUE before ~2004-05, and a common cohort that is mostly 2009+.

## Long runs on Colab
`sec.s05_max_wall_hours` (notebook default 8) makes the EDGAR download stop cleanly and say so; run the same cell again and finished
firms are skipped. Dependency pins are in `requirements.lock`; all 26 ChronoBERT vintages are pinned to commit SHAs.

## Labelling (Track C)
S12 writes `gold_*_filings_template.csv` and `gold_passages_template.csv` + `ANNOTATION_GUIDE.md` into the S12 folder.
Complete them (gold-test blind; a second rater on ≥ 30% of gold-test), copy them to `DATA_ROOT/inputs/gold/`, and re-run
S12 → S16 → S17 → S23 with force.

## Where things are
`configs/` parameters, pre-registration, model ledger, 8-K taxonomy, question bank · `src/secprice/` stages and modules ·
`tests/` unit + integration + fixtures · `docs/decision_log.md` every decision and deviation · `tools/` input preparation
and the notebook builder.
