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
| 1. full Part 1 | CPU High-RAM | `RUN_PROFILE="full"`, `PART="cpu1"` | S00-S12, S20-S21 (~9 h EDGAR at 6 req/s, resumable; ~2.5 h S11) |
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

## Pilot Part 1 results (engineering evidence; local CPU, 23 Oct 2026)
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
