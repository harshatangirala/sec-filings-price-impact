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
pytest                                   # 43 unit tests + offline integration smoke (gate G1)
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

## Labelling (Track C)
S12 writes `gold_*_filings_template.csv` and `gold_passages_template.csv` + `ANNOTATION_GUIDE.md` into the S12 folder.
Complete them (gold-test blind; a second rater on ≥ 30% of gold-test), copy them to `DATA_ROOT/inputs/gold/`, and re-run
S12 → S16 → S17 → S23 with force.

## Where things are
`configs/` parameters, pre-registration, model ledger, 8-K taxonomy, question bank · `src/secprice/` stages and modules ·
`tests/` unit + integration + fixtures · `docs/decision_log.md` every decision and deviation · `tools/` input preparation
and the notebook builder.
