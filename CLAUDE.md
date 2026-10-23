# CLAUDE.md - sec-filings-price-impact (digest of SPEC.md v5 + SPEC_ADDENDUM.md)

## Hard rules (SPEC.md §0.2, condensed)
1. SEC filings only (10-K, 10-Q, 8-K) as text. Never load news / FNSPID / FinMultiTime news / StockNet tweets.
2. Tracks stay separate: A event study (association), B prospective prediction, C content validity.
3. Four timestamps per event; features available no later than decision_time; prior documents strictly < decision_time.
4. RUN_PROFILE = smoke by default; pilot/full by explicit change; full needs G2 (pilot done + preflight + owner opt-in).
5. Never fabricate data, dates, results or metrics; the report contains computed values only.
6. Association is not causation; prediction is not profitability.
7. EAR never a feature for its own event/cluster; SUE only from the event's own filing/exhibit or earlier public accessions.
8. 8-Ks are event-driven (no expectation). 9. Never choose documents/events using later price moves.
10. High scores trigger an audit, not termination. 11. Provenance decides claims (strict vs retrospective leaderboards).
12. Verify before use; pin revisions; log substitutions. 13. SEC fair access: UA from env, <= 6 rps, backoff, pause on block.
14. No paid/hosted model APIs. 15. Seeds, config hash, commit, environment manifest, pinned deps. 16. Tests are part of the deliverable.

## Addendum (binding)
- Primary Track B target = **T_trade h=1** (first executable price after decision_time); T_info = Track A response.
- Stock and benchmark returns on identical intervals; never a close-to-close factor against an open-to-close return.
- Public availability = acceptance + lag is an **assumption**: wording "point-in-time evaluation under an assumed
  availability/processing-latency policy".
- accounting.predecessor_lookback >= 4 years; CPU smoke uses fixtures + mock LLM; gold-test single use enforced.

## Environment (§13)
- Local: `.venv` (uv, Python 3.12), `pip install -e .[dev]`, `pytest`, `secprice run --profile smoke --stages S00-S12`.
- Colab: `notebooks/SEC_Filings_Price_Impact_v5_EndToEnd.ipynb` (embeds this repo). DATA_ROOT = Drive folder
  `MyDrive/sec-price-study`; HOT_ROOT = /content/hot. Secrets: `SEC_USER_AGENT` ("Name email"), optional `HF_TOKEN`.
- Parts: CPU S00-S12, S20-S21 -> GPU S13-S17 -> CPU S18-S19, S22-S24. Restart the runtime when switching CPU/GPU.

## Layout (§14)
`configs/` (default + profiles + prereg + models ledger + taxonomy + question bank) | `src/secprice/` (stages_data,
stages_model, stages_report, pipeline/cli, sec, universe, market, audit, parse, events, targets, features, splits,
models, rag, gold, evaluation, track_a, backtest, viz, report, extensions, testing) | `tests/` (unit, integration,
fixtures) | `docs/decision_log.md` | `tools/` (input preparation, notebook builder).

## Current milestone
M0-M13 implemented; 44 tests pass on CPU (gate G1); notebook executes end to end on smoke (nbclient). Live smoke (5 firms)
and **pilot Part 1** (40 firms, 2012-2026-07, 11,370 filings, 10,105 events, all timestamps quality A) ran locally; the pilot
artifacts are on Drive (`MyDrive/sec-price-study/pilot`). Next: owner runs pilot Part 2 (A100) + Part 3 on Colab, then full
(G2: `owner_opt_in_full`), and annotates gold (Track C). Decisions D-001..D-051 in `docs/decision_log.md`.
