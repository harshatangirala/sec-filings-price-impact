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
M0-M13 implemented; 99 tests (unit, offline integration, opt-in live online smoke); notebook runs end to end on smoke on a
clean data root. **Full profile Part 1 (S00-S12, S20, S21, S24) ran on all 500 firms, 2000-2026, real data**; artifacts on
Drive (`MyDrive/sec-price-study/full`), seen as done by the notebook's own settings. Owner decisions: no pilot gate (D-058),
S&P / non-S&P / combined reporting (D-056), survivor-only wording (D-057). Next: owner runs Part 2 (A100, `PART="gpu"`) and
Part 3 on Colab, annotates gold (Track C), then unlocks the lockbox once. Decisions D-001..D-073; audit with closure table
in docs/audit_2026-10-23.md.
