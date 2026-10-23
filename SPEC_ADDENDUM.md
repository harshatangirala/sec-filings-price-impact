# V5 Pre-Coding Review Addendum
## SEC Filings → Price Impact

**Purpose:** Small corrective addendum to `SEC_Filings_Price_Impact_Build_Spec_v5.md`. This is not a replacement for v5. It clarifies issues to resolve before the relevant implementation milestones are accepted.

## Decision

**Approve milestone M0 and begin coding the repository scaffold.** Do not begin full data/model runs until the target definitions and return calculations below are resolved and covered by tests.

The v5 design is substantially improved: it separates event study, prediction and content validation; defines a repository and data contracts; narrows the default model roster; adds milestone gates; and makes smoke mode the default.

## P0 — Resolve before implementing prediction targets

### 1. Make the primary Track B target truly prospective

In v5, `T_info` starts from the last regular-session close before filing acceptance and measures the response after public availability. However, the model decision occurs later, at `decision_time = feature_ready_time`. The T_info target can therefore include price movement that occurred before the model could actually make its prediction.

Implement the following separation:

- **Track A / T_info:** retain as an information-response/event-study outcome. It answers how prices moved around a filing. Do not present a model scored on this target as an uncontaminated prospective forecast from `decision_time`.
- **Track B / T_trade:** make this the primary prospective prediction target and primary leaderboard for the trading-relevant task. Its outcome must begin at the first executable price after `decision_time`.
- Train a separate model per target if both targets are modeled. Do not use a model trained to predict T_info as the sole signal model for a T_trade backtest.
- Keep the same filing event eligible for Track A and Track B where possible, but report the different clocks, eligible cohorts and sample counts explicitly.
- Revisit the preregistration, primary metric and hypotheses so the primary Track B metric is defined on T_trade. T_info remains the complementary event-response analysis.

**Acceptance test:** demonstrate with synthetic PRE, POST, NONTRADING and INTRA events that no return component used as a Track B target occurs before its decision time.

### 2. Match stock returns, benchmark returns and factor intervals

The backtest enters at an open and exits at a close, while the project also uses daily close-to-close market and factor returns. These intervals must not be mixed.

- Keep the close-to-close total-return series for event-study outcomes where appropriate.
- Calculate executable T_trade returns from consistently adjusted entry and exit prices, with dividends and splits handled explicitly.
- Align benchmark returns to the exact same holding interval. For open-to-close entries, use a matching benchmark open-to-close component; for subsequent sessions, calculate the remaining holding-period return consistently.
- Do not subtract a close-to-close market/factor return from an open-to-close stock return.
- Preserve the separately specified factor-adjusted event-study method and backtest alpha method.
- Test a known split, a dividend during the holding period, overnight gaps and entry/exit across multiple sessions.

**Acceptance test:** unit tests establish the exact return for simple synthetic and known corporate-action examples, and verify that stock and benchmark intervals match.

### 3. Label estimated public availability honestly

The SEC documents that filing availability on sec.gov often follows the EDGAR acceptance timestamp by one to three minutes, with potentially longer delays and no guarantee. Therefore, historical `acceptance_time + 5 minutes` is an estimate, not a verified actual public-availability timestamp.

- Preserve acceptance time as observed metadata.
- Store actual public availability only when independently observed and supported; otherwise mark it as estimated.
- Preserve the configured public lag and feature-processing latency in the run manifest.
- Retain the 0 / 15 / 60-minute sensitivity (and the configured decision-latency sensitivity).
- Use wording such as **“point-in-time evaluation under an assumed availability/processing-latency policy”** when exact historical public availability is unavailable.
- Reserve “strict point-in-time” for model/data configurations whose chronology is actually documented and satisfied; do not imply the SEC publication delay was observed when it was inferred.

**Acceptance test:** the event table and report distinguish observed timestamps from assumed timestamps and label each relevant result with its timing policy.

## P1 — Resolve before accepting the corresponding milestones

### 4. Provide enough historical data for SUE

The config currently specifies a two-year predecessor lookback for filing-change features, while SUE's denominator uses up to eight prior seasonal EPS changes. Computing eight year-over-year seasonal changes requires more historical EPS observations than two years can normally supply.

- Separate `text.predecessor_lookback` from `accounting.predecessor_lookback`.
- For SUE, fetch enough earlier point-in-time quarterly/release EPS history to calculate the configured seasonal-difference history where available (a starting policy of at least four years before `STUDY_START` is safer than two).
- If the source cannot support the history, mark SUE unavailable for that event; do not fabricate values or silently shorten history without recording the actual number of observations.
- Report SUE coverage by year, filing type and company size.

**Acceptance test:** synthetic histories with 3, 4 and 8 valid seasonal differences produce the documented behavior and record the number of prior observations.

### 5. Keep the CPU smoke test realistically lightweight

The v5 smoke profile mentions a local LLM and RAG, but the LLM in the core roster is a 7B model and cannot be assumed to run quickly or fit in memory on every laptop CPU.

- The mandatory offline CPU smoke should use fixtures and a deterministic mock/stub generator for the LLM boundary; it must validate the same stage interfaces, schemas, caches and tests.
- Keep actual Qwen inference in an explicitly opt-in GPU smoke test on Colab.
- Use a tiny real model on CPU only if the machine passes a RAM/runtime preflight.
- Make the report unambiguous about which smoke stages use fixtures, real SEC/network responses, GPU or a local model.

**Acceptance test:** the mandatory offline CPU integration test completes without downloading a 7B model or requiring network/GPU access.

### 6. Guard the gold-test against repeated use

The gold-dev / gold-val / gold-test split is a major improvement. Make the code enforce it rather than relying on notebook discipline.

- Only gold-dev may be used for schema/prompt iteration.
- Gold-val selects retrieval, generator, detector and confidence thresholds.
- Gold-test remains blind until the winning pipeline is frozen; it is evaluated once for final content-validity reporting.
- Ensure the same filing or passage cannot appear across those partitions.
- Because rare categories are oversampled, do not treat the gold sample's class prevalence as population prevalence unless sampling weights are applied.

## Build order

Continue with v5's milestones:

1. **M0:** scaffold, config validation/hash, artifacts/manifests, CLI, logging, tests, `CLAUDE.md`, README.
2. **M1–M2:** SEC metadata, universe, prices/calendar/corporate actions and coverage audit.
3. **M3–M6:** parsing, taxonomy, accounting/SUE, event clocks, targets and leakage-safe features/splits.
4. **M7:** baselines and offline CPU integration smoke.
5. **M8–M9:** event studies and gold annotation tooling.
6. **M10 onward:** only then begin Colab GPU embeddings, tuning and gated RAG work.

Do not let M10 start expensive training if the target-timing tests or return-interval tests are still failing.

## Instruction to Claude Code

Apply this addendum alongside `SPEC.md`. Do not silently change v5: record each resolution in `docs/decision_log.md`, update the preregistration where primary-target definitions change, add automated tests for every item, and summarize the implementation choices at the end of each milestone.

**Bottom line:** v5 is ready for staged coding. The main remaining priority is to make the primary prediction target genuinely prospective, match tradable and benchmark return intervals, and be transparent that historical public-availability times are estimated when they are not observed.
