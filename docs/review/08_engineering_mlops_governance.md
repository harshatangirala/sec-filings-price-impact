# Engineering, reproducibility, governance and live operation

## 1. What we built

| Area | State | Evidence |
|---|---|---|
| Pipeline | 25 stages (S00 to S24), each a function with declared inputs, a content hash of relevant configuration, and an output folder named by that hash; `_SUCCESS` markers make reruns skip finished stages | `runner.py`, `storage.py`, `pipeline.py` |
| Configuration | One YAML default plus profiles (smoke, pilot, full); owner overrides logged; config, code and pre-registration hashes stored in `meta.json` | `config.py`, `configs/` |
| Parts | CPU part (S00 to S12, S20, S21), GPU part (S13 to S17), CPU part (S18, S19, S22 to S24); runtime restart between CPU and GPU | `CLAUDE.md`, notebook |
| Tests | 115 collected, 114 pass, 1 skipped; unit tests for timing, accounting, returns and targets, splits and metrics, lockbox, regex backtracking, reproducibility, speed mode; integration smoke tests; 4 dashboard tests | `tests/` |
| Dependencies | `pyproject.toml` with lower bounds, `requirements.lock`, `requirements.txt` | repo root |
| Notebook | Colab notebook embeds the repo, runs end to end on smoke | `notebooks/` |
| Dashboard | Streamlit app with 16 steps and 18 slides over a data snapshot | `dashboard/` |
| Docs | Decision log (79 decisions), stage cards, audit note, report, paper | `docs/` |

## 2. Strengths

1. **Content-addressed, resumable stages.** A Colab disconnect costs nothing; a changed parameter re-runs only what depends on it.
2. **Everything is logged:** seeds, config hash, code hash, git commit, hardware fields where recorded.
3. **Regression tests for real bugs**, including adversarial timing for regex backtracking, a shuffle-invariance test for the accounting lookup, and a planted-leak self-test.
4. **A strict lockbox mechanism** in code, not just policy.
5. **Documentation of decisions**, so a new owner can understand why.

## 3. Weaknesses, with evidence (W16)

| # | Finding | Evidence | Risk |
|---|---|---|---|
| 1 | No continuous integration | No `.github/workflows` in the repo | Regressions reach the main branch; the 4-minute test suite is run by hand |
| 2 | No container or lock of the OS environment | Only `requirements.lock` | "Works on my machine" differences (the LightGBM 80% identical result) |
| 3 | No data versioning | Data and run folders live outside git; hashes recorded in manifests only | Cannot restore a run from a commit |
| 4 | No prediction ledger | Predictions live in stage parquet files | No signed, time-stamped record for live use |
| 5 | Non-deterministic training | `reproducibility.csv` | Same code gives different classes on different machines |
| 6 | Environment manifest incomplete | Colab session not described; GPU model not recorded (`report` section 11) | Reproducibility gap |
| 7 | Large modules | `stages_model.py` 1,201 lines, `narrative.py` 1,148, `stages_data.py` 1,135, `stages_report.py` 1,127 | Harder to review and test |
| 8 | Test coverage is concentrated on pure functions | 115 tests over 12,940 lines; no coverage report in the repo | Stage-level logic is under-tested (for example `window_cars`, which hid the alpha issue) |
| 9 | No independent re-implementation of key numbers | None | A single author, a single codebase |
| 10 | Hard-coded Windows paths in analysis scripts | `analysis/` scripts default `RUN_ROOT` to a local folder | Portability |
| 11 | Dashboard deployment unreliable | Streamlit Cloud rendered a blank page when last tested | Public-facing failure |
| 12 | Duplicated logic across scripts | `paper/analysis` re-implements CAR code that exists in `track_a` | Divergence risk; the artifact appeared in the stored version and needed a parallel version to expose |
| 13 | Public repo contains the whole record | Earlier incorrect claims in earlier commits | Reputational; mitigated by D-079 and correction notes |
| 14 | No monitoring for data or model drift | None | Needed for live use |
| 15 | Secrets and licence handling informal | Environment variables for the SEC user agent and HF token; licence register absent | Compliance |

## 4. Governance and process

### 4.1 What exists

Pre-registration file with a hash, hypotheses H1 to H8, gates G3, G3b, G4, a hard-rule list in `CLAUDE.md`, an audit note, and a decision log.

### 4.2 Gaps a model-risk reviewer would raise

| Gap | Suggested control |
|---|---|
| Registration not externally timestamped | Register on OSF or AEA, or publish the hash through a notarised timestamp, before any new data are accessed |
| Freeze and lock were minutes apart without review | Written freeze checklist, a minimum 48-hour cooling-off, and a named second reviewer |
| Model selection compared scores from different runs | One run produces all contenders; tie-break rules written in advance |
| Exploratory work after the lock mixed with confirmatory claims | Separate exploratory and confirmatory sections in every output, enforced by a flag in the results table |
| A single author for code and analysis | Independent re-computation of headline numbers by a second person or an independent code path |
| No change management for data sources | A source register, versioned snapshots, vendor change alerts |
| No model card | One per frozen model |
| No incident log | A short log of data or model incidents with root cause and fix; D-068, D-071, D-072, D-077 and D-079 are the seeds |

## 5. World-class target state

| Area | Target |
|---|---|
| CI | GitHub Actions: lint, type check, unit and smoke tests, a 200-event golden-set regression, a notebook execution test on every pull request |
| Environment | Docker image for CPU and GPU parts; lock file for Python and system packages; recorded hardware |
| Data | DVC or lakeFS with content hashes, or an immutable object store with manifests; one command restores a run |
| Determinism | Fixed seeds and thread counts, deterministic flags, seed bagging; bit-for-bit reruns on the same image |
| Predictions | Append-only signed ledger (hash chain, timestamp service) of every prediction with feature hash, model hash and decision time |
| Monitoring | Daily data completeness, feature drift (population stability index), score distribution drift, latency and error rates; alerts |
| Quality | Coverage target above 80% on core modules; property-based tests for timing and splitting; mutation tests on leakage assertions |
| Documentation | Architecture note, data dictionary, model cards, runbook, incident log |
| Compliance | Licence register; secrets in a vault; audit trail |

## 6. Live shadow-test design (recommended first deliverable)

1. **Ingest.** Poll EDGAR for new 10-K, 10-Q and 8-K filings for the universe; record the receipt time and the SEC acceptance time.
2. **Parse and featurise** with the frozen pipeline; assert availability at the decision time.
3. **Predict** with the frozen models (L6 and L7 and the magnitude baseline).
4. **Commit** each prediction to a ledger line: event id, accession, acceptance time, decision time, feature hash, model hash, outputs. Chain each line with the hash of the previous one and timestamp the chain head daily (git commit to a separate repository, or a public timestamping service).
5. **Score** after outcomes realise: abnormal return over 1 and 5 sessions, rank IC, decile spread, calibration; monthly report; no changes to models during the test.
6. **Pre-state the success threshold** (for example direction rank IC at or above 0.02 with a lower bound above 0 after 12 months, and a magnitude rank IC above the metadata baseline by 0.03).
7. **Cost.** Mostly engineering time; compute is small.

## 7. Additions, ranked

| Rank | Addition | Effort |
|---|---|---|
| 1 | Prediction ledger and live shadow test | 2 weeks |
| 2 | CI with golden-set regression | 3 days |
| 3 | Fix the long-window alpha term in `window_cars`; add a calendar-time stage and a regression test that fails on the old code | 1 week |
| 4 | Deterministic training mode and seed bagging | 3 days |
| 5 | Docker image and hardware recording | 3 days |
| 6 | Split the four largest modules by responsibility; add coverage reporting | 1 to 2 weeks |
| 7 | Data versioning for run folders | 1 week |
| 8 | Independent re-implementation of headline numbers | 1 to 2 weeks |
| 9 | Model cards and a source and licence register | 1 week |
| 10 | Monitoring and alerting for the live path | 2 weeks |
| 11 | Fix and re-test the dashboard deployment; add a health check | 2 days |

## 8. Experiments and checks

| ID | Check | Pass condition |
|---|---|---|
| EXP-E1 | Re-run the full pipeline from the Docker image on a clean machine | Class predictions identical for the elastic net; LightGBM within seed-bag variance |
| EXP-E2 | Mutation test: break each leakage assertion and confirm a test fails | Every assertion has a failing test |
| EXP-E3 | Independent recompute of Table V and the drift table from raw price and filing data | All values within rounding |
| EXP-E4 | Replay one month of live ingestion against the batch pipeline | Identical features for every event |
