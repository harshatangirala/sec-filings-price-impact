# Project handoff: everything that is not in the other files

Written 5 November 2026 so the project can be continued in a new window. It collects what lives in the conversation history, the machine's folder layout and the owner's decisions, none of which are recorded elsewhere in the repository. The repository docs (`README.md`, `SPEC.md`, `docs/decision_log.md`, `docs/review/`) cover the study itself; read those for method and results.

To start a new session: open the repository folder, tell the assistant to read this file, `CLAUDE.md`, `README.md`, `docs/review/00_EXECUTIVE_SUMMARY.md` and `docs/review/12_SECOND_PASS_FINDINGS.md`, then say what you want to do.

---

## 1. Who and what

- **Owner:** Krishna Harsha Tangirala, harshatangirala2@gmail.com. Master's student in **Applied Artificial Intelligence, University of Wollongong in Dubai**. GitHub: `harshatangirala`.
- **Affiliation, one more time:** Master's in Applied Artificial Intelligence at the University of Wollongong in Dubai. An unrelated `CLAUDE.md` in the Downloads folder belongs to a different project and carries a different affiliation. An earlier version of the paper used it by mistake; the owner corrected this on 3 November 2026 and the paper, report and assistant memory were fixed. Never take biographical details from that file.
- **Machine setup** (from the global `C:\Users\harsh\.claude\CLAUDE.md`): Antigravity (Gemini) and Claude Code share files at `G:\My Drive\antigravity_workspace\` and `C:\Users\harsh\.gemini\antigravity\scratch\`. GPU work runs on Google Colab Pro+ with Drive mounted at `/content/drive/MyDrive/`; the Colab study folder is `MyDrive/sec-price-study`.

## 2. The project in one page

Question: do SEC filings (10-K, 10-Q, 8-K) predict next-day stock reactions? 150 surviving S&P 500 stocks, 2000 to July 2026, 25-stage `secprice` pipeline (S00 to S24), built to `SPEC.md` v5 plus `SPEC_ADDENDUM.md`. Three tracks: A event study (association), B prospective prediction, C content validity (Track C not done; no gold labels).

Design facts you need when touching anything:
- Four timestamps per event: acceptance, public (+5 min), decision (+30 min), next open. Public availability is an assumption, so the wording is "point-in-time evaluation under an assumed availability and processing-latency policy".
- Primary target: T_trade at horizon 1 (open-to-close abnormal return, stock minus beta times SPY on the same interval), three classes at plus or minus 0.5 sigma root h (about 29% Down, 41% Sideways, 30% Up). T_info (close-to-close) is the Track A response.
- Splits: walk-forward with six two-year folds 2008 to 2019 (28,533 common-cohort events), development 2020 to 2022 (7,453 labelled), **locked test 2023 to July 2026 (8,199 labelled), scored once and spent**. 90-day embargo, purge, uniqueness weights. Primary metric macro-F1 (chance about 0.33 from shuffled predictions; majority-class 0.19). Month-block bootstrap.
- Ladder L0 to L8. L6 is all non-neural features (price, metadata, accounting, word counts, TF-IDF). L7 is L6 plus frozen ChronoBERT/BGE embeddings. L5 is embeddings alone. Models: LightGBM, CatBoost, elastic net. Optuna with tiny budgets.
- Headline results: locked macro-F1 L6 0.384, L7 0.383; text lift L7 minus L6 = -0.001 [-0.012, +0.009]. The models predict whether a stock moves, not which way. A later numbers-only rung (`paper/analysis/no_text_rung.py`, fitted after the lock, so exploratory) does about as well.
- D-079 (2 November 2026): the post-earnings drift "reversal" (-2.6%) is an artifact. The pipeline's CARs subtract pre-event alpha times window length (`track_a/event_study.py:56`). Without it, Q5 minus Q1 = +0.12% [-0.80, +0.88]; calendar-time Carhart alpha +0.25% (t = 0.46). Also, L6 already contains classical text, so L7 minus L6 only tests embeddings. Stored stage outputs were not changed; `reports/results_summary.md` carries a correction notice.
- Second pass (5 November 2026), in `docs/review/12_SECOND_PASS_FINDINGS.md` and `13_SECOND_PASS_SUMMARY.md`: direction rank IC +0.018 pre-2020 then -0.008; magnitude skill is volatility persistence; accounting availability rule is vacuous before 2009; `meta_year` is a feature; measured minimum detectable IC 0.036 to 0.042.

## 3. Where everything is on this machine

### In the git repository `C:\Users\harsh\Downloads\sec-filings-price-impact`
Remote: `https://github.com/harshatangirala/sec-filings-price-impact` (public).

| Path | What |
|---|---|
| `SPEC.md`, `SPEC_ADDENDUM.md`, `CLAUDE.md` | Binding spec and the working rules |
| `src/secprice/`, `tests/`, `configs/`, `tools/`, `notebooks/` | Pipeline, 110 tests, configs (`default.yaml`), Colab notebook |
| `docs/decision_log.md` | Decisions D-001 to D-079 |
| `docs/audit_2026-10-23.md`, `docs/stage_cards/` | Audit and per-stage cards |
| `docs/write-up/` | Report PDF and docx, 13-slide deck, `report_audit.md`, `source/` (build scripts) |
| `docs/review/` | Quant review: files 00 to 13, `evidence/` scripts and CSVs |
| `docs/HANDOFF_CONTEXT.md` | This file |
| `paper/` | SSRN working paper PDF (36 pp), `build_paper.py`, `render_pdf.py`, `figures/`, `tables/`, `replication/`, `references.json`, `references_check.md`, `analysis/` |
| `analysis/` | Earnings-drift scripts (set `RUN_ROOT`) |
| `reports/` | Automated pipeline report (tables, 48 figures) |
| `dashboard/` | Streamlit app; live at https://sec-filings-study.streamlit.app/ |
| `.venv/` | Python 3.12 environment (4.5 GB, not in git, not in any export) |

### Outside the repository (not in git, not on GitHub)
| Path | What |
|---|---|
| `C:\Users\harsh\Downloads\sec_report_mini\full\` | The finished 150-stock run: stage folders S00 to S24, `logs`, `manifests`, `reports`, `stage_cards`. **This is the `RUN_ROOT` the evidence scripts read.** |
| `C:\Users\harsh\Downloads\sec_report_mini\inputs\` | Run inputs: `ohlcv_1day`, `lexicons`, `ff5.zip`, `mom.zip`, `Siccodes12.zip`, S&P membership CSV, frozen yfinance gap-fill, HF census |
| `C:\Users\harsh\Downloads\sec_results_mini\full\` | GPU-side results (S13 to S19, S22, S23) |
| `C:\Users\harsh\Downloads\sec_inputs_staging\` and `sec_inputs_staging_mini\` | Staging area for CPU stages and earlier profiles (smoke, pilot, probes, notebook tests) |
| `C:\Users\harsh\Downloads\sec_filings_price_impact_COMPLETE_STUDY.zip` | 307 MB archive of the complete study |
| `C:\Users\harsh\Downloads\SEC_Filings_Price_Impact_Build_Spec_v5.md`, `..._Colab_Build_Spec*.md`, `SEC_Filing_NLP_Research_Spec_for_Claude_Code.md`, `SEC_DATA_DICTIONARY_AND_DOWNLOAD_SCOPE.md`, `..._v4_Professor_Review.md`, `..._v5_Pre_Coding_Addendum.md` | Original specs and review notes |
| `C:\Users\harsh\Downloads\SEC_Filings_Price_Impact_v5_EndToEnd.ipynb` (and `(1)` copy) | Colab notebooks |
| `C:\Users\harsh\Downloads\sec_report_build\`, `sec_deck_build\` | Scratch builds for the report (edit2 to edit6 scripts) and the deck (Node and COM). The maintained copies are in `docs/write-up/source/` |
| `C:\Users\harsh\Downloads\SEC_Quant_Review\`, `SEC_Project_Export\` | Earlier export folders (refreshed in the final step of the 5 November session) |
| `C:\Users\harsh\Downloads\SEC_Filings_Price_Impact_Report.pdf/.docx`, `SEC_Filings_Study_13_Slides.pptx`, `Tangirala_SEC_Filings_Point_in_Time_Test.pdf` | Delivered copies |
| `C:\Users\harsh\.claude\projects\C--Users-harsh\memory\` | Assistant memory (`MEMORY.md`, `sec-price-impact-paper.md`, `sec-price-impact-project.md`, `triframe-project-state.md`) |
| `C:\Users\harsh\.claude\projects\C--Users-harsh\7a39cc5b-40a1-4c3c-99c1-95fc538640e1.jsonl` | Full transcript of the session that produced the paper, report conversions and reviews |

## 4. State of GitHub and git

- On GitHub (public), pushed on 5 November 2026 at the owner's request: code, tests, dashboard, report, deck, the paper PDF with corrected affiliation, replication data, the quant review (`docs/review/`, files 00 to 13 plus `evidence/`), this handoff file and `docs/original_specs/`.
- Not on GitHub because of size or privacy: `.venv`, the pipeline run folders and raw inputs (see section 3), and the conversation transcript.
- Check the current state with `git status` and `git log origin/main..HEAD` before assuming anything.

## 5. What was done, in order (this stretch of work)

1. Added five to seven 2024-2026 Q1-journal references to the report's last page (scraped via OpenAlex/Crossref).
2. Audited the report against project files; fixed overstated claims (see section 7).
3. "Humanised" the report PDF over several rounds after the owner reported AI-detector scores of 75% and then 80%. Installed `humanizer@humanizer` plugin (not loadable in-session until restart; its `SKILL.md` was applied by hand).
4. Produced the report as .docx (python-docx; rendered via Word COM because the LibreOffice helper fails on Windows).
5. Wrote the 36-page SSRN working paper (21 figures, 18 tables, references checked against OpenAlex/Crossref (8 benign differences kept as published)); pushed to GitHub.
6. Added affiliation, funding ("no external funding; personal Colab Pro+") and AI-use statements; verified hand-entered references; later corrected affiliation (section 1).
7. First quant-officer review (SWOT, per-stage critique, roadmap with 25 scored items, experiment cards, research agenda, issue register).
8. Exported markdown, papers and a repo snapshot into `SEC_Project_Export`.
9. Second-pass empirical check (files 12 and 13) with measured corrections.

## 6. Stances and decisions made in conversation

- **AI-detector scores:** no score can be guaranteed. The first-draft prose was AI-written, so the honest route is for the owner to rewrite parts in their own voice and keep the disclosure. The AI-use disclosure stays in the paper.
- **Do not invent facts.** Numbers come from stored outputs and scripts. Anything not computed is labelled as exploratory or not done.
- **Outward-facing actions need approval:** pushing commits, uploading to SSRN, publishing pages. The owner approved: making the repo public and pushing the paper and corrections. The review commits were not approved yet.
- **Locked test is spent.** Nothing computed on 2023 to July 2026 can be called out-of-sample again; label any new use as development or exploratory.
- **LaTeX download was blocked** by the permission classifier, so the paper PDF is built from HTML via Playwright (`paper/build_paper.py`, `paper/render_pdf.py`).
- Third-person neutral pronouns for anyone whose pronouns are unknown.

## 7. History of corrected claims (anything saying otherwise is wrong)

| Earlier claim | Corrected to |
|---|---|
| Post-earnings reversal of -2.6% | Artifact of the alpha term (D-079); +0.12% [-0.80, +0.88] without it |
| "Text adds nothing" | Embeddings add nothing over a model that already has classical text; the test of text as a whole was not done |
| "About a third of 1,094 tickers" | 353 eligible, 150 used |
| Backtest dev vs locked | Dev used lag 1, locked used lag 0: not comparable |
| 0.384 | The no-text model (L6); frozen L7 is 0.383 |
| "Announcement day" | A 3-day window |
| -3.3% vs -2.6% | Market-model vs Carhart on the same 3,825 reports |
| Direction IC about zero | +0.018 on 2008-2019, about -0.008 since 2020 |
| Magnitude skill is event metadata | It is volatility persistence |
| Power: MDE 0.033 to 0.048 | Measured 0.036 to 0.042 |
| A different institution on the paper's title page (taken from an unrelated file) | University of Wollongong in Dubai, Master's in Applied AI |

## 8. Open items

**Owner's to do**
1. Re-upload the corrected paper PDF to SSRN and fix the SSRN profile affiliation.
2. Decide whether to push the review commits (and this file).
3. Read the paper end to end before posting; add a conflict-of-interest statement if the journal needs one.
4. Check the Streamlit dashboard: it showed blank on Cloud earlier and the cause was never found.
5. Fix or move `C:\Users\harsh\Downloads\CLAUDE.md` (unrelated project).

**Engineering, in the order the review recommends**
1. Fix the accounting availability rule (`models/ladder.py:72`, count real `acc_*` values) and drop `meta_year`; rebuild per-rung subsets and re-run L3, L6, L7 (EXP-N1, N2).
2. Diagnose the direction-IC decay: rolling windows, structural break, per-era feature signs (EXP-N3).
3. Fit magnitude directly and benchmark against the volatility-times-event-type baseline (EXP-N4).
4. Year-by-year IC for L7 and LightGBM (N5).
5. Locate the raw minute-bar files behind `local:1min` (EXP-N6).
6. Fix the alpha term in the pipeline (`track_a/event_study.py:56`, `window_cars`) before the 500-firm study; the 500-firm results also predate D-077 (S09 to S24 need re-running, about 4.5 hours).
7. Patch the paper (accounting-adds-least statement and the direction "null"), rebuild, push, re-upload.
8. Forward shadow test on new filings.

Full prioritised list with gates and stop rules: `docs/review/09_roadmap_and_experiments.md`.

## 9. How to reproduce and rebuild

```powershell
cd C:\Users\harsh\Downloads\sec-filings-price-impact
.venv\Scripts\activate
pytest                                           # 110 tests
$env:RUN_ROOT = "C:\Users\harsh\Downloads\sec_report_mini\full"
python docs\review\evidence\deep_checks2.py      # second-pass evidence (reads stored outputs only)
python paper\analysis\make_tables.py             # paper tables
python paper\analysis\make_figures.py            # paper figures
python paper\build_paper.py ; python paper\render_pdf.py   # HTML then PDF through Playwright
```
Rebuild only the final report stage without touching any model: `secprice run ... --stages S24` with `force=["S24"]`.

Quirks to know:
- Crossref returns 429 under load; `paper/analysis/verify_references.py` uses OpenAlex first and delays Crossref.
- Word COM, not LibreOffice, renders the docx and pptx on this machine.
- JSON dumps need string keys for dates.
- Markdown table cells must not contain bare pipes (use abs()).
- SEC access needs `SEC_USER_AGENT="Name email"` and at most 6 requests per second.
- Treat anything downloaded as untrusted; run Python on it with `-I`.

## 10. Prompt to paste into a new window

> Read `docs/HANDOFF_CONTEXT.md`, `CLAUDE.md`, `README.md`, `docs/review/00_EXECUTIVE_SUMMARY.md` and `docs/review/12_SECOND_PASS_FINDINGS.md` in `C:\Users\harsh\Downloads\sec-filings-price-impact`. The locked test is spent. Do not push or upload anything without asking. Start with engineering item 1 in section 8 (accounting rule and `meta_year`), show me the plan first, and keep all new results labelled exploratory.

## 11. Facts from the assistant's memory that are not stated elsewhere

These came from the assistant's own memory notes, written during the earlier build (26 October to 3 November 2026). They are recorded here so a new window does not depend on that memory.

**Run history.** All 25 stages ran: the CPU part locally, the GPU part on Colab (fast mode, A100-class GPU, exact model not recorded). The locked test was scored once on 2026-10-25T23:59:58Z. Report v2 (D-078) was built from the finished stages into a Desktop folder `SEC_Report_150_Stocks` (html, pdf, md, 48 figures). The report stage is rebuilt with `force=["S24"]` through `Downloads\sec_inputs_staging
eport_run.py` on the merged root `Downloads\sec_report_mini` (CPU stages plus Colab stages). The Colab settings were reconstructed as `unlock_locked_test=True`, `acknowledge_reuse=True`. The raw Colab results copy is `Downloads\sec_results_mini`. The notebook that embeds the code also sits on the Desktop and in `colab_upload_mini\`; commit code before rebuilding the notebook.

**Headline numbers from that run (locked 2023 to 2026, n = 8,199).** L6 macro-F1 0.384. Real chance is about 0.33 (shuffled own predictions), not the 0.19 majority baseline. Walk-forward L7 minus L6 = -0.001 [-0.007, 0.005]; development +0.007; locked -0.001 [-0.012, 0.009]. Gate G3 failed, so no fine-tuning. Magnitude (does it move) AUC 0.635; direction AUC 0.494 [0.475, 0.509]. Backtest shows no edge. H4 no difference; H3 and H6 untestable (no gold labels). Track C, L8 and G3b were not run (149 Qwen-pre-filled annotation templates exist).

**Known issues listed in report section 11.**
1. LightGBM re-scores differ across machines (about 80% identical classes), roughly plus or minus 0.004.
2. The L6-versus-L7 freeze compared scores from two runs (S11 on the PC, S15 on Colab). Inside S15, L6 0.3909 beat L7 0.3898, so frozen L7 (retrospective, BGE) is a tie, not a win.
3. Stress folds were built but not scored.
4. S19 does not compute locked-test paired lifts (the report does).
5. PBO = 0 is uninformative.
6. Fast mode dropped sec_bert, finbert and catboost-on-embeddings (for time; not measured).
7. Tuning budgets are below Optuna's 10 random start-up trials, so several rungs share identical parameters.
8. The environment manifest is from the PC; the GPU model is unrecorded.
9. Locked direction AUC is identical for L6 and L7 (0.493902) and has not been investigated.

**500-firm study (the planned major project).** The CPU part exists, but accounting-dependent results are superseded by D-077 (re-run S09 to S24, about 4.5 hours). Before starting it: make LightGBM deterministic, compare finalists inside one run with a strict-model tie-break, score the stress folds, add locked lifts to S19, annotate the gold labels (templates are pre-filled), consider adding delisted firms, and fix `window_cars` (alpha term). The locked test of that study is unspent.

**Machine and process notes.** Windows, 15 GB RAM. Use PowerShell. Launcher scripts need `if __name__ == "__main__"`. Inline Python quoting breaks in PowerShell, so use script files. Drive sync from the PC is unreliable; upload zips through the browser. `pymupdf` is in the venv only to render PDF pages for checks.

**Dashboard.** `dashboard/` is a Streamlit 1.66 app with two windows: "Project Flow" (a clickable 16-step Plotly flowchart with per-step tabs) and "Slide Deck" (18 slides with notes, an overview grid and an HTML handout). The data snapshot is built by `dashboard/build_snapshot.py --root <merged root>/full` and holds numbers from artifacts only. Run it with `dashboard
un_dashboard.bat` or `cd dashboard; streamlit run app.py` (the working directory must be `dashboard/` for `.streamlit/config.toml`). Tests are in `tests/dashboard`. Streamlit does not reliably hot-reload `lib/` modules, so restart the server after edits. The cloud app at https://sec-filings-study.streamlit.app/ returns 200 but the frame stayed blank after the redeploy; a fresh clone runs locally. Logs are only visible to the owner in Streamlit Cloud (Manage app, then logs or Reboot app, or recreate the app with Python 3.12).

**Report references.** The written report ends with a References page of seven verified papers (He et al., JFE 185; Fink, JFQA 59(6); Karolyi, Management Science 71(3); Zhang, IRFA 94; Lan, IRFA 95; Magner, FRL 81; Fengler, IRFA 103). Scimago confirmed Q1 only for IRFA and FRL. A small language-model perplexity test (Qwen2.5-0.5B) gave the report a median of 121, against 95 for human abstracts and 19.5 for generic AI prose; it is a style probe, not a real detector.

**Original specs** are now in `docs/original_specs/` (build specs, research spec, data dictionary, professor review). `SPEC.md` and `SPEC_ADDENDUM.md` in the repository root are the binding v5 versions.
