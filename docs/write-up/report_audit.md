# Report audit, 1 November 2026

What this is: every number and factual claim in `SEC_Filings_Price_Impact_Report.pdf` checked against the project's own files
(`dashboard/data/*.csv|json`, `configs/`, `docs/decision_log.md`, the analysis scripts re-run on the finished run folder, and the
test suite). The report's tables and most figures are generated from those same files, so most checks confirm the build did
not drift from them. The hand-written sentences are where errors can hide, and that is where the problems were.

## Checked and matching
- Counts: 64,213 filings, 63,933 parsed, 280 without sections, 2,550,579 tables, 121,398 sections, 881,908 + 249,224 masks, about 900,694 chunks and 474M tokens, 304,348 XBRL facts, 14,981 releases, 1,024,405 bars, 150 firms in 12 industries, 743 eligible and 979 candidate firms, 60,261 events (45,193 8-K, 11,271 10-Q, 3,797 10-K).
- 128 features in seven groups (68/19/18/11/10/1/1); 0 availability violations; 4 of 4 planted leaks caught.
- Splits: six two-year folds 2008-2019, development 2020-22, locked test 2023 to July 2026, 90-day embargo, minimum 8 training years.
- Results tables (validation, development, locked): accuracy, macro-F1 and intervals, AUC, log-loss, calibration error. Locked: 0.384 without text, 0.383 with text.
- Text lift: validation -0.001 [-0.007, +0.005], development +0.007, locked -0.001 [-0.012, +0.010]. Gate rule (at least 0.01 and interval excluding zero) and "no fine-tuning ran" (`finetune_status.json`: ran = false).
- Move AUC 0.635 and direction AUC 0.494 on the locked test; Sideways recall 53% from the confusion table.
- Tuning: 6/8/15 trials, 32 rung-by-model combinations, elastic net certified as a global optimum (`enet_certificate.json`).
- Embeddings: ChronoBERT 60,030 events, 3,377 s (56 min); BGE 2,310 s (39 min); S14 total 5,946 s = 1.65 h.
- RAG: 12 questions, rerank top 30, 8 passages to the generator, 149 filings, 47 s each, about 2 h, 141 valid responses, no gold labels so unscored.
- Earnings drift: quintile means (+1.40% to -1.85%), announcement-window reaction (+0.81% / -0.54%), subperiods -4.4 / -2.0 / -4.3%, clean-event -2.1%, 1,908 best-news reports, 54.9% fell, 30.8% rose then fell by about 10%, 45% rose (re-run from `analysis/`).
- Backtest locked test (no delay, 10 bps): long 1.3%/yr, minus index -5.6%/yr, long-short -2.0%/yr; total long return 4.0%.
- Bugs: 63 filings lost to HTTP 503 and all recovered; SEC JSON feed with two time conventions (about 70% / 30%); benchmark that projected 19.3 GPU-hours against a 10-hour budget.
- 114 tests pass (1 skipped). Decision log runs to D-078. Dashboard has 16 steps and 18 slides.

## Wrong or misleading in the first version, now fixed
1. "My data covers about a third of the 1,094 tickers": the repo says the eligible data covers about 353 of 1,094 (32%), of which the study uses 150. Reworded to say that.
2. Backtest comparison used two different settings: the good development number (11.8%/yr) was the one-session-delay run, the locked numbers were the no-delay run. Now compares the same rules (development 7.1%/yr, +2.0% over the index; locked 1.3%, -5.6%) and reports the delayed version for both periods (development 11.8%, +3.7%; locked -0.9%, -7.3%). Also added the missing "hold for five sessions".
3. "The best model scored 0.384": that is the version without text; the frozen model (with text, used for the backtest) scored 0.383. The introduction now says both.
4. "Announcement day" returns are the three-day window around the announcement. Reworded.
5. "Each of the problems above has a test that fails on the old code": not true for the Streamlit and laptop-sleep items. Now says most of the code bugs.
6. Direction AUC was described as a coin flip using the locked value only; on the validation folds it was 0.51 [0.50, 0.52]. Added.
7. Elastic net "0.39 down to 0.35" is the pooled score (mean-fold score is 0.33). Said so.
8. Removed claims about feelings or timing I could not verify ("happiest with", "tripped me up", "a large part of the first week", "I spent a long time checking").

## Open items I could not settle
- Locked direction AUC is identical to six decimals (0.493902) for the all-numbers and the with-text model in `skill.csv`. Different models giving the same value is unusual. The report only quotes the single 0.49 figure, which is unaffected, but the cause is not checked.
- LightGBM re-scores differ for about 20% of class labels across machines (`reproducibility.csv`); already stated in the report.
- The live Streamlit app showed a blank page after wake-up when tested; the cause is not known (logs are only visible to the app owner).
- The first-person narrative of what happened, in what order and how it felt, was drafted from the project record. Anything you remember differently should be changed.
