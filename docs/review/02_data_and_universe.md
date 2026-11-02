# Data and universe

## 1. What we built (logic, code, result)

| Layer | Logic | Code | Result |
|---|---|---|---|
| Universe | Firms common to local OHLCV files and the Hugging Face 10-K corpus, mapped ticker to CIK; S&P 500 members first, then seeded, sector-stratified fill; selection never reads returns | `universe/build.py:95-143`, `config universe.*` | 979 candidates, 743 eligible, 150 selected (26 BusEq, 25 Other, 19 Manuf, 19 Shops, 17 Money, 14 Hlth, ...). All 150 are 2026 survivors |
| Filings | EDGAR submissions JSON, archives (SGML), `companyfacts`; politeness limits (6 requests a second, user agent, backoff, pause on repeated 403) | `sec/client.py`, `sec/archives.py`, `sec/submissions.py`, `parse/worker.py` | 64,213 filings; 63,933 parsed; 63 lost to HTTP 503 then recovered |
| Timestamps | SGML header `ACCEPTANCE-DATETIME` read for every filing; JSON feed corrected for double conversion; date-only quality B at 17:30 ET | `events/timestamps.py` | 57,817 events quality A, 1,672 B, 772 C |
| Prices | Local daily bars (split-adjusted, not dividend-adjusted) spliced with a frozen `yfinance` cache before the local history starts; dividends from Yahoo actions; total-return index built from split-adjusted close plus dividend | `market/prices.py:123-163` | 1,024,405 bars; 150 firms plus SPY |
| Factors | Kenneth French daily 5-factor and momentum files | `market/factors.py` | Used for Carhart in Track A and the backtest alpha |
| Accounting | XBRL `companyfacts` for 10 tags as first reported, with public availability time; EPS from press-release exhibits parsed by rules; seasonal-random-walk SUE | `features/accounting.py`, `parse/eps_release.py` | 304,348 facts; 14,981 results releases; SUE for 31% of events |
| Taxonomy | 8-K item and text categories (`taxonomy_8k_v1`) | `parse/taxonomy.py`, `configs/taxonomy_8k_v1.yaml` | 45,193 8-K events labelled |
| Lexicon | Loughran-McDonald 1993-2025 | `features/text.py:26-33` | 18 word-tone features |

## 2. Strengths

1. **Provenance is pinned.** Hugging Face dataset revision, S&P membership commit, Yahoo version, and French URLs are in `meta.json` and `default.yaml`. The frozen Yahoo cache means reruns reproduce.
2. **Timestamp quality is explicit.** Every event carries a quality flag (A, B, C) and the `json_rule_max_mismatch` guard drops the JSON rule to quality C if headers disagree by more than 1%.
3. **Accounting is as first reported.** `quarterly_eps` keeps the earliest public accession per period; `newest_period_facts` fixes the prior-year comparative bug (D-077).
4. **Sampling never reads outcomes** (`select_universe`, hard rule 9).

## 3. Weaknesses and risks, with evidence

### 3.1 Universe and survivorship (W1, W2)

| Issue | Evidence | Impact |
|---|---|---|
| Only 2026 survivors | `universe.csv`; D-057 | Firms that were acquired, delisted or failed are missing; bad-news firms that kept falling drop out |
| 150 of 1,094 index tickers ever in the index; eligible data cover about 353 | `build_report.py` fact and decision log | Coverage of the index is about one in seven, so both breadth and representativeness are low |
| Large-cap only | S&P 500 first | Filings matter most where analyst coverage is thin. Small and mid caps are where text might help |
| Events per firm range 86 to 967 (median 386) | `review_facts.json` | A few firms with many 8-Ks dominate some tests; firm clustering matters |

### 3.2 Prices

| Issue | Evidence | Impact |
|---|---|---|
| Free, Yahoo-derived bars | `market/prices.py` docstring | Free bars can differ from official opening and closing auction prices and may contain errors, especially early in the sample. Nothing in the repository audits them against a reference |
| Split-adjusted, not dividend-adjusted bars with a separately built total-return index | `build_series` | Open-to-close returns exclude dividends except on ex-dates handled in the backtest only. Small bias, but unaudited |
| No delisting returns | Survivors only | Backtest and drift overstate what a point-in-time universe would show |
| SPY as the only benchmark | `targets/build.py` | Industry and size factors are not removed from the target, adding noise |
| Daily bars only | Whole design | Cannot measure the reaction inside the session or in extended hours (W3) |
| Seam between local and Yahoo history | `splice` | Checked on 40 overlap days only |

### 3.3 Filings and parsing

| Issue | Evidence | Impact |
|---|---|---|
| Heuristic section extraction | `parse/sections.py`: longest span after a heading, title-only fallbacks | Hugging Face agreement for 10-K Item 7 is 0.83 Jaccard (D-070); other forms unchecked |
| 8-K body is "first item heading to signature" | `eightk_items_text` | Includes boilerplate; no item-level segmentation beyond codes |
| Press-release EPS parsed by rules | `parse/eps_release.py` | About 84% agreement with XBRL on a replay sample (D-066); 31% SUE coverage |
| Only three EPS tags and ten fact tags | `default.yaml:79-87` | Missing revenue surprise, margins, cash flow, segment data |
| Exhibits beyond EX-99 not parsed | `MAIN_SECTION` | Investor presentations and supplemental tables are not used |
| 15-MB cap on 8-K full submissions | `sec.max_8k_bytes` | Long exhibits are truncated |
| No XBRL-based text for "what changed" | Whole design | Missed the cheapest structured signal |

### 3.4 Timing assumptions (W3)

| Assumption | Where | Issue |
|---|---|---|
| Public = acceptance + 5 minutes; decision = public + 30 minutes | `timestamps.py:71-85` | Both are assumptions, not observed. Sensitivity targets exist (`T_info@pub0/15/60`, `T_trade@lat10/120`, `T_trade_lag1`) but no results are reported (`target_defs_stored`) |
| After-hours acceptance is traded at the next open | `T_trade` | Roughly half the event-window variance happens before that open (estimate, caveat in 01_SWOT W3) |
| 17:30 ET for date-only filings | `date_only_assumed_time` | Quality B events (1,672) are placed after the close even when they may not have been |

## 4. World-class target state

| Dimension | Target |
|---|---|
| Universe | Point-in-time constituents of a broad index (Russell 3000 or CRSP all common stocks above a liquidity floor) with delisting returns; 15 to 20 thousand events a year; small, mid and large strata |
| Prices | Licensed, survivorship-free, corporate-action-audited daily data plus intraday trades and quotes (TAQ, Polygon, Databento or similar), including extended hours |
| Reference data | Point-in-time identifiers (CRSP PERMNO or FIGI), historical GICS and SIC, historical shares outstanding, borrow and short-interest series |
| Expectations | Consensus EPS and revenue, guidance, revisions, with as-first-reported timestamps; earnings-call transcripts with call start times |
| Filings | Full submission archive with item-level segmentation, every exhibit parsed, XBRL financial-report tables, insider forms (3, 4, 5), ownership (13D, 13G, 13F), proxy |
| Quality gates | Automated reconciliation against a second vendor, outlier and split audits, a timestamp audit per source, and a data-quality dashboard |

## 5. Additions, ranked

| Rank | Addition | Data | Closes | Effort | Notes |
|---|---|---|---|---|---|
| 1 | Survivorship-free universe with delisting returns | CRSP or Sharadar or Norgate (verify licence and cost) | W2 | 3 to 6 weeks | Do before any drift or backtest claim |
| 2 | Broader universe (Russell 3000 equivalent) | Same vendors | W1 | 2 to 4 weeks after 1 | Breadth about 6x |
| 3 | Consensus estimates and guidance | I/B/E/S, Refinitiv or Visible Alpha (verify licence) | W4 | 4 to 6 weeks | Needs as-first-reported timestamps. See 10 |
| 4 | Earnings-call transcripts with timestamps | Vendor or EDGAR 8-K Item 7.01 exhibits where available | W4 | 3 to 5 weeks | Q&A tone is a known signal |
| 5 | Intraday trades and quotes including extended hours | TAQ, Polygon, Databento | W3 | 4 to 8 weeks | Redefines the entry clock |
| 6 | Report the existing sensitivity targets | Already stored | W3 | 1 to 2 days | Free. Shows how much the 5 and 30 minute assumptions matter |
| 7 | Full-submission exhibit parsing and item segmentation | EDGAR | Parsing | 2 to 3 weeks | Prerequisite for the text rebuild |
| 8 | Insider, ownership and short-interest data | EDGAR forms, FINRA | New signal | 3 to 4 weeks | See 10 |
| 9 | Second price source and automated reconciliation | Any | Quality | 1 week | Catches split and open errors |
| 10 | Point-in-time identifiers and historical GICS | CRSP or vendor | Joins | 1 to 2 weeks | Removes ticker-reuse risk |

## 6. Experiments for this layer

| ID | Question | Design | Accept or kill |
|---|---|---|---|
| EXP-D1 | How much does the 5 and 30 minute lag assumption move results? | Re-run L6 and L7 on the stored `T_trade@lat10`, `@lat120`, `T_info@pub0/15/60` targets | Report in the paper. If macro-F1 moves by more than 0.01, treat latency as a headline limit |
| EXP-D2 | How large is price-data error? | Reconcile 150 firms against a second source; count opens differing by more than 25 bps and split or dividend mismatches | If more than 0.5% of opens differ, move to licensed data before further work |
| EXP-D3 | How much does survivorship bias the drift and backtest? | Rebuild the 2010 to 2020 sample on a survivor-free universe | Quantify bias in bps; if it changes the sign of any reported effect, state it in all outputs |
| EXP-D4 | What do accounting-feature gaps cost? | Impute from XBRL tags not used (`Revenues`, cash flow) and compare coverage and L3 score | Accept if coverage rises above 85% and L3 improves |

## 7. Data-quality controls to add now

1. **Row-count and null monitors per stage** with thresholds in config (a stage fails if events drop more than 1% run to run).
2. **A golden-set regression test**: 200 hand-checked events (timestamps, sections, EPS) that must reproduce exactly.
3. **A timestamp audit report** per run: share by quality class, share of filings whose header and JSON disagree, and events moved to another trading day by each rule.
4. **Corporate-action audit**: flag adjacent-close ratios beyond 0.5 or 2 that are not explained by a recorded split.
5. **A data-licence register** listing each source, licence, redistribution right and renewal date.
