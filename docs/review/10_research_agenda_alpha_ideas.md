# Research agenda: ideas that could make this a world-class programme

This file collects ideas beyond fixing the existing design. Each has a mechanism, a testable claim, the data needed, the main risk, and an honest view of the chance it works. Probabilities are the reviewer's judgement, not estimates from data.

## 1. Why the current question is hard

Machine readers act on 8-Ks within seconds. By the time a 35-minute decision clock fires, most easily extractable information is priced. Slow-moving signals that fast readers do not trade are the realistic space: changes across filings, cross-firm effects, relative language, and content that needs context. The agenda below follows that logic.

## 2. Ideas

### A. Expectation-relative surprise (highest value)

| Item | Detail |
|---|---|
| Mechanism | Direction on an earnings release is surprise relative to what analysts and management expected. Our SUE is a seasonal random walk; no guidance or consensus is used |
| Claim | Consensus surprise, revenue surprise, and guidance-versus-consensus predict the sign of the first reaction and its drift better than seasonal SUE |
| Data | I/B/E/S or equivalent consensus with timestamps; guidance from press releases (parsed) and from transcripts |
| Test | L6-v2 with expectation features on earnings events only; direction rank IC and decile spread; compare with SUE baseline |
| Risk | Vendor timestamps and revisions; the first reaction happens before our entry (so this needs the intraday clock) |
| Chance | Moderate to high that it improves direction on earnings events; low that it survives costs at a 35-minute clock |

### B. Intraday and extended-hours reaction curves

| Item | Detail |
|---|---|
| Mechanism | The reaction begins at acceptance; the first minutes carry most variance. A reader with realistic latency captures part of it |
| Claim | A 1 to 5 minute reader, using the first quote reaction and text features, can capture a measurable fraction of the move; a 35-minute reader cannot |
| Data | Trades and quotes with extended hours |
| Test | Estimate the price path conditional on acceptance time; simulate latency distributions; net of spreads |
| Risk | Competition; execution quality; data cost |
| Chance | Reaction curves will be measurable. Profit depends on latency we may not match |

### C. Passage-level change and language novelty

| Item | Detail |
|---|---|
| Mechanism | Investors underreact to changes in boilerplate-heavy documents (Cohen, Malloy and Nguyen 2020) |
| Claim | Passage-level added and deleted content in risk factors, MD&A and legal proceedings predicts 5- to 60-day returns on alpha-free measures |
| Data | Existing filings with section alignment |
| Test | Calendar-time long-short on change score by tercile; control for size, momentum, industry |
| Risk | Published and likely decayed; our 150-firm sample cannot test it |
| Chance | Low to moderate in large caps; higher in small caps |

### D. Typed numeric extraction and guidance

| Item | Detail |
|---|---|
| Mechanism | Guidance changes, buyback sizes, dividend changes, impairment amounts are numbers in text. Dropping numeric lines removes them |
| Claim | Typed features (guidance midpoint change, buyback percent of market cap, dividend change) have direction information beyond tone |
| Data | Existing filings; hand labels for 500 filings |
| Test | Direction rank IC on the subset with each field present |
| Risk | Label cost; low frequency of some fields |
| Chance | Moderate for guidance and buybacks |

### E. Magnitude and volatility products

| Item | Detail |
|---|---|
| Mechanism | Predicted move size has rank IC 0.20 to 0.28 on the locked test |
| Claim | Beyond event-type and volatility, text and numbers predict move size, which matters for options, sizing and risk |
| Data | Options (OptionMetrics or similar); implied move for scheduled events |
| Test | Incremental IC over a baseline; trade or size against implied moves |
| Risk | The IC is close to what trailing volatility alone gives (file 12); implied vol already prices scheduled events |
| Chance | Low to moderate for scheduled events; moderate for unscheduled 8-Ks, where there is no implied move |

### F. Peer and network spillovers

| Item | Detail |
|---|---|
| Mechanism | A firm's filing carries information about suppliers, customers and competitors that react more slowly |
| Claim | Text-similarity or industry-linked peers earn abnormal returns in the days after a firm's earnings or guidance |
| Data | Industry codes, 10-K supplier and customer disclosures, text similarity |
| Test | Peer portfolio returns after event, alpha-free, calendar time |
| Risk | Documented in the literature; requires broad coverage |
| Chance | Moderate with a wide universe |

### G. Insider, ownership and short-interest context

| Item | Detail |
|---|---|
| Mechanism | Forms 3, 4, 5, 13D, 13G, 13F and short interest condition how information is absorbed |
| Claim | The same filing event has different expected drift when insiders are net buyers or short interest is high |
| Data | EDGAR forms, FINRA short interest |
| Test | Interaction features and conditional drift |
| Risk | Crowded; effect sizes small |
| Chance | Low as stand-alone alpha, moderate as conditioning variables |

### H. LLM-based reading under strict point-in-time rules

| Item | Detail |
|---|---|
| Mechanism | Language models can summarise, compare filings and extract structured content cheaply |
| Claim | A chronologically consistent LLM with structured outputs beats dictionary and frozen-embedding baselines on the earnings subset |
| Data | Existing filings; ChronoGPT or a dated model; labels for scoring |
| Test | Scored extraction accuracy first; then downstream rank IC |
| Risk | Contamination (use Glasserman-Lin and Sarkar-Vafa tests); cost; non-determinism |
| Chance | Moderate that extraction works; low to moderate that it adds direction skill |

### I. Cross-event memory

| Item | Detail |
|---|---|
| Mechanism | A firm's reaction to an 8-K depends on its recent filings (guidance history, litigation, credit events) |
| Claim | A sequence model over the last 8 filings outperforms flat features |
| Data | Existing |
| Test | EXP-M8 |
| Risk | Overfitting at this sample size |
| Chance | Low to moderate; revisit after the universe widens |

### J. Execution-side value

| Item | Detail |
|---|---|
| Mechanism | Predicted event magnitude helps market makers and execution desks size spreads and inventory around filings |
| Claim | Move forecasts reduce adverse selection cost |
| Data | Quote and trade data |
| Test | Realised spread and inventory loss versus forecast deciles |
| Risk | Needs microstructure expertise |
| Chance | Plausible; unexplored |

## 3. What to avoid

1. Adding encoders to the existing mean-pooled pipeline and rerunning L7 minus L6.
2. Tuning classifiers further on macro-F1.
3. Claims of alpha from a 150-name long-short book.
4. Any long-window abnormal return that subtracts a pre-event alpha.
5. Reading intraday information into daily-bar results.

## 4. A staged plan for the research agenda

| Stage | Ideas | Prerequisite |
|---|---|---|
| 1 | A (surprise) and D (typed numerics) on earnings events only | Consensus data; labels; event-type models |
| 2 | B (intraday) and E (magnitude) | Intraday and options data |
| 3 | C (passage change) and F (peers) | Broad universe |
| 4 | H (LLM) and I (memory) | Labels and compute |
| 5 | G and J | Operational maturity |

## 5. Success criteria for a world-class programme

| Dimension | Criterion |
|---|---|
| Evidence | A pre-registered primary result and a 12-month live shadow test agree in sign, with intervals that exclude zero |
| Breadth | At least 10,000 events a year across a survivor-free universe |
| Economics | Net IR above 0.5 at 2x modelled costs with a bootstrap lower bound above 0; stated capacity |
| Robustness | Positive in at least three of four regimes; stable across seeds and machines |
| Governance | Model cards, source register, incident log, independent recomputation, external registration |
| Communication | A paper, a replication package, and a public correction log |
