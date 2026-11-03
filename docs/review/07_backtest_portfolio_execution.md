# Backtest, portfolio construction and execution

## 1. What we built

`backtest/engine.py` turns the frozen model's probabilities into a daily calendar-time book.

| Step | Logic | Code |
|---|---|---|
| Signal | s = P(Up) - P(Down) from a frozen architecture refitted on `T_trade` h = 5 labels | `backtest_spec.json`, D-032 |
| Thresholds | 80th and 20th percentiles of training-fold signals | `engine.py:17-20` |
| Entry and exit | Enter at the open of the first session at or after decision time; hold 5 sessions; one position per firm, a newer event replaces the older one | `engine.py:23-39` |
| Weights | Equal weight per leg, capped at 5% per name; a leg with fewer than 10 names holds cash | `engine.py:76-79`, `default.yaml:204` |
| Costs | 10 or 20 bps per side applied to turnover; 50 bps a year borrow on shorts | `engine.py:85-89`, `default.yaml:205-206` |
| Benchmark | SPY on the identical intervals per position | `engine.py:57-68` |
| Statistics | Annualised return, volatility, Sharpe, maximum drawdown, deflated Sharpe, Carhart alpha on the holding component | `engine.py:107-134` |

Locked-test result (no delay, 10 bps): long book 1.3% a year, long minus matched S&P 500 -5.6%, hypothetical long-short -2.0%; the one-session-delay version is worse (-0.9%, -7.3%, -5.5%). Development results look better (7.1% and 11.8% a year for no delay and one-session delay). The long book holds 8.6 names on average and is in cash on 68% of locked-test days.

## 2. Strengths

1. Event-driven timing with an executable entry and an interval-matched benchmark.
2. Entry-day and holding components are separated so factor alpha is not computed against a mismatched interval (`engine.py` docstring).
3. A delay variant and two cost levels are reported.
4. Deflated Sharpe ratio is computed with the configuration counter.
5. The model's own claim is not oversold: the report labels it an economic-relevance check.

## 3. Weaknesses, with evidence (W10)

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| 1 | **Breadth is tiny.** About 2,379 events a year across 150 firms; the long book averages 8.6 names | `review_facts.json`, `backtest.csv` | Few positions, high idiosyncratic variance, wide Sharpe intervals |
| 2 | **Cash drag is dominant.** `min_names_per_side` = 10, so the long leg is uninvested on 68% of locked-test days | `engine.py:76`, `backtest.csv` `share_days_long_in_cash` | Annualised return and Sharpe are computed over a book that is mostly idle; the comparison with SPY on invested days only is correct but hides the capacity issue |
| 3 | **Signal is direction only.** It ignores the magnitude skill the model does have | `engine.py:3` | Wastes the one signal with rank IC above 0.2 |
| 4 | **Backtest model differs from the frozen model.** It is refitted on h = 5 labels | D-032 | The tested object is not the evaluated object |
| 5 | **Equal weights.** No volatility scaling, no signal weighting | `engine.py:78` | Dominated by high-volatility names |
| 6 | **No risk model, no neutralisation.** The long book carries market, size and momentum exposure; the long-short is not beta- or sector-neutral | `engine.py` | Long book returns reflect market exposure; the carhart alpha is estimated only on the holding component |
| 7 | **Flat costs.** 10 or 20 bps per side regardless of spread, size, volatility, time of day | `engine.py:85` | Opening-auction fills in mid caps cost more; earnings-day spreads are wide |
| 8 | **No market impact or capacity.** No participation limit, no ADV | None | Cannot state the size at which the strategy fails |
| 9 | **Short leg is hypothetical.** No borrow availability; flat 50 bps | `default.yaml:206` | Borrow on hard-to-borrow names around events can exceed 10% a year |
| 10 | **Survivors only; no delisting returns.** | W2 | Upward bias |
| 11 | **Entry at the open with daily bars.** No slippage versus the open print, no auction model | `engine.py:57-60` | The open is assumed fillable at the print |
| 12 | **Thresholds from training signals.** Percentiles are fixed in time while signal distributions drift (adversarial AUC 1.0) | `engine.py:17` | The share of days with positions varies with regime |
| 13 | **Sharpe and deflated Sharpe use daily returns of a mostly-idle book.** | `perf` | The deflated Sharpe probabilities (0.000 to 0.013) are not meaningful at this breadth |
| 14 | **No scenario or stress tests.** | None | |
| 15 | **No turnover constraint or cost-aware optimisation.** | None | |

## 4. What a world-class backtest needs

| Component | Specification |
|---|---|
| Universe and data | Survivor-free, point-in-time, with delisting returns; intraday prices; borrow and short-interest data |
| Signal | The same frozen model that was evaluated, scored once per event with a documented latency |
| Portfolio | Risk-model-based optimiser: maximise expected alpha minus risk aversion times variance minus transaction cost, subject to constraints (gross, net, beta, sector, size and momentum neutrality, name and ADV limits) |
| Sizing | Alpha scaled by volatility and confidence; magnitude used for risk budgeting |
| Costs | Spread/2 + square-root impact + fees + borrow, calibrated to participation: cost_bps ≈ a x spread_bps/2 + b x sigma_daily x sqrt(Q / ADV), with a and b calibrated on live or vendor execution data (initial values are assumptions to be calibrated) |
| Execution | Auction and VWAP fill models, partial fills, limit-order behaviour, time-of-day effects |
| Capacity | AUM at which expected cost equals half the gross alpha |
| Evidence | IC, ICIR, decile spreads, IR net of costs with bootstrap interval, hit rate, turnover, drawdown, factor-adjusted alpha (market, size, value, momentum, profitability, investment, short-term reversal), crowding and tail metrics |
| Robustness | Delay of 0, 1, 5, 30 minutes; cost multipliers of 1, 2, 3; subperiod and regime splits; leave-one-sector-out; bootstrap over events |
| Attribution | Return split by event type, session, size bucket and horizon |

## 5. Monetisation routes (decide before building)

| Route | Needs | What the evidence says | Risk | Verdict |
|---|---|---|---|---|
| **A. Stock long-short on direction** | Direction skill | Direction rank IC +0.018 on 2008-2019 and about -0.008 since 2020 (file 12) | Skill decayed or was never robust | Do not build until the decay is diagnosed (EXP-N3) and EXP-M1 passes |
| **B. Options on magnitude** (long or short straddle against implied move) | Options prices and implied moves; model abs(AR) forecasts | Magnitude rank IC 0.20 to 0.28, but a single trailing-volatility feature scores 0.16 to 0.26 (incremental skill about -0.01 to +0.06); the price-only classifier L1 scores about 0 only because it does not use volatility for magnitude | Implied vol already prices scheduled earnings. Edge, if any, is on unscheduled 8-Ks and relative mispricing | Test with EXP-P2 before any claim |
| **C. Sizing overlay on an existing book** | A portfolio | Predicted magnitude lowers or raises event risk | Needs a portfolio | Cheap to test as risk-reduction around events |
| **D. Execution and liquidity provision** | Intraday data | Predicted magnitude informs spread and inventory risk around events | Needs microstructure data | Potentially valuable for a market maker |
| **E. Signal sale or research product** | Evidence of persistence | None yet | Reputational | Not before forward evidence |

## 6. Additions, ranked

| Rank | Addition | Effort |
|---|---|---|
| 1 | Replace flat costs with a spread-and-impact model fitted to available bid-ask data; add ADV-based capacity | 2 to 3 weeks |
| 2 | Rebuild the backtest on the frozen model's own predictions (not a refit), with a prediction ledger | 1 week |
| 3 | Add a risk model (PCA or industry-factor) and neutralisation constraints | 3 to 4 weeks |
| 4 | Add magnitude-based sizing and a stand-alone magnitude strategy (route B or C) | 2 to 3 weeks |
| 5 | Replace the 10-name minimum with a minimum gross that targets capital deployment; report cash drag separately | 3 days |
| 6 | Report net IR with a block bootstrap interval and delays of 0, 1, 5, 30 minutes | 1 week |
| 7 | Borrow-cost and availability data for shorts; hard-to-borrow exclusion rules | 2 weeks |
| 8 | Turnover-aware optimisation and holding-period optimisation (1, 3, 5, 10 days) | 2 weeks |
| 9 | Stress and scenario tests: 2008, 2020, 2022 | 1 week |
| 10 | Live paper trading with the execution simulator | 4 weeks |

## 7. Experiments

| ID | Question | Design | Accept or kill |
|---|---|---|---|
| EXP-P1 | What does the book look like with realistic deployment? | Same signal, minimum names 3, volatility-scaled weights, full capital deployment on signal days | If net IR at 10 bps is below 0.3 with a lower bound below 0, stop direction strategies |
| EXP-P2 | Does the magnitude forecast beat implied moves? | For scheduled earnings, compare predicted abs(AR) with the front-month ATM straddle implied move; trade when the gap exceeds a threshold fixed in advance | Continue only if net P&L after options spreads has a positive lower bound over at least 2 years and 500 events |
| EXP-P3 | Is the magnitude signal more than known event timing? | Magnitude baseline with event type, session and trailing volatility versus L6 | See EXP-M2 |
| EXP-P4 | How sensitive are results to cost assumptions? | Multipliers 1x, 2x, 3x on a spread-impact model | Report break-even cost |
| EXP-P5 | What is capacity? | ADV participation at 1%, 5%, 10% | State AUM at half-alpha |
| EXP-P6 | Does a sizing overlay reduce event risk? | Reduce positions by predicted move before events and compare realised variance and drawdown | Adopt if tail loss falls by more than 10% without losing more than 10% of return |
