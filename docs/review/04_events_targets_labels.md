# Events, clocks, targets and labels

## 1. What we built

| Element | Logic | Code |
|---|---|---|
| Event | One filing, or the merged set of one firm's same-day 8-Ks; clock at the last acceptance | `events/build.py` |
| Clocks | acceptance, public = +5 min, feature-ready and decision = +30 min, next open | `events/timestamps.py:71-85` |
| Primary target `T_trade`, h = 1 | Entry at the open of the first session whose open is at or after the decision time; exit at the close of that session; abnormal return = stock return minus beta times SPY on the identical interval | `targets/build.py:149, 186-190` |
| Information target `T_info` | Close-to-close from the last close at or before acceptance to the close of the h-th session after public time; market-model residuals with alpha; intraday events excluded | `targets/build.py:148, 168-181` |
| Estimation window | Sessions -250 to -30 relative to the base or entry session, at least 120 observations | `targets/build.py:164-166`, `default.yaml:104-105` |
| Labels | Up if AR > k sigma sqrt(h), Down if AR < -k sigma sqrt(h), else Sideways; k = 0.5 (0.25 and 0.75 for sensitivity) | `targets/build.py:68-75` |
| Horizons | 1, 2, 5, 20, 60 sessions | `default.yaml:100` |
| Weights | Average label uniqueness per firm | `targets/build.py:95-110` |
| Sensitivity targets | `T_trade_lag1`, `T_trade@lat10`, `T_trade@lat120`, `T_info@pub0`, `@pub15`, `@pub60` | `targets/build.py:150-159` |

Class shares of the primary label: 29.4% Down, 40.5% Sideways, 30.0% Up.

## 2. Strengths

1. The executable target is the right idea: a trader can only earn what happens after the decision.
2. Benchmark and stock are measured over the same interval (S4).
3. Event definitions handle same-day 8-K merging and cluster membership; overlap flags use a 60-session window.
4. The target set separates "information" from "tradable".

## 3. Weaknesses, with evidence

### 3.1 The entry clock discards the first reaction (W3)

| Evidence | Number |
|---|---|
| Mean absolute `T_trade` for PRE events | 1.92% |
| Mean absolute `T_info` for PRE events | 2.84% (ratio 0.67) |
| POST events | 1.65% versus 2.30% (ratio 0.72) |
| Variance share before the open under independence | about 1 - ratio squared = 54% (PRE), 48% (POST) |
| Caveat | The gap includes ordinary overnight variance that exists without news. A non-event baseline is needed to separate the two |

Consequence: on after-hours filings, which are 54% of events, the model forecasts the afterglow. Where the information is richest, the first price available to the model is already informed. This is the strongest single explanation for weak direction skill, and it is a design property, not a modelling one.

INTRA events are a special case. The 35-minute decision time pushes entry to the next session's open, so an event accepted at 10:00 is traded about 23 hours later even though the market was open at the decision time. This loses a window where trading was possible.

### 3.2 Abnormal-return model

| Issue | Code | Effect |
|---|---|---|
| Beta from daily close-to-close pairs is applied to an open-to-close return | `build.py:164-166, 189` | Overnight and intraday betas differ; residual market exposure remains |
| No risk-free or alpha term on the one-day return | `build.py:189` | Minor at one day, but inconsistent with `T_info` |
| Estimation window includes earlier events | `build.py:164` | Prior earnings days inflate sigma and distort beta |
| Sigma is a daily residual volatility, not an intraday one | `build.py:166, 202` | T_trade sd is 1.43 times mean sigma overall, yet 59.5% of events exceed 0.5 sigma versus 61.7% under a normal with sd equal to sigma. The label shares are balanced by offsetting errors, not by calibration |
| One benchmark (SPY) | `build.py:187-190` | Industry and size effects stay in the target; idiosyncratic noise is higher than necessary |
| Beta estimated per event from 220 days | `build.py:164` | Noisy for volatile firms; no shrinkage |
| Open price is the first print on free data (unaudited) | Prices | Entry noise unknown |

### 3.3 Labels

| Issue | Evidence | Effect |
|---|---|---|
| Three classes throw away magnitude | `label3` | Magnitude is predictable (rank IC 0.20 to 0.28) and direction is not; one label mixes them |
| Class shares differ across contexts | 10-K 46.2% Sideways; PRE 32.6%; INTRA 44.9% (`sample_returns.csv`) | One threshold rule across all contexts means the "Sideways" class has a different meaning per event type |
| Fixed k = 0.5 for all firms and horizons | `default.yaml:111` | Arbitrary; sensitivity labels exist but are not reported |
| Macro-F1 optimises class balance, not a trade | See 05 | |

### 3.4 Event definition

| Issue | Evidence | Effect |
|---|---|---|
| All 8-K item types pooled | Mix (categories overlap): roughly 12.5k earnings, 5.8k management change, 8k Reg FD, plus agreements, votes and others | Heterogeneous information content under one model |
| Earnings release versus periodic filing for the same quarter are separate events | `_clusters`, `_earnings_types` | The 10-Q after the earnings 8-K carries little news; overlap flags handle part of this |
| No separation of scheduled and unscheduled events | None | Scheduled events have known timing and implied-volatility context |
| Merged same-day 8-Ks | `events/build.py` | A merged event takes the last clock, so earlier information is traded late |

### 3.5 Horizons

Targets exist for 1, 2, 5, 20 and 60 sessions, but only h = 1 is modelled in the main results. Slower content such as change in risk factors or guidance is unlikely to resolve in one session.

## 4. World-class target state

| Element | Target |
|---|---|
| Clock | Entry at the first executable quote after a measured latency, using intraday bars and extended-hours trades; latency distribution from a simulated reader; sensitivity to 1, 5, 15 and 30 minutes |
| Targets | (a) signed return from entry to a set of horizons (5 minutes, 1 hour, close, next close, 5 days), (b) absolute move and realised variance, (c) a ranking target such as the within-date percentile of abnormal return |
| Abnormal returns | Industry- and size-matched benchmark; intraday beta; shrinkage; excluded prior event windows from estimation |
| Labels | Continuous targets as primary; classes only for reporting. If classes are kept, thresholds from the matched interval's realised volatility and per event type |
| Event taxonomy | Separate models or heads for earnings releases, guidance, management changes, material agreements, Reg FD and periodic filings; scheduled and unscheduled flags |
| Pre-event baseline | A non-event baseline of volatility and overnight variance to separate news from routine movement |

## 5. Additions, ranked

| Rank | Addition | Effort | Notes |
|---|---|---|---|
| 1 | **Measure the first-reaction share with a non-event baseline** (same firms, matched non-event days) | 2 days | Turns the 48 to 54% estimate into a defensible number |
| 2 | **Report the six stored sensitivity targets** | 1 to 2 days | Free |
| 3 | **Intraday re-definition of the clock**, entry at decision time for PRE and INTRA, extended hours for POST | 4 to 6 weeks | Needs intraday data (see 02) |
| 4 | **Regression and rank targets**: `ar_mm`, `abs_std` and within-date percentile as primary with class labels derived | 1 week | The stage already builds `ar_mm`, `vol_ratio`, `abs_std` |
| 5 | **Industry-and-size-adjusted benchmark** using sector ETFs or Fama-French 12 industry portfolios | 1 week | Lowers target noise |
| 6 | **Intraday beta and volatility** from open-to-close pairs; exclude earlier event windows from estimation | 1 week | Removes mismatches |
| 7 | **Event-type models** with a shared encoder and type-specific heads | 3 weeks | See 05 |
| 8 | **Multi-horizon outcomes** for slower content (5 and 20 sessions) with alpha-free, calendar-time evaluation | 2 weeks | Avoids the artifact |
| 9 | **Scheduled versus unscheduled flags** from the earnings calendar | 1 week | Context for volatility pricing |
| 10 | **Shrunk betas** (Blume or Bayesian) | 2 days | Hygiene |

## 6. Experiments

| ID | Question | Design | Accept or kill |
|---|---|---|---|
| EXP-T1 | How much of the move happens before the first open, net of routine overnight variance? | Compare event overnight variance with matched non-event days for the same firms | Report the net share. If net share is above 30%, prioritise intraday data |
| EXP-T2 | Does regression or rank training beat classification for direction? | Same features, LightGBM with `lambdarank` or Huber on `ar_mm` and on within-date percentile; measure rank IC | Continue if direction rank IC exceeds 0.015 with a lower bound above 0 on pooled walk-forward |
| EXP-T3 | Is the label threshold material? | Report L6 and L7 on `label3_k025` and `label3_k075` (stored) | Report; if conclusions change, k becomes a pre-specified design choice |
| EXP-T4 | Is the benchmark noise material? | Industry-adjusted AR versus SPY-adjusted AR; compare target variance and model IC | Adopt if target variance falls by more than 10% |
| EXP-T5 | Is INTRA entry loss material? | For INTRA events, simulate entry at decision time with intraday data | If the intraday reaction after decision time carries more than 25% of the absolute move, redefine entry |
| EXP-T6 | Do slower horizons carry what h = 1 does not? | Run L6, L7 and change features on h = 5 and h = 20 with calendar-time portfolios | Report; promote horizons with a positive lower bound |
