"""Backtest (SPEC.md §11; addendum P0.2). An economic-relevance check, not an investable strategy.

Signal s = P(Up) - P(Down) from the frozen T_trade model. Long if s >= training-period 80th pct, short if
<= 20th pct (percentiles from training folds only). Entry at the open of the first session at or after
decision_time; exit at the close of session h (entry = session 1). Calendar-time daily portfolio, equal weight per
leg, max 5% per name, one position per firm (a newer event replaces the older one at the new entry open), legs with
fewer than `min_names_per_side` names hold cash. Benchmark SPY on the identical intervals per position. Carhart
alpha is estimated only on the close-to-close holding component; the open-to-close entry-day component is
market-adjusted with SPY open-to-close (never a close-to-close factor against an open-to-close return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def thresholds(train_scores: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    s = np.asarray(train_scores, float)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, lo_pct)), float(np.quantile(s, hi_pct))


def positions(sig: pd.DataFrame, lo: float, hi: float, h: int, lag: int = 0) -> pd.DataFrame:
    """sig: event_id, ticker, entry_idx, score. Returns one row per position with entry/exit indexes (after
    one-per-firm replacement)."""
    s = sig.copy()
    s["side"] = np.where(s.score >= hi, 1, np.where(s.score <= lo, -1, 0))
    s = s[s.side != 0].copy()
    s["entry"] = s.entry_idx + lag
    s["exit"] = s.entry + h - 1
    s["exit_at_open"] = False
    s = s.sort_values(["ticker", "entry"]).reset_index(drop=True)
    for tk, g in s.groupby("ticker"):
        idx = g.index.to_list()
        for a, b in zip(idx[:-1], idx[1:]):
            if s.at[b, "entry"] <= s.at[a, "exit"]:
                s.at[a, "exit"] = s.at[b, "entry"]      # replaced: old position exits at the new entry open
                s.at[a, "exit_at_open"] = True
    return s[s.exit >= s.entry]


def daily_book(pos: pd.DataFrame, panel, spy: dict, n_sessions: int, max_w: float, min_names: int,
               cost_bps: float, borrow_bps: float) -> pd.DataFrame:
    """Per-day leg returns with interval-matched SPY legs and entry/hold decomposition."""
    T = n_sessions
    rec = {k: np.zeros(T) for k in ("long_entry", "long_hold", "short_entry", "short_hold", "spy_long_entry",
                                     "spy_long_hold", "n_long", "n_short", "turnover_long", "turnover_short")}
    contrib = {1: [[] for _ in range(T)], -1: [[] for _ in range(T)]}
    s_open, s_close, s_tri = spy["open"], spy["close_split_adj"], spy["total_return_index"]
    for p in pos.itertuples(index=False):
        if not panel.has(p.ticker):
            continue
        P = panel[p.ticker]
        o, c, tri, dv = P["open"], P["close_split_adj"], P["total_return_index"], P["dividend"]
        e, x = int(p.entry), int(min(p.exit, T - 1))
        for t in range(e, x + 1):
            if t == e:
                r = c[t] / o[t] - 1
                rs = s_close[t] / s_open[t] - 1
                part = "entry"
            elif p.exit_at_open and t == x:
                r = (o[t] + dv[t]) / c[t - 1] - 1
                rs = (s_open[t] + spy["dividend"][t]) / s_close[t - 1] - 1
                part = "hold"
            else:
                r = tri[t] / tri[t - 1] - 1
                rs = s_tri[t] / s_tri[t - 1] - 1
                part = "hold"
            if np.isfinite(r):
                contrib[p.side][t].append((r, rs if np.isfinite(rs) else 0.0, part, t == e, t == x))
    for side, leg in ((1, "long"), (-1, "short")):
        for t in range(T):
            cs = contrib[side][t]
            n = len(cs)
            rec[f"n_{leg}"][t] = n
            if n < min_names or n == 0:
                continue
            w = min(1.0 / n, max_w)
            for r, rs, part, is_entry, is_exit in cs:
                rec[f"{leg}_{part}"][t] += w * r
                if side == 1:
                    rec[f"spy_long_{part}"][t] += w * rs
                rec[f"turnover_{leg}"][t] += w * (int(is_entry) + int(is_exit))
    d = pd.DataFrame(rec)
    c = cost_bps / 1e4
    d["long_ret"] = d.long_entry + d.long_hold - c * d.turnover_long
    d["short_ret"] = d.short_entry + d.short_hold
    borrow = borrow_bps / 1e4 / 252
    d["ls_ret"] = d.long_ret - d.short_ret - c * d.turnover_short - borrow * (d.n_short >= min_names)
    d["spy_matched"] = d.spy_long_entry + d.spy_long_hold
    d["long_excess_spy"] = d.long_ret - d.spy_matched
    return d


def exposure_stats(d: pd.DataFrame, min_names: int) -> dict:
    """Turnover and exposure of the daily book (SPEC.md §11.7): average names held, share of days each leg is invested,
    share of days a leg holds cash because it has fewer than min_names positions, mean daily one-way turnover."""
    nl, ns = d.n_long.to_numpy(float), d.n_short.to_numpy(float)
    return {"avg_names_long": float(nl[nl > 0].mean()) if (nl > 0).any() else 0.0,
            "avg_names_short": float(ns[ns > 0].mean()) if (ns > 0).any() else 0.0,
            "share_days_long_invested": float((nl >= min_names).mean()), "share_days_short_invested": float((ns >= min_names).mean()),
            "share_days_long_in_cash": float(((nl > 0) & (nl < min_names)).mean()),
            "share_days_short_in_cash": float(((ns > 0) & (ns < min_names)).mean()),
            "turnover_long_daily": float(d.turnover_long.mean()), "turnover_short_daily": float(d.turnover_short.mean())}


def perf(r: np.ndarray, n_trials: int, rf: np.ndarray | None = None) -> dict:
    from ..evaluation.mcs_pbo import deflated_sharpe
    from scipy.stats import kurtosis, skew
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 20 or np.std(r) == 0:
        return {"n_days": int(len(r)), "note": "too few active days"}
    sr_d = np.mean(r) / np.std(r, ddof=1)
    eq = np.cumprod(1 + r)
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    return {"n_days": int(len(r)), "ann_return": float(np.mean(r) * 252), "ann_vol": float(np.std(r, ddof=1) * np.sqrt(252)),
            "sharpe": float(sr_d * np.sqrt(252)), "max_drawdown": dd,
            "deflated_sharpe_prob": deflated_sharpe(sr_d, len(r), max(1, n_trials), float(skew(r)), float(kurtosis(r, fisher=False))),
            "total_return": float(eq[-1] - 1)}


def carhart_alpha(r_hold: np.ndarray, dates: pd.DatetimeIndex, factors: pd.DataFrame) -> dict:
    """Carhart alpha of the close-to-close holding component (interval-matched to daily factors)."""
    import statsmodels.api as sm
    f = factors.set_index(pd.to_datetime(factors["date"]))
    d = pd.DataFrame({"r": r_hold}, index=dates).join(f, how="inner")
    d = d[d.r != 0]
    if len(d) < 60:
        return {"alpha_ann": None, "note": "too few days"}
    X = sm.add_constant(d[["mkt_rf", "smb", "hml", "mom"]])
    m = sm.OLS(d.r - d.rf, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {"alpha_ann": float(m.params["const"] * 252), "alpha_t": float(m.tvalues["const"]), "n_days": int(len(d)),
            "betas": {k: float(m.params[k]) for k in ("mkt_rf", "smb", "hml", "mom")}}
