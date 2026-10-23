"""Drift tests and verdict rubric (SPEC.md §10.2). 'Supported' needs: consistent definitions, adequate coverage, an
FDR-adjusted p < 0.05 AND a CI excluding the pre-registered trivial threshold, the same sign in >= 2 of 3
subperiods, clean-event robustness, and survival under 10/20 bps costs and a one-session lag. Failing one test
never proves an anomaly is absent."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..evaluation.bootstrap import mean_ci
from .event_study import clustered_mean

VERDICTS = ["supported", "weak/mixed", "not supported", "inconclusive/low power", "not testable in scope"]


def trivial_bps(h: int, tb: dict) -> float:
    return float(tb["h_le_2"] if h <= 2 else tb["h_5"] if h <= 5 else tb["h_ge_20"])


def prior_quarter_quintiles(df: pd.DataFrame, col: str) -> pd.Series:
    """Quintile using breakpoints from the previous calendar quarter's distribution (tradable; no look-ahead)."""
    q = pd.to_datetime(df["trackA_day0"]).dt.to_period("Q")
    out = pd.Series(np.nan, index=df.index)
    for per, g in df.groupby(q):
        prev = df[(q < per) & (q >= per - 4)][col].dropna()
        if len(prev) < 25:
            continue
        bp = np.quantile(prev, [0.2, 0.4, 0.6, 0.8])
        out[g.index] = np.searchsorted(bp, g[col].to_numpy(), side="right") + 1
    out[df[col].isna()] = np.nan
    return out


def spread_test(df, qcol, car_col, firm, date, n_boot, seed):
    """Q5 - Q1 spread: OLS of CAR on a Q5 dummy within Q1 u Q5 with two-way (firm, date) clustered SE;
    CI from a month-block bootstrap of mean(Q5) - mean(Q1)."""
    import statsmodels.api as sm
    from scipy.stats import norm
    sub = df[df[qcol].isin([1, 5]) & df[car_col].notna()]
    hi, lo = sub[sub[qcol] == 5], sub[sub[qcol] == 1]
    if len(hi) < 10 or len(lo) < 10:
        return None
    from .event_study import cluster_fit
    y = sub[car_col].to_numpy(float)
    X = np.column_stack([np.ones(len(sub)), (sub[qcol] == 5).to_numpy(float)])
    res, _how = cluster_fit(y, X, sub[firm].to_numpy(), sub[date].astype(str).to_numpy())
    est, se = float(res.params[1]), float(res.bse[1])
    t = est / se if se > 0 else None
    months = pd.to_datetime(sub[date]).dt.to_period("M").astype(str).to_numpy()
    um, inv = np.unique(months, return_inverse=True)
    is_hi = (sub[qcol] == 5).to_numpy()
    s_hi = np.bincount(inv, weights=np.where(is_hi, y, 0), minlength=len(um))
    n_hi = np.bincount(inv, weights=is_hi.astype(float), minlength=len(um))
    s_lo = np.bincount(inv, weights=np.where(~is_hi, y, 0), minlength=len(um))
    n_lo = np.bincount(inv, weights=(~is_hi).astype(float), minlength=len(um))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(um), len(um))
        a, b = n_hi[s].sum(), n_lo[s].sum()
        if a > 0 and b > 0:
            bs.append(s_hi[s].sum() / a - s_lo[s].sum() / b)
    lo_ci, hi_ci = (np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan))
    return {"estimate": est, "t": t, "p_value": float(2 * (1 - norm.cdf(abs(t)))) if t is not None else None,
            "n": int(len(sub)), "ci_low": float(lo_ci), "ci_high": float(hi_ci),
            "q5_mean": float(hi[car_col].mean()), "q1_mean": float(lo[car_col].mean())}


def subperiod_signs(df, qcol, car_col, n_sub=3) -> tuple[int, list]:
    d = pd.to_datetime(df["trackA_day0"])
    edges = np.quantile(d.astype("int64"), np.linspace(0, 1, n_sub + 1))
    signs = []
    for a, b in zip(edges[:-1], edges[1:]):
        g = df[(d.astype("int64") >= a) & (d.astype("int64") <= b)]
        hi, lo = g[g[qcol] == 5][car_col].mean(), g[g[qcol] == 1][car_col].mean()
        signs.append(float(np.sign(hi - lo)) if np.isfinite(hi - lo) else 0.0)
    return signs


def verdict(res: dict | None, trivial: float, p_adj: float | None, signs: list, clean_ok: bool | None,
            cost_ok: bool | None, lag_ok: bool | None, min_n: int = 100) -> str:
    if res is None:
        return "inconclusive/low power"
    if res["n"] < min_n:
        return "inconclusive/low power"
    sgn = np.sign(res["estimate"])
    ci_excl = (res["ci_low"] is not None and res["ci_high"] is not None and
               ((res["ci_low"] > trivial / 1e4) or (res["ci_high"] < -trivial / 1e4)))
    same = sum(1 for s in signs if s == sgn) >= 2
    if p_adj is not None and p_adj < 0.05 and ci_excl and same and clean_ok and cost_ok and lag_ok:
        return "supported"
    if p_adj is not None and p_adj < 0.10 or (same and res.get("p_value") is not None and res["p_value"] < 0.10):
        return "weak/mixed"
    return "not supported"


DIV_UP = re.compile(r"(increas|rais)\w*\s+(its\s+|the\s+)?(regular\s+|quarterly\s+)*(cash\s+)?dividend|dividend\s+increase", re.I)
DIV_DN = re.compile(r"(reduc|cut|suspend|eliminat|lower)\w*\s+(its\s+|the\s+)?(regular\s+|quarterly\s+)*(cash\s+)?dividend", re.I)
DIV_INIT = re.compile(r"(initiat|first[- ]ever|inaugural)\w*\s+(a\s+|its\s+)?(quarterly\s+)?(cash\s+)?dividend", re.I)
SPIN = re.compile(r"spin-?off|spinoff|separation of .{0,60}(into|as) (an )?independent", re.I)


def detect_dividend(text: str) -> str | None:
    t = text[:20000]
    if DIV_DN.search(t):
        return "decrease"
    if DIV_INIT.search(t):
        return "initiation"
    if DIV_UP.search(t):
        return "increase"
    return None


def momentum_benchmark(panel, universe_tickers: list[str], start, end) -> pd.DataFrame:
    """Monthly equal-weight decile long-short returns: 1-month reversal, 2-12 (12-2), 12-1. Benchmark only."""
    cal = panel.cal
    me = pd.Series(np.arange(len(cal)), index=cal.sessions).groupby(cal.sessions.to_period("M")).max()
    me = me[(me.index >= pd.Period(start, "M")) & (me.index <= pd.Period(end, "M"))]
    tri = np.vstack([panel[t]["total_return_index"] for t in universe_tickers if panel.has(t)])
    rows = []
    idx = me.to_numpy()
    for k in range(13, len(idx) - 1):
        c, n1 = idx[k], idx[k + 1]
        fwd = tri[:, n1] / tri[:, c] - 1
        sig = {"rev_1m": -(tri[:, c] / tri[:, idx[k - 1]] - 1),
               "mom_12_2": tri[:, idx[k - 1]] / tri[:, idx[k - 12]] - 1,
               "mom_12_1": tri[:, c] / tri[:, idx[k - 12]] - 1}
        for name, s in sig.items():
            ok = np.isfinite(s) & np.isfinite(fwd)
            if ok.sum() < 50:
                continue
            q = pd.qcut(s[ok], 10, labels=False, duplicates="drop")
            ls = fwd[ok][q == q.max()].mean() - fwd[ok][q == 0].mean()
            rows.append({"month": str(me.index[k + 1]), "signal": name, "ls_ret": float(ls), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def newey_west_t(x: np.ndarray, lags: int = 3) -> float | None:
    import statsmodels.api as sm
    x = x[np.isfinite(x)]
    if len(x) < 12:
        return None
    r = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues[0])
