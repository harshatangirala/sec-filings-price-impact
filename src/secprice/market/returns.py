"""Interval returns with explicit dividend handling and interval-matched benchmark legs (addendum P0.2).

Rules
- close->close total return over sessions (b, x]: TRI[x] / TRI[b] - 1 (dividends with ex-date in (b, x]).
- open->close executable return from the entry open of session e to the close of session x >= e:
  (C[e] / O[e]) x TRI[x] / TRI[e] - 1. A dividend with ex-date e is excluded (an open buyer is not entitled);
  dividends with ex-date in (e, x] are included. All prices are on one split basis, so a split inside the
  holding interval has no effect.
- The benchmark (SPY) leg is computed on exactly the same interval with the same formula. A close-to-close
  benchmark is never subtracted from an open-to-close stock return.
"""
from __future__ import annotations

import numpy as np


def cc_return(tri: np.ndarray, b: int, x: int) -> float:
    if b < 0 or x >= len(tri) or x <= b:
        return np.nan
    return float(tri[x] / tri[b] - 1.0)


def oc_return(open_: np.ndarray, close: np.ndarray, tri: np.ndarray, e: int, x: int) -> float:
    if e < 0 or x >= len(tri) or x < e:
        return np.nan
    if not (np.isfinite(open_[e]) and open_[e] > 0 and np.isfinite(close[e]) and np.isfinite(tri[e]) and np.isfinite(tri[x])):
        return np.nan
    return float((close[e] / open_[e]) * (tri[x] / tri[e]) - 1.0)


def v_cc_return(tri: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(tri)
    ok = (b >= 0) & (x < n) & (x > b)
    out = np.full(len(b), np.nan)
    out[ok] = tri[x[ok]] / tri[b[ok]] - 1.0
    return out


def v_oc_return(open_: np.ndarray, close: np.ndarray, tri: np.ndarray, e: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(tri)
    ok = (e >= 0) & (x < n) & (x >= e)
    out = np.full(len(e), np.nan)
    ee, xx = e[ok], x[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = (close[ee] / open_[ee]) * (tri[xx] / tri[ee]) - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def daily_returns(tri: np.ndarray) -> np.ndarray:
    r = np.full(len(tri), np.nan)
    r[1:] = tri[1:] / tri[:-1] - 1.0
    return r
