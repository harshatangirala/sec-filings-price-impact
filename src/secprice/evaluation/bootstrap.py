"""Block bootstrap by calendar month (SPEC.md §9.2 Uncertainty). Paired: both models resample the same months."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import confusion, f1_from_cm


def month_ids(dates) -> np.ndarray:
    m = pd.to_datetime(pd.Series(dates)).dt.to_period("M").astype(str).to_numpy()
    _, inv = np.unique(m, return_inverse=True)
    return inv


def _per_month_cm(y, yhat, mid, n_months):
    cms = np.zeros((n_months, 3, 3), dtype=np.int64)
    np.add.at(cms, (mid, y.astype(int), yhat.astype(int)), 1)
    return cms


def macro_f1_ci(y, yhat, dates, n_boot=1000, seed=0, alpha=0.05) -> dict:
    mid = month_ids(dates)
    nm = mid.max() + 1 if len(mid) else 0
    cms = _per_month_cm(np.asarray(y), np.asarray(yhat), mid, nm)
    point = f1_from_cm(cms.sum(0))["macro_f1"]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = rng.integers(0, nm, nm)
        v = f1_from_cm(cms[s].sum(0))["macro_f1"]
        if v is not None:
            vals.append(v)
    lo, hi = (np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan))
    return {"value": point, "ci_low": float(lo), "ci_high": float(hi), "n_months": int(nm)}


def paired_macro_f1_diff(y, yhat_a, yhat_b, dates, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """Lift of model A over model B on identical rows; CI by month-block bootstrap; two-sided p from the
    bootstrap distribution (share of resamples on the other side of zero, doubled)."""
    y = np.asarray(y)
    mid = month_ids(dates)
    nm = mid.max() + 1 if len(mid) else 0
    ca = _per_month_cm(y, np.asarray(yhat_a), mid, nm)
    cb = _per_month_cm(y, np.asarray(yhat_b), mid, nm)
    point = f1_from_cm(ca.sum(0))["macro_f1"] - f1_from_cm(cb.sum(0))["macro_f1"]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        s = rng.integers(0, nm, nm)
        fa, fb = f1_from_cm(ca[s].sum(0))["macro_f1"], f1_from_cm(cb[s].sum(0))["macro_f1"]
        if fa is not None and fb is not None:
            d.append(fa - fb)
    d = np.array(d)
    lo, hi = (np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d) else (np.nan, np.nan))
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan
    return {"value": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_boot": p, "n_months": int(nm)}


def mean_ci(values, dates, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """CI of a mean (e.g. CAR) with month-block resampling."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v = v[ok]
    mid = month_ids(pd.Series(dates)[ok])
    if len(v) == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n_months": 0}
    nm = mid.max() + 1
    sums = np.bincount(mid, weights=v, minlength=nm)
    cnts = np.bincount(mid, minlength=nm)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, nm, nm)
        c = cnts[s].sum()
        if c:
            bs.append(sums[s].sum() / c)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return {"value": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_months": int(nm)}
