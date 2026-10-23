"""Price features at decision_time: only sessions whose close is at or before decision_time are used
(availability = that close). SPEC.md §9.1 L1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets.build import Estimator


def _ratio(tri, a, b):
    out = np.full(len(a), np.nan)
    ok = (b >= 0) & (a >= 0) & (a < len(tri)) & (b < len(tri))
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = tri[a[ok]] / tri[b[ok]] - 1
    return out


def _roll(fn, arr, ends, win, min_n):
    out = np.full(len(ends), np.nan)
    for i, c in enumerate(ends):
        if c - win + 1 < 0 or c >= len(arr):
            continue
        seg = arr[c - win + 1:c + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= min_n:
            out[i] = fn(seg)
    return out


def price_features(ev: pd.DataFrame, panel, spy: dict) -> pd.DataFrame:
    cal = panel.cal
    dec_ns = pd.DatetimeIndex(pd.to_datetime(ev["decision_time"])).tz_convert("UTC").asi8
    cidx = cal.v_last_close_at_or_before(dec_ns)
    ev = ev.assign(_cidx=cidx)
    rows = []
    s_tri, s_r = spy["total_return_index"], spy["ret"]
    for tk, g in ev.groupby("ticker", sort=False):
        c = g["_cidx"].to_numpy(int)
        f = pd.DataFrame({"event_id": g["event_id"].to_numpy()})
        if not panel.has(tk):
            rows.append(f)
            continue
        P = panel[tk]
        tri, r, close, vol = P["total_return_index"], P["ret"], P["close_split_adj"], P["volume"]
        for w in (1, 5, 20, 60, 250):
            f[f"px_ret_{w}"] = _ratio(tri, c, c - w)
        f["px_rev_1m"] = f["px_ret_20"]
        f["px_mom_12_2"] = _ratio(tri, c - 21, c - 252)
        f["px_vol_20"] = _roll(np.std, r, c, 20, 15)
        f["px_vol_60"] = _roll(np.std, r, c, 60, 40)
        dv = close * vol
        f["px_logdv_20"] = np.log(np.maximum(_roll(np.mean, dv, c, 20, 10), 1.0))
        f["px_volratio_5_60"] = np.log(np.maximum(_roll(np.mean, vol, c, 5, 3), 1) / np.maximum(_roll(np.mean, vol, c, 60, 30), 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            am = np.abs(r) / dv * 1e6
        f["px_amihud_20"] = np.log1p(_roll(np.mean, am, c, 20, 10))
        hi = _roll(np.max, close, c, 252, 100)
        f["px_dist_52w_high"] = close[np.clip(c, 0, len(close) - 1)] / hi - 1
        est = Estimator(r, s_r)
        cnt, a, b, sig = est.window(c - 250, c - 30)
        f["px_beta"] = np.where(cnt >= 120, b, np.nan)
        f["px_idio_vol"] = np.where(cnt >= 120, sig, np.nan)
        f["mkt_ret_20"] = _ratio(s_tri, c, c - 20)
        f["mkt_ret_60"] = _ratio(s_tri, c, c - 60)
        f["mkt_vol_20"] = _roll(np.std, s_r, c, 20, 15)
        f["mkt_dd_252"] = s_tri[np.clip(c, 0, len(s_tri) - 1)] / _roll(np.max, s_tri, c, 252, 100) - 1
        f["_avail_price"] = [cal.close_time(i) if 0 <= i < len(cal) else pd.NaT for i in c]
        rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    out["_avail_price"] = pd.to_datetime(out["_avail_price"], utc=True)
    return out
