"""S08 targets and labels (SPEC.md §5.4-5.5; addendum P0.1, P0.2).

T_info  (Track A information response; complementary Track B target): base = last regular close <= acceptance;
        exit = close of the h-th session whose close is after public time; INTRA excluded; close->close total
        return; abnormal = CAR of daily market-model residuals (alpha, beta from sessions -250..-30 before base).
T_trade (primary Track B target): entry = open of the first session whose open is at or after decision_time
        (all buckets); exit = close of session entry + h - 1; executable open->close total return; abnormal =
        R_stock - beta x R_SPY on the identical interval (alpha omitted on partial-day intervals; decision log).
        No close-to-close factor return is ever subtracted from an open-to-close return: Carhart is null here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..market.returns import v_cc_return, v_oc_return

NULL_CARHART_TTRADE = "carhart_not_interval_matched_for_open_entry"


class Estimator:
    """O(1) market-model OLS over any session window via prefix sums of valid (stock, market) return pairs."""

    def __init__(self, r: np.ndarray, rm: np.ndarray):
        ok = np.isfinite(r) & np.isfinite(rm)
        x, y = np.where(ok, rm, 0.0), np.where(ok, r, 0.0)
        self.c = [np.concatenate([[0.0], np.cumsum(a)]) for a in (ok.astype(float), x, y, x * y, x * x, y * y)]

    def window(self, lo: np.ndarray, hi: np.ndarray):
        """Sums over sessions [lo, hi] inclusive (clipped)."""
        n = len(self.c[0]) - 1
        lo_c, hi_c = np.clip(lo, 0, n), np.clip(hi + 1, 0, n)
        s = [c[hi_c] - c[lo_c] for c in self.c]
        cnt, sx, sy, sxy, sxx, syy = s
        with np.errstate(divide="ignore", invalid="ignore"):
            den = cnt * sxx - sx * sx
            beta = (cnt * sxy - sx * sy) / den
            alpha = (sy - beta * sx) / cnt
            ssr = syy - 2 * alpha * sy - 2 * beta * sxy + cnt * alpha ** 2 + 2 * alpha * beta * sx + beta ** 2 * sxx
            sigma = np.sqrt(np.maximum(ssr, 0) / (cnt - 2))
        bad = (cnt < 3) | ~np.isfinite(beta)
        beta[bad], alpha[bad], sigma[bad] = np.nan, np.nan, np.nan
        return cnt, alpha, beta, sigma


def _carhart_car(r, f, rf, base, x, lo, hi, min_obs):
    """CAR of Carhart residuals over (base, x] with loadings estimated on [lo, hi]; f = [mkt, smb, hml, mom]."""
    out = np.full(len(base), np.nan)
    for i in range(len(base)):
        if lo[i] < 1 or x[i] >= len(r) or base[i] < 0:
            continue
        yy = r[lo[i]:hi[i] + 1] - rf[lo[i]:hi[i] + 1]
        XX = f[lo[i]:hi[i] + 1]
        ok = np.isfinite(yy) & np.isfinite(XX).all(1)
        if ok.sum() < min_obs:
            continue
        A = np.column_stack([np.ones(ok.sum()), XX[ok]])
        coef, *_ = np.linalg.lstsq(A, yy[ok], rcond=None)
        seg = slice(base[i] + 1, x[i] + 1)
        ex = r[seg] - rf[seg]
        Xs = f[seg]
        if not (np.isfinite(ex).all() and np.isfinite(Xs).all()):
            continue
        out[i] = float(np.sum(ex - coef[0] - Xs @ coef[1:]))
    return out


def label3(ar: np.ndarray, sigma: np.ndarray, h: int, k: float) -> np.ndarray:
    th = k * sigma * np.sqrt(h)
    lab = np.full(len(ar), np.nan)
    ok = np.isfinite(ar) & np.isfinite(th)
    lab[ok] = 1.0                                   # Sideways
    lab[ok & (ar > th)] = 2.0                       # Up
    lab[ok & (ar < -th)] = 0.0                      # Down
    return lab


def _vol_ratio(r: np.ndarray, d1: np.ndarray, post: tuple, pre: tuple) -> np.ndarray:
    out = np.full(len(d1), np.nan)
    n = len(r)
    for i, s in enumerate(d1):
        a0, a1 = s + post[0] - 1, s + post[1] - 1
        b0, b1 = s - 1 + pre[0], s - 1 + pre[1]
        if b0 < 0 or a1 >= n:
            continue
        pa, pb = r[a0:a1 + 1], r[b0:b1 + 1]
        pa, pb = pa[np.isfinite(pa)], pb[np.isfinite(pb)]
        if len(pa) >= 10 and len(pb) >= 20:
            sa, sb = np.std(pa, ddof=1), np.std(pb, ddof=1)
            if sa > 0 and sb > 0:
                out[i] = np.log(sa / sb)
    return out


def uniqueness_weights(start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Average uniqueness of each event window within one firm (label-uniqueness weights)."""
    w = np.full(len(start), np.nan)
    ok = (start >= 0) & (stop >= start)
    if not ok.any():
        return w
    lo, hi = int(start[ok].min()), int(stop[ok].max())
    conc = np.zeros(hi - lo + 2)
    for a, b in zip(start[ok], stop[ok]):
        conc[a - lo] += 1
        conc[b - lo + 1] -= 1
    conc = np.cumsum(conc)[:-1]
    for i in np.where(ok)[0]:
        seg = conc[start[i] - lo: stop[i] - lo + 1]
        w[i] = float(np.mean(1.0 / seg)) if len(seg) else np.nan
    return w


def build_targets(ev: pd.DataFrame, panel, spy: dict, factors: pd.DataFrame | None, cfg: dict,
                  usable_cutoff_idx: int) -> pd.DataFrame:
    """ev needs: event_id, ticker, tinfo_base_idx, trackA_day0_idx, ttrade_entry_idx, tinfo_eligible,
    acceptance_time, public_available_time; plus sensitivity entry indexes (computed here from cal)."""
    hs = cfg["targets"]["horizons"]
    k = float(cfg["labels"]["k"])
    k_sens = [float(x) for x in cfg["labels"]["k_sensitivity"]]
    w0, w1 = cfg["targets"]["est_window"]
    min_obs = int(cfg["targets"]["est_min_obs"])
    post, pre = tuple(cfg["targets"]["vol_post"]), tuple(cfg["targets"]["vol_pre"])
    cal = panel.cal
    n = len(cal)
    rm = spy["ret"]
    fac = None
    if factors is not None and len(factors):
        fidx = cal.sessions.searchsorted(pd.to_datetime(factors["date"]).values)
        okf = (fidx < n) & (cal.sessions[np.clip(fidx, 0, n - 1)] == pd.to_datetime(factors["date"]).values)
        fac = np.full((n, 4), np.nan)
        rf = np.full(n, np.nan)
        fac[fidx[okf]] = factors.loc[okf, ["mkt_rf", "smb", "hml", "mom"]].to_numpy(float)
        rf[fidx[okf]] = factors.loc[okf, "rf"].to_numpy(float)
    out = []
    t = cfg["timing"]
    sens_h = [h for h in cfg["targets"].get("sensitivity_horizons", [1, 5]) if h in hs]
    for tk, g in ev.groupby("ticker", sort=False):
        if not panel.has(tk):
            continue
        P = panel[tk]
        est = Estimator(P["ret"], rm)
        defs = []
        base = g["tinfo_base_idx"].to_numpy(int)
        pub_first = g["trackA_day0_idx"].to_numpy(int)
        entry = g["ttrade_entry_idx"].to_numpy(int)
        elig = g["tinfo_eligible"].to_numpy(bool)
        for h in hs:
            defs.append(("T_info", h, base, pub_first + h - 1, elig, "cc"))
            defs.append(("T_trade", h, entry, entry + h - 1, np.ones(len(g), bool), "oc"))
        for h in sens_h:
            defs.append(("T_trade_lag1", h, entry + 1, entry + h, np.ones(len(g), bool), "oc"))
            for lat in t.get("decision_latency_sensitivity", []):
                dec = (pd.to_datetime(g["public_available_time"]) + pd.Timedelta(minutes=float(lat)))
                e2 = cal.v_first_open_at_or_after(pd.DatetimeIndex(dec).tz_convert("UTC").asi8)
                defs.append((f"T_trade@lat{lat}", h, e2, e2 + h - 1, np.ones(len(g), bool), "oc"))
            for lag in t.get("public_lag_sensitivity", []):
                pub = pd.to_datetime(g["acceptance_time"]) + pd.Timedelta(minutes=float(lag))
                d0 = cal.v_first_close_after(pd.DatetimeIndex(pub).tz_convert("UTC").asi8)
                defs.append((f"T_info@pub{lag}", h, base, d0 + h - 1, elig, "cc"))
        for name, h, b, x, eligible, kind in defs:
            b = np.asarray(b, dtype=int)
            x = np.asarray(x, dtype=int)
            anchor = b if kind == "cc" else b          # estimation window relative to the base / entry session
            cnt, alpha, beta, sigma = est.window(anchor + w0, anchor + w1)
            lowobs = cnt < min_obs
            alpha[lowobs], beta[lowobs], sigma[lowobs] = np.nan, np.nan, np.nan
            valid_idx = (b >= 0) & (x < n) & (x >= b)
            if kind == "cc":
                raw = v_cc_return(P["total_return_index"], b, x)
                bench = v_cc_return(spy["total_return_index"], b, x)
                # CAR of daily residuals over (b, x]
                fin_r, fin_m = np.isfinite(P["ret"]), np.isfinite(rm)
                cs_r = np.concatenate([[0.0], np.cumsum(np.where(fin_r, P["ret"], 0.0))])
                cs_m = np.concatenate([[0.0], np.cumsum(np.where(fin_m, rm, 0.0))])
                cs_bad = np.concatenate([[0], np.cumsum(~(fin_r & fin_m))])
                bb, xx = np.clip(b, 0, n - 1), np.clip(x, 0, n - 1)
                sum_r = cs_r[xx + 1] - cs_r[bb + 1]
                sum_m = cs_m[xx + 1] - cs_m[bb + 1]
                gaps = (cs_bad[xx + 1] - cs_bad[bb + 1]) > 0     # any missing day in (b, x] -> CAR undefined
                ar = sum_r - (alpha * (x - b) + beta * sum_m)
                ar[gaps] = np.nan
                ar_c = _carhart_car(P["ret"], fac, rf, b, x, anchor + w0, anchor + w1, min_obs) if fac is not None \
                    else np.full(len(b), np.nan)
                d1 = b + 1
                car_reason = None
            else:
                raw = v_oc_return(P["open"], P["close_split_adj"], P["total_return_index"], b, x)
                bench = v_oc_return(spy["open"], spy["close_split_adj"], spy["total_return_index"], b, x)
                ar = raw - beta * bench
                ar_c = np.full(len(b), np.nan)
                d1 = b
                car_reason = NULL_CARHART_TTRADE
            ar[~valid_idx] = np.nan
            raw[~valid_idx] = np.nan
            censored = (x > usable_cutoff_idx) | (x >= n)
            ok = eligible & ~censored
            for arr in (raw, ar, ar_c):
                arr[~ok] = np.nan
            vr = _vol_ratio(P["ret"], d1, post, pre)
            vr[~ok] = np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                abs_std = np.abs(ar) / (sigma * np.sqrt(h))
            rec = pd.DataFrame({
                "event_id": g["event_id"].to_numpy(), "ticker": tk, "target_def": name, "horizon": h,
                "start_idx": b, "exit_idx": x, "raw_ret": raw, "bench_ret": bench, "ar_mm": ar, "ar_carhart": ar_c,
                "alpha": alpha, "beta": beta, "sigma": sigma, "n_est": cnt, "label3": label3(ar, sigma, h, k),
                **{f"label3_k{str(kk).replace('.', '')}": label3(ar, sigma, h, kk) for kk in k_sens},
                "vol_ratio": vr, "abs_std": abs_std, "eligible": eligible, "censored": censored,
                "carhart_null_reason": car_reason,
            })
            rec["null_reason"] = np.select(
                [~eligible, censored, ~valid_idx, lowobs, ~np.isfinite(raw)],
                ["not_eligible_intra_or_C", "censored_after_usable_cutoff", "index_out_of_range",
                 f"est_obs_lt_{min_obs}", "missing_price"], default=None)
            rec["weight_uniq"] = uniqueness_weights(np.where(ok, b, -1), np.where(ok, x, -1))
            out.append(rec)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(res):
        res["start_date"] = [cal.date(i) if 0 <= i < n else pd.NaT for i in res["start_idx"]]
        res["exit_date"] = [cal.date(i) if 0 <= i < n else pd.NaT for i in res["exit_idx"]]
    return res
