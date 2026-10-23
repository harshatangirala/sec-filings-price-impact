"""Track A event study (SPEC.md §5.3, §10.1): CARs around day 0, abnormal volume and volatility response, by form /
8-K category / bucket, all-event and clean-event, pre-trend windows; errors clustered by firm and date.
Association only. INTRA day-0 returns are a mixed pre/post interval (labelled)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets.build import Estimator

WINDOWS = [(-1, 1), (0, 1), (0, 5), (2, 20), (2, 60), (-20, -2), (-5, -1), (3, 61)]   # (3, 61): one-session-lag drift


def wname(a, b) -> str:
    return f"car_{a}_{b}".replace("-", "m")


def window_cars(ev: pd.DataFrame, panel, spy: dict, factors: pd.DataFrame | None, cfg: dict,
                usable_cutoff_idx: int) -> pd.DataFrame:
    cal = panel.cal
    n = len(cal)
    w0, w1 = cfg["targets"]["est_window"]
    min_obs = int(cfg["targets"]["est_min_obs"])
    fac = rf = None
    if factors is not None and len(factors):
        fidx = cal.sessions.searchsorted(pd.to_datetime(factors["date"]).values)
        ok = (fidx < n) & (cal.sessions[np.clip(fidx, 0, n - 1)] == pd.to_datetime(factors["date"]).values)
        fac = np.full((n, 4), np.nan)
        rf = np.full(n, np.nan)
        fac[fidx[ok]] = factors.loc[ok, ["mkt_rf", "smb", "hml", "mom"]].to_numpy(float)
        rf[fidx[ok]] = factors.loc[ok, "rf"].to_numpy(float)
    rm = spy["ret"]
    out = []
    for tk, g in ev.groupby("ticker", sort=False):
        if not panel.has(tk):
            continue
        P = panel[tk]
        r = P["ret"]
        d0 = g["trackA_day0_idx"].to_numpy(int)
        est = Estimator(r, rm)
        cnt, al, be, sig = est.window(d0 + w0, d0 + w1)
        low = cnt < min_obs
        al[low], be[low], sig[low] = np.nan, np.nan, np.nan
        rec = {"event_id": g["event_id"].to_numpy(), "sigma": sig, "beta": be, "n_est": cnt}
        fin = np.isfinite(r) & np.isfinite(rm)
        cs = np.concatenate([[0.0], np.cumsum(np.where(fin, r - 0.0, 0.0))])
        csm = np.concatenate([[0.0], np.cumsum(np.where(fin, rm, 0.0))])
        bad = np.concatenate([[0], np.cumsum(~fin)])
        for a, b in WINDOWS:
            lo, hi = d0 + a, d0 + b
            okw = (lo >= 1) & (hi < n) & (hi <= usable_cutoff_idx)
            loc, hic = np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1)
            sr = cs[hic + 1] - cs[loc]
            sm_ = csm[hic + 1] - csm[loc]
            gaps = (bad[hic + 1] - bad[loc]) > 0
            car = sr - (al * (b - a + 1) + be * sm_)
            car[~okw | gaps] = np.nan
            rec[wname(a, b)] = car
            if fac is not None and (a, b) in ((-1, 1), (0, 1), (2, 60), (2, 20), (3, 61)):
                c4 = np.full(len(d0), np.nan)
                for i in range(len(d0)):
                    if not okw[i] or gaps[i] or d0[i] + w0 < 1:
                        continue
                    seg = slice(d0[i] + w0, d0[i] + w1 + 1)
                    y = r[seg] - rf[seg]
                    X = fac[seg]
                    m = np.isfinite(y) & np.isfinite(X).all(1)
                    if m.sum() < min_obs:
                        continue
                    A = np.column_stack([np.ones(m.sum()), X[m]])
                    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                    s2 = slice(lo[i], hi[i] + 1)
                    yy, XX = r[s2] - rf[s2], fac[s2]
                    if np.isfinite(yy).all() and np.isfinite(XX).all():
                        c4[i] = float(np.sum(yy - coef[0] - XX @ coef[1:]))
                rec[wname(a, b).replace("car_", "car4_")] = c4
        vol = P["volume"]
        lv = np.log(np.where(vol > 0, vol, np.nan))
        csv = np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(lv), lv, 0.0))])
        cnv = np.concatenate([[0], np.cumsum(np.isfinite(lv))])
        lo, hi = np.clip(d0 + w0, 0, n - 1), np.clip(d0 + w1, 0, n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            base_lv = (csv[hi + 1] - csv[lo]) / (cnv[hi + 1] - cnv[lo])
            e0, e1 = np.clip(d0, 0, n - 1), np.clip(d0 + 1, 0, n - 1)
            ev_lv = (csv[e1 + 1] - csv[e0]) / (cnv[e1 + 1] - cnv[e0])
        rec["abn_log_volume_0_1"] = ev_lv - base_lv
        with np.errstate(divide="ignore", invalid="ignore"):
            rec["vol_response_0_1"] = np.abs(rec[wname(0, 1)]) / (sig * np.sqrt(2))
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cluster_fit(y: np.ndarray, X: np.ndarray, firm, date):
    """OLS with two-way (firm, date) clustered errors; steps down to one-way and then to HC1 when a dimension has fewer
    than two clusters. Returns (result, method)."""
    import statsmodels.api as sm
    gf, gd = pd.factorize(pd.Series(firm).astype(str))[0], pd.factorize(pd.Series(date).astype(str))[0]
    nf, nd = len(np.unique(gf)), len(np.unique(gd))
    tries = []
    if nf >= 2 and nd >= 2:
        tries.append(("two-way firm x date", {"cov_type": "cluster", "cov_kwds": {"groups": np.column_stack([gf, gd])}}))
    if nf >= 2:
        tries.append(("firm", {"cov_type": "cluster", "cov_kwds": {"groups": gf}}))
    if nd >= 2:
        tries.append(("date", {"cov_type": "cluster", "cov_kwds": {"groups": gd}}))
    tries.append(("HC1 (fewer than 2 clusters)", {"cov_type": "HC1"}))
    for name, kw in tries:
        try:
            r = sm.OLS(y, X).fit(**kw)
            if np.all(np.isfinite(r.bse)):
                return r, name
        except Exception:
            continue
    return sm.OLS(y, X).fit(), "iid"


def clustered_mean(y: np.ndarray, firm: np.ndarray, date: np.ndarray) -> dict:
    """Mean with two-way (firm, date) clustered standard error (intercept-only OLS)."""
    from scipy.stats import norm
    ok = np.isfinite(y)
    y, firm, date = y[ok], np.asarray(firm)[ok], np.asarray(date)[ok]
    n = len(y)
    if n < 3:
        return {"mean": float(np.mean(y)) if n else None, "se": None, "t": None, "p": None, "n": n, "se_method": None}
    res, how = cluster_fit(y, np.ones((n, 1)), firm, date)
    se = float(res.bse[0])
    t = float(res.params[0] / se) if se > 0 else None
    return {"mean": float(res.params[0]), "se": se, "t": t, "p": float(2 * (1 - norm.cdf(abs(t)))) if t is not None else None,
            "n": int(n), "se_method": how}


def group_table(df: pd.DataFrame, group_col: str, cols: list[str], clean_col: str) -> pd.DataFrame:
    rows = []
    for gname, g in df.groupby(group_col):
        for es, gg in (("all", g), ("clean", g[~g[clean_col]])):
            for c in cols:
                if c not in gg:
                    continue
                st = clustered_mean(gg[c].to_numpy(float), gg["cik"].to_numpy(), gg["trackA_day0"].astype(str).to_numpy())
                rows.append({"group_by": group_col, "group": gname, "event_set": es, "measure": c, **st})
    return pd.DataFrame(rows)


def car_paths(ev: pd.DataFrame, panel, spy, cfg, rel=range(-20, 61), group_col: str = "event_type") -> pd.DataFrame:
    """Average cumulative market-model abnormal return path from day -20 to +60 by group (F11)."""
    cal = panel.cal
    n = len(cal)
    w0, w1 = cfg["targets"]["est_window"]
    rows = []
    rel = list(rel)
    for tk, g in ev.groupby("ticker", sort=False):
        if not panel.has(tk):
            continue
        r = panel[tk]["ret"]
        est = Estimator(r, spy["ret"])
        d0 = g["trackA_day0_idx"].to_numpy(int)
        cnt, al, be, _ = est.window(d0 + w0, d0 + w1)
        for i, (eid, grp) in enumerate(zip(g.event_id, g[group_col])):
            if cnt[i] < int(cfg["targets"]["est_min_obs"]) or d0[i] + rel[0] < 1 or d0[i] + rel[-1] >= n:
                continue
            idx = d0[i] + np.array(rel)
            ar = r[idx] - (al[i] + be[i] * spy["ret"][idx])
            if not np.isfinite(ar).all():
                continue
            rows.append(pd.DataFrame({"group": grp, "rel_day": rel, "ar": ar}))
    if not rows:
        return pd.DataFrame(columns=["group", "rel_day", "caar", "n"])
    d = pd.concat(rows)
    m = d.groupby(["group", "rel_day"])["ar"].agg(["mean", "count"]).reset_index()
    m["caar"] = m.groupby("group")["mean"].cumsum()
    return m.rename(columns={"count": "n"})[["group", "rel_day", "caar", "n"]]
