"""S06 accounting: XBRL facts with public-availability times, as-first-reported quarterly EPS, release EPS and
cluster-aware SUE (SPEC.md §5.6; addendum P1.4).

SUE = (EPS_q - EPS_{q-4}) / sd(seasonal differences over the prior quarters), using up to `sue_max_diffs`
(min `sue_min_diffs`) differences, each value as originally reported (earliest public accession for that
period). Every history value must satisfy public_available_time < decision_time of the event. Earnings-8-K
events take EPS_q from their own release exhibit; 10-Q/10-K events from their own XBRL. A later accession of the
same cluster can never feed an earnings-8-K event (assertion + planted-leak test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS_KIND = {"EarningsPerShareDiluted": "diluted", "EarningsPerShareBasicAndDiluted": "diluted",
            "EarningsPerShareBasic": "basic"}


def _qkey(period_end: pd.Series) -> pd.Series:
    """Calendar-quarter key of a fiscal period end, shifted 15 days back so 52/53-week ends (Jan 1-3, Oct 1-2)
    land in the quarter they close."""
    d = pd.to_datetime(period_end) - pd.Timedelta(days=15)
    return (d.dt.year * 4 + (d.dt.quarter - 1)).astype("Int64")


def attach_public_time(facts: pd.DataFrame, filing_times: pd.DataFrame, assumed_lag_min: float,
                       assumed_time: str = "17:30") -> pd.DataFrame:
    """filing_times: accession, public_available_time (tz ET). Facts from accessions outside the manifest get
    filed-date 17:30 ET + lag (quality B)."""
    f = facts.merge(filing_times[["accession", "public_available_time"]], on="accession", how="left")
    miss = f["public_available_time"].isna()
    if miss.any():
        hh, mm = assumed_time.split(":")
        fd = pd.to_datetime(f.loc[miss, "filed"]) + pd.Timedelta(hours=int(hh), minutes=int(mm) + float(assumed_lag_min))
        f.loc[miss, "public_available_time"] = fd.dt.tz_localize("America/New_York")
        f["pub_time_source"] = np.where(miss, "filed_date_assumed", "manifest")
    else:
        f["pub_time_source"] = "manifest"
    f["public_available_time"] = pd.to_datetime(f["public_available_time"], utc=True).dt.tz_convert("America/New_York")
    return f


def quarterly_eps(facts: pd.DataFrame) -> pd.DataFrame:
    """Direct quarterly EPS facts plus Q4 = FY - 9M YTD (both first-reported; availability = the later source)."""
    e = facts[facts["tag"].isin(EPS_KIND) & (facts["unit"] == "USD/shares")].copy()
    if e.empty:
        return pd.DataFrame(columns=["cik", "period_end", "qkey", "kind", "value", "source_accession",
                                     "public_available_time", "derivation", "form"])
    e["kind"] = e["tag"].map(EPS_KIND)
    e["start"] = pd.to_datetime(e["period_start"])
    e["end"] = pd.to_datetime(e["period_end"])
    e["days"] = (e["end"] - e["start"]).dt.days
    e = e.sort_values("public_available_time")
    first = lambda df, keys: df.drop_duplicates(keys, keep="first")  # noqa: E731
    q = first(e[(e.days >= 70) & (e.days <= 110)], ["cik", "kind", "end"])
    out = [pd.DataFrame({"cik": q.cik, "period_end": q.end, "kind": q.kind, "value": q.value,
                         "source_accession": q.accession, "public_available_time": q.public_available_time,
                         "derivation": "direct", "form": q.form})]
    fy = first(e[(e.days >= 340) & (e.days <= 380)], ["cik", "kind", "end"])
    m9 = first(e[(e.days >= 250) & (e.days <= 290)], ["cik", "kind", "start", "end"])
    if len(fy) and len(m9):
        j = fy.merge(m9, on=["cik", "kind", "start"], suffixes=("_fy", "_9m"))
        j = j[(j.end_fy - j.end_9m).dt.days.between(70, 110)]
        pub = np.maximum(j.public_available_time_fy.values, j.public_available_time_9m.values)
        out.append(pd.DataFrame({"cik": j.cik, "period_end": j.end_fy, "kind": j.kind, "value": j.value_fy - j.value_9m,
                                 "source_accession": j.accession_fy, "public_available_time": pub,
                                 "derivation": "fy_minus_9m", "form": j.form_fy}))
    r = pd.concat(out, ignore_index=True)
    r["public_available_time"] = pd.to_datetime(r["public_available_time"], utc=True).dt.tz_convert("America/New_York")
    r = r.sort_values(["cik", "kind", "period_end", "public_available_time"])
    r = r.drop_duplicates(["cik", "kind", "period_end"], keep="first")      # as first reported
    r["qkey"] = _qkey(r["period_end"])
    return r.reset_index(drop=True)


def sue_for(eps_q: float, qk: int, hist: pd.DataFrame, min_diffs: int, max_diffs: int) -> dict:
    """hist: qkey, value (already filtered to public < decision, one value per qkey)."""
    s = dict(zip(hist["qkey"].astype(int), hist["value"].astype(float)))
    out = {"sue": np.nan, "eps_q": eps_q, "eps_q4": s.get(qk - 4, np.nan), "n_prior_diffs": 0, "sue_reason": None}
    if eps_q is None or not np.isfinite(eps_q):
        out["sue_reason"] = "no_eps_q"
        return out
    if not np.isfinite(out["eps_q4"]):
        out["sue_reason"] = "no_eps_q_minus_4"
        return out
    diffs = []
    for j in range(1, max_diffs + 1):
        a, b = s.get(qk - j), s.get(qk - j - 4)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    out["n_prior_diffs"] = len(diffs)
    if len(diffs) < min_diffs:
        out["sue_reason"] = f"only_{len(diffs)}_prior_diffs"
        return out
    sd = float(np.std(diffs, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        out["sue_reason"] = "zero_dispersion"
        return out
    out["sue"] = (eps_q - out["eps_q4"]) / sd
    out["sue_reason"] = "ok"
    return out


def history_before(eps_hist: pd.DataFrame, cik: int, decision_time, exclude_accessions: set | None = None) -> pd.DataFrame:
    h = eps_hist[(eps_hist.cik == cik) & (eps_hist.public_available_time < decision_time)]
    if exclude_accessions:
        h = h[~h.source_accession.isin(exclude_accessions)]
    h = h.sort_values("public_available_time").drop_duplicates("qkey", keep="first")
    return h


def compute_sue_table(events: pd.DataFrame, xbrl_eps: pd.DataFrame, rel_eps: pd.DataFrame, min_diffs: int,
                      max_diffs: int, cluster_later: dict | None = None) -> pd.DataFrame:
    """events: accession, cik, kind ('release' | 'periodic'), decision_time, period_end (release), qkey.
    xbrl_eps / rel_eps: cik, qkey, value, source_accession, public_available_time (diluted preferred).
    cluster_later: {accession: set(later same-cluster accessions)} -> excluded from history (rule 7)."""
    hist_all = pd.concat([xbrl_eps.assign(src="xbrl"), rel_eps.assign(src="release")], ignore_index=True)
    hist_all = hist_all.sort_values("public_available_time")
    rows, sources = [], []
    for ev in events.itertuples(index=False):
        excl = (cluster_later or {}).get(ev.accession, set())
        h = history_before(hist_all, ev.cik, ev.decision_time, excl)
        if ev.kind == "release":
            eps_q, qk, src = ev.eps_q, ev.qkey, "release_exhibit"
        else:
            own = xbrl_eps[(xbrl_eps.cik == ev.cik) & (xbrl_eps.source_accession == ev.accession)]
            if len(own):
                own = own.sort_values("period_end").iloc[-1]
                eps_q, qk, src = float(own.value), int(own.qkey), "own_xbrl"
            else:
                eps_q, qk, src = np.nan, ev.qkey, "none"
        h = h[h.qkey != qk] if qk is not None and not pd.isna(qk) else h
        qki = int(qk) if qk is not None and not pd.isna(qk) else -10**6
        r = sue_for(eps_q, qki, h, min_diffs, max_diffs)
        used = {qki - 4} | {q for j in range(1, max_diffs + 1) for q in (qki - j, qki - j - 4)}
        hu = h[h.qkey.isin(list(used))]
        for s in hu.itertuples(index=False):
            sources.append((ev.accession, s.source_accession, s.public_available_time, s.src))
        r.update({"accession": ev.accession, "cik": ev.cik, "eps_q_source": src, "qkey": qk,
                  "hist_max_public_time": hu.public_available_time.max() if len(hu) else pd.NaT,
                  "hist_sources": ",".join(sorted(set(hu.src))) if len(hu) else ""})
        rows.append(r)
    src_df = pd.DataFrame(sources, columns=["accession", "source_accession", "source_public_time", "source_kind"])
    return pd.DataFrame(rows), src_df


def pit_facts(events: pd.DataFrame, facts: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """Latest value per tag with public time strictly before decision_time (merge_asof by cik)."""
    ev = events[["event_id", "cik", "decision_time"]].copy()
    ev["decision_time"] = pd.to_datetime(ev["decision_time"], utc=True)
    ev = ev.sort_values("decision_time")
    out = ev[["event_id"]].copy()
    pts = pd.DataFrame(index=ev.index)
    f = facts.copy()
    f["public_available_time"] = pd.to_datetime(f["public_available_time"], utc=True)
    f["days"] = (pd.to_datetime(f["period_end"]) - pd.to_datetime(f["period_start"])).dt.days
    for tag, kind in tags:
        sub = f[f.tag == tag]
        if kind == "fy":
            sub = sub[sub.days.between(340, 380)]
        elif kind == "q":
            sub = sub[sub.days.between(70, 110)]
        sub = sub.sort_values("public_available_time")[["cik", "public_available_time", "value", "period_end"]]
        if sub.empty:
            out[f"{tag}_{kind}"] = np.nan
            continue
        sub = sub.rename(columns={"public_available_time": "pt"})
        m = pd.merge_asof(ev, sub, left_on="decision_time", right_on="pt", by="cik", allow_exact_matches=False,
                          direction="backward")
        out[f"{tag}_{kind}"] = m["value"].values
        pts[f"{tag}_{kind}"] = pd.to_datetime(m["pt"].values, utc=True)
    out["_avail_accounting"] = pts.max(axis=1).values if pts.shape[1] else pd.NaT
    return out
