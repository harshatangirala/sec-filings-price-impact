"""S07 events and clocks (SPEC.md §5.1-5.3, §5.7, §16 events.parquet).

Periodic events: 10-K (incl. 405/KT) and 10-Q originals. 8-K events: same-firm, same-ET-day 8-Ks merged into one
multi-label event. Amendments are kept in the manifest for the audit only. Event construction never reads price
fields (hard rule 9 / leakage item 21).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .timestamps import derive_times, to_utc_ns

PERIODIC = {"10-K", "10-K405", "10-KT", "10-Q"}
EIGHTK = {"8-K"}


def event_id(cik: int, earliest_accession: str) -> str:
    return hashlib.sha1(f"{int(cik)}|{earliest_accession}".encode()).hexdigest()[:16]


def build_events(manifest: pd.DataFrame, cal, cfg: dict, tax, release_info: pd.DataFrame,
                 sections_avail: pd.DataFrame) -> pd.DataFrame:
    """manifest columns: accession, cik, form, filing_date, acceptance_time (ET), ts_quality, ts_source,
    report_date, items_list (list[str]), item_titles (list), in_study (bool), is_amendment.
    release_info: accession, is_results_release (bool), release_period_end.
    sections_avail: accession, section ids present (list)."""
    t = cfg["timing"]
    m = manifest[manifest["in_study"] & ~manifest["is_amendment"] & manifest["form"].isin(PERIODIC | EIGHTK)].copy()
    m["acceptance_time"] = pd.to_datetime(m["acceptance_time"])
    m = m[m["acceptance_time"].notna()]
    m["acc_date_et"] = m["acceptance_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    rel = release_info.set_index("accession") if len(release_info) else pd.DataFrame()
    sav = sections_avail.set_index("accession")["sections"].to_dict() if len(sections_avail) else {}

    rows = []
    per = m[m.form.isin(PERIODIC)]
    for r in per.itertuples(index=False):
        rows.append({"cik": r.cik, "accessions": [r.accession], "forms": [r.form], "acceptance_time": r.acceptance_time,
                     "ts_quality": r.ts_quality, "ts_source": r.ts_source,
                     "event_type": "periodic_10K" if r.form.startswith("10-K") else "periodic_10Q",
                     "categories": [], "items": [], "report_date": r.report_date, "filing_date": r.filing_date})
    ek = m[m.form.isin(EIGHTK)].sort_values(["cik", "acc_date_et", "acceptance_time", "accession"])
    for (cik, d), g in ek.groupby(["cik", "acc_date_et"], sort=False):
        items, cats = [], set()
        for r in g.itertuples(index=False):
            its = list(r.items_list or [])
            items += its
            cats |= set(tax.categories(its, r.filing_date, list(r.item_titles or [])))
        first = g.iloc[0]
        q = "A" if (g.ts_quality == "A").all() else ("B" if (g.ts_quality != "C").all() else "C")
        rows.append({"cik": cik, "accessions": g.accession.tolist(), "forms": g.form.tolist(),
                     "acceptance_time": first.acceptance_time, "ts_quality": q, "ts_source": first.ts_source,
                     "event_type": "8K", "categories": sorted(cats), "items": sorted(set(items)),
                     "report_date": first.report_date, "filing_date": first.filing_date})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev["event_id"] = [event_id(c, sorted(a)[0]) for c, a in zip(ev.cik, ev.accessions)]
    tt = derive_times(ev["acceptance_time"], t["public_lag_minutes"], t["decision_latency_minutes"])
    ev = pd.concat([ev.reset_index(drop=True), tt.reset_index(drop=True)], axis=1)
    acc_ns = to_utc_ns(ev["acceptance_time"])
    pub_ns = to_utc_ns(ev["public_available_time"])
    dec_ns = to_utc_ns(ev["decision_time"])
    valid = np.ones(len(ev), bool)
    ev["session_bucket"] = cal.v_bucket(acc_ns, valid)
    base = cal.v_last_close_at_or_before(acc_ns)
    d0 = cal.v_first_close_after(pub_ns)
    entry = cal.v_first_open_at_or_after(dec_ns)
    n = len(cal)
    ev["tinfo_base_idx"] = base
    ev["trackA_day0_idx"] = d0
    ev["ttrade_entry_idx"] = entry
    ev["tinfo_base_date"] = [cal.date(i) if 0 <= i < n else pd.NaT for i in base]
    ev["trackA_day0"] = [cal.date(i) if 0 <= i < n else pd.NaT for i in d0]
    ev["ttrade_entry_date"] = [cal.date(i) if 0 <= i < n else pd.NaT for i in entry]
    ev["tinfo_eligible"] = ev["session_bucket"].isin(["PRE", "POST", "NONTRADING"]) & (ev.ts_quality != "C")
    base_close_ns = np.where((base >= 0) & (base < n), cal.close_utc[np.clip(base, 0, n - 1)], np.nan)
    ev["contam_minutes"] = (acc_ns - base_close_ns) / 6e10   # pre-publication window base close -> acceptance
    # addendum P0.3: public availability is an assumption unless independently observed
    ev["timing_policy"] = np.where(ev["public_time_source"] == "observed", "observed_public_time",
                                   "assumed_availability_latency_policy")
    ev = _earnings_types(ev, tax, rel)
    ev = _clusters(ev, rel)
    ev = _overlaps(ev, cfg)
    ev["section_ids"] = [sorted({s for a in accs for s in (sav.get(a) or [])}) for accs in ev.accessions]
    ev["year"] = pd.to_datetime(ev["decision_time"]).dt.year
    return ev.sort_values(["decision_time", "cik"]).reset_index(drop=True)


def _earnings_types(ev: pd.DataFrame, tax, rel: pd.DataFrame) -> pd.DataFrame:
    et = []
    rel_flag = rel["is_results_release"].to_dict() if len(rel) else {}
    for r in ev.itertuples(index=False):
        if r.event_type != "8K":
            et.append(None)
            continue
        modern, legacy_eq = tax.earnings_candidate(r.items, r.filing_date)
        confirmed = any(bool(rel_flag.get(a, False)) for a in r.accessions)
        if (modern or legacy_eq) and confirmed and r.ts_quality == "A":
            et.append("earnings_announcement_verified")
        elif modern or (legacy_eq and confirmed):
            et.append("earnings_date_ambiguous")
        else:
            et.append(None)
    ev["earnings_type"] = et
    if "earnings" not in ev.columns:
        pass
    return ev


def _clusters(ev: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    """Link each verified release with the firm's next periodic filing for the same fiscal period."""
    ev["cluster_id"] = None
    ev["cluster_role"] = None
    ev["release_period_end"] = pd.NaT
    per_end = rel["release_period_end"].to_dict() if len(rel) else {}
    for i, r in ev.iterrows():
        if r.earnings_type == "earnings_announcement_verified":
            pe = next((per_end.get(a) for a in r.accessions if per_end.get(a)), None)
            ev.at[i, "release_period_end"] = pd.Timestamp(pe) if pe else pd.NaT
    ev["report_date_ts"] = pd.to_datetime(ev["report_date"], errors="coerce")
    for cik, g in ev.groupby("cik"):
        rels = g[g.earnings_type == "earnings_announcement_verified"].sort_values("decision_time")
        pers = g[g.event_type.str.startswith("periodic")].sort_values("decision_time")
        used = set()
        for i, r in rels.iterrows():
            ev.at[i, "cluster_id"] = "C_" + r.event_id
            ev.at[i, "cluster_role"] = "release"
            if pd.isna(r.release_period_end):
                continue
            cand = pers[(pers.decision_time > r.decision_time)
                        & (pers.decision_time <= r.decision_time + pd.Timedelta(days=120))
                        & ((pers.report_date_ts - r.release_period_end).abs() <= pd.Timedelta(days=20))]
            cand = cand[~cand.index.isin(used)]
            if len(cand):
                j = cand.index[0]
                used.add(j)
                ev.at[j, "cluster_id"] = "C_" + r.event_id
                ev.at[j, "cluster_role"] = "periodic_after_release"
    per_mask = ev.event_type.str.startswith("periodic")
    ev.loc[per_mask & ev.cluster_id.isna(), "earnings_type"] = "periodic_filing_only"
    return ev.drop(columns=["report_date_ts"])


def _overlaps(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """overlap_flag_h: another same-firm event's day 0 falls within (day0, day0 + h] (dates only)."""
    hs = sorted(set(cfg["targets"]["horizons"]) | {int(cfg["contam_window_sessions"])})
    for h in hs:
        ev[f"overlap_flag_h{h}"] = False
    for cik, g in ev.groupby("cik"):
        d = g["trackA_day0_idx"].to_numpy()
        order = np.argsort(d, kind="stable")
        ds = d[order]
        for h in hs:
            nxt = np.searchsorted(ds, ds, side="right")       # first index with day0 > own day0
            lim = np.searchsorted(ds, ds + h, side="right")   # first index with day0 > day0 + h
            same_day_other = np.searchsorted(ds, ds, side="right") - np.searchsorted(ds, ds, side="left") > 1
            flag = (lim > nxt) | same_day_other
            out = np.zeros(len(d), bool)
            out[order] = flag
            ev.loc[g.index, f"overlap_flag_h{h}"] = out
    return ev


def events_to_parquet_safe(ev: pd.DataFrame) -> pd.DataFrame:
    out = ev.copy()
    for c in ("accessions", "forms", "categories", "items", "section_ids"):
        if c in out:
            out[c] = out[c].map(lambda v: json.dumps(list(v)) if isinstance(v, (list, tuple, np.ndarray)) else v)
    return out


def events_from_parquet(ev: pd.DataFrame) -> pd.DataFrame:
    out = ev.copy()
    for c in ("accessions", "forms", "categories", "items", "section_ids"):
        if c in out:
            out[c] = out[c].map(lambda v: json.loads(v) if isinstance(v, str) else (list(v) if v is not None else []))
    for c in ("acceptance_time", "public_available_time", "feature_ready_time", "decision_time"):
        if c in out:
            out[c] = pd.to_datetime(out[c], utc=True).dt.tz_convert("America/New_York")
    return out
