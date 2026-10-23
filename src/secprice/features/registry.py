"""Feature registry + availability rules (SPEC.md §16 feature_registry.csv, §6.3 #9/#11/#21, hard rules 3 and 7).

Each family has an availability column `_avail_<family>` holding, per event, the latest timestamp of any input.
The rule is `_avail <= decision_time` (the current filing's own content is legitimate at decision time); prior
documents and history must be strictly earlier (enforced where they are built and re-checked here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILIES = {
    "price": {"prefixes": ["px_", "mkt_"], "rule": "closes at or before decision_time", "source": "prices (S03)"},
    "metadata": {"prefixes": ["meta_"], "rule": "own filing metadata (acceptance <= decision)", "source": "manifest/events"},
    "accounting": {"prefixes": ["acc_", "sue_"], "rule": "XBRL/release values with public_time < decision_time; "
                   "SUE never from a later same-cluster accession", "source": "S06"},
    "lexical": {"prefixes": ["lm_", "main_lm_"], "rule": "own filing text", "source": "S05 sections"},
    "change": {"prefixes": ["chg_", "rf_new", "rf_removed", "boilerplate_"], "rule": "prior same-firm documents with "
               "public_time < decision_time", "source": "S05 sections"},
    "novelty": {"prefixes": ["novelty_"], "rule": "prior same-firm documents with public_time < decision_time",
                "source": "S05 sections"},
    "ear": {"prefixes": ["ear_"], "rule": "EAR of a later, different cluster whose window closed before decision_time",
            "source": "S08/S20"},
}


def family_of(col: str) -> str | None:
    for fam, spec in FAMILIES.items():
        if any(col.startswith(p) for p in spec["prefixes"]):
            return fam
    return None


def registry_frame(feature_cols: list[str], strict_map: dict | None = None) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        fam = family_of(c)
        if fam is None:
            continue
        spec = FAMILIES[fam]
        rows.append({"name": c, "family": fam, "source": spec["source"], "availability_rule": spec["rule"],
                     "max_availability_col": f"_avail_{fam}", "leaderboard": (strict_map or {}).get(c, "strict")})
    return pd.DataFrame(rows)


def availability_violations(feat: pd.DataFrame, decision_time: pd.Series) -> pd.DataFrame:
    """Rows where any family's availability time is after decision_time."""
    dec = pd.to_datetime(decision_time, utc=True).reset_index(drop=True)
    out = []
    for c in [c for c in feat.columns if c.startswith("_avail_")]:
        av = pd.to_datetime(feat[c], utc=True).reset_index(drop=True)
        bad = av.notna() & (av > dec)
        if bad.any():
            out.append(pd.DataFrame({"event_id": feat["event_id"].reset_index(drop=True)[bad], "family": c[7:],
                                     "avail": av[bad], "decision": dec[bad]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["event_id", "family", "avail", "decision"])


def ear_rule_violations(feat: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """EAR features must come from a different cluster whose EAR window closed before decision_time."""
    if "ear_src_cluster" not in feat.columns:
        return pd.DataFrame(columns=["event_id", "reason"])
    m = feat[["event_id", "ear_src_cluster", "_avail_ear"]].merge(events[["event_id", "cluster_id", "decision_time"]],
                                                                   on="event_id")
    same = m["ear_src_cluster"].notna() & (m["ear_src_cluster"] == m["cluster_id"])
    late = pd.to_datetime(m["_avail_ear"], utc=True) > pd.to_datetime(m["decision_time"], utc=True)
    bad = m[same | late]
    return pd.DataFrame({"event_id": bad.event_id, "reason": np.where(same[same | late], "same_cluster", "window_not_closed")})


def sue_rule_violations(sue_src: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """SUE inputs must be public strictly before decision_time and never from a later same-cluster accession.
    sue_src: event_id, source_accession, source_public_time."""
    if sue_src is None or sue_src.empty:
        return pd.DataFrame(columns=["event_id", "reason"])
    acc2cl = {}
    for r in events[["accessions", "cluster_id", "decision_time"]].itertuples(index=False):
        for a in r.accessions:
            acc2cl[a] = (r.cluster_id, r.decision_time)
    m = sue_src.merge(events[["event_id", "cluster_id", "decision_time", "accessions", "earnings_type"]], on="event_id")
    rows = []
    for r in m.itertuples(index=False):
        if r.source_accession in (r.accessions or []):
            continue          # the event's own filing / release
        src_cl, _ = acc2cl.get(r.source_accession, (None, None))
        if pd.to_datetime(r.source_public_time, utc=True) >= pd.to_datetime(r.decision_time, utc=True):
            rows.append((r.event_id, "source_not_public_before_decision"))
        elif (r.earnings_type == "earnings_announcement_verified" and src_cl is not None
              and src_cl == r.cluster_id):
            rows.append((r.event_id, "later_same_cluster_source"))
    return pd.DataFrame(rows, columns=["event_id", "reason"])
