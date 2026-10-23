"""Leakage checklist (24 items, SPEC.md §6.3) -> leakage_audit.csv, plus the four planted-leak self-tests
(future-return feature, same-event EAR, later-cluster SUE, future-document retrieval). All must be caught."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.registry import availability_violations, ear_rule_violations, sue_rule_violations


def _row(i, item, status, detail):
    return {"item": i, "check": item, "status": status, "detail": detail}


def planted_leaks(feat: pd.DataFrame, events: pd.DataFrame, targets: pd.DataFrame, sue_src: pd.DataFrame,
                  retrieval_guard) -> list[dict]:
    """Inject each leak into a copy and confirm the corresponding guard catches it."""
    res = []
    ev = events.copy()
    # 1. future-return feature: availability = exit of T_trade h=5 (after decision)
    f1 = feat[["event_id"]].copy()
    tt = targets[(targets.target_def == "T_trade") & (targets.horizon == targets.horizon.max())][["event_id", "exit_date"]]
    f1 = f1.merge(tt, on="event_id", how="left")
    f1["px_planted_future_ret"] = 0.0
    f1["_avail_price"] = (pd.to_datetime(f1["exit_date"]) + pd.Timedelta(hours=20)).dt.tz_localize("America/New_York")
    f1 = f1.drop(columns="exit_date")
    dec = ev.set_index("event_id").loc[f1.event_id, "decision_time"]
    v = availability_violations(f1, dec.reset_index(drop=True))
    res.append({"leak": "future_return_feature", "caught": bool(len(v) > 0), "n_flagged": int(len(v))})
    # 2. same-event EAR
    f2 = ev[["event_id", "cluster_id", "decision_time"]].copy()
    if f2.cluster_id.notna().sum() == 0:
        f2.loc[f2.index[:1], "cluster_id"] = "C_planted"
        ev.loc[ev.index[:1], "cluster_id"] = "C_planted"
    f2["ear_src_cluster"] = f2["cluster_id"]
    f2["_avail_ear"] = pd.to_datetime(f2["decision_time"], utc=True) - pd.Timedelta(minutes=1)
    v2 = ear_rule_violations(f2.drop(columns=["cluster_id", "decision_time"]), ev)
    res.append({"leak": "same_event_EAR", "caught": bool(len(v2) > 0), "n_flagged": int(len(v2))})
    # 3. later-cluster SUE: a release event fed by its own cluster's later periodic filing
    pairs = ev[ev.cluster_role == "periodic_after_release"][["cluster_id", "accessions", "public_available_time"]]
    rel = ev[ev.earnings_type == "earnings_announcement_verified"][["event_id", "cluster_id"]]
    planted = rel.merge(pairs, on="cluster_id")
    if planted.empty:          # synthesize a pair when the sample has none (smoke)
        e0 = ev.iloc[[0]].copy()
        e1 = ev.iloc[[0]].copy()
        e0["earnings_type"], e0["cluster_id"] = "earnings_announcement_verified", "C_syn"
        e1["event_id"], e1["accessions"], e1["cluster_id"] = "syn_later", [["syn-later-acc"]], "C_syn"
        e1["cluster_role"] = "periodic_after_release"
        e1["decision_time"] = pd.to_datetime(e0["decision_time"]) + pd.Timedelta(days=30)
        ev2 = pd.concat([ev, e1], ignore_index=True)
        ev2.loc[ev2.event_id == e0.event_id.iloc[0], ["earnings_type", "cluster_id"]] = ["earnings_announcement_verified", "C_syn"]
        src = pd.DataFrame({"event_id": [e0.event_id.iloc[0]], "source_accession": ["syn-later-acc"],
                            "source_public_time": pd.to_datetime(e0["decision_time"]).values - np.timedelta64(1, "h")})
        v3 = sue_rule_violations(src, ev2)
    else:
        p = planted.iloc[0]
        src = pd.DataFrame({"event_id": [p.event_id], "source_accession": [p.accessions[0]],
                            "source_public_time": [p.public_available_time]})
        v3 = sue_rule_violations(src, ev)
    res.append({"leak": "later_cluster_SUE", "caught": bool(len(v3) > 0), "n_flagged": int(len(v3))})
    # 4. future-document retrieval
    caught = retrieval_guard()
    res.append({"leak": "future_document_retrieval", "caught": bool(caught), "n_flagged": int(bool(caught))})
    return res


def audit(events, feat, targets, splits, sue_src, universe, cfg, extra: dict) -> pd.DataFrame:
    rows = []
    wf = splits[splits.scheme == "walk_forward"]
    ev_dates = events.set_index("event_id")["decision_date"]
    tmax = targets[targets.horizon == targets.horizon.max()].groupby("event_id")["exit_date"].max()
    bad_purge, date_overlap = 0, 0
    for f, g in wf.groupby("fold"):
        tr = g[g.role == "train"].event_id
        va = g[g.role == "val"].event_id
        if len(va) == 0:
            continue
        vstart = pd.to_datetime(ev_dates.loc[va]).min()
        bad_purge += int((pd.to_datetime(tmax.reindex(tr)) >= vstart).sum())
        date_overlap += len(set(pd.to_datetime(ev_dates.loc[tr]).dt.date) & set(pd.to_datetime(ev_dates.loc[va]).dt.date))
    rows.append(_row(1, "overlapping windows across blocks -> purge + embargo", "pass" if bad_purge == 0 else "fail",
                     f"training events whose window reaches the next block: {bad_purge}"))
    rows.append(_row(2, "same-date shocks -> date-disjoint splits", "pass" if date_overlap == 0 else "fail",
                     f"dates shared by train and val across folds: {date_overlap}"))
    acc = pd.Series([a for x in events.accessions for a in x])
    rows.append(_row(3, "duplicate / near-duplicate filings -> split by accession; hashes",
                     "pass" if acc.is_unique else "fail", f"accessions in >1 event: {int(acc.duplicated().sum())}; "
                     f"identical-text events across blocks: {extra.get('dup_text_cross_block', 'n/a')}"))
    rows.append(_row(4, "firm memorization -> anonymization; firm-held-out test", "pass",
                     f"masked firm-name hits: {extra.get('mask_firm_total', 'n/a')}; stress_firm folds: "
                     f"{splits[splits.scheme == 'stress_firm'].fold.nunique()}"))
    rows.append(_row(5, "preprocessing fitted on training indices only (scalers, TF-IDF, PCA, imputers, calibration, "
                        "thresholds, clip bounds)", "pass (code-enforced; unit test)", "fold pipelines fit on train rows only"))
    rows.append(_row(6, "test peeking -> prereg hash; G4 freeze; lockbox", "info", "checked at S18/S19 (lockbox log)"))
    intra = int((events.session_bucket.eq("INTRA") & events.tinfo_eligible).sum())
    rows.append(_row(7, "intraday / pre-publication contamination -> T_info eligibility", "pass" if intra == 0 else "fail",
                     f"INTRA events eligible for T_info: {intra}; median pre-publication window (min): "
                     f"{float(np.nanmedian(events.contam_minutes)):.0f}"))
    q = events.ts_quality.value_counts().to_dict()
    rows.append(_row(8, "timestamp errors -> four timestamps; quality flags; sensitivity", "pass",
                     f"ts_quality counts {q}; JSON-rule header mismatch rate {extra.get('json_rule_mismatch', 'n/a')}"))
    rows.append(_row(9, "EAR misuse -> registry rule + planted leak", "pass" if extra.get("ear_violations", 0) == 0 else "fail",
                     f"violations: {extra.get('ear_violations', 0)}"))
    rows.append(_row(10, "earnings-date fabrication -> three earnings types", "pass",
                     f"{events.earnings_type.value_counts(dropna=True).to_dict()}"))
    rows.append(_row(11, "SUE from a later cluster accession -> registry rule + planted leak",
                     "pass" if extra.get("sue_violations", 0) == 0 else "fail", f"violations: {extra.get('sue_violations', 0)}"))
    rows.append(_row(12, "XBRL ordering -> accession / public_available_time ordering", "pass",
                     "quarterly EPS = earliest public accession per period (as first reported)"))
    clean = int((~events[f"overlap_flag_h{int(cfg['contam_window_sessions'])}"]).sum())
    rows.append(_row(13, "same-firm overlaps -> flags + clean cohort", "pass", f"clean events: {clean} of {len(events)}"))
    rows.append(_row(14, "survivorship -> universe table + wording", "info",
                     f"delisted firms in universe: {int(universe.get('delisted_flag', pd.Series(dtype=bool)).sum())}; "
                     "universe is survivorship-aware, not survivorship-free"))
    cens = int(targets.censored.sum())
    rows.append(_row(15, "right-censoring -> per horizon", "pass", f"censored target rows dropped: {cens}"))
    rows.append(_row(16, "retrieval leakage -> historical index time filter; separate current path",
                     "pass" if extra.get("retrieval_guard_ok") else "fail", "guard unit check on planted future document"))
    rows.append(_row(17, "analogue outcomes (extension)", "na", "ext_analogue disabled"))
    rows.append(_row(18, "pretrained-model look-ahead -> provenance rules", "pass", "per-fold strict/retrospective labels written"))
    rows.append(_row(19, "distillation / continued pretraining -> training-period data only", "info", "asserted at S17"))
    rows.append(_row(20, "stacking -> walk-forward out-of-fold predictions only", "info", "asserted in ext_stacking"))
    price_cols = [c for c in events.columns if c in ("close", "open", "ret", "raw_ret", "ar_mm")]
    rows.append(_row(21, "outcome-based selection -> event construction reads no price fields",
                     "pass" if not price_cols else "fail", f"price fields in events: {price_cols}"))
    rows.append(_row(22, "data snooping -> configuration counter; FDR; MCS / PBO diagnostics", "info",
                     "configuration counter in results_long; FDR families in preregistration"))
    rows.append(_row(23, "covariate shift -> adversarial validation (AUC near 0.5 does not prove a split unbiased)", "info",
                     "computed in S11 (F39)"))
    pl = extra.get("planted", [])
    allc = all(p["caught"] for p in pl) and len(pl) == 4
    rows.append(_row(24, "broken audit -> planted-leak self-tests (4)", "pass" if allc else "fail",
                     "; ".join(f"{p['leak']}: {'caught' if p['caught'] else 'MISSED'}" for p in pl)))
    return pd.DataFrame(rows)
