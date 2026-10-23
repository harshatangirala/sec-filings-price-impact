"""Index-membership buckets: S&P 500 vs non-S&P 500 vs combined (owner request, decision log D-056).

Firm bucket (`sp_bucket`): "SP500" if the firm was an S&P 500 member at any time in the study window according to the
pinned fja05680/sp500 list, else "NonSP500". The list is built from Wikipedia changes and is incomplete in early years,
so "NonSP500" means "not found in that list", never "verified never a member".
Point-in-time membership on the event's decision date (`sp500_member_at_event`) is reported alongside; it is not a model
feature. Every bucket result is computed from the same frozen models and walk-forward predictions as the combined result.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .bootstrap import macro_f1_ci, paired_macro_f1_diff
from .metrics import classification_metrics, power_label

BUCKETS = ("Combined", "SP500", "NonSP500")
LABELS = {"Combined": "Combined (all firms)", "SP500": "S&P 500 members (in the study window)",
          "NonSP500": "Non-S&P 500 (not in the fja05680 list)"}
PCOLS = ["p_down", "p_side", "p_up"]


def firm_bucket(ever_sp500) -> np.ndarray:
    return np.where(pd.Series(ever_sp500).fillna(False).astype(bool).to_numpy(), "SP500", "NonSP500")


def _member_on(intervals_json, day) -> bool:
    try:
        ivs = json.loads(intervals_json) if isinstance(intervals_json, str) else (intervals_json or [])
    except (TypeError, ValueError):
        return False
    for a, b in ivs:
        if pd.Timestamp(a) <= day and (b is None or pd.isna(pd.Timestamp(b)) or pd.Timestamp(b) >= day):
            return True
    return False


def event_buckets(ev: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """event_id -> cik, sp_bucket (firm level), sp500_member_at_event (point in time, decision date)."""
    uu = u.drop_duplicates("cik").set_index(u.drop_duplicates("cik").cik.astype(int))
    if "sp_bucket" in uu:
        fb = uu["sp_bucket"]
    else:
        fb = pd.Series(firm_bucket(uu.get("ever_sp500", pd.Series(False, index=uu.index))), index=uu.index)
    cik = ev["cik"].astype(int)
    out = pd.DataFrame({"event_id": ev["event_id"].values, "cik": cik.values,
                        "sp_bucket": cik.map(fb).fillna("NonSP500").values})
    iv = cik.map(uu["sp500_intervals"]) if "sp500_intervals" in uu else pd.Series([None] * len(ev))
    day = pd.to_datetime(ev["decision_date"]) if "decision_date" in ev else \
        pd.to_datetime(ev["decision_time"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    out["sp500_member_at_event"] = [_member_on(i, d) for i, d in zip(iv.values, day.values)]
    return out


def expand(df: pd.DataFrame, bk: pd.DataFrame, key: str = "event_id") -> pd.DataFrame:
    """Rows once per bucket they belong to: their own bucket and 'Combined'."""
    x = df.merge(bk[[key, "sp_bucket"]], on=key, how="left")
    x["sp_bucket"] = x["sp_bucket"].fillna("NonSP500")
    return pd.concat([x, x.assign(sp_bucket="Combined")], ignore_index=True)


def score_predictions(preds: pd.DataFrame, bk: pd.DataFrame, n_boot: int, seed: int, by=("run_id",)) -> pd.DataFrame:
    """Macro-F1 (month-block CI) and secondary metrics of existing predictions, per bucket. Uses the model's own decisions."""
    rows = []
    if preds is None or preds.empty:
        return pd.DataFrame()
    x = expand(preds, bk)
    for keys, g in x.groupby(list(by) + ["sp_bucket"], sort=False):
        g = g[g.y.notna()]
        if len(g) < 3:
            continue
        y = g.y.astype(int).values
        prior = np.bincount(y, minlength=3) / len(y)
        m = classification_metrics(y, g[PCOLS].values, prior, g.yhat.astype(int).values)
        ci = macro_f1_ci(y, g.yhat.astype(int).values, g.date.values, n_boot, seed)
        rk = dict(zip(list(by) + ["sp_bucket"], keys if isinstance(keys, tuple) else (keys,)))
        rows.append({**rk, "n_events": int(len(g)), "n_firms": int(g.event_id.map(bk.set_index("event_id").cik).nunique()),
                     "macro_f1": m.get("macro_f1"), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                     "accuracy": m.get("accuracy"), "balanced_accuracy": m.get("balanced_accuracy"), "auc_macro_ovr": m.get("auc_macro_ovr"),
                     "logloss": m.get("logloss"), "logloss_prior": m.get("logloss_prior"), "ece": m.get("ece"),
                     "share_up": float(prior[2]), "share_side": float(prior[1]), "share_down": float(prior[0]),
                     "power": power_label(len(g), ci["ci_high"] - ci["ci_low"] if np.isfinite(ci["ci_low"]) else None)})
    out = pd.DataFrame(rows)
    if len(out):
        order = {b: i for i, b in enumerate(BUCKETS)}
        out = out.assign(_o=out.sp_bucket.map(order)).sort_values(list(by) + ["_o"], kind="stable").drop(columns="_o")
    return out.reset_index(drop=True)


def pooled_vs_bucket(p_pooled: pd.DataFrame, p_bucket: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Paired comparison on identical rows: the combined (pooled) model vs a model trained on the bucket only."""
    m = p_pooled[["event_id", "y", "yhat", "date"]].merge(p_bucket[["event_id", "yhat"]], on="event_id",
                                                           suffixes=("_pooled", "_bucket"))
    if len(m) < 30:
        return {"n_events": int(len(m)), "note": "fewer than 30 shared rows"}
    d = paired_macro_f1_diff(m.y.astype(int).values, m.yhat_pooled.astype(int).values, m.yhat_bucket.astype(int).values,
                             m.date.values, n_boot, seed)
    return {"n_events": int(len(m)), "pooled_minus_bucket_macro_f1": d["value"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "p_boot": d["p_boot"]}


def composition(ev: pd.DataFrame, bk: pd.DataFrame, u: pd.DataFrame, labels: pd.DataFrame | None = None,
                features: pd.DataFrame | None = None) -> pd.DataFrame:
    """What differs between the buckets before any model: firms, events, event mix, timing, size, data availability."""
    x = expand(ev, bk)
    if "sp500_member_at_event" in bk and "sp500_member_at_event" not in x:
        x = x.merge(bk[["event_id", "sp500_member_at_event"]], on="event_id", how="left")
    if labels is not None and len(labels):
        x = x.merge(labels, on="event_id", how="left")
    if features is not None and len(features):
        keep = [c for c in ("acc_log_mcap", "sue_event", "has_text", "has_accounting", "px_vol_60") if c in features]
        x = x.merge(features[["event_id"] + keep], on="event_id", how="left")
    rows = []
    for b, g in x.groupby("sp_bucket", sort=False):
        r = {"sp_bucket": b, "firms": int(g.cik.nunique()), "events": int(len(g)),
             "events_per_firm": float(len(g) / max(1, g.cik.nunique())),
             "share_8k": float((g.event_type == "8K").mean()), "share_10q": float((g.event_type == "periodic_10Q").mean()),
             "share_10k": float((g.event_type == "periodic_10K").mean()),
             "share_verified_earnings": float((g.earnings_type == "earnings_announcement_verified").mean()),
             "share_ts_quality_A": float((g.ts_quality == "A").mean()),
             "member_at_event_share": float(g.sp500_member_at_event.mean()) if "sp500_member_at_event" in g else None,
             "first_year": int(pd.to_datetime(g.decision_date).dt.year.min()) if len(g) else None}
        if "label3" in g:
            y = g.label3.dropna()
            r.update({"labelled_T_trade_h1": int(len(y)), "share_up": float((y == 2).mean()) if len(y) else None,
                      "share_down": float((y == 0).mean()) if len(y) else None})
        for c, nm in (("acc_log_mcap", "median_log_mcap"), ("px_vol_60", "median_vol_60")):
            if c in g:
                r[nm] = float(g[c].median()) if g[c].notna().any() else None
        if "sue_event" in g:
            r["sue_available_share"] = float(g.sue_event.notna().mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    out["label"] = out.sp_bucket.map(LABELS)
    order = {b: i for i, b in enumerate(BUCKETS)}
    return out.sort_values("sp_bucket", key=lambda s: s.map(order)).reset_index(drop=True)
