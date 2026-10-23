"""ext_stacking (SPEC.md §7.3; H5): multi-tower stacking and a Caruana greedy ensemble, trained only on
walk-forward out-of-fold predictions in time order (each meta-model sees only folds strictly earlier than the fold
it predicts; SPEC.md §6.3 #20). Gate G3c: core complete + budget + the pre-stated question H5."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..evaluation.bootstrap import paired_macro_f1_diff
from ..evaluation.metrics import classification_metrics, confusion, decide, f1_from_cm

PCOLS = ["p_down", "p_side", "p_up"]


def _base_runs(ctx, prim) -> dict:
    frames = []
    for s in ("S11", "S15"):
        p = ctx.dep(s) / "predictions.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    pr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pr.empty:
        return {}
    pr = pr[(pr.target_def == prim["target_def"]) & (pr.horizon == prim["horizon"]) & (pr.tuned == "tuned")
            & ~pr.run_id.astype(str).str.startswith("dev|")]
    out = {}
    cohort = "common_emb" if (pr.cohort == "common_emb").any() else "common"
    sub = pr[pr.cohort == cohort]
    for rung in ("L6", "L7", "L8", "L8_ft"):
        g = sub[sub.rung == rung]
        if g.empty:
            continue
        best = None
        for rid, gg in g.groupby("run_id"):
            f = f1_from_cm(confusion(gg.y.values, gg.yhat.values))["macro_f1"]
            enc = gg.encoder.iloc[0]
            key = f"{rung}|{enc}" if rung in ("L7", "L8_ft") else rung
            if key not in out or f > out[key][1]:
                out[key] = (gg, f)
    return {k: v[0] for k, v in out.items()}


def _caruana(P: list[np.ndarray], y: np.ndarray, prior: np.ndarray, iters: int = 20) -> np.ndarray:
    w = np.zeros(len(P))
    cur = np.zeros_like(P[0])
    for _ in range(iters):
        best, bi = -1, 0
        for i, p in enumerate(P):
            cand = (cur * w.sum() + p) / (w.sum() + 1)
            f = f1_from_cm(confusion(y, decide(cand, prior)))["macro_f1"] or 0
            if f > best:
                best, bi = f, i
        cur = (cur * w.sum() + P[bi]) / (w.sum() + 1)
        w[bi] += 1
    return w / w.sum()


def run_stacking(ctx, lad, lab, common, res, dev_preds: pd.DataFrame) -> dict | None:
    from sklearn.linear_model import LogisticRegression
    cfg = ctx.cfg
    prim = cfg["targets"]["primary"]
    bases = _base_runs(ctx, prim)
    if len(bases) < 2:
        return {"ran": False, "reason": "fewer than two base models with OOF predictions"}
    keys = sorted(bases)
    ids = None
    for k in keys:
        s = set(bases[k].event_id)
        ids = s if ids is None else ids & s
    ids = sorted(ids)
    al = {k: bases[k].set_index("event_id").loc[ids] for k in keys}
    y = al[keys[0]].y.astype(int).values
    fold = al[keys[0]].fold.values
    dates = al[keys[0]].date.values
    Xlog = np.hstack([np.log(np.clip(al[k][PCOLS].values, 1e-9, 1)) for k in keys])
    folds = sorted(np.unique(fold))
    stack_p = np.full((len(ids), 3), np.nan)
    car_p = np.full((len(ids), 3), np.nan)
    for f in folds[1:]:
        tr, te = fold < f, fold == f
        if tr.sum() < 100 or len(np.unique(y[tr])) < 3:
            continue
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xlog[tr], y[tr])
        stack_p[te] = m.predict_proba(Xlog[te])
        prior = np.bincount(y[tr], minlength=3) / tr.sum()
        w = _caruana([al[k][PCOLS].values[tr] for k in keys], y[tr], prior)
        car_p[te] = sum(wi * al[k][PCOLS].values[te] for wi, k in zip(w, keys))
    ok = np.isfinite(stack_p).all(1)
    if ok.sum() < 50:
        return {"ran": False, "reason": "too few out-of-fold rows after the first fold"}
    prior_all = np.bincount(y[ok], minlength=3) / ok.sum()
    yh_stack = decide(stack_p[ok], prior_all)
    yh_car = decide(car_p[ok], prior_all)
    fold_scores = []
    for f in folds[1:]:
        m_ = ok & (fold == f)
        if m_.sum():
            fold_scores.append(f1_from_cm(confusion(y[m_], decide(stack_p[m_], prior_all)))["macro_f1"])
    single = {k: f1_from_cm(confusion(y[ok], al[k].yhat.values[ok].astype(int)))["macro_f1"] for k in keys}
    best_k = max(single, key=lambda k: single[k] or 0)
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    h5 = paired_macro_f1_diff(y[ok], yh_stack, al[best_k].yhat.values[ok].astype(int), dates[ok], nb, seed)
    h5c = paired_macro_f1_diff(y[ok], yh_car, al[best_k].yhat.values[ok].astype(int), dates[ok], nb, seed)
    pm = classification_metrics(y[ok], stack_p[ok], prior_all, yh_stack)
    res.add("ext_stacking|STACK|walk_forward_folds>=1", "STACK", "stack_logistic", "+".join(keys), "retrospective", "common_emb",
            prim["target_def"], prim["horizon"], "walk_forward_pooled_folds_ge1", "all", pm, None, int(ok.sum()), None, None, None, "tuned")
    out = {"ran": True, "bases": keys, "wf_score": float(np.mean(fold_scores)) if fold_scores else None,
           "h5_stack_vs_best_single": {"best_single": best_k, **h5}, "h5_caruana_vs_best_single": h5c,
           "spec": {"meta": "multinomial logistic on base log-probabilities", "bases": keys},
           "g3c": "core complete (S15 done); pre-stated question H5; budget checked in preflight"}
    # development holdout: meta trained on all walk-forward OOF rows, applied to the finalists' dev predictions
    dv = {}
    for k in keys:
        nm = k if k.startswith("L8_ft") else k.split("|")[0]
        if nm and len(dev_preds) and (dev_preds.name == nm).any():
            dv[k] = dev_preds[dev_preds.name == nm].set_index("event_id")
    if len(dv) == len(keys):
        j = sorted(set.intersection(*[set(d.index) for d in dv.values()]))
        Xd = np.hstack([np.log(np.clip(dv[k].loc[j, PCOLS].values, 1e-9, 1)) for k in keys])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xlog[ok | (fold == folds[0])], y[ok | (fold == folds[0])])
        P = m.predict_proba(Xd)
        yd = dv[keys[0]].loc[j, "y"].values.astype(int)
        yh = decide(P, np.bincount(y, minlength=3) / len(y))
        out["dev_preds"] = pd.DataFrame({"event_id": j, "y": yd, "date": dv[keys[0]].loc[j, "date"].values, "p_down": P[:, 0],
                                         "p_side": P[:, 1], "p_up": P[:, 2], "yhat": yh, "fold": 0, "run_id": "dev|STACK",
                                         "name": "STACK"})
    else:
        out["dev_preds"] = None
        out["dev_note"] = "dev predictions not available for every base (fine-tuned bases are walk-forward only)"
    return out
