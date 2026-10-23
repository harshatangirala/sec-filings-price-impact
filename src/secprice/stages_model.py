"""Track B stages: S11 baselines, S13 preflight, S14 embeddings, S15 heads/G3/fine-tune, S16 RAG, S17 distillation,
S18 dev holdout + G4, S19 locked test, S22 backtest (SPEC.md §9, §11, §13.3, §17.1)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .config import canonical_json, get
from .evaluation.bootstrap import macro_f1_ci, paired_macro_f1_diff
from .evaluation.metrics import classification_metrics, decide, power_label
from .models.ladder import RUNGS, Ladder, fit_verdict, summarize
from .runner import stage
from .storage import write_json

PCOLS_ = ["p_down", "p_side", "p_up"]
PRIMARY_METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1", "auc_macro_ovr", "prauc_macro", "kappa", "mcc",
                   "logloss", "logloss_prior", "brier", "ece", "acc_at_50", "macro_f1_at_50", "acc_at_25", "macro_f1_at_25",
                   "acc_at_10", "macro_f1_at_10", "f1_up", "f1_down", "f1_sideways", "auc_up", "auc_down"]


class Results:
    """results_long rows (SPEC.md §16)."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, run_id, rung, model_id, revision, leaderboard, cohort, target_def, horizon, block, event_set, metrics: dict,
            ci: dict | None, n_events, n_firms, n_dates, n_overlap, tuned: str, extra: dict | None = None):
        for k in PRIMARY_METRICS + ["r2", "oos_r2_vs_hist_mean", "mae", "rmse", "spearman", "directional_accuracy",
                                    "rank_ic_mean", "rank_ic_t", "rank_ic_icir", "qlike"]:
            if k in metrics and metrics[k] is not None and not isinstance(metrics[k], (list, dict)):
                row = {"run_id": run_id, "track": "B", "rung": rung, "model_id": model_id, "revision": revision,
                       "leaderboard": leaderboard, "cohort": cohort, "target_def": target_def, "horizon": horizon,
                       "block": block, "event_set": event_set, "coverage": 1.0, "metric": k, "value": float(metrics[k]),
                       "ci_low": None, "ci_high": None, "n_events": n_events, "n_firms": n_firms, "n_dates": n_dates,
                       "n_overlap": n_overlap, "tuned": tuned, **(extra or {})}
                if k == "macro_f1" and ci:
                    row["ci_low"], row["ci_high"] = ci.get("ci_low"), ci.get("ci_high")
                self.rows.append(row)

    COLS = ["run_id", "track", "rung", "model_id", "revision", "leaderboard", "cohort", "target_def", "horizon", "block",
            "event_set", "coverage", "metric", "value", "ci_low", "ci_high", "n_events", "n_firms", "n_dates", "n_overlap", "tuned"]

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for c in self.COLS:
            if c not in df.columns:
                df[c] = pd.Series(dtype=object)
        return df


def _counts(ev_idx: pd.DataFrame, ids, clean_col) -> tuple:
    e = ev_idx.loc[ids]
    return int(len(e)), int(e.cik.nunique()), int(e.decision_date.nunique()), int(e[clean_col].sum())


def _leaderboard(rung: str, enc: str | None, prov: pd.DataFrame, block: str) -> str:
    if enc is None and rung not in ("L8", "L8_ft"):
        return "strict"
    if rung == "L8":
        return "retrospective"      # extraction teacher (Qwen) has no documented cutoff
    key = enc
    r = prov[(prov.model == key) & (prov.block == block)]
    return r.leaderboard.iloc[0] if len(r) else "retrospective"


def build_ladder(ctx, with_emb: bool = False, with_ext: bool = False) -> Ladder:
    ev = C.events(ctx)
    F = C.read(ctx, "S09", "features.parquet")
    tg = C.read(ctx, "S08", "targets.parquet")
    sp = C.read(ctx, "S10", "splits.parquet")
    tx = C.read(ctx, "S09", "event_text.parquet").set_index("event_id")["text"]
    prov = pd.read_csv(ctx.dep("S10") / "fold_provenance.csv")
    embs = {}
    if with_emb:
        embs = load_embeddings(ctx)
        for enc, E in embs.items():
            extra = E.attrs.get("extra")
            if extra is not None and len(extra):
                ex = extra.set_index("event_id")
                for c in [c for c in ex.columns if c.startswith("tone_")]:
                    F = F.merge(ex[[c]].rename(columns={c: f"emb_{enc}_{c}"}), left_on="event_id", right_index=True, how="left")
    if with_ext:
        x = ctx.dep_or_none("S17")
        if x is not None and (x / "ext_features.parquet").exists():
            F = F.merge(pd.read_parquet(x / "ext_features.parquet"), on="event_id", how="left")
    return Ladder(ctx.cfg, ev, F, tg, sp, tx, embs, ctx.out, ctx.log, prov)


def load_embeddings(ctx) -> dict:
    d = ctx.dep_or_none("S14")
    out = {}
    if d is None:
        return out
    for enc_dir in sorted((d / "embeddings").glob("*")):
        f = enc_dir / "vectors.npy"
        if not f.exists():
            continue
        V = np.load(f).astype(np.float32)
        idx = pd.read_parquet(enc_dir / "index.parquet")
        dim = V.shape[1] // 2
        E = pd.DataFrame(V[:, :dim], index=idx.event_id.values)   # mean-pooled half (max half kept on disk)
        ex = pd.read_parquet(enc_dir / "extra.parquet") if (enc_dir / "extra.parquet").exists() else None
        E.attrs["extra"] = ex
        out[enc_dir.name] = E
    return out


def _run_and_record(lad: Ladder, res: Results, preds: list, rung, head, params, lab, ids, cohort, tdef, h, tuned, enc=None,
                    curves=False, task="cls", n_boot=1000, seed=0):
    r = lad.run(rung, head, params, lab, ids, "walk_forward", task, enc, keep=True, curves=curves)
    s = summarize(r, lab, n_boot, seed, task)
    run_id = f"{cohort}|{rung}|{head}|{enc or '-'}|{tdef}|h{h}|{tuned}"
    evx = lad.ev
    if len(r["preds"]):
        n = _counts(evx, r["preds"].event_id, lad.clean_col)
        lb = _leaderboard(rung, enc, lad.prov, "wf0") if lad.prov is not None else "strict"
        res.add(run_id, rung, head, enc or "n/a", lb, cohort, tdef, h, "walk_forward_pooled", "all", s["pooled"], s.get("ci"),
                *n, tuned, {"mean_fold_score": s["mean_fold"], "std_fold_score": s["std_fold"]})
        clean = r["preds"][~r["preds"].event_id.map(evx[lad.clean_col]).astype(bool)]
        if task == "cls" and len(clean) > 30:
            pri = np.bincount(clean.y.astype(int), minlength=3) / len(clean)
            mc = classification_metrics(clean.y.values, clean[["p_down", "p_side", "p_up"]].values, pri, clean.yhat.values)
            res.add(run_id, rung, head, enc or "n/a", lb, cohort, tdef, h, "walk_forward_pooled", "clean", mc, None,
                    *_counts(evx, clean.event_id, lad.clean_col), tuned)
        for fm in r["fold_metrics"]:
            if "skipped" in fm:
                continue
            res.add(run_id, rung, head, enc or "n/a", _leaderboard(rung, enc, lad.prov, f"wf{fm['fold']}") if lad.prov is not None else "strict",
                    cohort, tdef, h, f"wf{fm['fold']}", "all", fm, None, fm.get("n"), None, None, None, tuned)
        p = r["preds"].assign(run_id=run_id, model_id=head, rung=rung, encoder=enc, cohort=cohort, target_def=tdef,
                              horizon=h, tuned=tuned)
        preds.append(p)
    return r, s, run_id


# =====================================================================================================  S11
@stage("S11", "Baselines L0-L4, L6", "CPU", ["S10"], ["tuning", "targets", "labels", "bootstrap", "audit", "seed", "models.pca_dims"],
       outputs=["predictions.parquet", "results_long.parquet", "ladder_summary.csv", "optuna_trials.csv", "curves.json",
                "fit_diagnosis.csv", "best_params.json"],
       assertions=["common cohort recorded", "preprocessing fitted on training rows only"],
       card="Majority, price, metadata, accounting and lexical baselines and the best structured set (L6), default and tuned.")
def s11(ctx):
    cfg = ctx.cfg
    T = cfg["targets"]
    lad = build_ladder(ctx)
    prim = T["primary"]
    lab = lad.labelled(prim["target_def"], prim["horizon"])
    ids = lab.index
    res, preds = Results(), []
    rungs = ["L1", "L2", "L3", "L4", "L4_levels", "L4_change", "L6", "L6_nochange"]
    common = ids[lad.cohort(["L1", "L2", "L3", "L4", "L6"], [None], ids)]
    cohorts = {"common": common}
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    summ, fitd, best_params, trials_all = [], [], {}, []
    _run_and_record(lad, res, preds, "L0", "prior", None, lab, common, "common", prim["target_def"], prim["horizon"], "default",
                    n_boot=nb, seed=seed)
    maj = None
    rr = [r for r in res.rows if r["rung"] == "L0" and r["metric"] == "macro_f1" and r["block"] == "walk_forward_pooled"]
    if rr:
        maj = rr[0]["value"]
    study_db = ctx.storage.hot_root / f"optuna_S11_{ctx.stage_hash}.sqlite3"   # SQLite is unsafe on Drive FUSE
    tuned_rungs = set(cfg["tuning"].get("tuned_rungs") or rungs)
    parent = {"L4_levels": "L4", "L4_change": "L4", "L6_nochange": "L6"}
    for rung in sorted(rungs, key=lambda r: r in parent):        # tuned rungs first; ablation variants inherit
        for head in cfg["tuning"]["heads"]:
            r0, s0, _ = _run_and_record(lad, res, preds, rung, head, None, lab, common, "common", prim["target_def"],
                                        prim["horizon"], "default", curves=(rung == "L6"), n_boot=nb, seed=seed)
            if rung in tuned_rungs:
                n_tr = int(cfg["tuning"]["trials"].get(head, 10))
                bp, tr = lad.tune(rung, head, lab, common, n_tr, int(cfg["tuning"]["timeout_s_per_study"]), study_db,
                                  f"common|{rung}|{head}|{prim['target_def']}|h{prim['horizon']}")
                tr["rung"], tr["head"] = rung, head
                trials_all.append(tr)
                label = "tuned"
            else:
                bp = best_params.get(f"{parent.get(rung, rung)}|{head}", {})
                label = "tuned"                                  # parameters transferred from the parent rung (D-021)
            best_params[f"{rung}|{head}"] = bp
            r1, s1, rid = _run_and_record(lad, res, preds, rung, head, bp, lab, common, "common", prim["target_def"],
                                          prim["horizon"], label, curves=(rung == "L6"), n_boot=nb, seed=seed)
            summ.append({"rung": rung, "head": head, "default_mean_f1": s0["mean_fold"], "tuned_mean_f1": s1["mean_fold"],
                         "tuned_std": s1["std_fold"], "pooled_f1": (s1["pooled"] or {}).get("macro_f1"),
                         "ci_low": (s1.get("ci") or {}).get("ci_low"), "ci_high": (s1.get("ci") or {}).get("ci_high"),
                         "n": s1["n"], "run_id": rid})
            fitd.append({"run_id": rid, "rung": rung, "head": head, "mean_val": s1["mean_fold"], "mean_train": s1["mean_train"],
                         "std": s1["std_fold"], "verdict": fit_verdict(s1["mean_fold"], s1["mean_train"], s1["std_fold"], maj, cfg["audit"])})
            # maximum-available cohort with the tuned parameters
            mx = ids[lad.requirement_mask(rung, None, ids)]
            _run_and_record(lad, res, preds, rung, head, bp, lab, mx, "max_available", prim["target_def"], prim["horizon"],
                            "tuned", n_boot=nb, seed=seed)
    sm = pd.DataFrame(summ)
    sm.to_csv(ctx.out / "ladder_summary.csv", index=False)
    l6 = sm[sm.rung == "L6"].sort_values("tuned_mean_f1", ascending=False)
    best_l6 = l6.iloc[0].to_dict() if len(l6) else None
    write_json(ctx.out / "best_params.json", {"params": best_params, "best_L6": best_l6})
    # complementary T_info model (separate model; never the T_trade backtest signal) + regression heads
    comp = T["complementary"]
    lab_i = lad.labelled(comp["target_def"], comp["horizon"])
    if best_l6:
        for rung in ("L1", "L4", "L6"):
            ci_ids = lab_i.index[lad.cohort(["L1", "L2", "L3", "L4", "L6"], [None], lab_i.index)]
            _run_and_record(lad, res, preds, rung, best_l6["head"], best_params.get(f"{rung}|{best_l6['head']}"), lab_i, ci_ids,
                            "common", comp["target_def"], comp["horizon"], "tuned_transferred", n_boot=nb, seed=seed)
    for tdef, h, col in (("T_trade", 1, "ar_mm"), ("T_trade", 5, "ar_mm"), ("T_trade", 20, "vol_ratio"), ("T_trade", 1, "abs_std")):
        if h not in T["horizons"]:
            continue
        lr = lad.labelled(tdef, h, col)
        rid_ids = lr.index[lad.cohort(["L1", "L2", "L3", "L4", "L6"], [None], lr.index)]
        for rung in ("L0", "L1", "L6"):
            r, s, rid = _run_and_record(lad, res, preds, rung, "lightgbm", None, lr, rid_ids, "common", f"{tdef}:{col}", h,
                                        "default", task="reg", n_boot=nb, seed=seed)
    # learning curve + adversarial validation + permutation null + random-split demo (L6 best)
    diag = {}
    if best_l6:
        head, bp = best_l6["head"], best_params.get(f"L6|{best_l6['head']}")
        folds = lad.folds("walk_forward")
        lc = []
        if folds:
            fl = folds[-1]
            tr, te = lad.fold_ids("walk_forward", fl, common)
            trs = lab.loc[tr].sort_values("date")
            for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                keep = trs.index[int(len(trs) * (1 - frac)):]
                sub = keep.append(te)
                r = lad.run("L6", head, bp, lab, sub, "walk_forward", keep=False, folds=[fl])
                lc.append({"train_fraction": frac, "n_train": len(keep), "val_macro_f1": r["mean_score"],
                           "train_macro_f1": r["mean_train_score"]})
        pd.DataFrame(lc).to_csv(ctx.out / "learning_curves.csv", index=False)
        diag["adversarial"] = adversarial(lad, common, lab)
        pd.DataFrame(diag["adversarial"]).to_csv(ctx.out / "adversarial.csv", index=False)
        nperm = int(get(cfg, "tuning.permutations", 10))
        pm = permutation_null(lad, head, bp, lab, common, nperm, seed)
        pd.DataFrame(pm).to_csv(ctx.out / "permutation.csv", index=False)
        rd = lad.run("L6", head, bp, lab, common, "random_demo", keep=False)
        write_json(ctx.out / "random_demo.json", {"random_split_macro_f1": rd["mean_score"], "walk_forward_macro_f1": best_l6["tuned_mean_f1"],
                                                  "note": "pedagogical only"})
        if folds:
            tr, te = lad.fold_ids("walk_forward", folds[-1], common)
            if len(tr) > 100:
                from .models.heads import enet_certificate
                Xtr, _, _ = lad.matrices("L6", tr, [te[:1]])
                n = min(len(tr), 20000)
                cert = enet_certificate(Xtr[-n:], lab.loc[tr, "y"].to_numpy()[-n:], lab.loc[tr, "w"].to_numpy()[-n:],
                                        best_params.get("L6|enet") or {"C": 0.1, "l1_ratio": 0.5})
                write_json(ctx.out / "enet_certificate.json", cert)
    rl = res.frame()
    rl["config_counter"] = lad.config_counter
    ctx.write(rl, "results_long.parquet")
    ctx.write(pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), "predictions.parquet")
    pd.concat(trials_all, ignore_index=True).to_csv(ctx.out / "optuna_trials.csv", index=False) if trials_all else None
    (ctx.out / "curves.json").write_text(json.dumps(lad.curves[:200], default=float), encoding="utf-8")
    pd.DataFrame(fitd).to_csv(ctx.out / "fit_diagnosis.csv", index=False)
    if study_db.exists():
        import shutil
        shutil.copy2(study_db, ctx.out / "optuna.sqlite3")
    write_json(ctx.out / "cohorts.json", {"common_primary_n": int(len(common)), "labelled_primary_n": int(len(ids))})
    return {"common_cohort": int(len(common)), "labelled": int(len(ids)), "best_L6": best_l6, "configs_tried": lad.config_counter}


def adversarial(lad: Ladder, ids, lab) -> list[dict]:
    """Per fold: can a classifier tell training rows from validation rows? (AUC near 0.5 does not prove no shift.)"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    out = []
    for f in lad.folds("walk_forward"):
        tr, te = lad.fold_ids("walk_forward", f, ids)
        if len(tr) < 100 or len(te) < 50:
            continue
        X, _, names = lad.matrices("L6", tr, [te[:1]])
        Xt, _, _ = lad.matrices("L6", te, [te[:1]])
        Xa = np.vstack([X, Xt])
        ya = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
        from sklearn.metrics import roc_curve
        aucs = {}
        for tag, drop in (("all_features", lambda n: n == "meta_year"),
                          ("firm_level_only", lambda n: n.startswith("mkt_") or n in ("meta_year", "meta_month", "meta_dow"))):
            keep = [i for i, n in enumerate(names) if not drop(n)]
            Xk = Xa[:, keep]
            pr = np.zeros(len(ya))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(Xk, ya):
                m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, verbose=-1).fit(Xk[a], ya[a])
                pr[b] = m.predict_proba(Xk[b])[:, 1]
            aucs[tag] = (float(roc_auc_score(ya, pr)), pr)
            if tag == "firm_level_only":
                # which features separate the periods (gain share of a fit on all rows of this fold)
                g = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, verbose=-1).fit(Xk, ya)
                gain = g.booster_.feature_importance("gain")
                top = np.argsort(-gain)[:5]
                drivers = [[names[keep[i]], round(float(gain[i] / max(gain.sum(), 1e-12)), 3)] for i in top]
        pr = aucs["firm_level_only"][1]
        fpr, tpr, _ = roc_curve(ya, pr)
        out.append({"fold": f, "auc": aucs["firm_level_only"][0], "auc_all_features": aucs["all_features"][0],
                    "top_shift_features": json.dumps(drivers), "n_train": len(X), "n_val": len(Xt),
                    "fpr": json.dumps(fpr[:: max(1, len(fpr) // 50)].round(4).tolist()),
                    "tpr": json.dumps(tpr[:: max(1, len(tpr) // 50)].round(4).tolist())})
    return out


def permutation_null(lad: Ladder, head, params, lab, ids, n_perm, seed) -> list[dict]:
    """Labels permuted within month blocks (class counts per month preserved), refit, score: null of macro-F1."""
    rng = np.random.default_rng(seed)
    out = []
    obs = lad.run("L6", head, params, lab, ids, "walk_forward", keep=False)["mean_score"]
    out.append({"perm": -1, "macro_f1": obs, "kind": "observed"})
    month = lab["date"].dt.to_period("M")
    for k in range(n_perm):
        lp = lab.copy()
        for _, g in lp.groupby(month):
            lp.loc[g.index, "y"] = rng.permutation(g["y"].to_numpy())
        out.append({"perm": k, "macro_f1": lad.run("L6", head, params, lp, ids, "walk_forward", keep=False)["mean_score"],
                    "kind": "permuted"})
    return out


# =====================================================================================================  S13
@stage("S13", "Preflight (GPU)", "GPU", ["S10"], ["models", "text", "budget", "rag", "distill", "tuning.finetune_grid"],
       outputs=["preflight.json"], assertions=["budgets OK (full refuses when exceeded)"],
       card="Workload, storage and GPU-time projection from a timed micro-benchmark.")
def s13(ctx):
    from .models.embeddings import benchmark, chunk_words, device_and_dtype, load_encoder, select_chunks, unload
    cfg = ctx.cfg
    ev = C.events(ctx)
    si = C.read(ctx, "S05", "sections_index.parquet")
    est_chunks = int(np.minimum(np.ceil(si.n_tokens_est / 1.3 / max(1, cfg["text"]["chunk_words"] - cfg["text"]["chunk_overlap_words"])),
                                cfg["text"]["max_chunks_per_section"]).sum()) if len(si) else 0
    encs = list(cfg["models"]["encoders"])
    dim, nb = 768, 2
    storage_gb = len(ev) * dim * 2 * nb * len(encs) / 1e9 * 1.1
    from huggingface_hub import HfApi
    api = HfApi()
    rost = C.roster(cfg)
    sizes = {}
    for k, v in {**rost["encoders"], **rost["reranker"], **rost["llm"], **rost["student"]}.items():
        mid = v.get("model_id") or v.get("model_id_template", "").format(year=2010)
        try:
            info = api.model_info(mid, files_metadata=True)
            sizes[k] = round(sum((s.size or 0) for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin"))) / 1e9, 2)
        except Exception as e:
            sizes[k] = f"unknown ({type(e).__name__})"
    dev, _ = device_and_dtype()
    rate = None
    try:
        b = rost["encoders"]["bge_base"]
        tok, mdl, d = load_encoder(b["model_id"], b["revision"], "base")
        rate = benchmark(tok, mdl, d, 256 if dev == "cuda" else 32)
        unload(mdl)
    except Exception as e:
        ctx.warn(f"micro-benchmark failed: {e}")
    n_encode = est_chunks * (len(encs))
    gpu_h_b = (n_encode / rate / 3600) if rate else None
    llm_prompts = int(cfg["budget"]["llm_teacher_filings"]) + int(cfg["budget"]["llm_gold_filings"])
    gpu_h_c = llm_prompts * 20 / 3600 if dev == "cuda" else None     # ~20 s per filing prompt (estimate, logged)
    rep = {"device": dev, "firms": int(ev.cik.nunique()), "events": int(len(ev)), "sections": int(len(si)),
           "est_chunks": est_chunks, "est_tokens": int(si.n_tokens_est.sum()) if len(si) else 0,
           "embedding_storage_gb": round(storage_gb, 2), "model_sizes_gb": sizes, "llm_prompts": llm_prompts,
           "disk_free_gb": {"data_root": C.disk_free_gb(ctx.storage.data_root), "hot_root": C.disk_free_gb(ctx.storage.hot_root)},
           "bench_chunks_per_s": rate, "projected_gpu_hours_encoders": gpu_h_b, "projected_gpu_hours_llm": gpu_h_c,
           "budget": cfg["budget"]}
    write_json(ctx.out / "preflight.json", rep)
    if cfg["run_profile"] == "full":
        if storage_gb > float(cfg["budget"]["max_storage_gb"]):
            raise RuntimeError(f"preflight: embedding storage {storage_gb:.1f} GB > budget")
        if gpu_h_b is not None and gpu_h_b > float(cfg["budget"]["max_gpu_hours_b"]):
            raise RuntimeError(f"preflight: projected encoder GPU hours {gpu_h_b:.1f} > budget.max_gpu_hours_b")
    return rep


# =====================================================================================================  S14
@stage("S14", "Core embeddings + vintage ChronoBERT", "GPU", ["S10"], ["models", "text", "seed"],
       outputs=["embeddings/<encoder>/vectors.npy", "index.parquet", "extra.parquet"],
       assertions=["pooling + change vectors", "revisions pinned", "one model in memory at a time"],
       card="Frozen encoder vectors per event (mean and max pooled), FinBERT native tone, change cosine to the prior same-type event.")
def s14(ctx):
    from .models.embeddings import change_cosine, chrono_vintage, encode, event_chunks, load_encoder, pool_events, unload
    cfg = ctx.cfg
    rost = C.roster(cfg)
    ev = C.events(ctx)
    ev = ev[pd.to_datetime(ev.decision_date) >= pd.Timestamp(cfg["study"]["start"])]
    store = _chunk_store(ctx, ev)                    # one pass over the firm files -> chunks partitioned by event year
    out_root = ctx.out / "embeddings"
    stats = {"chunk_store": str(store)}
    years = sorted(int(p.name.split("=")[1]) for p in store.glob("year=*"))
    for enc in cfg["models"]["encoders"]:
        spec = rost["encoders"][enc]
        d = out_root / enc
        if (d / "vectors.npy").exists():
            stats[enc] = "exists"
            continue
        d.mkdir(parents=True, exist_ok=True)
        ids_all, V_all, X_all = [], [], []
        t0 = time.time()
        loaded, mdl = None, None
        for y in years:                               # years in order: ChronoBERT loads each vintage once
            vint = chrono_vintage(y, spec["vintages"]) if enc == "chrono_bert" else None
            mid = spec["model_id_template"].format(year=vint) if vint else spec["model_id"]
            rev = (spec.get("revisions") or {}).get(str(vint)) if vint else spec.get("revision")
            if not rev:
                raise RuntimeError(f"{mid}: no pinned revision in configs/models.yaml (hard rule 12)")
            if loaded != mid:
                if mdl is not None:
                    unload(mdl)
                tok, mdl, dev = load_encoder(mid, rev, "classifier" if spec.get("native_outputs") else "base")
                loaded = mid
            ch = pd.read_parquet(store / f"year={y}")
            if ch.empty:
                continue
            sizes = ch.groupby("event_id", sort=False).size()
            slice_of = (sizes.cumsum() - sizes) // 40000          # whole events per slice (bounded memory)
            for _, part in ch.groupby(ch.event_id.map(slice_of), sort=True):
                V, P = encode(tok, mdl, dev, part.text.tolist(), int(spec.get("max_len", 512)), int(cfg["models"]["embed_batch"]),
                              spec.get("pooling", "mean"), classifier=bool(spec.get("native_outputs")), log=ctx.log)
                ids, M, extra = pool_events(part, V, P)
                extra["vintage"], extra["model_id"] = vint, mid
                ids_all.append(ids)
                V_all.append(M)
                X_all.append(extra)
            ctx.log.info(f"S14 {enc} year {y}: {len(ch):,} chunks ({time.time() - t0:.0f}s)")
        if mdl is not None:
            unload(mdl)
        if not V_all:
            continue
        ids = pd.Index(np.concatenate([np.asarray(i) for i in ids_all]), name="event_id")
        M = np.vstack(V_all)
        ex = pd.concat(X_all, ignore_index=True)
        # an event split across slices is pooled per slice: keep the first occurrence only (rare; logged)
        dup = ids.duplicated()
        if dup.any():
            ctx.warn(f"{enc}: {int(dup.sum())} events split across slices; first slice kept")
            ids, M, ex = ids[~dup], M[~dup], ex[~ex.event_id.duplicated()]
        dim = M.shape[1] // 2
        cc = change_cosine(ev, ids, M, dim)
        ex["chg_emb_cosine"] = ex.event_id.map(cc)
        np.save(d / "vectors.npy", M.astype(np.float16))
        pd.DataFrame({"event_id": ids}).to_parquet(d / "index.parquet", index=False)
        ex.to_parquet(d / "extra.parquet", index=False)
        stats[enc] = {"events": int(len(ids)), "dim": int(M.shape[1]), "seconds": round(time.time() - t0, 1)}
        ctx.log.info(f"S14 {enc}: {stats[enc]}")
    return stats


def _chunk_store(ctx, ev: pd.DataFrame) -> Path:
    """Masked, section-aware chunks for every event, written once per firm into year partitions on fast local disk."""
    from .models.embeddings import event_chunks
    store = ctx.storage.hot_root / f"s14_chunks_{ctx.stage_hash}"
    done = store / "_DONE"
    if done.exists():
        return store
    si_dir = ctx.dep("S05") / "sections"
    for cik, g in ev.groupby("cik"):
        p = si_dir / f"{int(cik):010d}.parquet"
        if not p.exists():
            continue
        ch = event_chunks(g, pd.read_parquet(p, columns=["accession", "section_id", "text_masked"]), ctx.cfg)
        if ch.empty:
            continue
        ch["year"] = ch.event_id.map(g.set_index("event_id")["year"]).astype(int)
        for y, gy in ch.groupby("year"):
            q = store / f"year={int(y)}"
            q.mkdir(parents=True, exist_ok=True)
            gy.drop(columns="year").to_parquet(q / f"{int(cik):010d}.parquet", index=False)
    store.mkdir(parents=True, exist_ok=True)
    done.write_text("ok")
    return store


# =====================================================================================================  S15
@stage("S15", "L5 / L7 heads + tuning; G3; fine-tune top 2", "GPU", ["S11", "S14"],
       ["tuning", "gates", "models", "bootstrap", "seed"],
       outputs=["results_long.parquet", "predictions.parquet", "g3.json", "h4.csv", "ft_curves.json"],
       assertions=["finalist caps", "default and tuned results", "fine-tune only if G3 passes (or labelled owner override)"],
       card="Embedding-only (L5) and structured + embedding (L7) heads per encoder; G3 decision; fine-tuning curves.")
def s15(ctx):
    cfg = ctx.cfg
    lad = build_ladder(ctx, with_emb=True)
    prim = cfg["targets"]["primary"]
    lab = lad.labelled(prim["target_def"], prim["horizon"])
    encs = [e for e in cfg["models"]["encoders"] if e in lad.embs]
    ids = lab.index
    common = ids[lad.cohort(["L1", "L2", "L3", "L4", "L6", "L5", "L7"], encs, ids)]
    res, preds = Results(), []
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    s11 = json.loads((ctx.dep("S11") / "best_params.json").read_text())
    bp6 = s11["params"]
    summ = []
    db = ctx.storage.hot_root / f"optuna_S15_{ctx.stage_hash}.sqlite3"
    # re-score L6 on this (embedding-inclusive) common cohort with its tuned parameters: identical rows for the lift
    l6runs = {}
    for head in cfg["tuning"]["heads"]:
        r, s, rid = _run_and_record(lad, res, preds, "L6", head, bp6.get(f"L6|{head}"), lab, common, "common_emb", prim["target_def"],
                                    prim["horizon"], "tuned", n_boot=nb, seed=seed)
        l6runs[head] = (r, s, rid)
    for enc in encs:
        for rung in ("L5", "L7"):
            for head in cfg["tuning"]["heads"]:
                r0, s0, _ = _run_and_record(lad, res, preds, rung, head, None, lab, common, "common_emb", prim["target_def"],
                                            prim["horizon"], "default", enc=enc, curves=(rung == "L7"), n_boot=nb, seed=seed)
                bp, _ = lad.tune(rung, head, lab, common, int(cfg["tuning"]["trials"].get(head, 10)),
                                 int(cfg["tuning"]["timeout_s_per_study"]), db, f"{rung}|{head}|{enc}", enc=enc)
                r1, s1, rid = _run_and_record(lad, res, preds, rung, head, bp, lab, common, "common_emb", prim["target_def"],
                                              prim["horizon"], "tuned", enc=enc, n_boot=nb, seed=seed)
                summ.append({"rung": rung, "encoder": enc, "head": head, "params": json.dumps(bp), "default_mean_f1": s0["mean_fold"],
                             "tuned_mean_f1": s1["mean_fold"], "tuned_std": s1["std_fold"], "run_id": rid,
                             "pooled_f1": (s1["pooled"] or {}).get("macro_f1")})
    sm = pd.DataFrame(summ)
    sm.to_csv(ctx.out / "emb_ladder_summary.csv", index=False)
    if db.exists():
        import shutil
        shutil.copy2(db, ctx.out / "optuna.sqlite3")
    pr = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    # G3: best L7 vs L6 (same head, identical rows), paired month-block bootstrap
    g3 = {"passed": False, "reason": "no L7 runs"}
    if len(sm):
        b7 = sm[sm.rung == "L7"].sort_values("tuned_mean_f1", ascending=False).iloc[0]
        p7 = pr[pr.run_id == b7.run_id].set_index("event_id")
        p6 = pr[pr.run_id == l6runs[b7["head"]][2]].set_index("event_id")
        j = p7.index.intersection(p6.index)
        d = paired_macro_f1_diff(p7.loc[j, "y"].values, p7.loc[j, "yhat"].values, p6.loc[j, "yhat"].values,
                                 p7.loc[j, "date"].values, nb, seed)
        passed = d["value"] >= float(cfg["gates"]["text_lift"]) and d["ci_low"] > 0
        g3 = {"passed": bool(passed), "lift": d, "best_L7": b7.to_dict(), "n_events": int(len(j)),
              "rule": f"lift >= {cfg['gates']['text_lift']} macro-F1 and CI excludes 0"}
    write_json(ctx.out / "g3.json", g3)
    # H4: ChronoBERT vintage vs each static encoder, identical rows and heads
    h4 = []
    if "chrono_bert" in encs and len(sm):
        for rung in ("L5", "L7"):
            for head in cfg["tuning"]["heads"]:
                a = sm[(sm.rung == rung) & (sm.encoder == "chrono_bert") & (sm["head"] == head)]
                if a.empty:
                    continue
                pa = pr[pr.run_id == a.run_id.iloc[0]].set_index("event_id")
                for enc in encs:
                    if enc == "chrono_bert":
                        continue
                    b = sm[(sm.rung == rung) & (sm.encoder == enc) & (sm["head"] == head)]
                    if b.empty:
                        continue
                    pb = pr[pr.run_id == b.run_id.iloc[0]].set_index("event_id")
                    j = pa.index.intersection(pb.index)
                    d = paired_macro_f1_diff(pa.loc[j, "y"].values, pa.loc[j, "yhat"].values, pb.loc[j, "yhat"].values,
                                             pa.loc[j, "date"].values, nb, seed)
                    h4.append({"rung": rung, "head": head, "vintage": "chrono_bert", "static": enc, "diff_vintage_minus_static": d["value"],
                               "ci_low": d["ci_low"], "ci_high": d["ci_high"], "p_boot": d["p_boot"], "n": int(len(j))})
    pd.DataFrame(h4).to_csv(ctx.out / "h4.csv", index=False)
    # fine-tuning: top-2 encoders by L5 walk-forward F1, only if G3 passes (or owner override, labelled exploratory)
    ft_info = {"ran": False}
    override = bool(cfg["gates"].get("owner_override_g3_finetune"))
    if (g3.get("passed") or override) and len(sm):
        top2 = (sm[sm.rung == "L5"].groupby("encoder").tuned_mean_f1.max().sort_values(ascending=False).index[:2].tolist())
        ft_info = finetune_encoders(ctx, lad, lab, common, top2, res, preds, label="G3 passed" if g3.get("passed") else
                                    "EXPLORATORY: owner override, G3 not passed")
    write_json(ctx.out / "finetune_status.json", ft_info)
    rl = res.frame()
    rl["config_counter"] = lad.config_counter
    ctx.write(rl, "results_long.parquet")
    ctx.write(pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), "predictions.parquet")
    (ctx.out / "curves.json").write_text(json.dumps(lad.curves[:200], default=float), encoding="utf-8")
    return {"common_cohort": int(len(common)), "encoders": encs, "g3": {k: v for k, v in g3.items() if k != "best_L7"},
            "finetune": ft_info.get("ran")}


def finetune_encoders(ctx, lad: Ladder, lab, ids, encs, res: Results, preds: list, label: str) -> dict:
    from .models.embeddings import chrono_vintage
    from .models.finetune import grid, train_eval
    cfg = ctx.cfg
    rost = C.roster(cfg)
    folds = lad.folds("walk_forward")
    texts = lad.texts
    cap = int(cfg["tuning"]["finetune_max_train_events"])
    ml = int(cfg["models"]["max_seq_len"])
    seed = int(cfg["seed"])
    curves, info = [], {"ran": True, "label": label, "encoders": encs, "grid": {}}
    blocks = pd.read_csv(ctx.dep("S10") / "blocks.csv", parse_dates=["start"])

    def mid_for(enc, block_start):
        sp = rost["encoders"][enc]
        if enc == "chrono_bert":
            v = chrono_vintage(pd.Timestamp(block_start).year, sp["vintages"])
            return sp["model_id_template"].format(year=v), (sp.get("revisions") or {}).get(str(v))
        return sp["model_id"], sp.get("revision")

    def one(enc, scheme, fold, hp, keep=True):
        tr, te = lad.fold_ids(scheme, fold, ids)
        trs = lab.loc[tr].sort_values("date").iloc[-cap:]
        cut = int(len(trs) * 0.9)
        itr, iva = trs.index[:cut], trs.index[cut:]
        bname = {"walk_forward": f"wf{fold}", "dev_holdout": "dev", "locked_test": "locked"}[scheme]
        bs = blocks.loc[blocks.block == bname, "start"]
        mid, rev = mid_for(enc, bs.iloc[0] if len(bs) else lab.date.max())
        out = train_eval(mid, rev, texts.reindex(itr).fillna("").tolist(), lab.loc[itr, "y"].astype(int).to_numpy(),
                         lab.loc[itr, "w"].to_numpy(), texts.reindex(iva).fillna("").tolist(),
                         lab.loc[iva, "y"].astype(int).to_numpy(), [texts.reindex(te).fillna("").tolist()], hp, ml, seed, ctx.log)
        prior = np.bincount(lab.loc[itr, "y"].astype(int), minlength=3) / max(1, len(itr))
        from .models.heads import prior_correct
        P = prior_correct(out["preds"][0], prior)       # class-weighted CE -> true-frequency probabilities
        yhat = decide(P, prior)
        m = classification_metrics(lab.loc[te, "y"].to_numpy(), P, prior, yhat)
        return out, m, te, P, yhat

    for enc in encs:
        if not folds:
            break
        g = grid(cfg)
        best = None
        for hp in g:
            out, m, te, P, yhat = one(enc, "walk_forward", folds[-1], hp)
            curves.append({"label": f"{enc} lr{hp['lr']} b{hp['batch']} wd{hp['weight_decay']}", "curve": out["curve"],
                           "val_macro_f1": m.get("macro_f1"), "best_step": out["best_step"]})
            if best is None or (m.get("macro_f1") or 0) > (best[1].get("macro_f1") or 0):
                best = (hp, m)
        info["grid"][enc] = {"trials": len(g), "best_hp": best[0], "best_val_macro_f1": best[1].get("macro_f1")}
        rid = f"common_emb|L8_ft|ft|{enc}|{cfg['targets']['primary']['target_def']}|h{cfg['targets']['primary']['horizon']}|tuned"
        allp = []
        for f in folds:
            out, m, te, P, yhat = one(enc, "walk_forward", f, best[0])
            allp.append(pd.DataFrame({"event_id": te, "fold": f, "y": lab.loc[te, "y"].values, "date": lab.loc[te, "date"].values,
                                      "p_down": P[:, 0], "p_side": P[:, 1], "p_up": P[:, 2], "yhat": yhat}))
            res.add(rid, "L8_ft", "ft", enc, _leaderboard("L8_ft", enc, lad.prov, f"wf{f}"), "common_emb",
                    cfg["targets"]["primary"]["target_def"], cfg["targets"]["primary"]["horizon"], f"wf{f}", "all", m, None,
                    m.get("n"), None, None, None, "tuned", {"label": label})
        p = pd.concat(allp, ignore_index=True)
        prior = np.bincount(p.y.astype(int), minlength=3) / len(p)
        pm = classification_metrics(p.y.values, p[["p_down", "p_side", "p_up"]].values, prior, p.yhat.values)
        ci = macro_f1_ci(p.y.values, p.yhat.values, p.date.values, int(cfg["bootstrap"]["n"]), seed)
        res.add(rid, "L8_ft", "ft", enc, "retrospective" if enc != "chrono_bert" else "strict", "common_emb",
                cfg["targets"]["primary"]["target_def"], cfg["targets"]["primary"]["horizon"], "walk_forward_pooled", "all", pm, ci,
                len(p), None, None, None, "tuned", {"label": label})
        preds.append(p.assign(run_id=rid, model_id="ft", rung="L8_ft", encoder=enc, cohort="common_emb",
                              target_def=cfg["targets"]["primary"]["target_def"], horizon=cfg["targets"]["primary"]["horizon"], tuned="tuned"))
        # development-holdout predictions: trained on all pre-holdout data with the selected grid point (used by S18)
        out, m, te, P, yhat = one(enc, "dev_holdout", 0, best[0])
        if len(te):
            dp = pd.DataFrame({"event_id": te, "fold": 0, "y": lab.loc[te, "y"].values, "date": lab.loc[te, "date"].values,
                               "p_down": P[:, 0], "p_side": P[:, 1], "p_up": P[:, 2], "yhat": yhat})
            preds.append(dp.assign(run_id=f"dev|L8_ft|{enc}", model_id="ft", rung="L8_ft", encoder=enc, cohort="common_emb",
                                   target_def=cfg["targets"]["primary"]["target_def"],
                                   horizon=cfg["targets"]["primary"]["horizon"], tuned="tuned"))
            info["grid"][enc]["dev_macro_f1"] = m.get("macro_f1")
            info["grid"][enc]["dev_curve_best_step"] = out["best_step"]
    (ctx.out / "ft_curves.json").write_text(json.dumps(curves, default=float), encoding="utf-8")
    return info


# =====================================================================================================  S16
@stage("S16", "Core RAG: retrieval eval, extraction (gold-val), freeze", "GPU", ["S12", "S14"], ["rag", "budget", "seed"],
       outputs=["rag_tournament.csv", "g3b.json", "rag_frozen.json"],
       assertions=["historical path time-filtered", "current path separate"],
       card="BM25 vs hybrid vs hybrid+rerank on gold-val qrels; Qwen extraction scored on gold-val; G3b decision.")
def s16(ctx):
    from .rag.core import Retriever, historical_candidates, ir_metrics, retrieval_guard
    from .rag.extract import MockGenerator, ResponseCache, build_prompt, cache_key, field_schema, guard
    cfg = ctx.cfg
    assert retrieval_guard(), "historical retrieval filter failed its self-test"
    qb = C.question_bank(cfg)
    gold_dir = ctx.dep("S12")
    sample = pd.read_parquet(gold_dir / "gold_sample.parquet")
    labels = pd.read_parquet(gold_dir / "gold_labels.parquet") if (gold_dir / "gold_labels.parquet").exists() else pd.DataFrame()
    have_val = len(labels) and (labels.gold_split == "gold_val").any()
    gen, gen_name = _generator(ctx, qb)
    cache = ResponseCache(ctx.out / "llm_cache.jsonl")
    ev = C.events(ctx).set_index("event_id")
    chunks = _gold_chunks(ctx, sample)
    rows, status = [], {}
    setups = ["bm25", "hybrid", "hybrid_rerank"]
    dense, rerank = _dense_and_reranker(ctx) if gen_name != "mock" else (None, None)
    prefill = {}
    ret = Retriever("hybrid_rerank" if rerank else "bm25", dense, rerank, cfg["rag"]["bm25_k1"], cfg["rag"]["bm25_b"],
                    cfg["rag"]["rrf_k"], cfg["rag"]["rerank_top_n"])
    q = " ".join(f["question"] for f in qb["fields"].values())
    todo, pmap = [], {}
    for split in ("gold_dev", "gold_val"):                  # LLM pre-fill: gold-dev / gold-val only (never gold-test)
        for eid in sample[sample.gold_split == split].event_id:
            c = chunks[chunks.event_id == eid]
            if c.empty:
                continue
            top = ret.rank(q, c, int(cfg["rag"]["top_k"]))
            passages = [{"chunk_id": r.chunk_id, "text": r.text} for r in top.itertuples(index=False)]
            k = cache_key(getattr(gen, "revision", gen_name), build_prompt(qb, passages), passages)
            pmap[eid] = (k, passages)
            if cache.get(k) is None:
                todo.append(eid)
    t_gen = time.time()
    for b0 in range(0, len(todo), 64):
        ids_b = todo[b0:b0 + 64]
        for e, raw in zip(ids_b, gen.batch(qb, [pmap[e][1] for e in ids_b])):
            cache.put(pmap[e][0], raw)
    for eid, (k, passages) in pmap.items():
        clean, _flags = guard(cache.get(k) or {}, qb, passages, field_schema(qb))
        prefill[eid] = clean
    gen_seconds = time.time() - t_gen
    if cfg["rag"].get("prefill_dev_val", True):
        from .gold.tools import export_templates
        export_templates(sample, qb, None, ctx.out / "prefilled_templates", prefill=prefill)
    if have_val:
        # retrieval tournament on gold-val qrels (evidence chunk ids) + extraction F1 on gold-val
        val = labels[labels.gold_split == "gold_val"]
        for setup in setups:
            if setup != "bm25" and dense is None:
                rows.append({"setup": setup, "status": "not run (dense model unavailable on CPU)"})
                continue
            ret = Retriever(setup, dense, rerank)
            ms = []
            for (eid, field), g in val[val.evidence_chunk_id != ""].groupby(["event_id", "field"]):
                c = chunks[chunks.event_id == eid]
                top = ret.rank(qb["fields"][field]["question"], c, 10)
                ms.append(ir_metrics(top.chunk_id.tolist(), set(g.evidence_chunk_id), 10))
            d = pd.DataFrame(ms)
            rows.append({"setup": setup, "n_queries": len(d), **(d.mean(numeric_only=True).to_dict() if len(d) else {})})
        ext = []
        for (eid, field), g in val.groupby(["event_id", "field"]):
            pred = (prefill.get(eid) or {}).get(field) or {}
            ext.append({"field": field, "y": bool(g.present_b.iloc[0]), "p": bool(pred.get("present") or False)})
        e = pd.DataFrame(ext)
        from sklearn.metrics import f1_score
        f1 = float(f1_score(e.y, e.p)) if len(e) and e.y.any() else None
        status = {"gold_val_extraction_f1": f1, "passed": bool(f1 is not None and f1 >= float(cfg["gates"]["extraction_min_f1"])),
                  "generator": gen_name}
    else:
        rows.append({"setup": "all", "status": "not evaluable: no gold-val labels imported (annotate and re-run S12/S16)"})
        status = {"passed": False, "reason": "no gold-val labels", "generator": gen_name}
    pd.DataFrame(rows).to_csv(ctx.out / "rag_tournament.csv", index=False)
    write_json(ctx.out / "g3b.json", status)
    write_json(ctx.out / "rag_frozen.json", {"setup": "hybrid_rerank" if rerank else "bm25", "generator": gen_name,
                                             "top_k": cfg["rag"]["top_k"], "frozen": bool(status.get("passed"))})
    return {"g3b": status, "prefilled_events": len(prefill), "generation_seconds": round(gen_seconds, 1),
            "seconds_per_filing": round(gen_seconds / max(1, len(todo)), 2)}


def _generator(ctx, qb):
    from .rag.extract import MockGenerator, QwenGenerator, field_schema
    import torch
    if ctx.services.get("mock_llm") or not torch.cuda.is_available():
        return MockGenerator(), "mock"
    q = C.roster(ctx.cfg)["llm"]["qwen"]
    return QwenGenerator(q["model_id"], q["revision"], field_schema(qb), int(ctx.cfg["rag"]["max_new_tokens"]), ctx.log), "qwen2.5-7b"


def _dense_and_reranker(ctx):
    import torch
    from .models.embeddings import encode, load_encoder
    rost = C.roster(ctx.cfg)
    b = rost["encoders"]["bge_base"]
    tok, mdl, dev = load_encoder(b["model_id"], b["revision"], "base")
    inst = "Represent this sentence for searching relevant passages: "

    def dense(q, texts):
        V, _ = encode(tok, mdl, dev, [inst + q] + texts, 512, 64, "cls")
        V = V.astype(np.float32)
        return V[1:] @ V[0]

    rr = rost["reranker"]["bge_reranker"]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rt = AutoTokenizer.from_pretrained(rr["model_id"], revision=rr["revision"])
    rm = AutoModelForSequenceClassification.from_pretrained(rr["model_id"], revision=rr["revision"]).to(dev).eval()

    def rerank(q, texts):
        with torch.no_grad():
            e = rt([q] * len(texts), texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            return rm(**e).logits.view(-1).float().cpu().numpy()

    return dense, rerank


def _gold_chunks(ctx, sample: pd.DataFrame) -> pd.DataFrame:
    from .models.embeddings import event_chunks
    ev = C.events(ctx)
    g = ev[ev.event_id.isin(sample.event_id)]
    secs = []
    for cik in g.cik.unique():
        p = ctx.dep("S05") / "sections" / f"{int(cik):010d}.parquet"
        if p.exists():
            secs.append(pd.read_parquet(p, columns=["accession", "section_id", "text_masked"]))
    if not secs:
        return pd.DataFrame(columns=["event_id", "section_id", "chunk_no", "text", "chunk_id"])
    ch = event_chunks(g, pd.concat(secs, ignore_index=True), ctx.cfg)
    ch["chunk_id"] = ch.event_id + ":" + ch.section_id + ":" + ch.chunk_no.astype(str)
    return ch


# =====================================================================================================  S17
@stage("S17", "Teacher sample + student distillation -> L8", "GPU", ["S16", "S15"], ["distill", "rag", "budget", "seed"],
       outputs=["extractions.parquet", "ext_features.parquet", "student_report.json"],
       assertions=["training-period only"],
       card="Teacher (frozen RAG + Qwen) on a training-period sample; ModernBERT student applied to all events; L8 = L7 + extraction.")
def s17(ctx):
    cfg = ctx.cfg
    g3b = json.loads((ctx.dep("S16") / "g3b.json").read_text())
    if not (g3b.get("passed") or cfg["rag"].get("run_extraction_without_gold")):
        write_json(ctx.out / "student_report.json", {"ran": False, "reason": "G3b not passed (no gold-val evidence); set "
                                                     "rag.run_extraction_without_gold: true to run as exploratory"})
        ctx.write(pd.DataFrame(columns=["event_id"]), "ext_features.parquet")
        return {"ran": False}
    from .rag.distill import run_distillation
    info = run_distillation(ctx, exploratory=not g3b.get("passed"))
    return info


# =====================================================================================================  S18
@stage("S18", "Development holdout + diagnostics (MCS / PBO) + G4 freeze", "CPU", ["S15"],
       ["tuning", "bootstrap", "audit", "extensions", "seed"],
       outputs=["dev_results.parquet", "frozen_architecture.json", "mcs_pbo.json", "review_triggers.json"],
       assertions=["no test access", "prereg hash present"],
       card="Finalists retrained on all pre-holdout data and scored on 2020-2022; MCS/PBO diagnostics; G4 manifest freeze.")
def s18(ctx):
    from .evaluation.mcs_pbo import model_confidence_set, pbo_cscv
    cfg = ctx.cfg
    ph = ctx.storage.manifests / "prereg_hash.txt"
    if not ph.exists():
        raise RuntimeError("pre-registration not frozen: run secprice prereg-freeze (notebook cell 'Freeze pre-registration') first")
    lad = build_ladder(ctx, with_emb=True, with_ext=True)
    prim = cfg["targets"]["primary"]
    lab = lad.labelled(prim["target_def"], prim["horizon"])
    encs = list(lad.embs)
    common = lab.index[lad.cohort(["L1", "L2", "L3", "L4", "L6", "L5", "L7"], encs, lab.index)] if encs else \
        lab.index[lad.cohort(["L1", "L2", "L3", "L4", "L6"], [None], lab.index)]
    s11 = json.loads((ctx.dep("S11") / "best_params.json").read_text())
    s15sum = pd.read_csv(ctx.dep("S15") / "emb_ladder_summary.csv") if (ctx.dep("S15") / "emb_ladder_summary.csv").exists() else pd.DataFrame()
    fin = []
    b6 = s11.get("best_L6")
    if b6:
        fin.append({"name": "L6", "rung": "L6", "head": b6["head"], "enc": None, "params": s11["params"].get(f"L6|{b6['head']}"),
                    "wf_score": b6["tuned_mean_f1"]})
        fin.append({"name": "L6_nochange", "rung": "L6_nochange", "head": b6["head"], "enc": None,
                    "params": s11["params"].get(f"L6_nochange|{b6['head']}"), "wf_score": None})
        fin.append({"name": "L1", "rung": "L1", "head": b6["head"], "enc": None, "params": s11["params"].get(f"L1|{b6['head']}"),
                    "wf_score": None})
    if len(s15sum):
        b7 = s15sum[s15sum.rung == "L7"].sort_values("tuned_mean_f1", ascending=False).iloc[0]
        fin.append({"name": "L7", "rung": "L7", "head": b7["head"], "enc": b7["encoder"], "params": json.loads(b7.params),
                    "wf_score": float(b7.tuned_mean_f1)})
        if any(c.startswith("ext_") for c in lad.F.columns):
            fin.append({"name": "L8", "rung": "L8", "head": b7["head"], "enc": b7["encoder"], "params": json.loads(b7.params),
                        "wf_score": None})
    res, preds = Results(), []
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    for f in fin:
        r = lad.run(f["rung"], f["head"], f["params"], lab, common, "dev_holdout", "cls", f["enc"], keep=True)
        if r["preds"].empty:
            continue
        p = r["preds"]
        prior = np.bincount(p.y.astype(int), minlength=3) / len(p)
        m = classification_metrics(p.y.values, p[["p_down", "p_side", "p_up"]].values, prior, p.yhat.values)
        ci = macro_f1_ci(p.y.values, p.yhat.values, p.date.values, nb, seed)
        lb = _leaderboard(f["rung"], f["enc"], lad.prov, "dev")
        res.add(f"dev|{f['name']}", f["rung"], f["head"], f["enc"] or "n/a", lb, "common_emb", prim["target_def"], prim["horizon"],
                "dev", "all", m, ci, *_counts(lad.ev, p.event_id, lad.clean_col), "tuned")
        preds.append(p.assign(run_id=f"dev|{f['name']}", name=f["name"]))
    pr = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    # fine-tuned finalists: development predictions were produced in S15 (trained on all pre-holdout data)
    s15p = ctx.dep("S15") / "predictions.parquet"
    if s15p.exists():
        fp = pd.read_parquet(s15p)
        fp = fp[fp.run_id.astype(str).str.startswith("dev|L8_ft")]
        for rid, g in fp.groupby("run_id"):
            enc = g.encoder.iloc[0]
            prior = np.bincount(g.y.astype(int), minlength=3) / len(g)
            m = classification_metrics(g.y.values, g[PCOLS_].values, prior, g.yhat.values)
            ci = macro_f1_ci(g.y.values, g.yhat.values, g.date.values, nb, seed)
            res.add(rid, "L8_ft", "ft", enc, _leaderboard("L8_ft", enc, lad.prov, "dev"), "common_emb", prim["target_def"],
                    prim["horizon"], "dev", "all", m, ci, *_counts(lad.ev, g.event_id, lad.clean_col), "tuned")
            pr = pd.concat([pr, g.assign(name=f"L8_ft|{enc}")[["event_id", "fold", "y", "date", *PCOLS_, "yhat", "run_id", "name"]]],
                           ignore_index=True)
            wfs = pd.read_parquet(s15p)
            wfs = wfs[(wfs.rung == "L8_ft") & (wfs.encoder == enc) & ~wfs.run_id.astype(str).str.startswith("dev|")]
            if len(wfs):
                from .evaluation.metrics import confusion, f1_from_cm
                sc = [f1_from_cm(confusion(x.y.values, x.yhat.values))["macro_f1"] for _, x in wfs.groupby("fold")]
                fin.append({"name": f"L8_ft|{enc}", "rung": "L8_ft", "head": "ft", "enc": enc, "params": None,
                            "wf_score": float(np.mean(sc)), "dev_from": "S15"})
    # stacking extension (H5), walk-forward OOF only (SPEC.md §6.3 #20)
    stack = None
    if cfg["extensions"].get("ext_stacking"):
        from .extensions.stacking import run_stacking
        stack = run_stacking(ctx, lad, lab, common, res, pr)
        if stack and stack.get("dev_preds") is not None:
            pr = pd.concat([pr, stack["dev_preds"]], ignore_index=True)
    # lifts on the development holdout (identical rows)
    lifts = []
    names = set(pr.name) if len(pr) else set()
    for a, b, hyp in (("L7", "L6", "H1"), ("L6", "L6_nochange", "H2"), ("L8", "L7", "H3"), ("STACK", "L7", "H5")):
        if a in names and b in names:
            pa, pb = pr[pr.name == a].set_index("event_id"), pr[pr.name == b].set_index("event_id")
            j = pa.index.intersection(pb.index)
            d = paired_macro_f1_diff(pa.loc[j, "y"].values, pa.loc[j, "yhat"].values, pb.loc[j, "yhat"].values, pa.loc[j, "date"].values, nb, seed)
            lifts.append({"hypothesis": hyp, "a": a, "b": b, "block": "dev", **d, "n": int(len(j)), "power": power_label(len(j))})
    pd.DataFrame(lifts).to_csv(ctx.out / "dev_lifts.csv", index=False)
    # MCS over finalists (monthly mean log-loss on dev) and PBO over walk-forward configurations
    mcs, pbo = {}, {}
    if len(pr):
        ll = []
        for nm, g in pr.groupby("name"):
            P = np.clip(g[["p_down", "p_side", "p_up"]].values, 1e-9, 1)
            g = g.assign(ll=-np.log(P[np.arange(len(g)), g.y.astype(int).values]), m=pd.to_datetime(g.date).dt.to_period("M"))
            ll.append(g.groupby("m").ll.mean().rename(nm))
        mcs = model_confidence_set(pd.concat(ll, axis=1), reps=500)
    wf = pd.concat([pd.read_parquet(ctx.dep(s) / "predictions.parquet") for s in ("S11", "S15")
                    if (ctx.dep(s) / "predictions.parquet").exists()], ignore_index=True)
    wf = wf[(wf.target_def == prim["target_def"]) & (wf.horizon == prim["horizon"]) & wf.yhat.notna()
            & ~wf.run_id.astype(str).str.startswith("dev|")] if len(wf) else wf
    if len(wf):
        from .evaluation.metrics import confusion, f1_from_cm
        wf["m"] = pd.to_datetime(wf.date).dt.to_period("M")
        perf = wf.groupby(["m", "run_id"]).apply(lambda g: f1_from_cm(confusion(g.y.values, g.yhat.values))["macro_f1"],
                                                 include_groups=False).unstack()
        pbo = pbo_cscv(perf)
        pbo["streams"] = "monthly walk-forward macro-F1 of every stored configuration (S11, S15)"
    mcs["streams"] = "monthly mean log-loss on the development holdout, finalists only"
    write_json(ctx.out / "mcs_pbo.json", {"mcs": mcs, "pbo": pbo})
    # review triggers (audit, never a stop)
    rl = res.frame()
    trig = []
    for _, r in rl[(rl.metric == "auc_macro_ovr")].iterrows():
        if r.value > float(cfg["audit"]["auc_trigger"]):
            trig.append(f"{r.run_id}: dev AUC {r.value:.3f} > {cfg['audit']['auc_trigger']} -> open the audit checklist")
    for l in lifts:
        if l["value"] > float(cfg["audit"]["f1_lift_trigger"]):
            trig.append(f"{l['hypothesis']}: dev lift {l['value']:.3f} > {cfg['audit']['f1_lift_trigger']} -> audit")
    write_json(ctx.out / "review_triggers.json", {"triggers": trig, "note": "review triggers open an audit; they never terminate"})
    # G4: select the primary model by the pre-registered rule (walk-forward mean macro-F1), freeze and hash
    cand = [f for f in fin if f.get("wf_score") is not None]
    if stack and stack.get("wf_score") is not None:
        cand.append({"name": "STACK", "rung": "STACK", "head": "stack", "enc": None, "params": stack.get("spec"),
                     "wf_score": stack["wf_score"]})
    chosen = max(cand, key=lambda f: f["wf_score"]) if cand else None
    frozen = {"primary_model": chosen, "finalists": fin, "selection_rule": "walk-forward mean macro-F1 (validation-selected)",
              "prereg_hash": ph.read_text().strip(), "config_hash": ctx.storage.global_config_hash,
              "common_cohort_n": int(len(common)), "frozen_at": pd.Timestamp.utcnow().isoformat()}
    h = hashlib.sha256(canonical_json(frozen).encode()).hexdigest()
    frozen["g4_hash"] = h
    write_json(ctx.out / "frozen_architecture.json", frozen)
    write_json(ctx.storage.manifests / "frozen_architecture.json", frozen)
    (ctx.storage.manifests / "g4_hash.txt").write_text(h)
    rl["config_counter"] = lad.config_counter
    ctx.write(rl, "dev_results.parquet")
    ctx.write(pr, "dev_predictions.parquet")
    return {"finalists": [f["name"] for f in fin], "chosen": chosen and chosen["name"], "g4_hash": h, "triggers": trig,
            "pbo": pbo.get("pbo")}


# =====================================================================================================  S19
@stage("S19", "unlock-test: retrain + locked test + clean test", "CPU-GPU", ["S18"], ["lockbox", "bootstrap", "seed"],
       outputs=["locked_results.parquet", "lockbox_log.json"], assertions=["single use", "logged"],
       card="The locked test (2023-01-01 to the usable cutoff), scored once after the G4 freeze.")
def s19(ctx):
    cfg = ctx.cfg
    if not cfg["lockbox"].get("unlock_locked_test"):
        raise RuntimeError("lockbox sealed: set lockbox.unlock_locked_test: true in the owner settings to score the locked test once")
    g4 = ctx.storage.manifests / "g4_hash.txt"
    frozen = json.loads((ctx.dep("S18") / "frozen_architecture.json").read_text())
    if not g4.exists() or g4.read_text().strip() != frozen["g4_hash"]:
        raise RuntimeError("G4 hash missing or does not match the frozen architecture")
    log_p = ctx.storage.manifests / "lockbox_log.json"
    relabel = False
    if log_p.exists():
        if not cfg["lockbox"].get("acknowledge_reuse"):
            raise RuntimeError(f"locked test already scored ({log_p.read_text()}); set lockbox.acknowledge_reuse: true "
                               "to rescore it as a DEVELOPMENT block")
        relabel = True
    lad = build_ladder(ctx, with_emb=True, with_ext=True)
    prim = cfg["targets"]["primary"]
    lab = lad.labelled(prim["target_def"], prim["horizon"])
    encs = list(lad.embs)
    common = lab.index[lad.cohort(["L1", "L2", "L3", "L4", "L6", "L5", "L7"], encs, lab.index)] if encs else \
        lab.index[lad.cohort(["L1", "L2", "L3", "L4", "L6"], [None], lab.index)]
    rost = C.roster(cfg)
    from .splits.provenance import model_cutoffs
    cut = model_cutoffs(rost)
    res, preds = Results(), []
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    block = "locked_reused_as_development" if relabel else "locked"
    notes = []
    for f in frozen["finalists"]:
        if f["rung"] == "L8_ft":
            p = _ft_locked(ctx, lad, lab, common, f)
            if p is None:
                notes.append(f"{f['name']}: not scored on the locked test (GPU required for fine-tuning)")
                continue
        else:
            r = lad.run(f["rung"], f["head"], f["params"], lab, common, "locked_test", "cls", f["enc"], keep=True)
            if r["preds"].empty:
                continue
            p = r["preds"]
        prior = np.bincount(p.y.astype(int), minlength=3) / len(p)
        m = classification_metrics(p.y.values, p[["p_down", "p_side", "p_up"]].values, prior, p.yhat.values)
        ci = macro_f1_ci(p.y.values, p.yhat.values, p.date.values, nb, seed)
        lb = _leaderboard(f["rung"], f["enc"], lad.prov, "locked")
        res.add(f"{block}|{f['name']}", f["rung"], f["head"], f["enc"] or "n/a", lb, "common_emb", prim["target_def"],
                prim["horizon"], block, "all", m, ci, *_counts(lad.ev, p.event_id, lad.clean_col), "tuned")
        c = cut.get(f["enc"]) if f["enc"] else "folds"
        if c not in (None,):
            cs = p if c in ("vintage", "folds") else p[pd.to_datetime(p.date) > c]
            if len(cs) > 30:
                pri = np.bincount(cs.y.astype(int), minlength=3) / len(cs)
                mc = classification_metrics(cs.y.values, cs[["p_down", "p_side", "p_up"]].values, pri, cs.yhat.values)
                res.add(f"{block}|{f['name']}|clean_test", f["rung"], f["head"], f["enc"] or "n/a", lb, "common_emb",
                        prim["target_def"], prim["horizon"], f"{block}_clean_after_cutoff", "all", mc, None, len(cs), None, None, None, "tuned")
        preds.append(p.assign(run_id=f"{block}|{f['name']}", name=f["name"]))
    write_json(log_p, {"scored_at": pd.Timestamp.utcnow().isoformat(), "g4_hash": frozen["g4_hash"], "reused": relabel,
                       "previous": json.loads(log_p.read_text()) if log_p.exists() else None})
    rl = res.frame()
    ctx.write(rl, "locked_results.parquet")
    ctx.write(pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), "locked_predictions.parquet")
    return {"block": block, "rows": int(len(rl)), "notes": notes}


def _ft_locked(ctx, lad, lab, ids, f) -> pd.DataFrame | None:
    """Retrain a fine-tuned finalist on all pre-test data with its selected grid point; None without a GPU."""
    import torch
    if not torch.cuda.is_available():
        return None
    from .models.embeddings import chrono_vintage
    from .models.finetune import train_eval
    cfg = ctx.cfg
    st = json.loads((ctx.dep("S15") / "finetune_status.json").read_text())
    hp = st["grid"][f["enc"]]["best_hp"]
    sp = C.roster(cfg)["encoders"][f["enc"]]
    tr, te = lad.fold_ids("locked_test", 0, ids)
    if f["enc"] == "chrono_bert":
        v = chrono_vintage(pd.Timestamp(cfg["splits"]["locked_test_start"]).year, sp["vintages"])
        mid, rev = sp["model_id_template"].format(year=v), (sp.get("revisions") or {}).get(str(v))
    else:
        mid, rev = sp["model_id"], sp.get("revision")
    trs = lab.loc[tr].sort_values("date").iloc[-int(cfg["tuning"]["finetune_max_train_events"]):]
    cut = int(len(trs) * 0.9)
    itr, iva = trs.index[:cut], trs.index[cut:]
    tx = lad.texts
    out = train_eval(mid, rev, tx.reindex(itr).fillna("").tolist(), lab.loc[itr, "y"].astype(int).to_numpy(),
                     lab.loc[itr, "w"].to_numpy(), tx.reindex(iva).fillna("").tolist(), lab.loc[iva, "y"].astype(int).to_numpy(),
                     [tx.reindex(te).fillna("").tolist()], hp, int(cfg["models"]["max_seq_len"]), int(cfg["seed"]), ctx.log)
    prior = np.bincount(lab.loc[itr, "y"].astype(int), minlength=3) / max(1, len(itr))
    from .models.heads import prior_correct
    P = prior_correct(out["preds"][0], prior)
    return pd.DataFrame({"event_id": te, "fold": 0, "y": lab.loc[te, "y"].values, "date": lab.loc[te, "date"].values,
                         "p_down": P[:, 0], "p_side": P[:, 1], "p_up": P[:, 2], "yhat": decide(P, prior)})


# =====================================================================================================  S22
@stage("S22", "Backtest", "CPU", ["S18"], ["backtest", "seed"], outputs=["backtest.csv", "backtest_daily.parquet"],
       assertions=["entry-rule assertions"],
       card="Economic-relevance check of the frozen T_trade model (signal P(Up)-P(Down), horizon h=5).")
def s22(ctx):
    from .backtest.engine import carhart_alpha, daily_book, perf, positions, thresholds
    cfg, B = ctx.cfg, ctx.cfg["backtest"]
    frozen = json.loads((ctx.dep("S18") / "frozen_architecture.json").read_text())
    ch = frozen.get("primary_model")
    note = None
    if not ch or ch["rung"] in ("STACK", "L8_ft"):
        note = (f"primary model {ch and ch['name']} needs a horizon-{B['horizon']} refit that is not tabular; the backtest "
                "uses the best tabular finalist (L7, else L6) with the frozen hyperparameters")
        ch = max([f for f in frozen["finalists"] if f["name"] in ("L7", "L6") and f.get("wf_score") is not None],
                 key=lambda f: f["wf_score"], default=None)
        if ch is None:
            ch = next((f for f in frozen["finalists"] if f["name"] in ("L7", "L6")), None)
        ctx.warn(note)
    lad = build_ladder(ctx, with_emb=True, with_ext=True)
    h = int(B["horizon"])
    lab = lad.labelled("T_trade", h)
    encs = list(lad.embs)
    ids = lab.index[lad.requirement_mask(ch["rung"], ch["enc"], lab.index)]
    wf = lad.run(ch["rung"], ch["head"], ch["params"], lab, ids, "walk_forward", "cls", ch["enc"], keep=True)["preds"]
    dv = lad.run(ch["rung"], ch["head"], ch["params"], lab, ids, "dev_holdout", "cls", ch["enc"], keep=True)["preds"]
    blocks = {"dev": dv}
    lpath = ctx.storage.manifests / "lockbox_log.json"
    if lpath.exists():
        blocks["locked"] = lad.run(ch["rung"], ch["head"], ch["params"], lab, ids, "locked_test", "cls", ch["enc"], keep=True)["preds"]
    cal = C.calendar(ctx)
    panel, spy = C.panel_and_spy(ctx, cal)
    fac = C.factors(ctx)
    ev = lad.ev
    lo, hi = thresholds((wf.p_up - wf.p_down).values, float(B["short_pct"]), float(B["long_pct"]))
    rows, daily = [], []
    for bname, p in blocks.items():
        if p.empty:
            continue
        sig = pd.DataFrame({"event_id": p.event_id, "ticker": ev.loc[p.event_id, "ticker"].values,
                            "entry_idx": ev.loc[p.event_id, "ttrade_entry_idx"].values.astype(int), "score": (p.p_up - p.p_down).values})
        for lag in (0, int(B["lag_sessions_sensitivity"])):
            pos = positions(sig, lo, hi, h, lag)
            # entry-rule assertion: the open of the entry session is at or after decision time
            if lag == 0 and len(pos):
                ent_open = pd.Series([cal.open_time(int(i)) for i in pos.entry], index=pos.index)
                dec = pd.to_datetime(ev.loc[pos.event_id, "decision_time"]).values
                assert (ent_open.values >= dec).all(), "backtest entry before decision_time"
            for cost in B["costs_bps"]:
                d = daily_book(pos, panel, spy, len(cal), float(B["max_weight"]), int(B["min_names_per_side"]), float(cost),
                               float(B["borrow_bps_per_year"]))
                d["date"] = cal.sessions
                act = d[(d.n_long > 0) | (d.n_short > 0)]
                if act.empty:
                    continue
                a0, a1 = act.index.min(), act.index.max()
                d = d.loc[a0:a1]
                n_trials = int(sum(pd.read_parquet(ctx.dep(s) / "results_long.parquet").config_counter.max() for s in ("S11", "S15")))
                for leg, col in (("long_only", "long_ret"), ("long_minus_spy_matched", "long_excess_spy"), ("long_short_hypothetical", "ls_ret")):
                    pm = perf(d[col].values, n_trials)
                    rows.append({"block": bname, "lag": lag, "cost_bps": cost, "leg": leg, **pm})
                ca = carhart_alpha(d.long_hold.values, pd.DatetimeIndex(d.date), fac)
                rows.append({"block": bname, "lag": lag, "cost_bps": cost, "leg": "long_hold_component_carhart",
                             "ann_return": ca.get("alpha_ann"), "sharpe": None, "note": json.dumps(ca)})
                ent = d.long_entry - d.spy_long_entry
                rows.append({"block": bname, "lag": lag, "cost_bps": cost, "leg": "long_entry_day_vs_spy_open_close",
                             "ann_return": float(ent.mean() * 252), "n_days": int((d.n_long > 0).sum())})
                daily.append(d.assign(block=bname, lag=lag, cost_bps=cost))
    bt = pd.DataFrame(rows)
    bt["wording"] = "economic-relevance check, not an investable strategy (borrow, capacity and delisting data unknown)"
    bt.to_csv(ctx.out / "backtest.csv", index=False)
    if daily:
        ctx.write(pd.concat(daily, ignore_index=True), "backtest_daily.parquet")
    write_json(ctx.out / "backtest_spec.json", {"signal_model": ch, "thresholds": {"short_le": lo, "long_ge": hi},
                                                "threshold_source": "walk-forward OOF signals (training period)", "horizon": h})
    return {"rows": int(len(bt)), "blocks": list(blocks)}
