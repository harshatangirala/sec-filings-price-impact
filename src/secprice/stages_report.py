"""S12 gold tooling, S20 Track A, S21 drift tests, S23 Track C final, S24 figures + report (SPEC.md §10, §12, §20, §21)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .config import get
from .evaluation.fdr import apply_families
from .runner import stage
from .storage import write_json


# =====================================================================================================  S12
@stage("S12", "Gold export / import", "CPU", ["S10"], ["seed", "splits.dev_start"], outputs=["gold_sample.parquet", "templates/"],
       assertions=["gold-test blind", "split integrity"],
       card="Stratified gold sample from training-period filings, CSV/Excel annotation templates, validated import.")
def s12(ctx):
    from .gold.tools import cohen_kappa, export_templates, import_gold, sample_gold
    cfg = ctx.cfg
    ev = C.events(ctx)
    qb = C.question_bank(cfg)
    prof = cfg["run_profile"]
    sizes = {"gold_dev": 50, "gold_val": 100, "gold_test": 100} if prof != "smoke" else {"gold_dev": 3, "gold_val": 4, "gold_test": 4}
    smp = sample_gold(ev, pd.Timestamp(cfg["splits"]["dev_start"]), sizes, int(cfg["seed"]))
    ctx.write(smp.drop(columns=[c for c in ("accessions", "forms", "categories", "items", "section_ids") if c in smp]).assign(
        accessions=smp.accessions.map(json.dumps), forms=smp.forms.map(json.dumps)), "gold_sample.parquet")
    from .models.embeddings import event_chunks
    secs = []
    for cik in smp.cik.unique():
        p = ctx.dep("S05") / "sections" / f"{int(cik):010d}.parquet"
        if p.exists():
            secs.append(pd.read_parquet(p, columns=["accession", "section_id", "text_masked"]))
    ch = event_chunks(smp, pd.concat(secs, ignore_index=True), cfg) if secs else None
    paths = export_templates(smp, qb, ch, ctx.out / "templates", prefill=None, n_passages=200 if prof != "smoke" else 10,
                             seed=int(cfg["seed"]))
    gdir = ctx.storage.inputs / "gold"
    status = {"templates": paths, "labels_imported": False, "second_rater": False}
    if gdir.exists():
        g, errs = import_gold(gdir, smp, qb)
        status["import_errors"] = errs
        if len(g):
            ctx.write(g, "gold_labels.parquet")
            status["labels_imported"] = True
            status["n_rows"] = int(len(g))
            ann = g.groupby(["item_id"]).annotator.nunique()
            if (ann > 1).any():
                gg = g[g.item_id.isin(ann[ann > 1].index)]
                a, b = gg.annotator.unique()[:2]
                x = gg[gg.annotator == a].set_index("item_id").present_b
                y = gg[gg.annotator == b].set_index("item_id").present_b
                j = x.index.intersection(y.index)
                status["second_rater"] = True
                status["kappa_present"] = cohen_kappa(x.loc[j], y.loc[j])
    status["rater_status"] = "double-rater" if status.get("second_rater") else "single-rater, provisional"
    write_json(ctx.out / "gold_status.json", status)
    return {"sample": int(len(smp)), "labels_imported": status["labels_imported"], "rater": status["rater_status"]}


# =====================================================================================================  S20
@stage("S20", "Track A event study + correlations + topics", "CPU", ["S10"], ["targets", "contam_window_sessions", "bootstrap",
                                                                            "splits.dev_start", "seed"],
       outputs=["track_a_cars.parquet", "track_a_groups.csv", "car_paths.csv", "correlations.csv", "topics.csv"],
       assertions=["clean / all", "pre-trends"], card="CARs, abnormal volume and volatility around disclosures; association only.")
def s20(ctx):
    from .track_a.event_study import WINDOWS, car_paths, group_table, wname, window_cars
    cfg = ctx.cfg
    ev = C.events(ctx)
    cal = C.calendar(ctx)
    panel, spy = C.panel_and_spy(ctx, cal)
    uc = C.usable_cutoff(ctx)
    wc = window_cars(ev, panel, spy, C.factors(ctx), cfg, int(uc["usable_cutoff_idx"]))
    clean_col = f"overlap_flag_h{int(cfg['contam_window_sessions'])}"
    d = ev[["event_id", "cik", "event_type", "session_bucket", "earnings_type", "categories", "trackA_day0", clean_col,
            "decision_date"]].merge(wc, on="event_id", how="inner")
    d["intra_mixed_day0"] = d.session_bucket == "INTRA"
    ctx.write(d.drop(columns=["categories"]), "track_a_cars.parquet")
    cols = [wname(a, b) for a, b in WINDOWS] + [c for c in d.columns if c.startswith("car4_")] + ["abn_log_volume_0_1", "vol_response_0_1"]
    tabs = [group_table(d, "event_type", cols, clean_col), group_table(d, "session_bucket", [wname(0, 1), wname(-1, 1)], clean_col),
            group_table(d.assign(earnings_type=d.earnings_type.fillna("none")), "earnings_type", [wname(-1, 1), wname(2, 60)], clean_col)]
    ex = d.explode("categories")
    ex = ex[ex.event_type == "8K"]
    if len(ex):
        tabs.append(group_table(ex.rename(columns={"categories": "category"}), "category", [wname(-1, 1), wname(0, 1), wname(2, 20)], clean_col))
    gt = pd.concat(tabs, ignore_index=True)
    gt["fdr_family"] = "track_a_car"
    gt["p_value"] = gt["p"]
    gt = apply_families(gt[gt.p_value.notna()], q=float(cfg["fdr"]["q"]))
    gt.to_csv(ctx.out / "track_a_groups.csv", index=False)
    cp = car_paths(ev, panel, spy, cfg)
    cp_clean = car_paths(ev[~ev[clean_col]], panel, spy, cfg)
    pd.concat([cp.assign(event_set="all"), cp_clean.assign(event_set="clean")]).to_csv(ctx.out / "car_paths.csv", index=False)
    # correlations: tone persistence, release vs later periodic filing, filing tone vs CAR (Spearman + partial)
    F = C.read(ctx, "S09", "features.parquet")
    m = d.merge(F[["event_id"] + [c for c in F.columns if c in ("lm_net_tone", "chg_lm_net_tone", "chg_cosine", "novelty_share",
                                                                 "lm_uncertainty", "lm_negative", "acc_log_mcap", "px_mom_12_2")]],
                on="event_id", how="left")
    from scipy.stats import spearmanr
    corr = []
    for f in ("lm_net_tone", "chg_lm_net_tone", "chg_cosine", "novelty_share", "lm_uncertainty", "lm_negative"):
        for c in (wname(-1, 1), wname(0, 1), wname(2, 60)):
            if f in m and c in m:
                x = m[[f, c]].dropna()
                if len(x) > 30:
                    rho, p = spearmanr(x[f], x[c])
                    corr.append({"feature": f, "car": c, "spearman": float(rho), "p": float(p), "n": len(x)})
                    pc = _partial(m, f, c, ["acc_log_mcap", "px_mom_12_2"])
                    if pc is not None:
                        corr.append({"feature": f, "car": c, "partial_spearman_size_mom": pc[0], "p": pc[1], "n": pc[2]})
    evs = ev.merge(F[["event_id", "lm_net_tone"]], on="event_id", how="left").sort_values("decision_time")
    evs["prev_tone"] = evs.groupby(["cik", "event_type"]).lm_net_tone.shift(1)
    x = evs[["lm_net_tone", "prev_tone"]].dropna()
    if len(x) > 30:
        corr.append({"feature": "tone_persistence(prior same type)", "car": "-", "spearman": float(spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]),
                     "n": len(x)})
    cl = evs[evs.cluster_id.notna()]
    rr = cl[cl.cluster_role == "release"][["cluster_id", "lm_net_tone"]].merge(
        cl[cl.cluster_role == "periodic_after_release"][["cluster_id", "lm_net_tone"]], on="cluster_id", suffixes=("_rel", "_10q"))
    rr = rr.dropna()
    if len(rr) > 30:
        corr.append({"feature": "release_tone vs later 10-Q/10-K tone", "car": "-",
                     "spearman": float(spearmanr(rr.lm_net_tone_rel, rr.lm_net_tone_10q)[0]), "n": len(rr)})
    pd.DataFrame(corr).to_csv(ctx.out / "correlations.csv", index=False)
    # topics fitted on training years only
    topics = _topics(ctx, ev, d, pd.Timestamp(cfg["splits"]["dev_start"]))
    topics.to_csv(ctx.out / "topics.csv", index=False)
    return {"events_with_cars": int(len(d)), "group_rows": int(len(gt))}


def _partial(m, f, c, ctrls):
    from scipy.stats import spearmanr
    import statsmodels.api as sm
    cols = [f, c] + [x for x in ctrls if x in m]
    x = m[cols].dropna()
    if len(x) < 50 or len(cols) < 3:
        return None
    r = x.rank()
    Z = sm.add_constant(r[cols[2:]])
    e1 = sm.OLS(r[f], Z).fit().resid
    e2 = sm.OLS(r[c], Z).fit().resid
    rho, p = spearmanr(e1, e2)
    return float(rho), float(p), int(len(x))


def _topics(ctx, ev, cars, train_end, k=12):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    tx = C.read(ctx, "S09", "event_text.parquet").set_index("event_id")["text"]
    tr = ev[pd.to_datetime(ev.decision_date) < train_end].event_id
    tr = [e for e in tr if e in tx.index][:40000]
    if len(tr) < 50:
        return pd.DataFrame({"note": ["too few training documents for topics"]})
    vec = TfidfVectorizer(max_features=8000, min_df=3, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", sublinear_tf=True)
    M = vec.fit_transform(tx.loc[tr].fillna(""))
    kk = min(k, max(2, M.shape[0] // 10))
    nmf = NMF(n_components=kk, random_state=0, init="nndsvda", max_iter=300).fit(M)
    W = nmf.transform(vec.transform(tx.reindex(cars.event_id).fillna("")))
    words = np.array(vec.get_feature_names_out())
    top = {i: ", ".join(words[np.argsort(-nmf.components_[i])[:8]]) for i in range(kk)}
    dom = W.argmax(1)
    c = cars.assign(topic=dom)
    rows = []
    for t, g in c.groupby("topic"):
        rows.append({"topic": int(t), "top_words": top[t], "n": len(g), "car_m1_1": g.car_m1_1.mean(), "car_2_60": g.car_2_60.mean(),
                     "fit_on": "training years only (before development holdout)"})
    return pd.DataFrame(rows)


# =====================================================================================================  S21
@stage("S21", "Drift tests", "CPU", ["S20"], ["trivial_effect_bps", "fdr", "bootstrap", "seed", "backtest.costs_bps"],
       outputs=["hypothesis_tests.csv"], assertions=["FDR families", "verdict rubric"],
       card="SUE-conditioned post-earnings drift and other drift tests with pre-registered verdict rules.")
def s21(ctx):
    from .track_a.drift import (detect_dividend, momentum_benchmark, newey_west_t, prior_quarter_quintiles, spread_test,
                                subperiod_signs, trivial_bps, verdict)
    from .track_a.event_study import window_cars
    cfg = ctx.cfg
    tb = cfg["trivial_effect_bps"]
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    d = C.read(ctx, "S20", "track_a_cars.parquet")
    ev = C.events(ctx)
    F = C.read(ctx, "S09", "features.parquet")
    clean_col = f"overlap_flag_h{int(cfg['contam_window_sessions'])}"
    rows = []
    car = "car4_2_60" if "car4_2_60" in d and d["car4_2_60"].notna().sum() > 100 else "car_2_60"
    rel = d[d.earnings_type == "earnings_announcement_verified"].merge(F[["event_id", "sue_event"]], on="event_id", how="left")
    cal = C.calendar(ctx)
    panel, spy = C.panel_and_spy(ctx, cal)
    uc = C.usable_cutoff(ctx)
    def add(test, design, res, trivial, signs=None, clean_ok=None, cost_ok=None, lag_ok=None, note=""):
        r = {"test": test, "fdr_family": "drift", "design": design, "trivial_bps": trivial, "note": note,
             "subperiod_signs": json.dumps(signs) if signs else None, "clean_robust": clean_ok, "cost_robust": cost_ok,
             "lag_robust": lag_ok}
        if res:
            r.update({k: res.get(k) for k in ("estimate", "ci_low", "ci_high", "p_value", "n", "t")})
        rows.append(r)

    if rel.sue_event.notna().sum() >= 50:
        rel["q_sue"] = prior_quarter_quintiles(rel, "sue_event")
        res = spread_test(rel, "q_sue", car, "cik", "trackA_day0", nb, seed)
        signs = subperiod_signs(rel.dropna(subset=["q_sue"]), "q_sue", car) if res else None
        cres = spread_test(rel[~rel[clean_col]], "q_sue", car, "cik", "trackA_day0", nb, seed)
        clean_ok = bool(cres and res and np.sign(cres["estimate"]) == np.sign(res["estimate"]) and (cres["p_value"] or 1) < 0.10)
        cost = 4 * max(cfg["backtest"]["costs_bps"]) / 1e4
        cost_ok = bool(res and abs(res["estimate"]) - cost > trivial_bps(60, tb) / 1e4)
        lag_col = "car4_3_61" if car.startswith("car4") and "car4_3_61" in rel else "car_3_61"
        lres = spread_test(rel, "q_sue", lag_col, "cik", "trackA_day0", nb, seed)
        lag_ok = bool(lres and res and np.sign(lres["estimate"]) == np.sign(res["estimate"]) and (lres["p_value"] or 1) < 0.10)
        add("SUE-conditioned post-earnings drift (Q5-Q1, SUE_release)", f"verified releases; {car}; prior-quarter breakpoints",
            res, trivial_bps(60, tb), signs, clean_ok, cost_ok, lag_ok,
            f"one-session lag = {lag_col}; costs = 4 x {max(cfg['backtest']['costs_bps'])} bps round trip on the spread; "
            "never a consensus-based PEAD replication")
        x_sue = _xbrl_sue_for_releases(ctx, rel)
        if x_sue is not None and x_sue.notna().sum() >= 50:
            rx = rel.assign(sue_x=x_sue.values)
            rx["q_x"] = prior_quarter_quintiles(rx, "sue_x")
            add("drift sorted on SUE_xbrl (EX-POST sensitivity, not tradable)", f"verified releases; {car}",
                spread_test(rx, "q_x", car, "cik", "trackA_day0", nb, seed), trivial_bps(60, tb),
                note="XBRL EPS comes from the LATER 10-Q/10-K of the same cluster: labelled ex-post, never a feature")
        ear = rel.assign(q_ear=prior_quarter_quintiles(rel, "car_m1_1"))
        add("drift sorted on EAR (observed before +2)", f"verified releases; {car}", spread_test(ear, "q_ear", car, "cik", "trackA_day0", nb, seed),
            trivial_bps(60, tb), note="sensitivity")
    else:
        add("SUE-conditioned post-earnings drift", "verified releases", None, trivial_bps(60, tb),
            note=f"only {int(rel.sue_event.notna().sum())} verified releases with SUE_release")
    # momentum benchmarks (never attributed to filings)
    u = C.universe(ctx)
    mb = momentum_benchmark(panel, u.ticker.tolist(), cfg["study"]["start"], cfg["study"]["end"])
    for sig, g in mb.groupby("signal") if len(mb) else []:
        x = g.ls_ret.to_numpy()
        t = newey_west_t(x)
        rows.append({"test": f"momentum benchmark: {sig} (monthly decile long-short)", "fdr_family": "drift_benchmark",
                     "design": "equal-weight deciles within our universe", "estimate": float(np.mean(x)), "t": t,
                     "p_value": None if t is None else float(2 * (1 - __import__("scipy.stats").stats.norm.cdf(abs(t)))),
                     "n": len(x), "trivial_bps": trivial_bps(20, tb), "note": "benchmark only; never attributed to filings"})
    # does text add value beyond momentum? CAR[+2,+60] on text features with momentum controls (two-way clustered)
    import statsmodels.api as sm
    mm = d.merge(F[["event_id", "lm_net_tone", "chg_cosine", "novelty_share", "px_mom_12_2", "px_rev_1m", "acc_log_mcap"]],
                 on="event_id", how="left")
    cols = ["lm_net_tone", "chg_cosine", "novelty_share", "px_mom_12_2", "px_rev_1m"]
    x = mm[cols + ["car_2_60", "cik", "trackA_day0"]].dropna()
    if len(x) > 200:
        from .track_a.event_study import cluster_fit
        sd = x[cols].std().replace(0, np.nan)
        Z = sm.add_constant(((x[cols] - x[cols].mean()) / sd).fillna(0.0))
        fit, _how = cluster_fit(x.car_2_60.to_numpy(float), Z, x.cik.to_numpy(), x.trackA_day0.astype(str).to_numpy())
        fit.params.index = Z.columns
        fit.tvalues.index = Z.columns
        fit.pvalues.index = Z.columns
        for c in ("lm_net_tone", "chg_cosine", "novelty_share"):
            rows.append({"test": f"text beyond momentum: {c} -> CAR[+2,+60]", "fdr_family": "drift", "design": "OLS with momentum "
                         "controls, standardized, two-way clustered", "estimate": float(fit.params[c]), "t": float(fit.tvalues[c]),
                         "p_value": float(fit.pvalues[c]), "n": int(len(x)), "trivial_bps": trivial_bps(60, tb)})
    # post-dividend announcement drift (detector provisional until validated on gold)
    k8 = d[d.event_type == "8K"]
    tx = C.read(ctx, "S09", "event_text.parquet").set_index("event_id")["text"]
    k8 = k8.assign(div=k8.event_id.map(lambda e: detect_dividend(tx.get(e, "") or "")))
    for kind in ("increase", "decrease", "initiation"):
        g = k8[k8["div"] == kind]
        if len(g) >= 100:
            from .track_a.event_study import clustered_mean
            st = clustered_mean(g["car_2_60"].to_numpy(float), g.cik.to_numpy(), g.trackA_day0.astype(str).to_numpy())
            rows.append({"test": f"post-dividend-announcement drift: {kind}", "fdr_family": "drift", "design": "8-K text detector; CAR[+2,+60]",
                         "estimate": st["mean"], "t": st["t"], "p_value": st["p"], "n": st["n"], "trivial_bps": trivial_bps(60, tb),
                         "note": "detector provisional (precision not validated on gold); announcement != ex-date; total return"})
        else:
            rows.append({"test": f"post-dividend-announcement drift: {kind}", "fdr_family": "drift", "n": int(len(g)),
                         "verdict": "inconclusive/low power", "note": "fewer than 100 detected announcements"})
    # spin-off / restructuring drift (parent only)
    ex = d[d.event_type == "8K"].merge(ev[["event_id", "categories"]], on="event_id")
    for cat in ("restructuring_exit", "impairment", "acquisition_disposition"):
        g = ex[ex.categories.map(lambda c: cat in c)]
        if len(g) >= 100:
            from .track_a.event_study import clustered_mean
            st = clustered_mean(g["car_2_60"].to_numpy(float), g.cik.to_numpy(), g.trackA_day0.astype(str).to_numpy())
            rows.append({"test": f"restructuring / spin-off drift: 8-K {cat}", "fdr_family": "drift", "design": "parent only; CAR[+2,+60]",
                         "estimate": st["mean"], "t": st["t"], "p_value": st["p"], "n": st["n"], "trivial_bps": trivial_bps(60, tb)})
    for t, why in (("analyst revisions drift", "no consensus data; a management-guidance proxy is a different construct"),
                   ("insider trading drift", "Form 4 is an extension (ext_data_modules)"),
                   ("institutional ownership drift", "13F is an extension (ext_data_modules)")):
        rows.append({"test": t, "fdr_family": "not_testable", "verdict": "not testable in scope", "note": why})
    ht = pd.DataFrame(rows)
    for c in ("p_value", "estimate", "ci_low", "ci_high", "n", "verdict", "subperiod_signs", "clean_robust", "cost_robust",
              "lag_robust", "trivial_bps"):
        if c not in ht:
            ht[c] = None
    ht["p_adj_bh"] = None
    tst = ht[ht.p_value.notna()]
    if len(tst):
        adj = apply_families(tst, q=float(cfg["fdr"]["q"]))
        ht.loc[adj.index, "p_adj_bh"] = adj.p_adj_bh
    for i, r in ht.iterrows():
        if pd.notna(r.get("verdict")) and r.get("verdict"):
            continue
        if r.get("fdr_family") == "drift_benchmark":
            ht.at[i, "verdict"] = "benchmark (not a filing effect)"
            continue
        res = None if pd.isna(r.get("estimate")) else {"estimate": r.estimate, "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                                                       "p_value": r.get("p_value"), "n": int(r.n) if pd.notna(r.n) else 0}
        signs = json.loads(r.subperiod_signs) if isinstance(r.get("subperiod_signs"), str) else []
        ht.at[i, "verdict"] = verdict(res, float(r.trivial_bps or 50), r.get("p_adj_bh"), signs,
                                      r.get("clean_robust"), r.get("cost_robust"), r.get("lag_robust"))
    ht["rule"] = "supported needs FDR p<0.05, CI excluding the trivial threshold, >=2/3 subperiod signs, clean, cost and lag robustness"
    ht.to_csv(ctx.out / "hypothesis_tests.csv", index=False)
    ht.to_csv(ctx.storage.reports / "hypothesis_tests.csv", index=False)
    return {"tests": int(len(ht)), "verdicts": ht.verdict.value_counts().to_dict()}


def _xbrl_sue_for_releases(ctx, rel: pd.DataFrame):
    """SUE of each release computed with the later same-cluster periodic filing's own XBRL (ex-post sensitivity)."""
    ev = C.events(ctx)
    sue = C.read(ctx, "S06", "sue_by_accession.parquet")
    if sue.empty:
        return None
    a2s = dict(zip(sue.accession, sue.sue))
    per = ev[ev.cluster_role == "periodic_after_release"].drop_duplicates("cluster_id").set_index("cluster_id")
    cl = ev.set_index("event_id")["cluster_id"]
    out = []
    for eid in rel.event_id:
        c = cl.get(eid)
        v = np.nan
        if c is not None and c in per.index:
            v = a2s.get(list(per.loc[c, "accessions"])[0], np.nan)
        out.append(v)
    return pd.Series(out, index=rel.index)


# =====================================================================================================  S23
@stage("S23", "Track C final (gold-test)", "CPU", ["S12"], ["gates"], outputs=["content_validity.csv"],
       assertions=["gold-test touched once"], card="Content validity on held-out human labels (never against prices).")
def s23(ctx):
    from .gold.tools import GoldTestLock
    gd = ctx.dep("S12")
    s16 = ctx.dep_or_none("S16")
    lab = pd.read_parquet(gd / "gold_labels.parquet") if (gd / "gold_labels.parquet").exists() else pd.DataFrame()
    test = lab[lab.gold_split == "gold_test"] if len(lab) else lab
    if test.empty or s16 is None or not json.loads((s16 / "rag_frozen.json").read_text()).get("frozen"):
        pd.DataFrame([{"metric": "all", "status": "not evaluated: gold-test labels and a frozen RAG pipeline are both required"}]).to_csv(
            ctx.out / "content_validity.csv", index=False)
        return {"evaluated": False}
    lock = GoldTestLock(ctx.storage.manifests / "gold_test_lock.json")
    frozen = json.loads((ctx.storage.manifests / "frozen_architecture.json").read_text()) if (ctx.storage.manifests / "frozen_architecture.json").exists() else {}
    lock.consume(frozen.get("g4_hash", "no-g4"))
    ext = pd.read_parquet(ctx.dep("S17") / "extractions.parquet") if ctx.dep_or_none("S17") and (ctx.dep("S17") / "extractions.parquet").exists() else pd.DataFrame()
    rows = []
    from sklearn.metrics import f1_score, precision_score, recall_score
    for src in ("teacher", "student"):
        e = ext[ext.source == src] if len(ext) else ext
        if e.empty:
            continue
        m = test.merge(e[["event_id", "field", "present"]].rename(columns={"present": "pred"}), on=["event_id", "field"], how="inner")
        for f, g in m.groupby("field"):
            y, p = g.present_b.astype(bool), g.pred.fillna(False).astype(bool)
            rows.append({"source": src, "field": f, "n": len(g), "precision": precision_score(y, p, zero_division=0),
                         "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0)})
    cv = pd.DataFrame(rows)
    st = json.loads((gd / "gold_status.json").read_text())
    cv["rater_status"] = st.get("rater_status")
    cv.to_csv(ctx.out / "content_validity.csv", index=False)
    return {"evaluated": True, "rows": len(cv)}


# =====================================================================================================  S24
@stage("S24", "Figures + report", "CPU", ["S10", "S11", "S20", "S21"], ["run_profile"],
       outputs=["reports/results_summary.md", "reports/results_report.html", "reports/results_report.pdf", "figures/"],
       assertions=["computed values only"], card="All figures and the full extracted report, built only from artifacts.")
def s24(ctx):
    from .report.build import build_html, build_markdown, build_pdf
    from .viz.figures import Figs
    facts, figs = collect(ctx)
    md = build_markdown(facts)
    rep = ctx.storage.reports
    (rep / "results_summary.md").write_text(md, encoding="utf-8")
    (ctx.out / "results_summary.md").write_text(md, encoding="utf-8")
    html = build_html(md, figs)
    (rep / "results_report.html").write_text(html, encoding="utf-8")
    err = build_pdf(md, figs, rep / "results_report.pdf")
    if err:
        ctx.warn(err)
    return {"figures_ok": sum(f["status"] == "ok" for f in figs), "figures_skipped": sum(f["status"] != "ok" for f in figs),
            "report": str(rep / "results_summary.md")}


def _csv(path, **kw):
    """read_csv that returns an empty frame for an empty file (tiny smoke folds can write empty tables)."""
    try:
        return pd.read_csv(path, **kw)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _opt(ctx, stage_id, name, kind="parquet"):
    d = ctx.dep_or_none(stage_id)
    if d is None or not (d / name).exists():
        return None
    p = d / name
    if kind == "parquet":
        return pd.read_parquet(p)
    if kind == "csv":
        x = _csv(p)
        return x if len(x.columns) else None
    return json.loads(p.read_text(encoding="utf-8"))


def collect(ctx):
    """Assemble report facts + draw all figures from artifacts (missing inputs -> 'not produced: reason')."""
    from .evaluation.metrics import classification_metrics
    from .events.timestamps import timing_policy_label
    from .viz.figures import Figs
    cfg = ctx.cfg
    F = Figs(ctx.storage.figures)
    ev = C.events(ctx)
    u = C.universe(ctx, selected_only=False)
    usel = u[u.selected]
    man = C.manifest(ctx)
    led = C.read(ctx, "S05", "parse_ledger.parquet")
    tg = C.read(ctx, "S08", "targets.parquet")
    fpy = _csv(ctx.dep("S02") / "firms_per_year.csv")
    uc = C.usable_cutoff(ctx)
    facts = {"profile": cfg["run_profile"], "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
             "config_hash": ctx.storage.global_config_hash, "timing_policy": timing_policy_label(cfg)}
    from .storage import git_commit
    facts["code"] = git_commit()
    ph = ctx.storage.manifests / "prereg_hash.txt"
    facts["prereg_hash"] = ph.read_text().strip() if ph.exists() else "not frozen"
    # ---------------- data integrity
    facts["universe_text"] = (
        f"Universe: **{len(usel)} unique firms** (target {cfg['universe']['n_stocks']}) common to our OHLCV files and the HF 10-K corpus, "
        f"{int(usel.ever_sp500.sum())} of them S&P 500 members at some point in {cfg['study']['start']} to {cfg['study']['end']} "
        f"(fja05680/sp500 @ {cfg['universe']['sp500_commit'][:8]}). Candidates: {len(u)}; eligible: {int(u.eligible.sum())}. "
        f"Exclusions: {u.exclusion_reason.value_counts().to_dict()}. Usable cutoff {uc['usable_cutoff']} "
        f"(last price {uc['last_price_date']}, last document {uc['last_document_date']}); final month partial: {uc['final_month_partial']}. "
        "Firms per year are reported separately below; 500 firms per year is never claimed.")
    facts["firms_per_year"] = fpy
    mm = man.merge(led[["accession", "status", "n_sections"]], on="accession", how="left")
    mm["year"] = pd.to_datetime(mm.filing_date).dt.year
    cov = mm[mm.download_task].groupby("base_form").agg(observed=("accession", "size"), parsed=("n_sections", lambda s: int((s.fillna(0) > 0).sum())))
    cov["parsed_share"] = cov.parsed / cov.observed
    facts["coverage_by_form"] = cov.reset_index()
    cfy = _csv(ctx.dep("S04") / "coverage_firm_year.csv")
    facts["reason_codes"] = cfy.reason_code.value_counts().rename_axis("reason_code").reset_index(name="firm_years") if len(cfy) else None
    k8 = ev[ev.event_type == "8K"].explode("categories")
    cby = k8.groupby(["year", "categories"]).size().unstack(fill_value=0) if len(k8) else pd.DataFrame()
    facts["category_by_year"] = cby.T.reset_index().rename(columns={"categories": "category"}) if len(cby) else None
    tq = ev.groupby(["year", "ts_quality"]).size().unstack(fill_value=0)
    facts["ts_quality"] = tq.reset_index()
    facts["timestamp_text"] = (f"Acceptance times: SGML header when the download carried one; otherwise the submissions-JSON value under the "
                               f"verified double-conversion rule (header disagreement rate {uc.get('json_rule_mismatch')} over "
                               f"{uc.get('json_rule_n')} checks); pre-2002 filings have midnight placeholders (quality B, assumed 17:30 ET). "
                               "Public availability is ASSUMED (acceptance + lag), not observed.")
    hfa = ctx.dep("S01") / "hf_dataset_audit.md"
    facts["hf_text"] = hfa.read_text(encoding="utf-8").split("## Filings by year")[0] if hfa.exists() else "n/a"
    facts["data_limitations"] = ["Daily bars only; the universe is survivorship-aware, not survivorship-free (all firms are current listings).",
                                 "Free prices are weak on delistings; local OHLCV is split-adjusted and dividends come from Yahoo actions.",
                                 "Selected sections only (10-K Items 1/1A/7/7A, 10-Q MD&A/market risk/risk factors, 8-K items + EX-99).",
                                 "SUE coverage is limited before the XBRL era (2009); release-EPS parsing is heuristic (QA vs XBRL reported).",
                                 "The HF corpus contains 10-Ks only (1994-2021); it defines the firm overlap and QA, never the study text."]
    # ---------------- design
    facts["design_text"] = ("Four timestamps per event; T_trade (primary, prospective: entry at the first open at or after decision time, "
                            "SPY on the identical interval) and T_info (information response; INTRA excluded); 3-class labels with "
                            f"k={cfg['labels']['k']} x sigma x sqrt(h); per-horizon censoring at the usable cutoff; overlap flags "
                            f"(clean events = no same-firm event within {cfg['contam_window_sessions']} sessions).")
    facts["event_counts"] = ev.groupby(["event_type", "session_bucket"]).size().unstack(fill_value=0).reset_index()
    # ---------------- leakage / provenance
    facts["leakage_audit"] = _csv(ctx.dep("S10") / "leakage_audit.csv")
    facts["ledger"] = _csv(ctx.dep("S10") / "model_provenance_ledger.csv")[["key", "model_id", "revision", "license", "stated_cutoff"]]
    sanity = []
    perm = _opt(ctx, "S11", "permutation.csv", "csv")
    if perm is not None and len(perm):
        obs = perm[perm.kind == "observed"].macro_f1.iloc[0]
        null = perm[perm.kind == "permuted"].macro_f1
        sanity.append({"check": "permutation test (within-month, class-stratified)", "observed": obs, "null_mean": null.mean(),
                       "p_value": float((null >= obs).mean()) if len(null) else None, "n_perm": len(null)})
    rd = _opt(ctx, "S11", "random_demo.json", "json")
    if rd:
        sanity.append({"check": "random-split demo (pedagogical)", "observed": rd["random_split_macro_f1"], "null_mean": rd["walk_forward_macro_f1"]})
    adv = _opt(ctx, "S11", "adversarial.csv", "csv")
    if adv is not None and len(adv):
        sanity.append({"check": "adversarial validation AUC (mean over folds)", "observed": adv.auc.mean(), "null_mean": 0.5})
    trig = _opt(ctx, "S18", "review_triggers.json", "json")
    if trig:
        sanity.append({"check": "review triggers", "observed": len(trig["triggers"]), "null_mean": None, "p_value": None,
                       "n_perm": None})
    facts["sanity"] = pd.DataFrame(sanity)
    # ---------------- Track A
    ta = _opt(ctx, "S20", "track_a_groups.csv", "csv")
    if ta is not None and len(ta):
        main = ta[(ta.group_by == "event_type") & ta.measure.isin(["car_m1_1", "car_0_1", "car_2_60", "car4_m1_1", "car4_2_60"])]
        facts["track_a_main"] = main[["group", "event_set", "measure", "mean", "t", "p_adj_bh", "n"]]
    facts["hypothesis_tests"] = _opt(ctx, "S21", "hypothesis_tests.csv", "csv")
    if facts["hypothesis_tests"] is not None:
        facts["hypothesis_tests"] = facts["hypothesis_tests"][[c for c in ("test", "estimate", "ci_low", "ci_high", "p_value", "p_adj_bh", "n",
                                                                           "verdict") if c in facts["hypothesis_tests"]]]
    # ---------------- Track B
    rl = pd.concat([x for x in (_opt(ctx, "S11", "results_long.parquet"), _opt(ctx, "S15", "results_long.parquet")) if x is not None],
                   ignore_index=True)
    prim = cfg["targets"]["primary"]
    lad = rl[(rl.metric == "macro_f1") & (rl.block == "walk_forward_pooled") & (rl.event_set == "all")
             & (rl.target_def == prim["target_def"]) & (rl.horizon == prim["horizon"])]
    facts["ladder"] = lad[["rung", "model_id", "revision", "cohort", "tuned", "leaderboard", "mean_fold_score", "value", "ci_low",
                           "ci_high", "n_events"]].rename(columns={"value": "pooled_macro_f1", "revision": "encoder"}).sort_values(["cohort", "rung"])
    tb = _csv(ctx.dep("S11") / "ladder_summary.csv")
    t15 = _opt(ctx, "S15", "emb_ladder_summary.csv", "csv")
    tba = pd.concat([tb.assign(encoder="-"), t15] if t15 is not None else [tb.assign(encoder="-")], ignore_index=True)
    facts["tuning_before_after"] = tba[["rung", "encoder", "head", "default_mean_f1", "tuned_mean_f1"]]
    hyp = []
    g3 = _opt(ctx, "S15", "g3.json", "json")
    if g3 and g3.get("lift"):
        d = g3["lift"]
        bl0 = _csv(ctx.dep("S10") / "blocks.csv")
        wfb = bl0[bl0.kind == "walk_forward_val"] if len(bl0) else bl0
        span = (f"walk-forward validation blocks {str(wfb.start.min())[:10]} to {str(wfb.end_exclusive.max())[:10]}"
                if len(wfb) else "walk-forward validation")
        facts["h1"] = {"value": d["value"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "n_events": g3.get("n_events"),
                       "block": span}
        hyp.append({"hypothesis": "H1 (walk-forward): best L7 - L6", **{k: d[k] for k in ("value", "ci_low", "ci_high", "p_boot")},
                    "n": g3.get("n_events"), "G3": g3["passed"]})
    dl = _opt(ctx, "S18", "dev_lifts.csv", "csv")
    if dl is not None:
        for r in dl.itertuples(index=False):
            hyp.append({"hypothesis": f"{r.hypothesis} (development holdout): {r.a} - {r.b}", "value": r.value, "ci_low": r.ci_low,
                        "ci_high": r.ci_high, "p_boot": r.p_boot, "n": r.n})
    h4 = _opt(ctx, "S15", "h4.csv", "csv")
    if h4 is not None and len(h4):
        for r in h4.itertuples(index=False):
            hyp.append({"hypothesis": f"H4 {r.rung}/{r.head}: ChronoBERT vintage - {r.static}", "value": r.diff_vintage_minus_static,
                        "ci_low": r.ci_low, "ci_high": r.ci_high, "p_boot": r.p_boot, "n": r.n})
    h2 = lad[(lad.rung.isin(["L6", "L6_nochange"])) & (lad.cohort == "common") & (lad.tuned == "tuned")]
    facts["hypotheses_core"] = pd.DataFrame(hyp)
    facts["trackb_text"] = (f"Primary target: {prim['target_def']} h={prim['horizon']} 3-class; primary metric: mean walk-forward macro-F1 "
                            "(guardrails: log-loss no worse than the class prior; ECE). Results are validation-selected. Strict leaderboard = "
                            "models whose documented cutoff precedes every evaluated event (ChronoBERT vintages, models trained on our folds, "
                            "lexical baselines; SEC-BERT only after 2019); retrospective = all models.")
    facts["dev_results"] = _filter_main(_opt(ctx, "S18", "dev_results.parquet"))
    lr = _opt(ctx, "S19", "locked_results.parquet")
    lock = ctx.storage.manifests / "lockbox_log.json"
    facts["locked_text"] = ("Locked test scored once: " + lock.read_text()) if lock.exists() else \
        "The locked test (2023-01-01 to the usable cutoff) is **sealed**: it has not been scored. Set lockbox.unlock_locked_test: true after reviewing the development results."
    facts["locked_results"] = _filter_main(lr)
    prs = _opt(ctx, "S11", "predictions.parquet")
    sel_rows = rl[(rl.block == "walk_forward_pooled") & (rl.event_set == "all") & (rl.target_def == prim["target_def"])
                  & (rl.horizon == prim["horizon"]) & rl.metric.str.contains("_at_")]
    facts["selective"] = sel_rows.pivot_table(index=["rung", "model_id", "revision", "tuned"], columns="metric", values="value").reset_index().head(20) \
        if len(sel_rows) else None
    ti = rl[(rl.metric == "macro_f1") & (rl.block == "walk_forward_pooled") & (rl.event_set == "all") & rl.target_def.isin(["T_info", "T_trade"])
            & (rl.horizon == 1) & (rl.cohort == "common")]
    facts["tinfo_vs_ttrade"] = ti[["rung", "model_id", "target_def", "tuned", "value", "n_events"]] if len(ti) else None
    facts["backtest"] = _opt(ctx, "S22", "backtest.csv", "csv")
    if facts["backtest"] is not None:
        facts["backtest"] = facts["backtest"][[c for c in ("block", "lag", "cost_bps", "leg", "ann_return", "ann_vol", "sharpe",
                                                           "deflated_sharpe_prob", "max_drawdown", "n_days") if c in facts["backtest"]]]
    # ---------------- Track C, fit, extensions, limitations
    gs = _opt(ctx, "S12", "gold_status.json", "json") or {}
    cv = _opt(ctx, "S23", "content_validity.csv", "csv")
    g3b = _opt(ctx, "S16", "g3b.json", "json")
    facts["trackc_text"] = (f"Gold set: {gs.get('rater_status', 'n/a')}; labels imported: {gs.get('labels_imported')}. "
                            f"G3b: {g3b}. Price outcomes never validate semantic labels.")
    facts["trackc"] = cv
    fd = _csv(ctx.dep("S11") / "fit_diagnosis.csv")
    facts["fit_diag"] = fd[["rung", "head", "mean_val", "mean_train", "std", "verdict"]]
    cert = _opt(ctx, "S11", "enet_certificate.json", "json")
    facts["optim_text"] = ("\n**Optimisation diagnostics.** " + (
        f"Elastic-net certificate: objective spread across random starts {cert['spread']:.2e} -> global optimum of the convex problem "
        f"{'certified' if cert['certified_global_min'] else 'NOT certified'}. " if cert else "") +
        "Tree and transformer heads are non-convex: their optima are validation-selected (best Optuna trial / early-stopped checkpoint), "
        "not global minima.")
    ext = [{"extension": k, "enabled": v} for k, v in cfg["extensions"].items() if v]
    mp = _opt(ctx, "S18", "mcs_pbo.json", "json")
    facts["extensions"] = pd.DataFrame(ext) if ext else None
    facts["limitations"] = ["Association is not causation; prediction is not profitability; the backtest is an economic-relevance check only.",
                            "Public availability times are assumed (acceptance + 5 min), not observed; results carry that timing policy.",
                            "Pretrained encoders without documented cutoffs (FinBERT-tone, BGE, Qwen, ModernBERT) appear only on the "
                            "retrospective leaderboard and support no clean-test claim.",
                            f"MCS / PBO are diagnostics only: {json.dumps(mp)[:400] if mp else 'not run'}"]
    man_run = json.loads((ctx.storage.manifests / "run_manifest.json").read_text()) if (ctx.storage.manifests / "run_manifest.json").exists() else {}
    facts["resources"] = pd.DataFrame([{"stage": k, "wall_s": v.get("wall_s"), "status": v.get("status")} for k, v in man_run.get("stages", {}).items()])
    facts["decision_log"] = "docs/decision_log.md"
    # executive bullets (computed values only)
    eb = [f"{len(usel)} firms, {len(ev):,} events ({ev.event_type.value_counts().to_dict()}), {cfg['study']['start']} to {uc['usable_cutoff']}."]
    if len(lad):
        best = lad[lad.cohort.str.startswith("common")].sort_values("mean_fold_score", ascending=False).head(1)
        if len(best):
            b = best.iloc[0]
            eb.append(f"Best walk-forward model: {b.rung}/{b.model_id}{'/' + str(b.revision) if b.revision != 'n/a' else ''} "
                      f"mean macro-F1 {b.mean_fold_score:.3f} (pooled {b.value:.3f}, CI {b.ci_low if b.ci_low is None else round(b.ci_low, 3)} to "
                      f"{b.ci_high if b.ci_high is None else round(b.ci_high, 3)}).")
        l0 = lad[lad.rung == "L0"]
        if len(l0):
            eb.append(f"Majority-class baseline macro-F1: {l0.value.iloc[0]:.3f}.")
    if g3:
        eb.append(f"G3 (text lift >= {cfg['gates']['text_lift']} with CI excluding 0): {'passed' if g3.get('passed') else 'not passed'}.")
    ht = facts.get("hypothesis_tests")
    if ht is not None and len(ht):
        eb.append("Drift verdicts: " + "; ".join(f"{r.test}: {r.verdict}" for r in ht.head(3).itertuples(index=False)))
    facts["exec_bullets"] = eb
    figures(ctx, F, facts, ev, u, usel, man, mm, led, tg, rl, fpy)
    facts["figures"] = F.write_index()
    return facts, facts["figures"]


def _filter_main(df):
    if df is None or not len(df):
        return None
    d = df[(df.metric.isin(["macro_f1", "auc_macro_ovr", "logloss", "ece", "accuracy"])) & (df.event_set == "all")]
    return d.pivot_table(index=["run_id", "leaderboard", "n_events"], columns="metric", values="value").reset_index()


def figures(ctx, F, facts, ev, u, usel, man, mm, led, tg, rl, fpy):
    cfg = ctx.cfg
    prim = cfg["targets"]["primary"]
    # F01-F06 data and coverage
    piv = mm[mm.download_task].assign(parsed=lambda d: d.n_sections.fillna(0) > 0).groupby(["year", "base_form"]).parsed.mean().unstack()
    F.safe("F01", "Coverage heatmap (parsed share)", lambda i, t: F.heat(i, t, piv.T, vmin=0, vmax=1))
    cfy = _csv(ctx.dep("S04") / "coverage_firm_year.csv")
    F.safe("F02", "Audit reason codes over time", lambda i, t: F.stacked(i, t, cfy.groupby(["year", "reason_code"]).size().unstack(fill_value=0)))
    k8 = ev[ev.event_type == "8K"].explode("categories")
    F.safe("F03", "8-K category availability by year", lambda i, t: F.heat(i, t, k8.groupby(["categories", "year"]).size().unstack(fill_value=0),
                                                                             fmt="{:.0f}", cmap="magma"))
    F.safe("F04", "Universe: unique firms vs firms per year", lambda i, t: F.lines(i, t, {
        "unique firms": (fpy.year, fpy.unique_firms), "firms with prices": (fpy.year, fpy.firms_with_prices),
        "firms in S&P 500": (fpy.year, fpy.firms_in_sp500)}, "year", "firms"))
    F.safe("F05", "Timestamp quality and session buckets by year", lambda i, t: _f05(F, i, t, ev))
    F.safe("F06", "Pre-publication window and lag sensitivity", lambda i, t: _f06(F, i, t, ev, tg))
    F.safe("F07", "Event-clock diagram", lambda i, t: F.f07_clock(i, t))
    F.safe("F08", "Earnings types over time", lambda i, t: F.stacked(i, t, ev.groupby(["year", ev.earnings_type.fillna("none")]).size().unstack(fill_value=0)))
    oc = {h: float(ev[f"overlap_flag_h{h}"].mean()) for h in cfg["targets"]["horizons"] if f"overlap_flag_h{h}" in ev}
    F.safe("F09", "Overlap counts and clean share", lambda i, t: F.bars_ci(i, t, pd.DataFrame({"label": [f"h={h}" for h in oc],
                                                                                              "value": [1 - v for v in oc.values()],
                                                                                              "ci_low": [1 - v for v in oc.values()],
                                                                                              "ci_high": [1 - v for v in oc.values()]})))
    F.safe("F10", "Target distributions and class balance", lambda i, t: _f10(F, i, t, tg))
    # F11-F16 Track A
    cp = _opt(ctx, "S20", "car_paths.csv", "csv")
    F.safe("F11", "CAR paths by event type (all vs clean)", lambda i, t: F.lines(i, t, {f"{g} ({es})": (x.rel_day, x.caar * 100)
                                                                                      for (g, es), x in cp.groupby(["group", "event_set"])},
                                                                              "session relative to day 0", "CAAR (%)") if cp is not None and len(cp)
           else F.skip(i, t, "no CAR paths"))
    ta = _opt(ctx, "S20", "track_a_groups.csv", "csv")
    if ta is not None and len(ta):
        b = ta[(ta.group_by == "session_bucket") & (ta.measure == "car_0_1") & (ta.event_set == "all")]
        F.safe("F12", "CAR[0,+1] by session bucket", lambda i, t: F.bars_ci(i, t, b.assign(label=b.group, value=b["mean"] * 100,
                                                                                            ci_low=(b["mean"] - 1.96 * b.se) * 100,
                                                                                            ci_high=(b["mean"] + 1.96 * b.se) * 100), ref=0))
        pt = ta[(ta.group_by == "event_type") & ta.measure.isin(["car_m20_m2", "car_m5_m1"]) & (ta.event_set == "all")]
        F.safe("F13", "Pre-trend diagnostics", lambda i, t: F.bars_ci(i, t, pt.assign(label=pt.group, value=pt["mean"] * 100,
                                                                                       ci_low=(pt["mean"] - 1.96 * pt.se) * 100,
                                                                                       ci_high=(pt["mean"] + 1.96 * pt.se) * 100), hue="measure", ref=0))
        av = ta[(ta.group_by == "event_type") & ta.measure.isin(["abn_log_volume_0_1", "vol_response_0_1"]) & (ta.event_set == "all")]
        F.safe("F14", "Abnormal volume and volatility response", lambda i, t: F.bars_ci(i, t, av.assign(label=av.group, value=av["mean"],
                                                                                                         ci_low=av["mean"] - 1.96 * av.se,
                                                                                                         ci_high=av["mean"] + 1.96 * av.se), hue="measure"))
    else:
        for i in ("F12", "F13", "F14"):
            F.skip(i, "", "Track A not run")
    corr = _opt(ctx, "S20", "correlations.csv", "csv")
    if corr is not None and len(corr) and "spearman" in corr:
        cm = corr.dropna(subset=["spearman"]).pivot_table(index="feature", columns="car", values="spearman")
        F.safe("F15", "Correlation heatmap", lambda i, t: F.heat(i, t, cm, cmap="coolwarm", vmin=-0.2, vmax=0.2))
    else:
        F.skip("F15", "Correlation heatmap", "no correlations")
    F.safe("F16", "SUE-drift quintile CAR[+2,+60]", lambda i, t: _f16(ctx, F, i, t))
    # F17-F31 Track B
    bl = _csv(ctx.dep("S10") / "blocks.csv", parse_dates=["start", "end_exclusive"])
    prov = _csv(ctx.dep("S10") / "fold_provenance.csv")
    F.safe("F17", "Split timeline", lambda i, t: F.f17_timeline(i, t, bl, prov, cfg["study"]["start"], float(cfg["splits"]["embargo_days"])))
    lad = facts["ladder"]
    if lad is not None and len(lad):
        L = lad[lad.tuned.isin(["tuned", "default"]) & lad.cohort.isin(["common", "common_emb", "max_available"])].copy()
        L = L.sort_values("pooled_macro_f1", ascending=False).drop_duplicates(["rung", "cohort", "encoder"])
        L["label"] = L.rung + np.where(L.encoder.astype(str) != "n/a", "/" + L.encoder.astype(str), "") + " [" + L.leaderboard.str[0].str.upper() + "]"
        F.safe("F18", "Ladder lift with CIs", lambda i, t: F.bars_ci(i, t, L.rename(columns={"pooled_macro_f1": "value"}), hue="cohort",
                                                                       ref=float(L[L.rung == "L0"].value.iloc[0]) if (L.rung == "L0").any() and "value" in L else None))
    else:
        F.skip("F18", "Ladder lift with CIs", "no ladder results")
    h4 = _opt(ctx, "S15", "h4.csv", "csv")
    F.safe("F19", "Vintage vs static (H4)", lambda i, t: F.bars_ci(i, t, h4.assign(label=h4.rung + "/" + h4["head"] + " vs " + h4.static,
                                                                                  value=h4.diff_vintage_minus_static), ref=0)
           if h4 is not None and len(h4) else F.skip(i, t, "ChronoBERT or embeddings not run"))
    curves = json.loads((ctx.dep("S11") / "curves.json").read_text())
    c15 = _opt(ctx, "S15", "curves.json", "json") or []
    ft = _opt(ctx, "S15", "ft_curves.json", "json") or []
    F.safe("F20", "Loss curves", lambda i, t: F.f20_loss(i, t, (curves + c15)[:6], ft))
    tri = _opt(ctx, "S11", "optuna_trials.csv", "csv")
    F.safe("F21", "Validation curves", lambda i, t: _f21(F, i, t, tri) if tri is not None and len(tri) else F.skip(i, t, "no trials"))
    lc = _opt(ctx, "S11", "learning_curves.csv", "csv")
    F.safe("F22", "Learning curves", lambda i, t: F.lines(i, t, {"validation": (lc.train_fraction, lc.val_macro_f1),
                                                                 "train": (lc.train_fraction, lc.train_macro_f1)}, "training fraction", "macro-F1")
           if lc is not None and len(lc) else F.skip(i, t, "no learning curve"))
    tba = facts["tuning_before_after"]
    F.safe("F23", "Before / after tuning", lambda i, t: _f23(F, i, t, tba))
    preds = pd.concat([x for x in (_opt(ctx, "S11", "predictions.parquet"), _opt(ctx, "S15", "predictions.parquet")) if x is not None],
                      ignore_index=True)
    pp = preds[(preds.target_def == prim["target_def"]) & (preds.horizon == prim["horizon"]) & preds.yhat.notna()
               & ~preds.run_id.astype(str).str.startswith("dev|")] if len(preds) else preds
    cms, best_run = {}, None
    if len(pp) and lad is not None and len(lad):
        from .evaluation.metrics import confusion
        for rung in ("L0", "L1", "L6", "L7", "L8_ft"):
            cand = lad[(lad.rung == rung) & lad.cohort.str.startswith("common")].sort_values("mean_fold_score", ascending=False)
            if len(cand):
                rid_rows = rl[(rl.rung == rung) & (rl.metric == "macro_f1") & (rl.block == "walk_forward_pooled")
                              & (rl.target_def == prim["target_def"]) & (rl.horizon == prim["horizon"])
                              & rl.cohort.astype(str).str.startswith("common") & (rl.event_set == "all")]
                rid_rows = rid_rows.sort_values("mean_fold_score", ascending=False)
                if len(rid_rows):
                    rid = rid_rows.run_id.iloc[0]
                    g = pp[pp.run_id == rid]
                    if len(g):
                        cms[f"{rung}"] = confusion(g.y.values.astype(int), g.yhat.values.astype(int))
                        if rung in ("L6", "L7", "L8_ft"):
                            best_run = g
    F.safe("F24", "Confusion matrices", lambda i, t: F.f24_cm(i, t, cms))
    if best_run is not None and len(best_run):
        y = best_run.y.values.astype(int)
        P = best_run[["p_down", "p_side", "p_up"]].values
        from .evaluation.metrics import ece_top
        F.safe("F25", "ROC / PR curves", lambda i, t: F.f25_roc(i, t, y, P))
        F.safe("F26", "Calibration", lambda i, t: F.f26_calib(i, t, y, P, ece_top(P, y)))
        F.safe("F27", "Coverage-accuracy curve", lambda i, t: F.f27_coverage(i, t, y, P, best_run.yhat.values.astype(int)))
        F.safe("F31", "Performance by slice", lambda i, t: _f31(F, i, t, best_run, ctx))
    else:
        for i in ("F25", "F26", "F27", "F31"):
            F.skip(i, "", "no primary predictions")
    reg = preds[preds.target_def.astype(str).str.contains(":ar_mm") & preds.yhat_reg.notna()] if len(preds) and "yhat_reg" in preds else pd.DataFrame()
    F.safe("F28", "Rolling IC", lambda i, t: _f28(F, i, t, reg) if len(reg) else F.skip(i, t, "no regression predictions"))
    ti = facts.get("tinfo_vs_ttrade")
    if ti is not None and len(ti):
        ti = ti.sort_values("value", ascending=False).drop_duplicates(["rung", "model_id", "target_def"])
    F.safe("F29", "T_info vs T_trade", lambda i, t: F.bars_ci(i, t, ti.assign(label=ti.rung + "/" + ti.model_id, ci_low=ti.value, ci_high=ti.value),
                                                                hue="target_def") if ti is not None and len(ti) else F.skip(i, t, "no rows"))
    bd = _opt(ctx, "S22", "backtest_daily.parquet")
    F.safe("F30", "Backtest", lambda i, t: _f30(F, i, t, bd) if bd is not None and len(bd) else F.skip(i, t, "backtest not run"))
    rt = _opt(ctx, "S16", "rag_tournament.csv", "csv")
    if rt is not None and "recall" in rt:
        F.safe("F32", "Retrieval metrics by setup", lambda i, t: F.bars_ci(i, t, rt.assign(label=rt.setup, value=rt.recall, ci_low=rt.recall,
                                                                                           ci_high=rt.recall)))
    else:
        F.skip("F32", "Retrieval metrics by setup", "not evaluable without gold-val qrels")
    cv = _opt(ctx, "S23", "content_validity.csv", "csv")
    if cv is not None and "f1" in cv:
        F.safe("F33", "Extraction F1 (gold-test)", lambda i, t: F.bars_ci(i, t, cv.assign(label=cv.field, value=cv.f1, ci_low=cv.f1, ci_high=cv.f1),
                                                                           hue="source"))
    else:
        F.skip("F33", "Extraction F1 / faithfulness (gold-test)", "gold-test not evaluated")
    for i, t in (("F34", "Numeric fidelity"), ("F35", "Judge vs human"), ("F36", "Student vs teacher vs human")):
        F.skip(i, t, "needs gold labels (Track C)")
    perm = _opt(ctx, "S11", "permutation.csv", "csv")
    F.safe("F37", "Permutation null", lambda i, t: _f37(F, i, t, perm) if perm is not None and len(perm) else F.skip(i, t, "not run"))
    rd = _opt(ctx, "S11", "random_demo.json", "json")
    F.safe("F38", "Random-split demo", lambda i, t: F.bars_ci(i, t, pd.DataFrame({"label": ["random split (demo)", "walk-forward"],
                                                                                  "value": [rd["random_split_macro_f1"], rd["walk_forward_macro_f1"]],
                                                                                  "ci_low": [rd["random_split_macro_f1"], rd["walk_forward_macro_f1"]],
                                                                                  "ci_high": [rd["random_split_macro_f1"], rd["walk_forward_macro_f1"]]}))
           if rd else F.skip(i, t, "not run"))
    adv = _opt(ctx, "S11", "adversarial.csv", "csv")
    F.safe("F39", "Adversarial-validation ROC", lambda i, t: F.lines(i, t, {f"fold {r.fold} AUC {r.auc:.2f}": (json.loads(r.fpr), json.loads(r.tpr))
                                                                            for r in adv.itertuples(index=False)}, "FPR", "TPR")
           if adv is not None and len(adv) else F.skip(i, t, "not run"))
    mp = _opt(ctx, "S18", "mcs_pbo.json", "json")
    F.safe("F40", "MCS / PBO diagnostics", lambda i, t: _f40(F, i, t, mp) if mp else F.skip(i, t, "S18 not run"))
    F.safe("FX1", "Optimisation diagnostics", lambda i, t: _fx1(ctx, F, i, t, tri))


def _f05(F, i, t, ev):
    fig, (a, b) = F.plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col in ((a, "ts_quality"), (b, "session_bucket")):
        d = ev.groupby(["year", col]).size().unstack(fill_value=0)
        bottom = np.zeros(len(d))
        for c in d.columns:
            ax.bar(d.index.astype(str), d[c], bottom=bottom, label=c)
            bottom += d[c].values
        ax.tick_params(axis="x", rotation=90)
        ax.legend(fontsize=7)
        ax.set_title(col, fontsize=8)
    F.save(fig, i, t)


def _f06(F, i, t, ev, tg):
    fig, (a, b) = F.plt.subplots(1, 2, figsize=(12, 4.2))
    x = ev.contam_minutes.dropna() / 60
    a.hist(x.clip(upper=96), bins=60)
    a.set_xlabel("hours from last close to acceptance (clipped at 96)")
    rows = []
    base = tg[(tg.target_def == "T_info") & (tg.horizon == 1)].set_index("event_id").exit_idx
    for lag in (0, 15, 60):
        s = tg[(tg.target_def == f"T_info@pub{lag}") & (tg.horizon == 1)].set_index("event_id").exit_idx
        j = base.index.intersection(s.index)
        rows.append((f"lag {lag} min", float((base.loc[j] != s.loc[j]).mean()) if len(j) else np.nan))
    b.bar([r[0] for r in rows], [r[1] for r in rows])
    b.set_ylabel("share of events whose T_info exit changes")
    F.save(fig, i, t)


def _f10(F, i, t, tg):
    fig, (a, b) = F.plt.subplots(1, 2, figsize=(12, 4.2))
    for td in ("T_trade", "T_info"):
        x = tg[(tg.target_def == td) & (tg.horizon == 1)].ar_mm.dropna()
        a.hist(x.clip(-0.2, 0.2), bins=80, alpha=0.5, label=td)
    a.legend(fontsize=7)
    a.set_xlabel("abnormal return, h=1 (clipped +-20%)")
    cb = tg[tg.target_def.isin(["T_trade", "T_info"])].groupby(["target_def", "horizon"]).label3.value_counts(normalize=True).unstack()
    cb.plot(kind="bar", stacked=True, ax=b)
    b.set_ylabel("class share (0=down, 1=side, 2=up)")
    F.save(fig, i, t)


def _f16(ctx, F, i, t):
    from .track_a.drift import prior_quarter_quintiles
    d = C.read(ctx, "S20", "track_a_cars.parquet")
    Fe = C.read(ctx, "S09", "features.parquet")
    r = d[d.earnings_type == "earnings_announcement_verified"].merge(Fe[["event_id", "sue_event"]], on="event_id")
    if r.sue_event.notna().sum() < 50:
        return F.skip(i, t, "too few verified releases with SUE")
    r["q"] = prior_quarter_quintiles(r, "sue_event")
    g = r.dropna(subset=["q"]).groupby("q").car_2_60.agg(["mean", "sem", "size"]).reset_index()
    F.bars_ci(i, t, g.assign(label="Q" + g.q.astype(int).astype(str), value=g["mean"] * 100, ci_low=(g["mean"] - 1.96 * g["sem"]) * 100,
                             ci_high=(g["mean"] + 1.96 * g["sem"]) * 100), ref=0)


def _f21(F, i, t, tri):
    pcols = [c for c in tri.columns if c.startswith("params_")]
    k = min(6, len(pcols))
    if k == 0:
        return F.skip(i, t, "no tuned parameters")
    fig, axes = F.plt.subplots(1, k, figsize=(3.2 * k, 3.4), squeeze=False)
    for ax, c in zip(axes[0], pcols[:k]):
        d = tri[[c, "value", "head"]].dropna()
        for h, g in d.groupby("head"):
            ax.scatter(g[c], g.value, s=8, label=h)
        ax.set_xlabel(c.replace("params_", ""), fontsize=7)
        ax.set_ylabel("walk-forward macro-F1", fontsize=7)
    axes[0][0].legend(fontsize=6)
    F.save(fig, i, t)


def _f23(F, i, t, tba):
    d = tba.dropna(subset=["tuned_mean_f1"]).copy()
    d["label"] = d.rung + "/" + d["head"] + np.where(d.encoder.astype(str) != "-", "/" + d.encoder.astype(str), "")
    fig, ax = F.plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.default_mean_f1, 0.4, label="default")
    ax.bar(x + 0.2, d.tuned_mean_f1, 0.4, label="tuned")
    ax.set_xticks(x, d.label, rotation=90, fontsize=6)
    ax.legend(fontsize=7)
    ax.set_ylabel("mean walk-forward macro-F1")
    F.save(fig, i, t)


def _f28(F, i, t, reg):
    from scipy.stats import spearmanr
    g = reg[reg.rung == reg.rung.iloc[-1]]
    g = g.assign(m=pd.to_datetime(g.date).dt.to_period("M"))
    ic = g.groupby("m").apply(lambda x: spearmanr(x.y, x.yhat_reg)[0] if len(x) > 10 else np.nan, include_groups=False).dropna()
    roll = ic.rolling(12, min_periods=6).mean()
    F.lines(i, t, {"monthly rank IC": (ic.index.to_timestamp(), ic.values), "12-month mean": (roll.index.to_timestamp(), roll.values)},
            "month", "rank IC")


def _f30(F, i, t, bd):
    fig, ax = F.plt.subplots(figsize=(11, 4.6))
    for (blk, lag, cost), g in bd.groupby(["block", "lag", "cost_bps"]):
        ax.plot(pd.to_datetime(g.date), np.cumprod(1 + g.long_ret.fillna(0)) - 1, lw=1, label=f"{blk} long lag{lag} {cost}bps")
        ax.plot(pd.to_datetime(g.date), np.cumprod(1 + g.ls_ret.fillna(0)) - 1, lw=1, ls="--", label=f"{blk} L/S lag{lag} {cost}bps")
        ax.plot(pd.to_datetime(g.date), np.cumprod(1 + g.spy_matched.fillna(0)) - 1, lw=0.8, ls=":", color="k")
    ax.set_ylabel("cumulative return")
    ax.legend(fontsize=6, ncol=2)
    F.save(fig, i, t)


def _f31(F, i, t, best, ctx):
    from .evaluation.metrics import confusion, f1_from_cm
    ev = C.events(ctx).set_index("event_id")
    b = best.join(ev[["event_type", "session_bucket", "ts_quality", "ff12_industry", "year"]], on="event_id")
    fig, axes = F.plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, col in zip(axes, ("event_type", "session_bucket", "ff12_industry", "year")):
        s = b.groupby(col).apply(lambda g: f1_from_cm(confusion(g.y.values.astype(int), g.yhat.values.astype(int)))["macro_f1"],
                                 include_groups=False)
        ax.bar(s.index.astype(str), s.values)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.set_title(col, fontsize=8)
    F.save(fig, i, t)


def _f37(F, i, t, perm):
    fig, ax = F.plt.subplots(figsize=(6, 4))
    null = perm[perm.kind == "permuted"].macro_f1.dropna()
    obs = perm[perm.kind == "observed"].macro_f1.iloc[0]
    ax.hist(null, bins=max(5, len(null) // 2), alpha=0.7, label="permuted labels")
    ax.axvline(obs, color="C3", lw=2, label=f"observed {obs:.3f}")
    ax.legend(fontsize=7)
    ax.set_xlabel("walk-forward macro-F1")
    F.save(fig, i, t)


def _f40(F, i, t, mp):
    fig, (a, b) = F.plt.subplots(1, 2, figsize=(11, 4))
    pv = (mp.get("mcs") or {}).get("pvalues") or {}
    if pv:
        a.barh(list(pv), list(pv.values()))
        a.axvline(0.10, color="k", ls="--", lw=0.8)
        a.set_xlabel("MCS p-value (in set if > 0.10)")
    else:
        a.text(0.1, 0.5, (mp.get("mcs") or {}).get("note", "MCS unavailable"), fontsize=8)
    pbo = (mp.get("pbo") or {}).get("pbo")
    b.bar(["PBO"], [pbo if pbo is not None else 0])
    b.set_ylim(0, 1)
    b.set_title(f"PBO = {pbo}" if pbo is not None else (mp.get("pbo") or {}).get("note", "n/a"), fontsize=8)
    F.save(fig, i, t)


def _fx1(ctx, F, i, t, tri):
    fig, (a, b) = F.plt.subplots(1, 2, figsize=(12, 4.2))
    if tri is not None and len(tri):
        for (rung, head), g in tri.groupby(["rung", "head"]):
            if rung not in ("L6", "L4", "L1"):
                continue
            v = g.sort_values("number").value.cummax()
            a.plot(np.arange(len(v)), v.values, label=f"{rung}/{head}")
        a.set_xlabel("trial")
        a.set_ylabel("best-so-far walk-forward macro-F1")
        a.legend(fontsize=7)
    cert = _opt(ctx, "S11", "enet_certificate.json", "json")
    if cert:
        b.bar([f"start {k}" for k in range(len(cert["objective_by_start"]))], cert["objective_by_start"])
        lo = min(cert["objective_by_start"])
        b.set_ylim(lo - max(1e-4, abs(lo) * 1e-3), lo + max(1e-4, abs(lo) * 1e-3))
        b.set_title(f"elastic-net objective by random start (spread {cert['spread']:.1e}; "
                    f"{'global optimum certified' if cert['certified_global_min'] else 'not certified'})", fontsize=8)
    F.save(fig, i, t)
