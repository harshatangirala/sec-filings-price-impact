"""Build the dashboard's data snapshot from a finished run (every number on the dashboard comes from here).

    python dashboard/build_snapshot.py --root <DATA_ROOT>/<profile> [--out dashboard/data]

Reads stage artifacts (S01-S24), the report figures and the decision log, and writes small CSV / JSON / parquet files plus the
figures and the PDF report. Nothing is typed in by hand; a missing artifact leaves its file out and the dashboard says so.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


class Run:
    def __init__(self, root: Path):
        self.root = Path(root)

    def stage_dir(self, sid: str) -> Path | None:
        c = [Path(p).parent for p in glob.glob(str(self.root / sid / "*" / "_SUCCESS"))]
        return max(c, key=lambda p: (p / "_SUCCESS").stat().st_mtime) if c else None

    def path(self, sid: str, name: str) -> Path | None:
        d = self.stage_dir(sid)
        return d / name if d is not None and (d / name).exists() else None

    def read(self, sid: str, name: str, **kw):
        p = self.path(sid, name)
        if p is None:
            return None
        if name.endswith(".parquet"):
            return pd.read_parquet(p, **kw)
        if name.endswith(".csv"):
            try:
                return pd.read_csv(p, **kw)
            except pd.errors.EmptyDataError:
                return None
        return json.loads(p.read_text(encoding="utf-8"))

    def success(self, sid: str) -> dict:
        p = self.path(sid, "_SUCCESS")
        return json.loads(p.read_text(encoding="utf-8")) if p else {}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


class Out:
    def __init__(self, out: Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []

    def csv(self, name: str, df: pd.DataFrame | None):
        if df is not None and len(df):
            df.to_csv(self.out / name, index=False)
            self.written.append(name)

    def json(self, name: str, obj):
        if obj is not None:
            (self.out / name).write_text(json.dumps(_jsonable(obj), indent=1, default=str), encoding="utf-8")
            self.written.append(name)

    def parquet(self, name: str, df: pd.DataFrame | None):
        if df is not None and len(df):
            df.to_parquet(self.out / name, index=False)
            self.written.append(name)


# ---------------------------------------------------------------------------------------------------------------- builders
def stages(run: Run) -> pd.DataFrame:
    import secprice.stages_data  # noqa: F401  (registers stages)
    import secprice.stages_model  # noqa: F401
    import secprice.stages_report  # noqa: F401
    from secprice.runner import REGISTRY
    rows = []
    for sid in sorted(REGISTRY, key=lambda s: int(s[1:])):
        sp, sx = REGISTRY[sid], run.success(sid)
        note = "done" if sx else "not run"
        if sid == "S16" and (sx.get("g3b") or {}).get("passed") is False:
            note = "ran, not scored (no gold labels)"
        if sid == "S17" and sx.get("ran") is False:
            note = "not run (needs gold labels)"
        if sid == "S23" and sx.get("evaluated") is False:
            note = "not evaluated (needs gold labels)"
        rows.append({"stage": sid, "name": sp.name, "part": sp.part, "needs": ", ".join(sp.deps), "outputs": ", ".join(sp.outputs[:4]),
                     "card": sp.card, "hours": (sx.get("wall_s") or 0) / 3600 if sx else None, "result": note,
                     "finished": sx.get("finished")})
    return pd.DataFrame(rows)


def eda(run: Run, O: Out) -> dict:
    ev = run.read("S07", "events.parquet", columns=["event_id", "cik", "event_type", "ts_quality", "session_bucket", "earnings_type", "year",
                                                    "ticker", "ff12_industry", "categories", "overlap_flag_h60", "decision_time"])
    u = run.read("S02", "universe.parquet")
    us = u[u.selected].copy()
    O.csv("universe.csv", us[["ticker", "cik", "sec_name", "ff12_industry", "sic_desc", "ever_sp500", "price_start", "price_end", "n_bars",
                             "hf_10k_count", "n_periodic_in_window"]].sort_values(["ff12_industry", "ticker"]))
    O.csv("firms_per_year.csv", run.read("S02", "firms_per_year.csv"))
    O.csv("events_by_year.csv", ev.groupby(["year", "event_type"]).size().rename("events").reset_index())
    O.csv("ts_quality_by_year.csv", ev.groupby(["year", "ts_quality"]).size().rename("events").reset_index())
    O.csv("session_buckets.csv", ev.groupby(["event_type", "session_bucket"]).size().rename("events").reset_index())
    O.csv("earnings_types.csv", ev.groupby(["year", ev.earnings_type.fillna("not earnings")]).size().rename("events").reset_index())
    k8 = ev[ev.event_type == "8K"].explode("categories")
    O.csv("categories_8k.csv", k8.groupby("categories").size().rename("events").reset_index().sort_values("events", ascending=False))
    O.csv("events_by_firm.csv", ev.groupby(["ticker", "ff12_industry"]).size().rename("events").reset_index())
    man = run.read("S04", "filing_manifest.parquet")
    led = run.read("S05", "parse_ledger.parquet", columns=["accession", "status", "n_sections", "bytes", "n_tables", "form"])
    cov = None
    if man is not None and led is not None:
        mm = man[man.download_task].merge(led[["accession", "status", "n_sections"]], on="accession", how="left")
        cov = mm.groupby("base_form").agg(observed=("accession", "size"), parsed=("n_sections", lambda s: int((s.fillna(0) > 0).sum()))).reset_index()
        cov["parsed_share"] = cov.parsed / cov.observed
        mm["year"] = pd.to_datetime(mm.filing_date).dt.year
        O.csv("filings_by_year.csv", mm.groupby(["year", "base_form"]).size().rename("filings").reset_index())
    O.csv("coverage_by_form.csv", cov)
    si = run.read("S05", "sections_index.parquet")
    if si is not None:
        sec = si.groupby("section_id").agg(sections=("accession", "size"), median_tokens=("n_tokens_est", "median"),
                                           mean_tokens=("n_tokens_est", "mean"), truncated=("truncated", "sum"),
                                           by_reference=("by_reference", "sum"), firm_masks=("mask_firm", "sum"),
                                           ticker_masks=("mask_ticker", "sum")).reset_index()
        O.csv("sections.csv", sec)
        # token-length distribution (log bins) per section family for the EDA histogram
        si["log10_tokens"] = np.log10(si.n_tokens_est.clip(lower=1))
        bins = np.linspace(0, 5.5, 45)
        rows = []
        for sid, g in si.groupby("section_id"):
            h, e = np.histogram(g.log10_tokens, bins=bins)
            rows += [{"section_id": sid, "log10_tokens": float((e[i] + e[i + 1]) / 2), "sections": int(h[i])} for i in range(len(h))]
        O.csv("section_length_hist.csv", pd.DataFrame(rows))
    O.csv("hf_parser_qa.csv", run.read("S05", "hf_parser_qa.csv"))
    O.csv("section_found_counts.csv", run.read("S05", "section_found_counts.csv"))
    tg = run.read("S08", "targets.parquet", columns=["event_id", "target_def", "horizon", "ar_mm", "label3", "eligible", "start_date"])
    O.csv("target_summary.csv", run.read("S08", "target_summary.csv"))
    if tg is not None:
        p = tg[(tg.target_def == "T_trade") & (tg.horizon == 1) & tg.eligible.fillna(False) & tg.label3.notna()].copy()
        p["year"] = pd.to_datetime(p.start_date).dt.year
        lab = p.groupby(["year", "label3"]).size().rename("events").reset_index()
        lab["label"] = lab.label3.map({0: "Down", 1: "Sideways", 2: "Up"})
        O.csv("labels_by_year.csv", lab)
        x = p.ar_mm.clip(-0.1, 0.1) * 100
        h, e = np.histogram(x.dropna(), bins=np.linspace(-10, 10, 81))
        O.csv("ar_hist.csv", pd.DataFrame({"abnormal_return_pct": (e[:-1] + e[1:]) / 2, "events": h}))
    O.csv("sue_coverage.csv", run.read("S06", "sue_coverage.csv"))
    xf = run.read("S06", "xbrl_facts.parquet", columns=["tag"])
    er = run.read("S06", "eps_release.parquet", columns=["is_results_release", "parse_confidence"])
    su = run.read("S06", "sue_by_accession.parquet", columns=["sue"])
    pc = run.read("S03", "price_coverage.csv")
    led_ok = int((led.status == "ok").sum()) if led is not None else None
    pf = run.read("S13", "preflight.json") or {}
    return {
        "firms": int(len(us)), "events": int(len(ev)), "events_by_type": ev.event_type.value_counts().to_dict(),
        "filings": int(len(led)) if led is not None else None, "filings_parsed_ok": led_ok,
        "tables_flattened": int(led.n_tables.sum()) if led is not None else None,
        "sections": int(len(si)) if si is not None else None,
        "firm_name_masks": int(si.mask_firm.sum()) if si is not None else None,
        "ticker_masks": int(si.mask_ticker.sum()) if si is not None else None,
        "chunks_est": pf.get("est_chunks"), "tokens_est": pf.get("est_tokens"),
        "xbrl_facts": int(len(xf)) if xf is not None else None, "xbrl_tags": int(xf.tag.nunique()) if xf is not None else None,
        "release_rows": int(len(er)) if er is not None else None,
        "results_releases": int(er.is_results_release.sum()) if er is not None else None,
        "sue_filings": int(len(su)) if su is not None else None, "sue_available": int(su.sue.notna().sum()) if su is not None else None,
        "price_first": str(pc["first"].min()) if pc is not None else None, "price_last": str(pc["last"].max()) if pc is not None else None,
        "price_bars": int(pc.n_bars.sum()) if pc is not None else None,
        "eligible_firms": int(u.eligible.sum()), "candidate_firms": int(len(u)),
        "exclusions": u.exclusion_reason.value_counts().to_dict(),
        "sectors": us.ff12_industry.value_counts().to_dict(),
        "event_first": str(pd.to_datetime(ev.decision_time).min())[:10], "event_last": str(pd.to_datetime(ev.decision_time).max())[:10],
        "ts_quality": ev.ts_quality.value_counts().to_dict(), "session_buckets": ev.session_bucket.value_counts().to_dict(),
        "clean_share_h60": float(1 - ev.overlap_flag_h60.mean()),
        "availability_violations": run.success("S09").get("availability_violations"),
        "sue_rule_violations": run.success("S09").get("sue_rule_violations"),
        "feature_families": run.success("S09").get("families"),
        "audit_counts": run.success("S10").get("audit"), "planted_leaks_caught": run.success("S10").get("planted_leaks_caught"),
        "configs_tried_s11": run.success("S11").get("configs_tried"), "common_cohort": run.success("S11").get("common_cohort"),
        "labelled_events": run.success("S11").get("labelled"),
    }


def modelling(run: Run, O: Out) -> dict:
    rl = pd.concat([x for x in (run.read("S11", "results_long.parquet"), run.read("S15", "results_long.parquet")) if x is not None], ignore_index=True)
    keep = ["macro_f1", "accuracy", "balanced_accuracy", "auc_macro_ovr", "logloss", "logloss_prior", "brier", "ece"]
    w = rl[(rl.block == "walk_forward_pooled") & (rl.event_set == "all") & (rl.target_def == "T_trade") & (rl.horizon == 1)
           & rl.metric.isin(keep) & rl.cohort.isin(["common", "common_emb"])]
    pv = w.pivot_table(index=["run_id", "rung", "model_id", "revision", "cohort", "tuned", "leaderboard"], columns="metric",
                       values="value", aggfunc="first").reset_index()
    ci = w[w.metric == "macro_f1"].drop_duplicates("run_id").set_index("run_id")[["ci_low", "ci_high", "n_events", "mean_fold_score", "std_fold_score"]]
    lad = pv.join(ci, on="run_id").rename(columns={"model_id": "head", "revision": "encoder"})
    O.csv("ladder.csv", lad)
    folds = rl[rl.block.str.match(r"^wf\d$") & (rl.metric == "macro_f1") & (rl.event_set == "all") & (rl.target_def == "T_trade")
               & (rl.horizon == 1) & rl.cohort.isin(["common", "common_emb"])]
    O.csv("ladder_folds.csv", folds[["run_id", "rung", "model_id", "revision", "tuned", "block", "value"]].rename(
        columns={"model_id": "head", "revision": "encoder", "value": "macro_f1"}))
    tb = run.read("S11", "ladder_summary.csv")
    t15 = run.read("S15", "emb_ladder_summary.csv")
    tba = pd.concat([tb.assign(encoder="-")] + ([t15] if t15 is not None else []), ignore_index=True)
    O.csv("tuning.csv", tba[["rung", "encoder", "head", "default_mean_f1", "tuned_mean_f1", "tuned_std"] + (["params"] if "params" in tba else [])])
    O.csv("fit_diagnosis.csv", run.read("S11", "fit_diagnosis.csv"))
    O.csv("learning_curves.csv", run.read("S11", "learning_curves.csv"))
    O.csv("permutation.csv", run.read("S11", "permutation.csv"))
    tri = run.read("S11", "optuna_trials.csv")
    O.csv("optuna_trials.csv", tri)
    adv = run.read("S11", "adversarial.csv")
    if adv is not None:
        O.csv("adversarial.csv", adv[["fold", "auc", "auc_all_features", "top_shift_features", "n_train", "n_val"]])
    O.json("best_params.json", run.read("S11", "best_params.json"))
    O.json("enet_certificate.json", run.read("S11", "enet_certificate.json"))
    O.csv("bucket_composition.csv", run.read("S11", "bucket_composition.csv"))
    O.csv("emb_ladder_summary.csv", t15)
    O.json("g3.json", run.read("S15", "g3.json"))
    O.csv("h4.csv", run.read("S15", "h4.csv"))
    O.json("finetune_status.json", run.read("S15", "finetune_status.json"))
    O.json("preflight.json", run.read("S13", "preflight.json"))
    O.json("s14.json", run.success("S14"))
    O.json("s15.json", run.success("S15"))
    # development holdout and locked test
    out = {}
    for sid, name, tag in (("S18", "dev_results.parquet", "dev"), ("S19", "locked_results.parquet", "locked")):
        r = run.read(sid, name)
        if r is None:
            continue
        m = r[r.metric.isin(keep) & (r.event_set == "all")]
        p = m.pivot_table(index=["run_id", "rung", "model_id", "revision", "leaderboard", "block"], columns="metric", values="value",
                          aggfunc="first").reset_index()
        c = m[m.metric == "macro_f1"].drop_duplicates("run_id").set_index("run_id")[["ci_low", "ci_high", "n_events"]]
        O.csv(f"{tag}_results.csv", p.join(c, on="run_id").rename(columns={"model_id": "head", "revision": "encoder"}))
    O.csv("dev_lifts.csv", run.read("S18", "dev_lifts.csv"))
    fz = run.read("S18", "frozen_architecture.json")
    O.json("frozen_architecture.json", fz)
    O.json("mcs_pbo.json", run.read("S18", "mcs_pbo.json"))
    lk = run.root / "manifests" / "lockbox_log.json"
    O.json("lockbox_log.json", json.loads(lk.read_text(encoding="utf-8")) if lk.exists() else None)
    # results in context (same functions as the report, section 5a)
    from secprice.evaluation.context import agreement, class_shares, paired_lifts, skill_decomposition
    p11, p15 = run.read("S11", "predictions.parquet"), run.read("S15", "predictions.parquet")
    pdv, plk = run.read("S18", "dev_predictions.parquet"), run.read("S19", "locked_predictions.parquet")
    head = ((run.read("S11", "best_params.json") or {}).get("best_L6") or {}).get("head")
    l7 = t15[t15.rung == "L7"].sort_values("tuned_mean_f1", ascending=False).iloc[0] if t15 is not None and (t15.rung == "L7").any() else None
    blocks = {"walk-forward": {}, "development": {}, "locked": {}}
    if p11 is not None and head:
        for rung in ("L1", "L6", "L6_nochange"):
            x = p11[p11.run_id == f"common|{rung}|{head}|-|T_trade|h1|tuned"]
            if len(x):
                blocks["walk-forward"][rung] = x
    if p15 is not None and l7 is not None:
        x = p15[p15.run_id == f"common_emb|L7|{l7['head']}|{l7['encoder']}|T_trade|h1|tuned"]
        y = p15[p15.run_id == f"common_emb|L6|{l7['head']}|-|T_trade|h1|tuned"]
        if len(x):
            blocks["walk-forward"]["L7"] = x
        if len(y):
            blocks["walk-forward"]["L6@L7head"] = y
    for key, df in (("development", pdv), ("locked", plk)):
        if df is not None and "name" in df:
            blocks[key] = {nm: df[df.name == nm] for nm in ("L1", "L6", "L6_nochange", "L7") if (df.name == nm).any()}
    skill, lifts = [], []
    for blk, mods in blocks.items():
        for nm in ("L1", "L6", "L7"):
            if nm in mods:
                skill.append({"period": blk, "model": nm, **skill_decomposition(mods[nm], 300, 0)})
        for a, b, label in (("L7", "L6", "L7 - L6 (text)"), ("L6", "L6_nochange", "L6 - L6_nochange (change features)"),
                            ("L6", "L1", "L6 - L1 (structured over price)")):
            mm = dict(mods)
            if blk == "walk-forward" and a == "L7" and "L6@L7head" in mm:
                mm["L6"] = mm["L6@L7head"]
            r = paired_lifts(mm, [(a, b, label)], 1000, 0)
            if len(r):
                lifts.append({"period": blk, **r.iloc[0].to_dict()})
    O.csv("skill.csv", pd.DataFrame(skill))
    O.csv("lifts.csv", pd.DataFrame(lifts))
    out["class_shares"] = {blk: class_shares(m["L6"]) for blk, m in blocks.items() if "L6" in m}
    rep = []
    if p11 is not None and p15 is not None:
        for h in ("lightgbm", "enet", "catboost"):
            a = p11[p11.run_id == f"common|L6|{h}|-|T_trade|h1|tuned"]
            b = p15[p15.run_id == f"common_emb|L6|{h}|-|T_trade|h1|tuned"]
            if len(a) and len(b):
                rep.append({"head": h, **agreement(a, b)})
    O.csv("reproducibility.csv", pd.DataFrame(rep))
    # confusion matrices of the locked test
    if plk is not None:
        cms = []
        for nm, g in plk.groupby("name"):
            ct = pd.crosstab(g.y.astype(int), g.yhat.astype(int)).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            for i in range(3):
                for j in range(3):
                    cms.append({"model": nm, "true": ["Down", "Sideways", "Up"][i], "pred": ["Down", "Sideways", "Up"][j], "events": int(ct.iloc[i, j])})
        O.csv("locked_confusion.csv", pd.DataFrame(cms))
    perm = run.read("S11", "permutation.csv")
    l0 = lad[(lad.rung == "L0")].macro_f1.max() if (lad.rung == "L0").any() else None
    out.update({"majority_baseline_f1": l0, "permutation_null_mean": float(perm[perm.kind == "permuted"].macro_f1.mean()) if perm is not None else None,
                "l6_head": head, "l7": None if l7 is None else {"head": l7["head"], "encoder": l7["encoder"], "wf": float(l7["tuned_mean_f1"])}})
    s15_l6 = lad[lad.run_id.str.startswith("common_emb|L6|") & (lad.tuned == "tuned")]
    out["s15_l6_best_mean_fold"] = float(s15_l6.mean_fold_score.max()) if len(s15_l6) else None
    return out


def track_a(run: Run, O: Out) -> None:
    O.csv("track_a_groups.csv", run.read("S20", "track_a_groups.csv"))
    O.csv("car_paths.csv", run.read("S20", "car_paths.csv"))
    O.csv("correlations.csv", run.read("S20", "correlations.csv"))
    O.csv("topics.csv", run.read("S20", "topics.csv"))
    O.csv("hypothesis_tests.csv", run.read("S21", "hypothesis_tests.csv"))
    d = run.read("S20", "track_a_cars.parquet")
    fe = run.read("S09", "features.parquet", columns=["event_id", "sue_event"])
    if d is not None and fe is not None:
        from secprice.track_a.drift import prior_quarter_quintiles
        r = d[d.earnings_type == "earnings_announcement_verified"].merge(fe, on="event_id")
        if r.sue_event.notna().sum() >= 50:
            r["q"] = prior_quarter_quintiles(r, "sue_event")
            g = r.dropna(subset=["q"]).groupby("q").car_2_60.agg(["mean", "sem", "size"]).reset_index()
            O.csv("sue_quintiles.csv", g.rename(columns={"size": "events"}))


def backtest(run: Run, O: Out) -> None:
    O.csv("backtest.csv", run.read("S22", "backtest.csv"))
    O.json("backtest_spec.json", run.read("S22", "backtest_spec.json"))
    bd = run.read("S22", "backtest_daily.parquet")
    if bd is not None:
        bd = bd[bd.sp_bucket == "Combined"] if "sp_bucket" in bd else bd
        O.parquet("backtest_daily.parquet", bd[["date", "block", "lag", "cost_bps", "long_ret", "short_ret", "ls_ret", "spy_matched",
                                                "long_excess_spy", "n_long", "n_short"]])


def leakage(run: Run, O: Out) -> None:
    from secprice.report.narrative import relabel_stress_audit
    rl = pd.concat([x for x in (run.read("S11", "results_long.parquet"), run.read("S15", "results_long.parquet")) if x is not None], ignore_index=True)
    O.csv("leakage_audit.csv", relabel_stress_audit(run.read("S10", "leakage_audit.csv"), rl))
    O.json("planted_leaks.json", run.read("S10", "planted_leaks.json"))
    O.csv("feature_registry.csv", run.read("S09", "feature_registry.csv"))
    O.csv("model_ledger.csv", run.read("S10", "model_provenance_ledger.csv"))
    O.csv("fold_provenance.csv", run.read("S10", "fold_provenance.csv"))
    bl = run.read("S10", "blocks.csv")
    ev = run.read("S07", "events.parquet", columns=["decision_time"])
    if bl is not None and ev is not None:
        dt = pd.to_datetime(ev.decision_time, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        bl["events"] = [int(((dt >= pd.Timestamp(r.start)) & (dt < pd.Timestamp(r.end_exclusive))).sum()) for r in bl.itertuples()]
        O.csv("blocks.csv", bl)


def rag(run: Run, O: Out) -> dict:
    import yaml
    qb = yaml.safe_load((REPO / "configs" / "question_bank_v1.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
    s16 = run.success("S16")
    rows, examples = [], []
    p = run.path("S16", "llm_cache.jsonl")
    if p is not None:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                v = json.loads(line)["v"]
            except Exception:
                continue
            pres = 0
            for f, o in v.items():
                if isinstance(o, dict):
                    rows.append({"field": f, "present": bool(o.get("present")), "certainty": o.get("certainty"),
                                 "direction": o.get("direction"), "n_quotes": len(o.get("evidence") or [])})
                    pres += bool(o.get("present"))
            if pres >= 3:
                examples.append((pres, v))
    O.csv("rag_prevalence_rows.csv", pd.DataFrame(rows))
    examples.sort(key=lambda t: -t[0])
    info = {"question_bank": qb, "rag_config": cfg.get("rag"), "distill_config": cfg.get("distill"), "gates": cfg.get("gates"),
            "s16": s16, "g3b": run.read("S16", "g3b.json"), "rag_frozen": run.read("S16", "rag_frozen.json"),
            "tournament": (run.read("S16", "rag_tournament.csv") if run.path("S16", "rag_tournament.csv") else pd.DataFrame()).to_dict("records"),
            "student": run.read("S17", "student_report.json"), "gold_status": run.read("S12", "gold_status.json"),
            "examples": [v for _, v in examples[:3]], "valid_responses": len({i for i in range(len(rows) // 12)}) if rows else 0}
    O.json("rag.json", info)
    return info


def decisions(O: Out) -> None:
    dl = REPO / "docs" / "decision_log.md"
    rows = []
    if dl.exists():
        for line in dl.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\|\s*(D-\d+)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$", line)
            if m:
                txt = re.sub(r"\*\*|`", "", m.group(2))
                rows.append({"id": m.group(1), "summary": txt.split(". ")[0][:220], "decision": txt, "why": m.group(3), "spec": m.group(4)})
    O.csv("decisions.csv", pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the profile folder of a finished run, e.g. <DATA_ROOT>/full")
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    a = ap.parse_args()
    run, O = Run(Path(a.root)), Out(Path(a.out))
    meta = {"source_root": str(run.root), "built_utc": pd.Timestamp.utcnow().isoformat()}
    man = run.root / "manifests" / "run_manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        meta.update({k: m.get(k) for k in ("profile", "config_hash", "git_commit", "code_hash", "timing_policy", "data_sources")})
    st = stages(run)
    O.csv("stages.csv", st)
    meta["eda"] = eda(run, O)
    meta["modelling"] = modelling(run, O)
    track_a(run, O)
    backtest(run, O)
    leakage(run, O)
    rg = rag(run, O)
    meta["rag_summary"] = {"prefilled": rg["s16"].get("prefilled_events"), "seconds_per_filing": rg["s16"].get("seconds_per_filing"),
                           "generation_hours": (rg["s16"].get("generation_seconds") or 0) / 3600}
    decisions(O)
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
    prof = yaml.safe_load((REPO / "configs" / "profiles" / "full.yaml").read_text(encoding="utf-8")) or {}
    meta["config"] = {k: cfg.get(k) for k in ("study", "timing", "targets", "labels", "splits", "text", "tuning", "models", "backtest", "speed",
                                               "contam_window_sessions", "gates", "bootstrap")}
    meta["profile_overrides"] = prof
    meta["gpu_hours"] = float(st[st.part.isin(["GPU", "CPU-GPU"])].hours.fillna(0).sum())
    meta["cpu_hours"] = float(st[st.part == "CPU"].hours.fillna(0).sum())
    O.json("meta.json", meta)
    # figures and the report
    fig_src = run.root / "reports" / "figures"
    fig_dst = Path(a.out) / "figures"
    if fig_src.exists():
        shutil.rmtree(fig_dst, ignore_errors=True)
        shutil.copytree(fig_src, fig_dst)
    for f in ("results_report.pdf", "results_summary.md"):
        if (run.root / "reports" / f).exists():
            shutil.copy2(run.root / "reports" / f, Path(a.out) / f)
    print(f"wrote {len(O.written)} files to {a.out}")


if __name__ == "__main__":
    main()
