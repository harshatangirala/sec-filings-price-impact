"""S24 narrative sections (architecture, data, pre-processing, models, RAG, evaluation design, results in context, known issues).

Every number is read from a stage artifact or from the configuration; prose only describes what the code does. A section whose
inputs are missing says so instead of being left blank (hard rule 5). Called once from stages_report.collect.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .build import fmt, md_table

BLOCK_LABEL = {"wf": "walk-forward 2008-2019", "dev": "development 2020-2022", "locked": "locked 2023-2026"}
BLOCK_LABEL_ = BLOCK_LABEL


# ---------------------------------------------------------------------------------------------------- helpers
def _safe(ctx, name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:                                           # a missing input must never break the report
        ctx.warn(f"report section '{name}' not built: {type(e).__name__}: {str(e)[:160]}")
        return f"_(section not built: {type(e).__name__}: {str(e)[:120]})_"


def _success(ctx, sid) -> dict:
    d = ctx.dep_or_none(sid)
    if d is None or not (Path(d) / "_SUCCESS").exists():
        return {}
    try:
        return json.loads((Path(d) / "_SUCCESS").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(x, nd=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}%"


def _f(x, nd=3) -> str:
    return fmt(x, nd)


def _ci(lo, hi, nd=3) -> str:
    return f"[{_f(lo, nd)}, {_f(hi, nd)}]"


# ---------------------------------------------------------------------------------------------------- A2 architecture
def stage_table(ctx) -> pd.DataFrame:
    from ..runner import REGISTRY
    rows = []
    for sid in sorted(REGISTRY, key=lambda s: int(s[1:])):
        sp = REGISTRY[sid]
        sx = _success(ctx, sid)
        wall = sx.get("wall_s")
        note = None
        if sid == "S16" and (sx.get("g3b") or {}).get("passed") is False:
            note = "ran, not scored"
        if sid == "S17" and sx.get("ran") is False:
            note = "not run"
        if sid == "S23" and sx.get("evaluated") is False:
            note = "not evaluated"
        rows.append({"stage": sid, "name": sp.name, "part": sp.part, "needs": ", ".join(sp.deps) or "-",
                     "main outputs": ", ".join(o for o in sp.outputs[:3]), "hours": None if wall is None else wall / 3600.0,
                     "result": note or ("done" if sx else "not run")})
    return pd.DataFrame(rows)


def draw_pipeline(ctx, F, stages: pd.DataFrame):
    from ..viz import diagrams
    st = [{"id": r.stage, "name": r["name"], "part": r.part, "hours": r.hours,
           "note": r.result if r.result not in ("done",) else None} for _, r in stages.iterrows()]
    F.safe("A01", "Pipeline architecture: 25 stages in three parts", lambda i, t: diagrams.pipeline(F, i, t, st))


# ---------------------------------------------------------------------------------------------------- A1 hypotheses
def hypotheses_table(facts) -> pd.DataFrame:
    rows = []
    lt = facts.get("lifts_period")
    def lift(label, col):
        if lt is None or not len(lt):
            return None
        r = lt[lt.comparison == label]
        if r.empty or pd.isna(r.iloc[0].get(col)):
            return None
        x = r.iloc[0]
        return f"{x[col]:+.3f} {_ci(x[col + '_lo'], x[col + '_hi'])}"
    h4 = facts.get("h4_summary") or "n/a"
    stack = facts.get("stack_summary") or "not run"
    rows.append({"id": "H1", "claim": "SEC text adds out-of-time value beyond price, metadata and accounting",
                 "test": "paired macro-F1 lift L7 - L6, month-block bootstrap, identical events",
                 "walk-forward": lift("L7 - L6", "wf"), "development": lift("L7 - L6", "dev"), "locked": lift("L7 - L6", "locked"),
                 "verdict": facts.get("h1_verdict", "n/a")})
    rows.append({"id": "H2", "claim": "Changes in filing language add more than absolute sentiment",
                 "test": "L6 (with change features) - L6_nochange", "walk-forward": lift("L6 - L6_nochange", "wf"),
                 "development": lift("L6 - L6_nochange", "dev"), "locked": lift("L6 - L6_nochange", "locked"),
                 "verdict": facts.get("h2_verdict", "n/a")})
    rows.append({"id": "H3", "claim": "Validated structured extraction adds value beyond embeddings", "test": "L8 - L7",
                 "walk-forward": "not run", "development": "not run", "locked": "not run",
                 "verdict": "not testable: no gold labels, so gate G3b could not pass and the student model was not trained"})
    rows.append({"id": "H4", "claim": "Point-in-time (vintage) encoders differ from static ones", "test": "ChronoBERT vintage - BGE, same rows and heads",
                 "walk-forward": h4, "development": "n/a", "locked": "n/a", "verdict": facts.get("h4_verdict", "n/a")})
    rows.append({"id": "H5", "claim": "A stacked ensemble beats the best single model", "test": "STACK - L7 on the development holdout",
                 "walk-forward": "n/a", "development": stack, "locked": "n/a", "verdict": facts.get("h5_verdict", "n/a")})
    rows.append({"id": "H6", "claim": "Richer RAG designs beat hybrid + rerank", "test": "retrieval tournament on gold-val qrels",
                 "walk-forward": "n/a", "development": "n/a", "locked": "n/a", "verdict": "not testable: needs gold-val labels"})
    rows.append({"id": "H7", "claim": "Analogue retrieval or time-series models add value", "test": "extensions (default off)",
                 "walk-forward": "n/a", "development": "n/a", "locked": "n/a", "verdict": "not run (extension disabled)"})
    g = facts.get("gain_period")
    rows.append({"id": "H8", "claim": "Gains shrink under strict timestamps, costs, clean events and the locked test",
                 "test": "L6 - L1 (all structured vs price-only) by period; backtest net of cost",
                 "walk-forward": g and g.get("wf"), "development": g and g.get("dev"), "locked": g and g.get("locked"),
                 "verdict": facts.get("h8_verdict", "n/a")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- A3 data
def data_sources_table(ctx, facts, usel) -> pd.DataFrame:
    cfg = ctx.cfg
    man = {}
    p = ctx.storage.manifests / "run_manifest.json"
    if p.exists():
        man = json.loads(p.read_text(encoding="utf-8")).get("data_sources", {})
    cov = facts.get("coverage_by_form")
    forms = "; ".join(f"{r.base_form}: {int(r.observed):,} filings, {int(r.parsed):,} parsed" for r in cov.itertuples()) if cov is not None else "n/a"
    pc = ctx_read_csv(ctx, "S03", "price_coverage.csv")
    span = f"{pc['first'].min()} to {pc['last'].max()}; {int(pc.n_bars.sum()):,} daily bars for {len(pc)} firms" if pc is not None else "n/a"
    xf = _read_pq(ctx, "S06", "xbrl_facts.parquet", ["tag"])
    er = _read_pq(ctx, "S06", "eps_release.parquet", ["is_results_release", "eps_diluted"])
    su = _read_pq(ctx, "S06", "sue_by_accession.parquet", ["sue"])
    ex = {
        "xbrl": f"{len(xf):,} facts, {xf.tag.nunique()} tags" if xf is not None else "n/a",
        "eps": f"{len(er):,} exhibit rows, {int(er.is_results_release.sum()):,} results releases" if er is not None else "n/a",
        "sue": f"{len(su):,} filings, SUE available for {int(su.sue.notna().sum()):,}" if su is not None else "n/a"}
    hf = man.get("hf_dataset", f"{cfg['hf']['repo']}@{cfg['hf']['revision']}")
    rows = [
        {"source": "SEC EDGAR (10-K, 10-Q, 8-K, EX-99 exhibits)", "used for": "all study text; filing lists and acceptance times (SGML header, else submissions JSON)",
         "scope in this run": forms, "access": "public; User-Agent from the environment, <= 6 requests/s"},
        {"source": "EDGAR XBRL company facts", "used for": "accounting features, quarterly EPS as first reported, SUE", "scope in this run": ex["xbrl"], "access": "public"},
        {"source": "Earnings-release exhibits (EX-99)", "used for": "release EPS parsed for SUE and verified earnings dates",
         "scope in this run": f"{ex['eps']}; {ex['sue']}", "access": "parsed from EDGAR"},
        {"source": "Hugging Face SEC corpus " + str(hf).split("@")[0], "used for": "firm overlap with the study universe and parser QA only, never the study text (10-K only)",
         "scope in this run": f"revision {str(hf).split('@')[-1][:10]}", "access": "public dataset"},
        {"source": "Daily prices (OHLCV): owner's local files, spliced with Yahoo Finance", "used for": "returns, volatility, price features, benchmark and backtest",
         "scope in this run": span, "access": "local files + yfinance (auto_adjust off; actions kept)"},
        {"source": "Fama-French / Carhart factors and the 12-industry map", "used for": "Carhart alpha in the backtest, industry grouping",
         "scope in this run": "factor files in the inputs folder (ff5, momentum, Siccodes12)", "access": "Kenneth French data library"},
        {"source": "S&P 500 membership history " + str(man.get("sp500_membership", "")).split("@")[0], "used for": "index bucket and point-in-time membership flag",
         "scope in this run": f"{len(usel)} firms, all ever members in the window", "access": "public GitHub list, pinned commit"},
        {"source": "Loughran-McDonald master dictionary 1993-2025", "used for": "lexical tone features", "scope in this run": "sha256 pinned",
         "access": "academic use"},
    ]
    return pd.DataFrame(rows)


def ctx_read_csv(ctx, sid, name):
    d = ctx.dep_or_none(sid)
    if d is None or not (Path(d) / name).exists():
        return None
    try:
        return pd.read_csv(Path(d) / name)
    except Exception:
        return None


def _read_pq(ctx, sid, name, cols=None):
    d = ctx.dep_or_none(sid)
    if d is None or not (Path(d) / name).exists():
        return None
    return pd.read_parquet(Path(d) / name, columns=cols)


def tickers_text(usel) -> str:
    by = {}
    for r in usel.itertuples():
        by.setdefault(str(r.ff12_industry), []).append(str(r.ticker))
    lines = [f"- **{k}** ({len(v)}): " + ", ".join(sorted(v)) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- A4 pre-processing
def preprocessing(ctx, facts) -> tuple[str, pd.DataFrame]:
    cfg, tx = ctx.cfg, ctx.cfg["text"]
    si = _read_pq(ctx, "S05", "sections_index.parquet")
    led = _read_pq(ctx, "S05", "parse_ledger.parquet", ["status", "form", "bytes", "n_tables", "truncated", "fallback"])
    pf = _success(ctx, "S13")
    lines = []
    if si is not None:
        t = si.groupby("section_id").agg(sections=("accession", "size"), median_tokens=("n_tokens_est", "median"),
                                          truncated=("truncated", "sum"), by_reference=("by_reference", "sum")).reset_index()
        sec_tab = t
    else:
        sec_tab = pd.DataFrame()
    steps = [
        ("1 Download", "Full-submission text per filing from EDGAR (SGML header read for every filing; exhibits for 8-K). Requests are rate-limited, cached on disk (compressed) and retried per filing.",
         f"{len(led):,} filings processed; {int((led.status == 'ok').sum()):,} parsed ok, {int((led.status != 'ok').sum()):,} without a target section" if led is not None else "n/a"),
        ("2 HTML to text", "HTML / SGML to normalised plain text; tables are flattened to 'cell | cell' rows so numeric parsers can read them; page-number and table-border debris is removed.",
         f"{int(led.n_tables.sum()):,} tables flattened" if led is not None and "n_tables" in led else "n/a"),
        ("3 Section extraction", "Headings found by item number plus title keywords; of several candidate starts (table-of-contents lines repeat headings) the longest span wins; sections incorporated by reference are flagged and recovered from EX-13 where possible. Targets: "
         + "; ".join(f"{k}: {', '.join(v)}" for k, v in tx["sections"].items()) + ".",
         f"{len(si):,} sections kept" if si is not None else "n/a"),
        ("4 Firm-name masking", "The firm's own legal names, former names and tickers are masked in every section (firm-memorisation control, leakage item 4). A named-entity masker (bert-base-NER) exists for the GPU stage and is off by default.",
         f"{int(si.mask_firm.sum()):,} firm-name and {int(si.mask_ticker.sum()):,} ticker replacements" if si is not None and "mask_firm" in si else "n/a"),
        ("5 Numeric-line filter and caps", f"Numeric-heavy lines are dropped from narrative text (drop_numeric_lines={tx['drop_numeric_lines']}); each section is capped at {tx['max_chars_per_section']:,} characters and the text passed to models at {tx['model_text_chars']:,} characters.",
         f"{int(si.truncated.sum()):,} sections hit the cap" if si is not None and "truncated" in si else "n/a"),
        ("6 Chunking", f"Sliding windows of {tx['chunk_words']} words with {tx['chunk_overlap_words']} words overlap, at most {tx['max_chunks_per_section']} chunks per section; the same chunks feed every encoder and the retriever.",
         f"about {pf.get('est_chunks', 0):,} chunks, {pf.get('est_tokens', 0):,} tokens (preflight estimate)" if pf else "n/a"),
        ("7 Tokenisation", f"Each encoder's own tokenizer, truncated to {cfg['models']['max_seq_len']} tokens; batches are length-sorted (batch {cfg['models']['embed_batch']}, halved automatically on out-of-memory).", "-"),
        ("8 Pooling to event vectors", "Chunk vectors are mean- and max-pooled to section vectors; an event vector is the chunk-weighted mean of its section means; the 'change' vector is the cosine to the firm's previous event of the same type (decision time strictly earlier).",
         f"{_success(ctx, 'S14').get('bge_base', {}).get('events', 'n/a'):,} events embedded per encoder" if isinstance(_success(ctx, 'S14').get('bge_base', {}).get('events'), int) else "n/a"),
        ("9 Classic text features", "Loughran-McDonald tone counts, filing-change similarity (fixed hashing space, so no vocabulary is learned from future data) and a novelty share; fold-fitted TF-IDF + SVD features are added inside each training fold only.", "see feature table"),
    ]
    return "\n".join(lines), pd.DataFrame(steps, columns=["step", "what is done", "measured in this run"]), sec_tab


def features_table(ctx) -> pd.DataFrame:
    fr = ctx_read_csv(ctx, "S09", "feature_registry.csv")
    if fr is None:
        return pd.DataFrame()
    g = fr.groupby("family").agg(n=("name", "size"), examples=("name", lambda s: ", ".join(list(s)[:4])), source=("source", "first"),
                                 availability=("availability_rule", "first"), leaderboard=("leaderboard", lambda s: "/".join(sorted(set(s))))).reset_index()
    return g.rename(columns={"n": "features"})


# ---------------------------------------------------------------------------------------------------- A7 models
def encoder_table(ctx) -> pd.DataFrame:
    from .. import common as C
    cfg = ctx.cfg
    ros = C.roster(cfg)["encoders"]
    rows = []
    in_use = set(cfg["models"]["encoders"])
    for k, v in ros.items():
        rows.append({"encoder": k, "model": v.get("model_id") or v.get("model_id_template"), "training data": v.get("training_corpus", "n/a"),
                     "cutoff": v.get("stated_cutoff", "unknown"), "pooling": v.get("pooling"),
                     "in this run": "yes" if k in in_use else "no (fast mode)" if (cfg.get("speed") or {}).get("mode") == "fast" else "no"})
    return pd.DataFrame(rows)


def head_params_table(ctx) -> pd.DataFrame:
    bp = ctx_json(ctx, "S11", "best_params.json") or {}
    p15 = ctx_read_csv(ctx, "S15", "emb_ladder_summary.csv")
    rows = []
    for key, par in (bp.get("params") or {}).items():
        if key.split("|")[0] in ("L1", "L2", "L3", "L4", "L6") and par:
            rows.append({"model": key, "tuned hyperparameters": json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in par.items()})})
    if p15 is not None:
        for r in p15.itertuples():
            if r.rung == "L7":
                par = json.loads(r.params) if isinstance(r.params, str) else {}
                rows.append({"model": f"L7|{r.head}|{r.encoder}", "tuned hyperparameters": json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in par.items()})})
    return pd.DataFrame(rows)


def ctx_json(ctx, sid, name):
    d = ctx.dep_or_none(sid)
    if d is None or not (Path(d) / name).exists():
        return None
    return json.loads((Path(d) / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------- A8 RAG
def rag_facts(ctx) -> dict:
    s16, s12 = _success(ctx, "S16"), ctx_json(ctx, "S12", "gold_status.json") or {}
    out = {"s16": s16}
    d = ctx.dep_or_none("S16")
    cache_rows = []
    if d is not None and (Path(d) / "llm_cache.jsonl").exists():
        for line in (Path(d) / "llm_cache.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                cache_rows.append(json.loads(line)["v"])
            except Exception:
                continue
    out["cache_n"] = len(cache_rows)
    prev = []
    for v in cache_rows:
        for field, o in v.items():
            if isinstance(o, dict):
                prev.append({"field": field, "present": bool(o.get("present")), "certainty": o.get("certainty"), "n_quotes": len(o.get("evidence") or []),
                             "direction": o.get("direction")})
    out["prevalence"] = pd.DataFrame(prev)
    # one real extraction as an illustration (largest number of present fields, first such filing)
    best = None
    for v in cache_rows:
        pres = [k for k, o in v.items() if isinstance(o, dict) and o.get("present")]
        if len(pres) >= 3 and (best is None or len(pres) > best[0]):
            best = (len(pres), v)
    out["example"] = best[1] if best else None
    out["gold"] = s12
    return out


def draw_rag(ctx, F, rg: dict):
    from ..viz import diagrams
    s16 = rg["s16"]
    c = {"sections": f"{_success(ctx, 'S05').get('sections', '')}",
         "chunks": "", "gen": f"{s16.get('seconds_per_filing', 'n/a')} s per filing" if s16 else "",
         "cache": f"{rg['cache_n']} filings", "templates": f"{s16.get('prefilled_events', 'n/a')} filings"}
    pf = _success(ctx, "S13")
    c["sections"] = f"{pf.get('sections', ''):,} sections" if isinstance(pf.get("sections"), int) else ""
    c["chunks"] = f"~{pf.get('est_chunks', 0):,}" if pf else ""
    F.safe("A04", "RAG and extraction system", lambda i, t: diagrams.rag(F, i, t, c))


def prevalence_figure(ctx, F, prev: pd.DataFrame):
    if prev is None or prev.empty:
        F.skip("R06", "Qwen pre-fill: how often each field was extracted", "no cached extractions")
        return
    g = prev.groupby("field").present.mean().sort_values()
    def draw(i, t):
        fig, ax = F.plt.subplots(figsize=(9, 4.6))
        ax.barh(g.index, g.values * 100, color="#2563eb")
        ax.set_xlabel("% of filings where the field was extracted as present (unvalidated model output)")
        F.save(fig, i, t)
    F.safe("R06", "Qwen pre-fill: how often each field was extracted", draw)


# ---------------------------------------------------------------------------------------------------- results in context
def _wf_runs(ctx, preds11, preds15, head_l6, l7row):
    """Walk-forward prediction frames for L1, L6, L6_nochange (S11, best-L6 head) and L7 + its L6 twin (S15)."""
    out = {}
    for rung in ("L1", "L6", "L6_nochange"):
        rid = f"common|{rung}|{head_l6}|-|T_trade|h1|tuned"
        if preds11 is not None and (preds11.run_id == rid).any():
            out[rung] = preds11[preds11.run_id == rid]
    if preds15 is not None and l7row is not None:
        r7 = f"common_emb|L7|{l7row['head']}|{l7row['encoder']}|T_trade|h1|tuned"
        r6 = f"common_emb|L6|{l7row['head']}|-|T_trade|h1|tuned"
        if (preds15.run_id == r7).any():
            out["L7"] = preds15[preds15.run_id == r7]
        if (preds15.run_id == r6).any():
            out["L6@L7head"] = preds15[preds15.run_id == r6]
    return out


def results_in_context(ctx, facts) -> dict:
    from ..evaluation.context import agreement, class_shares, paired_lifts, skill_decomposition
    cfg = ctx.cfg
    nb, seed = int(cfg["bootstrap"]["n"]), int(cfg["seed"])
    nboot_auc = min(300, nb)
    d11, d15, d18, d19 = (ctx.dep_or_none(s) for s in ("S11", "S15", "S18", "S19"))
    p11 = pd.read_parquet(d11 / "predictions.parquet") if d11 is not None and (d11 / "predictions.parquet").exists() else None
    p15 = pd.read_parquet(d15 / "predictions.parquet") if d15 is not None and (d15 / "predictions.parquet").exists() else None
    pdv = pd.read_parquet(d18 / "dev_predictions.parquet") if d18 is not None and (d18 / "dev_predictions.parquet").exists() else None
    plk = pd.read_parquet(d19 / "locked_predictions.parquet") if d19 is not None and (d19 / "locked_predictions.parquet").exists() else None
    bp = ctx_json(ctx, "S11", "best_params.json") or {}
    head_l6 = (bp.get("best_L6") or {}).get("head")
    sm15 = ctx_read_csv(ctx, "S15", "emb_ladder_summary.csv")
    l7row = None
    if sm15 is not None and (sm15.rung == "L7").any():
        l7row = sm15[sm15.rung == "L7"].sort_values("tuned_mean_f1", ascending=False).iloc[0].to_dict()
    wf = _wf_runs(ctx, p11, p15, head_l6, l7row) if head_l6 else {}
    blocks = {"wf": wf}
    for key, df in (("dev", pdv), ("locked", plk)):
        if df is not None and "name" in df:
            blocks[key] = {nm: df[df.name == nm] for nm in ("L1", "L6", "L6_nochange", "L7") if (df.name == nm).any()}
    skill = []
    for blk, models in blocks.items():
        for nm in ("L1", "L6", "L7"):
            if nm in models and len(models[nm]):
                sd = skill_decomposition(models[nm], nboot_auc, seed)
                skill.append({"period": BLOCK_LABEL[blk], "blk": blk, "model": nm, **sd})
    skill = pd.DataFrame(skill)
    shares = {blk: class_shares(m["L6"]) for blk, m in blocks.items() if "L6" in m}
    # paired lifts by period
    lifts = []
    for label, a, b in (("L7 - L6", "L7", "L6"), ("L6 - L6_nochange", "L6", "L6_nochange"), ("L6 - L1", "L6", "L1")):
        row = {"comparison": label}
        for blk, models in blocks.items():
            mm = dict(models)
            if blk == "wf" and label == "L7 - L6" and "L6@L7head" in mm:
                mm["L6"] = mm["L6@L7head"]                      # same head and rows as L7 (S15 re-score)
            r = paired_lifts(mm, [(a, b, label)], nb, seed)
            if len(r):
                row.update({blk: float(r.lift.iloc[0]), blk + "_lo": float(r.ci_low.iloc[0]), blk + "_hi": float(r.ci_high.iloc[0]),
                            blk + "_p": float(r.p_boot.iloc[0]), blk + "_n": int(r.n_events.iloc[0])})
        lifts.append(row)
    lifts = pd.DataFrame(lifts)
    out = {"skill": skill, "shares": shares, "lifts": lifts, "head_l6": head_l6, "l7": l7row}
    # reproducibility: the same L6 model re-scored in S11 (CPU stage) and S15 (GPU stage), same rows and folds
    rep = []
    if p11 is not None and p15 is not None:
        for head in ("lightgbm", "enet", "catboost"):
            a = p11[p11.run_id == f"common|L6|{head}|-|T_trade|h1|tuned"]
            b = p15[p15.run_id == f"common_emb|L6|{head}|-|T_trade|h1|tuned"]
            if len(a) and len(b):
                rep.append({"head": head, **agreement(a, b)})
    out["repro"] = pd.DataFrame(rep)
    return out


def apply_context_to_facts(facts, rc: dict) -> None:
    lt = rc["lifts"]
    if lt is not None and len(lt):
        flat = []
        for _, r in lt.iterrows():
            d = {"comparison": r.comparison}
            for blk in ("wf", "dev", "locked"):
                if blk in r and pd.notna(r.get(blk)):
                    d.update({blk: r[blk], blk + "_lo": r[blk + "_lo"], blk + "_hi": r[blk + "_hi"], blk + "_p": r[blk + "_p"]})
            flat.append(d)
        facts["lifts_period"] = pd.DataFrame(flat)

        def verdict(label):
            r = lt[lt.comparison == label]
            if r.empty:
                return "n/a"
            r = r.iloc[0]
            cis = [(r.get(b + "_lo"), r.get(b + "_hi")) for b in ("wf", "dev", "locked") if pd.notna(r.get(b))]
            if not cis:
                return "n/a"
            if all(lo is not None and lo > 0 for lo, hi in cis):
                return "supported in every period"
            if all(not (lo > 0 or hi < 0) for lo, hi in cis):
                return "not supported: every confidence interval includes 0"
            return "mixed: see table"
        facts["h1_verdict"] = verdict("L7 - L6")
        facts["h2_verdict"] = verdict("L6 - L6_nochange")
        g = {}
        r = lt[lt.comparison == "L6 - L1"]
        if len(r):
            r = r.iloc[0]
            for b in ("wf", "dev", "locked"):
                if pd.notna(r.get(b)):
                    g[b] = f"{r[b]:+.3f} {_ci(r[b + '_lo'], r[b + '_hi'])}"
        facts["gain_period"] = g
        bt = facts.get("backtest")
        bt_txt = ""
        if bt is not None and len(bt) and {"block", "leg", "lag", "cost_bps", "ann_return"} <= set(bt.columns):
            x = bt[(bt.block == "locked") & (bt.leg == "long_minus_spy_matched") & (bt.lag == 0)].sort_values("cost_bps")
            if len(x):
                bt_txt = f"; backtest on the locked test, long book minus matched SPY: {x.ann_return.iloc[0]:+.1%} a year at {int(x.cost_bps.iloc[0])} bps"
        lo_l = float(r.get("locked_lo")) if "locked_lo" in r and pd.notna(r.get("locked_lo")) else None
        facts["h8_verdict"] = (("the classification gain persists on the locked test (CI above 0) but varies strongly by period" if lo_l is not None and lo_l > 0
                                else "the classification gain is not detectable on the locked test") + bt_txt) if g.get("locked") else "n/a"


# ---------------------------------------------------------------------------------------------------- figures for the context section
def draw_context_figures(ctx, F, rc: dict, facts):
    sk = rc["skill"]
    if sk is None or sk.empty:
        for fid, t in (("R01", "Macro-F1 against its real chance level"), ("R02", "Move versus direction skill"), ("R03", "Paired lifts by period")):
            F.skip(fid, t, "no predictions")
        return
    l0 = None
    rl = facts.get("_rl")
    if rl is not None:
        x = rl[(rl.rung == "L0") & (rl.metric == "macro_f1") & (rl.block == "walk_forward_pooled")]
        l0 = float(x.value.iloc[0]) if len(x) else None
    perm = facts.get("_perm_null")
    order = ["wf", "dev", "locked"]
    models = [m for m in ("L1", "L6", "L7") if m in set(sk.model)]
    colors = {"L1": "#9ca3af", "L6": "#2563eb", "L7": "#dc2626"}

    def r01(i, t):
        fig, ax = F.plt.subplots(figsize=(10, 4.8))
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            vals = [sk[(sk.blk == b) & (sk.model == m)].macro_f1.mean() for b in order]
            ax.bar(np.arange(3) + k * w - 0.4 + w / 2, vals, w, label=m, color=colors[m])
        ch = _chance(sk)
        ax.axhline(ch, color="k", ls="--", lw=1.2, label=f"real chance level (shuffled own predictions) {ch:.3f}")
        if l0 is not None:
            ax.axhline(l0, color="#6b7280", ls=":", lw=1.2, label=f"majority-class baseline {l0:.3f}")
        if perm is not None:
            ax.axhline(perm, color="#9ca3af", ls="-.", lw=1.2, label=f"permutation null {perm:.3f}")
        ax.set_xticks(range(3), [BLOCK_LABEL[b] for b in order])
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0.15, max(0.5, sk.macro_f1.max() + 0.09))
        ax.legend(fontsize=8, ncol=2, loc="upper center")
        F.save(fig, i, t)
    F.safe("R01", "Macro-F1 against its real chance level", r01)

    def r02(i, t):
        fig, axs = F.plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        for ax, col, ttl in ((axs[0], "move_auc", "does the stock MOVE? (move vs sideways)"), (axs[1], "direction_auc", "which WAY? (up vs down, among moved)")):
            w = 0.8 / len(models)
            for k, m in enumerate(models):
                s = [sk[(sk.blk == b) & (sk.model == m)].iloc[0] for b in order]
                v = np.array([x[col] for x in s])
                err = np.vstack([v - np.array([x[col + "_lo"] for x in s]), np.array([x[col + "_hi"] for x in s]) - v])
                ax.bar(np.arange(3) + k * w - 0.4 + w / 2, v, w, yerr=err, capsize=3, label=m, color=colors[m])
            ax.axhline(0.5, color="k", ls="--", lw=1)
            ax.set_xticks(range(3), [BLOCK_LABEL[b].split(" ")[0] for b in order])
            ax.set_title(ttl, fontsize=9)
            ax.set_ylim(0.4, 0.75)
        axs[0].set_ylabel("AUC (0.5 = no information)")
        axs[1].legend()
        F.save(fig, i, t)
    F.safe("R02", "Move versus direction skill", r02)

    lt = rc["lifts"]

    def r03(i, t):
        fig, ax = F.plt.subplots(figsize=(10, 4.8))
        y = 0
        ticks, labs, seen = [], [], set()
        for _, r in lt.iterrows():
            for b, c in (("wf", "#6b7280"), ("dev", "#d97706"), ("locked", "#dc2626")):
                if pd.notna(r.get(b)):
                    ax.errorbar(r[b], y, xerr=[[r[b] - r[b + "_lo"]], [r[b + "_hi"] - r[b]]], fmt="o", color=c, capsize=3,
                                label=BLOCK_LABEL[b] if BLOCK_LABEL[b] not in seen else None)
                    seen.add(BLOCK_LABEL[b])
                    ticks.append(y)
                    labs.append(f"{r.comparison}  ({BLOCK_LABEL[b].split(' ')[0]})")
                    y -= 1
            y -= 0.6
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(ticks, labs, fontsize=8)
        ax.set_xlabel("paired lift in macro-F1 (95% month-block bootstrap CI)")
        ax.legend(fontsize=7, loc="upper right")
        F.save(fig, i, t)
    F.safe("R03", "Paired lifts by period", r03)


# ---------------------------------------------------------------------------------------------------- other result figures
def draw_resources(ctx, F, stages: pd.DataFrame):
    s = stages.dropna(subset=["hours"])
    if s.empty:
        F.skip("R04", "Compute time by stage", "no stage timings")
        return
    def draw(i, t):
        fig, ax = F.plt.subplots(figsize=(11, 4.6))
        col = {"CPU": "#60a5fa", "GPU": "#f59e0b", "CPU-GPU": "#ec4899"}
        ax.bar(s.stage, s.hours, color=[col.get(p, "#60a5fa") for p in s.part])
        ax.set_ylabel("hours (wall clock)")
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("blue CPU, orange GPU stage, pink CPU/GPU. Part 1 ran on a 12-core PC; Parts 2-3 on Colab.", fontsize=8)
        F.save(fig, i, t)
    F.safe("R04", "Compute time by stage", draw)


def draw_sector(ctx, F, usel):
    c = usel.ff12_industry.value_counts()
    def draw(i, t):
        fig, ax = F.plt.subplots(figsize=(8, 4.6))
        ax.barh(c.index[::-1], c.values[::-1], color="#2563eb")
        for k, v in enumerate(c.values[::-1]):
            ax.text(v + 0.3, k, str(int(v)), va="center", fontsize=8)
        ax.set_xlabel("firms")
        F.save(fig, i, t)
    F.safe("R05", "Universe by Fama-French 12 industry", draw)


def draw_arch_figs(ctx, F, facts, usel, rl):
    from ..viz import diagrams
    cfg = ctx.cfg
    cov = facts.get("coverage_by_form")
    tot = int(cov.observed.sum()) if cov is not None else 0
    ev = facts.get("_ev")
    pf = _success(ctx, "S13")
    ftab = features_table(ctx)
    nf = dict(zip(ftab.family, ftab.features)) if len(ftab) else {}
    uni = _read_pq(ctx, "S02", "universe.parquet", ["eligible", "selected"])
    n_elig = f" (of {int(uni.eligible.sum())} eligible)" if uni is not None else ""
    pcv = ctx_read_csv(ctx, "S03", "price_coverage.csv")
    px_span = f"{str(pcv['first'].min())[:4]}-{str(pcv['last'].max())[:4]}" if pcv is not None else ""
    xf = _read_pq(ctx, "S06", "xbrl_facts.parquet", ["tag"])
    er = _read_pq(ctx, "S06", "eps_release.parquet", ["is_results_release"])
    xb = (f"{len(xf):,} XBRL facts; " if xf is not None else "") + (f"{int(er.is_results_release.sum()):,} parsed releases" if er is not None else "")
    c = {"src_edgar": f"{tot:,} filings (10-K/10-Q/8-K)", "src_hf": "10-K overlap + parser QA", "src_px": "daily bars, local + Yahoo",
         "src_fac": "Fama-French/Carhart, 12-industry map", "src_sp": "fja05680 membership list", "src_lm": "Loughran-McDonald",
         "universe": f"{len(usel)} S&P 500 firms{n_elig}", "manifest": f"{tot:,} filings listed", "prices": f"{len(usel)} firms, {px_span}",
         "xbrl": xb, "download": f"{tot:,} filings", "sections": f"{pf.get('sections', 0):,} sections kept",
         "mask": "regex on every section", "chunks": f"~{pf.get('est_chunks', 0):,} chunks",
         "events": f"{len(ev):,} events" if ev is not None else "", "targets": "horizons 1, 2, 5, 20, 60",
         "overlap": "clean = no same-firm event within 60 sessions",
         "ft_a": f"Price ({nf.get('price', 0)}) + metadata ({nf.get('metadata', 0)})\n+ accounting ({nf.get('accounting', 0)})",
         "ft_b": f"Lexical tone ({nf.get('lexical', 0)}) + filing change ({nf.get('change', 0)})\n+ novelty ({nf.get('novelty', 0)}); EAR registered, unused",
         "footer": "Every arrow is a stage dependency; each stage writes only to its own hash-named folder and a _SUCCESS marker (resume = skip finished stages)."}
    F.safe("A02", "Data and pre-processing flow", lambda i, t: diagrams.data_flow(F, i, t, c))
    fr = facts.get("ladder")
    wf = {}
    if fr is not None and len(fr):
        for rung in ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"):
            x = fr[(fr.rung == rung) & (fr.tuned.isin(["tuned", "default"]))]
            if len(x):
                wf[rung] = float(x.mean_fold_score.dropna().max()) if x.mean_fold_score.notna().any() else float(x.pooled_macro_f1.max())
    lb = {"L5": "retrospective", "L7": "retrospective"}
    rungs = [
        {"rung": "L0", "adds": "Class prior only (majority-class reference).", "wf": wf.get("L0"), "lb": "strict"},
        {"rung": "L1", "adds": "+ price: returns over 1-250 days, reversal, momentum, volatility, dollar volume, illiquidity, 52-week-high distance, beta, market state (known at decision time).", "wf": wf.get("L1"), "lb": "strict"},
        {"rung": "L2", "adds": "+ metadata: event type, 8-K item categories, which sections exist, industry, year/month/weekday, session bucket, timestamp quality, earnings-date type, filing lag.", "wf": wf.get("L2"), "lb": "strict"},
        {"rung": "L3", "adds": "+ accounting: assets, leverage, ROA, revenue, latest EPS, market cap, book-to-market from XBRL, and SUE where available.", "wf": wf.get("L3"), "lb": "strict"},
        {"rung": "L4", "adds": "+ lexical tone (and its change versus the firm's previous filing), filing-change similarity, novelty and fold-fitted TF-IDF+SVD.", "wf": wf.get("L4"), "lb": "strict"},
        {"rung": "L5", "adds": "Embeddings only (ChronoBERT vintages or BGE): tests the text signal on its own.", "wf": wf.get("L5"), "lb": "retrospective"},
        {"rung": "L6", "adds": "Price + metadata + accounting + lexical + change + novelty + TF-IDF together: the best non-neural model.", "wf": wf.get("L6"), "lb": "strict"},
        {"rung": "L7", "adds": "L6 + embedding PCA components (+ change vectors): the test of H1.", "wf": wf.get("L7"), "lb": "retrospective"},
        {"rung": "L8", "adds": "L7 + features from validated RAG extraction / fine-tuned encoder (needs gold labels).", "wf": None, "lb": "retrospective", "status": "not run"},
    ]
    F.safe("A03", "Ablation ladder L0-L8", lambda i, t: diagrams.ladder(F, i, t, rungs))


# ---------------------------------------------------------------------------------------------------- composer
GLOSSARY = [
    ("Event", "One filing (or one same-day group of 8-Ks) by one firm. The unit of analysis."),
    ("Acceptance / public / decision time", "EDGAR acceptance time; an assumed public time (acceptance + 5 min unless observed); the decision time (public + 30 min), the moment a forecast would have to be made. Features may only use information available by then."),
    ("T_trade (primary target)", "Abnormal return from the first market open at or after the decision time (what a trader could have captured), benchmarked on the identical interval."),
    ("T_info", "Information response measured from the last close before acceptance (Track A, and a complementary Track B target)."),
    ("Abnormal return, CAR", "Stock return minus the market-model expectation (beta estimated on sessions -250 to -30 before the event; T_trade uses beta x SPY on the identical interval); CAR cumulates it over a window."),
    ("Three classes", "Down / Sideways / Up: the abnormal return beyond -/+ k x sigma x sqrt(h), with k = 0.5, sigma the stock's own estimation-window volatility, h the horizon in sessions."),
    ("SUE", "Standardised unexpected earnings: (EPS this quarter - EPS four quarters ago) / standard deviation of past seasonal differences, using first-reported values only. No analyst forecasts."),
    ("EAR", "Earnings-announcement return. Used only as a Track A sort variable, never as a feature for its own event."),
    ("Macro-F1", "Mean of the Down, Sideways and Up F1 scores. Treats the three classes equally. Chance level is about 0.33, not 0.19 (see section 5a)."),
    ("AUC", "Probability that a random positive event is ranked above a random negative one. 0.5 = no information."),
    ("ECE / log-loss", "Calibration error of the predicted probabilities / their average negative log-likelihood (lower is better; the class-prior log-loss is the reference)."),
    ("Walk-forward, development, locked test", "Expanding-window validation folds, a development holdout used to choose the final model, and a final period scored exactly once."),
    ("Strict vs retrospective leaderboard", "Strict = models whose training cutoff precedes every evaluated event. Retrospective = models with an unknown or later cutoff; they support no clean-test claim."),
    ("Gates G3 / G3b / G4", "G3: text lift >= 0.01 macro-F1 with CI excluding 0 before fine-tuning. G3b: extraction quality on gold labels before the teacher run. G4: freeze and hash of the final architecture before the locked test."),
]

CONTENTS = [
    "Executive summary and key findings",
    "A. Study design and system: A1 hypotheses, A2 architecture, A3 data, A4 pre-processing, A5 events/targets, A6 features and ladder, "
    "A7 models, A8 RAG and extraction, A9 evaluation design",
    "1-3. Data integrity, design, leakage and provenance audit",
    "4. Track A: event study and post-earnings drift",
    "5. Track B: prediction results, 5a accuracy in context, development holdout, locked test, backtest",
    "6-9. Track C, fit diagnostics, extensions, limitations and compute",
    "10. What the results mean (answers to the research questions)",
    "11. Known issues, reproducibility and next steps",
    "Appendix: decision log, file map, how to re-run",
]


def _block_labels(ctx):
    bl = ctx_read_csv(ctx, "S10", "blocks.csv")
    if bl is None or bl.empty:
        return
    wf = bl[bl.kind == "walk_forward_val"]
    if len(wf):
        BLOCK_LABEL["wf"] = f"walk-forward {str(wf.start.min())[:4]}-{int(str(wf.end_exclusive.max())[:4]) - 1}"
    for k, nm in (("dev", "development_holdout"), ("locked", "locked_test")):
        x = bl[bl.kind == nm]
        if len(x):
            last = (pd.Timestamp(x.end_exclusive.iloc[0]) - pd.Timedelta(days=1)).year
            BLOCK_LABEL[k] = f"{'development' if k == 'dev' else 'locked'} {str(x.start.iloc[0])[:4]}-{last}"


def split_table(ctx, ev) -> pd.DataFrame:
    bl = ctx_read_csv(ctx, "S10", "blocks.csv")
    if bl is None:
        return pd.DataFrame()
    dt = pd.to_datetime(ev.decision_time, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    rows = []
    for r in bl.itertuples():
        m = (dt >= pd.Timestamp(r.start)) & (dt < pd.Timestamp(r.end_exclusive))
        rows.append({"block": r.block, "role": {"walk_forward_val": "validation fold (train on everything earlier)",
                                                 "development_holdout": "development holdout: choose the final model",
                                                 "locked_test": "locked test: scored once after the freeze"}.get(r.kind, r.kind),
                     "from": str(r.start)[:10], "to (exclusive)": str(r.end_exclusive)[:10], "events": int(m.sum())})
    return pd.DataFrame(rows)


METRICS = pd.DataFrame([
    ("Macro-F1 (primary)", "Down/Sideways/Up classes are roughly balanced and equally important; macro-F1 weights them equally and cannot be inflated by predicting only the biggest class.", "all prediction tables"),
    ("Accuracy, balanced accuracy", "Plain and class-balanced hit rate; accuracy alone rewards the majority class.", "ladder, holdout, locked test"),
    ("AUC (one-vs-rest, macro)", "Ranking quality of the probabilities, independent of the decision threshold.", "ladder, section 5a"),
    ("Log-loss and Brier score", "Quality of the probabilities themselves; a model must beat the class prior (pre-registered guardrail).", "guardrails table"),
    ("Expected calibration error (ECE)", "Whether predicted probabilities match observed frequencies; guardrail ECE <= 0.10.", "guardrails, calibration figure"),
    ("Move AUC and direction AUC", "Separates 'will the stock move?' from 'which way?' (section 5a).", "section 5a"),
    ("Month-block bootstrap CI", "Events in one month share market shocks; resampling whole months keeps that dependence. Paired versions resample the same months for two models.", "every CI"),
    ("Benjamini-Hochberg FDR", "Controls the false-discovery rate within pre-declared families of Track A tests.", "Track A tables"),
    ("Two-way clustered errors", "Track A standard errors cluster by firm and by date.", "Track A tables"),
    ("Deflated Sharpe, Carhart alpha", "Backtest sanity: Sharpe corrected for the number of configurations tried; alpha after market, size, value and momentum.", "backtest"),
    ("Permutation null, adversarial AUC", "Sanity checks: does the pipeline score above chance only when labels are real, and are train and validation periods separable?", "section 3"),
    ("MCS and PBO", "Model Confidence Set and Probability of Backtest Overfitting; diagnostics only, they never select a model.", "section 3, F40"),
], columns=["metric", "why it is used", "where it appears"])


def compose(ctx, facts, usel, ev, tg, rc, rg, stages) -> dict:
    cfg = ctx.cfg
    N: dict = {}
    _block_labels(ctx)
    n_ev, n_firms = len(ev), len(usel)

    # ---------------- reader's guide
    def _guide():
        return ("**Contents**\n\n" + "\n".join(f"- {c}" for c in CONTENTS) + "\n\n**Glossary**\n"
                      + md_table(pd.DataFrame(GLOSSARY, columns=["term", "meaning"]), 40))

    N["guide"] = _safe(ctx, "guide", _guide)

    # ---------------- A1
    def _a1():
        return ("The study asks whether SEC filings (10-K, 10-Q, 8-K) carry information about the next stock-price reaction, whether text adds anything "
                   "beyond numbers the market already has, and whether the well-known post-earnings drift appears in our sample. It is organised in three tracks "
                   "that are never mixed: **Track A** (event study, association only), **Track B** (prospective prediction from information available at decision "
                   "time) and **Track C** (does extracted text content match human labels; never judged against prices). The pre-registered hypotheses and what "
                   "this run found:\n\n" + md_table(hypotheses_table(facts), 12))

    N["a1"] = _safe(ctx, "a1", _a1)

    # ---------------- A2
    def _a2():
        return ("The system is a pipeline of **25 stages (S00-S24)**, each a function with declared inputs, a content hash of the configuration keys that "
                   "matter to it, and an output folder named by that hash. A finished stage writes a `_SUCCESS` marker; re-running skips finished stages, which is how a "
                   "Colab disconnect costs nothing. Stages run in three parts: Part 1 on CPU (data, events, features, baselines, event study), Part 2 on GPU (embeddings, "
                   "text models, RAG), Part 3 on CPU (holdout, locked test, backtest, report). Safety checks are assertions inside stages: availability of every feature "
                   "before decision time, SUE and EAR registry rules, a single-use locked test, and four deliberately planted leaks that the audit must catch.\n\n"
                   + md_table(stages.assign(hours=stages.hours.map(lambda h: "n/a" if pd.isna(h) else f"{h:.2f}")), 30)
                + "\n\n**Reading the hours.** Hours are the wall time of the run that produced each result. In this 150-firm study S05 reused the filings already downloaded and parsed for "
                  "these firms in the earlier 500-firm run (copied, not fetched again; decision D-076), so its time here is a copy, not the cost of downloading and parsing the filings. "
                  "Part 1 ran on a 12-core PC; S13-S19 and S22-S24 ran on Colab.")

    N["a2"] = _safe(ctx, "a2", _a2)

    # ---------------- A3
    def _a3():
        sp = ctx_json(ctx, "S02", "sp500_coverage.json") or {}
        n_cand = None
        return (md_table(data_sources_table(ctx, facts, usel), 12) +
                   f"\n\n**Universe.** {n_firms} firms: S&P 500 members at some point between {cfg['study']['start']} and {cfg['study']['end']}, drawn with a fixed seed and "
                   "stratified by Fama-French industry from the eligible set (a firm needs prices in the local files and at least one 10-K in the Hugging Face corpus). "
                   f"All {n_firms} are *current* listings, so the sample is survivor-only (see limitations). The eligible data covers about {sp.get('covered_by_eligible_universe', 'n/a')} "
                   f"of the {sp.get('sp500_tickers_in_window', 'n/a')} tickers that were ever S&P 500 members in the window.\n\n**Tickers by industry**\n\n" + tickers_text(usel))

    N["a3"] = _safe(ctx, "a3", _a3)

    # ---------------- A4
    def _a4():
        _, steps, sec_tab = preprocessing(ctx, facts)
        return ("Text is cleaned, sectioned, masked and chunked identically for every model, so differences between encoders cannot come from different inputs.\n\n"
                   + md_table(steps, 12) + "\n\n**Sections kept, by section id**\n\n" + md_table(sec_tab, 12))

    N["a4"] = _safe(ctx, "a4", _a4)

    # ---------------- A5
    def _a5():
        sb = ev.session_bucket.value_counts().to_dict()
        tq = ev.ts_quality.value_counts().to_dict()
        lab = tg[(tg.target_def == cfg["targets"]["primary"]["target_def"]) & (tg.horizon == cfg["targets"]["primary"]["horizon"]) & (tg.eligible.fillna(False))] if tg is not None else None
        sh = lab.label3.value_counts(normalize=True).sort_index().to_dict() if lab is not None and len(lab) else {}
        return (f"**Events.** {n_ev:,} events: {ev.event_type.value_counts().to_dict()}. Same-firm, same-day 8-Ks are merged into one multi-label event whose clock is the "
                   "LAST acceptance of the group (found as a look-ahead leak in the full-scale run, decision D-071). **Four timestamps** per event: acceptance, assumed public "
                   f"time (+{cfg['timing']['public_lag_minutes']} min), feature-ready time and decision time (+{cfg['timing']['decision_latency_minutes']} min). Timestamp quality "
                   f"{tq} (A = exact from the SGML header, B = date only, C = uncertain). Session buckets {sb}.\n\n"
                   f"**Targets.** T_trade (primary): entry at the open of the first session whose open is at or after the decision time, exit at the close of session "
                   "entry + h - 1 (an executable open-to-close return); abnormal return = stock return minus beta x SPY return on the identical interval (no close-to-close "
                   "factor return is ever subtracted from an open-to-close return, so Carhart abnormal returns are null for T_trade). Beta comes from a market model estimated on sessions "
                   f"{cfg['targets']['est_window'][0]} to {cfg['targets']['est_window'][1]} before the event (at least {cfg['targets']['est_min_obs']} observations). T_info is the information "
                   "response: close-to-close from the last close at or before acceptance to the close of the h-th session after public time (INTRA events excluded). Horizons: " + ", ".join(map(str, cfg["targets"]["horizons"])) + " sessions. "
                   "Labels: Up/Down beyond k x sigma x sqrt(h) with k = " + str(cfg["labels"]["k"]) + " (sensitivity 0.25 and 0.75). "
                   + (f"Class shares of the primary labels: Down {sh.get(0, float('nan')):.3f}, Sideways {sh.get(1, float('nan')):.3f}, Up {sh.get(2, float('nan')):.3f}. " if sh else "")
                   + "Events after the usable cutoff are censored per horizon. **Overlap:** an event is 'clean' if the firm has no other event within "
                   f"{cfg['contam_window_sessions']} sessions; flags exist per horizon and ignore members of the event's own cluster (decision D-073).")

    N["a5"] = _safe(ctx, "a5", _a5)

    # ---------------- A6
    def _a6():
        return ("Features are grouped in families; each family carries an availability column and the rule `latest input <= decision time` is asserted for every event "
                   "(0 violations in this run). The ablation ladder adds information step by step and every comparison uses identical events (the 'common cohort').\n\n"
                   + md_table(features_table(ctx), 10))

    N["a6"] = _safe(ctx, "a6", _a6)

    # ---------------- A7
    def _a7():
        fa = (cfg.get("speed") or {})
        fast = fa.get("mode") == "fast"
        return (
            "**Encoders (frozen, used to embed filing chunks).**\n\n" + md_table(encoder_table(ctx), 10) +
            "\n\nWhy these: ChronoBERT gives a point-in-time encoder (event year Y uses the vintage trained to the end of year Y-1; 26 vintages with pinned revisions), so its results can go "
            "on the strict leaderboard; BGE-base is a strong general-purpose encoder that also serves as the dense retriever; SEC-BERT (10-K domain) and FinBERT-tone (tone) are in "
            "the spec's core roster but were " + ("**dropped in this run to save time (fast mode, decision D-074); this was a time decision, not a measured result**" if fast else "used") + ".\n\n"
            "**Heads (trained on the features and embeddings).** LightGBM and CatBoost (gradient-boosted trees) and an elastic-net logistic regression. Trees capture interactions "
            "and non-linearities in tabular data; the elastic net is a convex, regularised reference whose optimum can be certified. "
            + ("CatBoost was used for the structured baselines but not for the embedding stage (fast mode). " if fast else "")
            + "\n\n**Training.** Loss: class-weighted cross-entropy (balanced class weights x label-uniqueness weights, so overlapping events count less). Early stopping and "
            "probability calibration use an inner, time-ordered slice of each training fold. Hyperparameters: Optuna (TPE sampler, median pruner), tuned on the most recent "
            f"{cfg['tuning']['tune_last_folds']} walk-forward folds with at most {cfg['tuning']['tune_max_train_rows']:,} training rows per fold, then the chosen setting is refitted "
            "on every fold. Decision rule: probabilities are corrected back to true frequencies and the class is `argmax p(c|x) / prior(c)`, which keeps predicted classes balanced. "
            "Fold-level preprocessing (TF-IDF + SVD, embedding PCA, imputers, scalers, calibration, clip bounds) is fitted on the training rows of that fold only. "
            "Fine-tuning of the encoders is allowed only if gate G3 passes; here it did not, so no encoder was fine-tuned.\n\n**Tuned hyperparameters**\n\n"
            + md_table(head_params_table(ctx), 30) +
            "\n\nThe elastic net's convex problem has a certified global optimum; tree models are validation-selected, never claimed to be global optima. "
        f"**Search budget.** Each Optuna study first tries the default setting, then draws candidates with a fixed seed: {cfg['tuning']['trials']} trials per study. Optuna's TPE sampler draws "
        "its first 10 candidates at random, so CatBoost and elastic-net studies (fewer than 10 trials) are small random searches that see the SAME random candidates in every rung; this is why several rungs "
        "share identical settings in the table, and why 'tuned' is often equal to 'default' (see 'Tuning: default vs tuned'). Tuning gains are small by design and should not be read as a search for the best possible model.")

    N["a7"] = _safe(ctx, "a7", _a7)

    # ---------------- A8
    def _a8():
        g3b = rg["s16"].get("g3b") or {}
        pv = rg.get("prevalence")
        prev_tab = None
        if pv is not None and len(pv):
            prev_tab = pv.groupby("field").apply(lambda g: pd.Series({
                "filings": len(g), "extracted_present": g.present.mean(),
                "high_certainty_when_present": (g[g.present].certainty == "high").mean() if g.present.any() else float("nan")}),
                include_groups=False).reset_index()
            prev_tab["extracted_present"] = prev_tab.extracted_present.map(lambda x: f"{100 * x:.1f}%")
            prev_tab["high_certainty_when_present"] = prev_tab.high_certainty_when_present.map(lambda x: "n/a" if pd.isna(x) else f"{100 * x:.1f}%")
        ex = rg.get("example")
        ex_txt = ""
        if ex:
            shown = []
            for k, o in ex.items():
                if isinstance(o, dict) and o.get("present"):
                    q = (o.get("evidence") or [{}])[0].get("quote", "")
                    shown.append(f"- **{k}**: {o.get('direction')} (certainty {o.get('certainty')}, magnitude {o.get('magnitude')} {o.get('unit')}); evidence: \"{q[:220]}\"")
            ex_txt = ("\n\n**One real extraction (illustration only; unvalidated model output on a development filing).**\n" + "\n".join(shown)
                  + "\n\nRead the quotes against the fields yourself: without gold labels nothing here is checked, and typical flaws are a magnitude left at 0 when a range is quoted, "
                    "a quote cut off mid-sentence, or a generic sentence offered as evidence of a change.")
        return (
            "The RAG system turns a filing into structured, evidence-cited events (guidance change, impairment, restructuring, M&A, dividends, liquidity stress, "
            "litigation, management change and so on; 12 fields in question bank v1). It has two retrieval paths: the **current-document path** (chunks of the filing being "
            "processed, allowed at decision time) and a **historical path** (the firm's earlier filings with public time strictly before decision time; the current or any later "
            "filing can never enter; checked by a self-test and a planted leak). In S16 all 12 questions are asked together as one query, the top 8 passages are retrieved "
            f"(BM25 k1 {cfg['rag']['bm25_k1']}, b {cfg['rag']['bm25_b']}; dense BGE; reciprocal-rank fusion k {cfg['rag']['rrf_k']}; cross-encoder rerank of the top "
            f"{cfg['rag']['rerank_top_n']}) and Qwen2.5-7B-Instruct, run locally with JSON-schema-constrained decoding, fills the 12 fields. Guardrails: schema validation with one "
            "retry; a numeric magnitude must appear in its cited evidence; every quote must be a substring of its cited chunk or the field is set to null; citation offsets are "
            "verified. Responses are cached by model revision, prompt and passages.\n\n"
            f"**What ran in this study.** Qwen produced pre-filled annotation templates for {rg['s16'].get('prefilled_events', 'n/a')} development/validation filings "
        f"({int(pv.groupby('field').size().max()) if pv is not None and len(pv) else 'n/a'} returned a valid structured response) "
            f"({rg['s16'].get('seconds_per_filing', 'n/a')} s per filing, {rg['s16'].get('generation_seconds', 0) / 3600:.1f} h). Gold-test filings were never sent to the model. "
            f"**What could not run:** there are no human gold labels (status: {rg['gold'].get('rater_status', 'n/a')}; imported: {rg['gold'].get('labels_imported')}), so the retrieval tournament "
            f"(BM25 vs hybrid vs hybrid+rerank), extraction F1, faithfulness and numeric-fidelity evaluation (F32-F36), gate G3b ({g3b}), the teacher/student distillation (S17) and "
            "feature set L8 were not produced, and H3 and H6 are untested. The model outputs below are therefore **not evidence of extraction quality**; they only show what the pipeline produces."
            + ("\n\n**How often each field was extracted as present** (descriptive; unvalidated)\n\n" + md_table(prev_tab, 14) if prev_tab is not None else "") + ex_txt)

    N["a8"] = _safe(ctx, "a8", _a8)

    # ---------------- A9
    def _a9():
        return ("**Splits.** Expanding walk-forward validation folds (training starts in 2000, first fold needs at least "
                   f"{cfg['splits']['min_train_years']} years, validation blocks of {cfg['splits']['val_block_years']} years), then a development holdout, then the locked test. "
                   f"Events whose outcome window reaches the next block are purged and a {cfg['splits']['embargo_days']}-day embargo separates blocks. Models selected on the "
                   "walk-forward folds are retrained on all earlier data before the development holdout; the final choice is frozen and hashed (G4) before the locked test, which is scored once.\n\n"
                   + md_table(split_table(ctx, ev), 12) +
                   "\n\n**Leakage controls.** A 24-item checklist with an assertion per item, plus four planted leaks (a future-return feature, a same-event EAR, a later-cluster SUE, a "
                   "future document in retrieval) that the audit must catch; all four were caught.\n\n**Metrics and statistics**\n\n" + md_table(METRICS, 14))
    N["a9"] = _safe(ctx, "a9", _a9)

    return N



# ---------------------------------------------------------------------------------------------------- 5a / answers / issues
def _chance(sk) -> float:
    """Chance macro-F1 of a model with a balanced predicted-class mix: the mean over the L6 and L7 rows (L1's predictions are
    lopsided in some periods, which lowers its shuffled score)."""
    x = sk[sk.model.isin(["L6", "L7"])]
    return float((x if len(x) else sk).chance_f1_shuffled.mean())


def _skill_row(sk, blk, model):
    x = sk[(sk.blk == blk) & (sk.model == model)]
    return None if x.empty else x.iloc[0]


def context_text(ctx, facts, rc) -> str:
    sk, lt, rep = rc["skill"], rc["lifts"], rc["repro"]
    if sk is None or sk.empty:
        return "_(predictions not available: section not produced)_"
    t = sk.assign(model=sk.model, **{"macro-F1": sk.macro_f1.map(lambda v: f"{v:.3f}"),
                                      "chance (shuffled)": sk.chance_f1_shuffled.map(lambda v: f"{v:.3f}"),
                                      "move AUC": [f"{r.move_auc:.3f} {_ci(r.move_auc_lo, r.move_auc_hi)}" for r in sk.itertuples()],
                                      "direction AUC": [f"{r.direction_auc:.3f} {_ci(r.direction_auc_lo, r.direction_auc_hi)}" for r in sk.itertuples()]})
    tab = t[["period", "model", "n", "macro-F1", "chance (shuffled)", "move AUC", "direction AUC"]]
    rl = facts.get("_rl")
    l0 = None
    if rl is not None:
        x = rl[(rl.rung == "L0") & (rl.metric == "macro_f1") & (rl.block == "walk_forward_pooled")]
        l0 = float(x.value.iloc[0]) if len(x) else None
    perm = facts.get("_perm_null")
    ch = _chance(sk)
    lock6 = _skill_row(sk, "locked", "L6")
    out = [f"**Chance level.** The report's majority-class baseline (macro-F1 {l0:.3f}) and the permutation null ({perm:.3f}) are both below the macro-F1 that "
           f"any classifier with balanced predictions gets by luck: shuffling a model's own predicted classes against the true labels gives about {ch:.3f} for the L6 and L7 models. "
           "Read model scores against that line, not against the majority-class baseline." if l0 is not None and perm is not None else
           f"**Chance level.** Shuffling a model's own predicted classes against the true labels gives a macro-F1 of about {ch:.3f}; read model scores against that line."]
    if lock6 is not None:
        d_ci = (lock6.direction_auc_lo, lock6.direction_auc_hi)
        dir_txt = ("**no detectable directional information** (the CI includes 0.5)" if d_ci[0] <= 0.5 <= d_ci[1]
                   else ("a small but detectable directional signal" if d_ci[0] > 0.5 else "a below-chance directional signal"))
        out.append(f"**Move versus direction.** On the locked test L6 scores macro-F1 {lock6.macro_f1:.3f}, {lock6.macro_f1 - lock6.chance_f1_shuffled:+.3f} above its chance level. "
                   f"Almost all of that comes from telling moving stocks from quiet ones (move AUC {lock6.move_auc:.3f} {_ci(lock6.move_auc_lo, lock6.move_auc_hi)}); "
                   f"among stocks that did move, the probabilities carry {dir_txt} (direction AUC {lock6.direction_auc:.3f} {_ci(*d_ci)}). "
                   "The models predict how much a stock will react, which is largely a volatility and event-type effect, not which way it will go.")
    sh = rc["shares"].get("locked")
    if sh:
        out.append(f"Locked-test class shares: Down {sh['down']:.3f}, Sideways {sh['sideways']:.3f}, Up {sh['up']:.3f}; always predicting Sideways would be right {sh['always_sideways_accuracy']:.3f} of the time.")
    l1 = {b: _skill_row(sk, b, "L1") for b in ("wf", "dev", "locked")}
    if all(v is not None for v in l1.values()):
        out.append("The structured-feature gain over price-only varies strongly by period because the price-only baseline L1 does: its macro-F1 is "
                   + ", ".join(f"{l1[b].macro_f1:.3f} ({BLOCK_LABEL[b]})" for b in ("wf", "dev", "locked")) + ". L6 is stable (see table).")
    out.append("\n" + md_table(tab, 20))
    out.append("\n**Paired lifts by period** (macro-F1 difference on identical events, 95% month-block bootstrap CI). The pipeline itself computes H1 on the walk-forward folds (gate G3) "
               "and H1, H2 and H5 on the development holdout; every other cell, including the whole locked-test column, is computed here from the stored predictions with the same bootstrap.\n")
    rows = []
    for _, r in lt.iterrows():
        d = {"comparison": r.comparison}
        for b in ("wf", "dev", "locked"):
            if pd.notna(r.get(b)):
                d[BLOCK_LABEL[b]] = f"{r[b]:+.3f} {_ci(r[b + '_lo'], r[b + '_hi'])} p={r[b + '_p']:.2f}"
        rows.append(d)
    out.append(md_table(pd.DataFrame(rows), 8))
    return "\n\n".join(out)


def selection_text(ctx, facts, rc) -> str:
    fa = ctx_json(ctx, "S18", "frozen_architecture.json")
    if not fa:
        return ""
    pm = fa.get("primary_model") or {}
    fin = {f["name"]: f for f in fa.get("finalists", [])}
    s6 = fin.get("L6", {}).get("wf_score")
    sp = pm.get("wf_score")
    rep = rc["repro"]
    lb = "retrospective" if pm.get("rung") in ("L7", "L5", "L8", "STACK") else "strict"
    txt = (f"**Which model is 'best'?** The pre-registered rule freezes the model with the highest walk-forward mean macro-F1. That is **{pm.get('name')}** "
           f"({pm.get('head')}{'/' + pm.get('enc') if pm.get('enc') else ''}, {lb} leaderboard) with {sp:.4f}" if sp is not None else
           f"**Which model is 'best'?** The frozen primary model is {pm.get('name')}.")
    if sp is not None and s6 is not None:
        txt += (f", against {s6:.4f} for the best strict model L6, a margin of {sp - s6:+.4f}. ")
        if rep is not None and len(rep):
            mx = float((rep.macro_f1_a - rep.macro_f1_b).abs().max())
            txt += (f"Re-scoring the same L6 model in two stages (same events and folds, different machines) changed its pooled macro-F1 by up to {mx:.4f} and its predicted class "
                    f"for up to {100 * (1 - float(rep.same_class_share.min())):.1f}% of events ({', '.join(f'{r.head} {100 * r.same_class_share:.1f}% identical' for r in rep.itertuples())}). "
                    "The margin is therefore smaller than the measurement noise: **L6 and L7 are statistically tied**, and on the locked test they are tied as well. "
                    "The backtest uses the frozen model, which is on the retrospective leaderboard.")
    return txt


def mcs_text(ctx) -> str:
    mp = ctx_json(ctx, "S18", "mcs_pbo.json")
    if not mp:
        return ""
    inc = (mp.get("mcs") or {}).get("included")
    pbo = (mp.get("pbo") or {})
    return (f"**Reading MCS and PBO here.** The Model Confidence Set on development log-loss keeps {inc}; the Probability of Backtest Overfitting is {pbo.get('pbo')} over "
            f"{pbo.get('n_configs')} stored configurations. PBO = 0 mainly reflects that the stored configurations mix rungs with very different information content (L0 to L7), "
            "so the best in-sample configuration is trivially the best out-of-sample one; it does not test overfitting of the close call between L6 and L7. Both are diagnostics only.")


def answers_text(ctx, facts, rc) -> str:
    sk, lt = rc["skill"], rc["lifts"]
    cfg = ctx.cfg
    lock6, lock7, lock1 = (_skill_row(sk, "locked", m) for m in ("L6", "L7", "L1")) if sk is not None and len(sk) else (None, None, None)
    ht = facts.get("hypothesis_tests")
    sue = None
    if ht is not None and len(ht):
        x = ht[ht.test.str.startswith("SUE-conditioned post-earnings drift (Q5-Q1, SUE_release)")]
        sue = x.iloc[0] if len(x) else None
    acc_txt = ""
    lr = facts.get("locked_results")
    if lr is not None and len(lr) and "run_id" in lr:
        x = lr[lr.run_id.astype(str) == "locked|L6"]
        if len(x):
            x = x.iloc[0]
            acc_txt = f", accuracy {x.accuracy:.3f} (always-Sideways {rc['shares'].get('locked', {}).get('always_sideways_accuracy', float('nan')):.3f}), AUC {x.auc_macro_ovr:.3f}, calibration error {x.ece:.3f}"
    q = []
    q.append("**Are the accuracies good?** " + (
        f"On the locked test the best structured model (L6) scores macro-F1 {lock6.macro_f1:.3f} (chance {_chance(sk):.3f}){acc_txt}. "
        "That is a modest but real above-chance result of the size usually seen in daily return prediction, it holds up from the walk-forward folds to the locked test, and it is almost entirely a "
        "'will it move' signal, not a 'which way' signal." if lock6 is not None else "n/a"))
    q.append("**Were the spec requirements met?** The ablation ladder L0-L7, point-in-time availability rules, purge and embargo, the planted-leak audit, the G4 freeze and the "
             "single-use locked test were executed and are documented. Not met: Track C (no gold labels), so L8, G3b, RAG evaluation, H3 and H6 are untested; the firm- and sector-held-out "
             "stress folds were built but not scored; the run used the fast-mode roster (two of four encoders) on 150 firms. Details in section 11.")
    q.append("**Did we answer the post-earnings-drift question?** " + (
        f"Sorting earnings releases by SUE, the high-surprise quintile underperformed the low-surprise quintile by {sue.estimate:+.2%} [{sue.ci_low:+.2%}, {sue.ci_high:+.2%}] over days +2 to +60 "
        f"(n = {int(sue.n):,}, verdict '{sue.verdict}'). That is a **reversal**, the opposite of classic continuation. Because the universe contains only firms that survived to 2026, firms that "
        "kept falling after bad news and were delisted are missing, which biases low-SUE drift upwards; these data cannot separate that bias from a genuine reversal. The backtest does not turn this into a "
        "tradable edge." if sue is not None else "n/a"))
    h1l = lt[lt.comparison == "L7 - L6"].iloc[0] if lt is not None and (lt.comparison == "L7 - L6").any() else None
    q.append("**Did text add value beyond structured numbers (H1)?** " + (
        "No. Paired lift of the best text model over the best structured model: " + "; ".join(
            f"{BLOCK_LABEL[b]} {h1l[b]:+.3f} {_ci(h1l[b + '_lo'], h1l[b + '_hi'])}" for b in ("wf", "dev", "locked") if pd.notna(h1l.get(b))) +
        ". Every interval includes zero and the gate G3 (+0.01 with CI above 0) failed, so no encoder was fine-tuned. Annual ChronoBERT vintages did not differ detectably from the static BGE encoder (H4)."
        if h1l is not None else "n/a"))
    q.append("**Is the study complete and defensible?** It is a complete, reproducible empirical study of Tracks A and B with a credible null result for text, and an incomplete Track C. "
             "See 'Smallest defensible claim' above and the caveats in section 11.")
    return "\n\n".join(q)


def issues_text(ctx, facts, rc, rl) -> str:
    rep = rc["repro"]
    lines = []
    if rep is not None and len(rep):
        lines.append("**Cross-machine reproducibility of tree models.** The same L6 model re-scored in S11 (CPU stage, local PC) and S15 (GPU stage, Colab) on identical events and folds: "
                     + "; ".join(f"{r.head}: {100 * r.same_class_share:.1f}% identical predicted classes, mean |dp| {r.mean_abs_prob_diff:.4f}, macro-F1 {r.macro_f1_a:.3f} vs {r.macro_f1_b:.3f}" for r in rep.itertuples())
                     + ". Multi-threaded boosting and library versions are the probable causes (not isolated). Effect sizes in this report are far larger than this noise except for the L6/L7 selection margin.")
    lines.append("**Final-model selection.** The rule compares the L6 walk-forward score from S11 with the L7 score from S15, computed in different runs, and has no tie-break in favour of the strict leaderboard "
                 "(see section 5a).")
    stress = bool(rl is not None and rl.run_id.astype(str).str.contains("stress").any())
    lines.append("**Stress folds.** Firm-held-out and sector-held-out folds are built in S10 "
                 + ("and scored." if stress else "but their scores were not produced in this run; the leakage audit item 4 is relabelled 'built, not scored' in this report."))
    lines.append("**Locked-test comparisons.** S19 scores the frozen finalists but does not compute the paired H1/H2 lifts; they are computed in section 5a from the stored locked predictions.")
    fa, lk = ctx_json(ctx, "S18", "frozen_architecture.json"), None
    lp = ctx.storage.manifests / "lockbox_log.json"
    if fa and lp.exists():
        lk = json.loads(lp.read_text(encoding="utf-8"))
        gap = (pd.Timestamp(lk["scored_at"]) - pd.Timestamp(fa["frozen_at"]).tz_localize("UTC") if pd.Timestamp(fa["frozen_at"]).tzinfo is None
               else pd.Timestamp(lk["scored_at"]) - pd.Timestamp(fa["frozen_at"]))
        lines.append(f"**Lockbox.** Scored once ({lk['scored_at'][:19]}Z, reused={lk['reused']}, previous={lk['previous']}), {gap.total_seconds() / 60:.1f} minutes after the freeze, in the same "
                     "run and straight after the development stage. The spec requires the freeze (G4) to precede scoring, which holds; the project's own guidance was to review the development results "
                     "before unlocking, which did not happen in this run. The period is now spent: any re-run can only score it as development data.")
    lines.append("**Track C and RAG.** No gold labels were imported, so content validity, retrieval and extraction quality, the student model, L8 and hypotheses H3 and H6 are not evaluated (figures F32-F36 are marked 'not produced').")
    lines.append("**Fast mode.** Two of four core encoders (SEC-BERT, FinBERT-tone) and CatBoost on embeddings were dropped, tuning trials were reduced and fine-tuning was disabled (decision D-074); all conclusions about text refer to ChronoBERT and BGE only.")
    lines.append("**Environment record.** The environment manifest was written by the local PC run and does not describe the Colab session; the GPU model was not recorded. S13 records device 'cuda' and a benchmark in chunks/s.")
    lines.append("**Survivor-only universe, daily bars, 150 firms.** See sections 1 and 9. Drift and backtest results are biased towards survivors; confidence intervals are about twice as wide as with 500 firms.")
    lines.append("**History.** An earlier 500-firm run predates the point-in-time accounting fix (D-077); its accounting-dependent results are superseded and are not used here.")
    nxt = ["Move to the 500-firm study (S&P and non-S&P buckets, more power) after the fixes below.",
           "Make LightGBM deterministic (fixed thread count, `deterministic=True`) and compare finalists inside one run.",
           "Add a tie-break that prefers the strict model when walk-forward scores differ by less than the measured noise.",
           "Score the firm- and sector-held-out stress folds and add the locked-test paired lifts to S19.",
           "Annotate the gold set (templates are pre-filled) to enable G3b, L8, H3, H6 and Track C.",
           "Add delisted firms' prices to remove the survivor-only limitation."]
    return "\n\n".join(f"- {x}" for x in lines) + "\n\n**Recommended next steps**\n\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(nxt))


def appendix_text(ctx, stages) -> str:
    cfg = ctx.cfg
    root = Path(cfg.get("_meta", {}).get("repo_root", "."))
    dl = root / "docs" / "decision_log.md"
    rows = []
    if dl.exists():
        for line in dl.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\|\s*(D-\d+)\s*\|\s*(.+?)\s*\|", line)
            if m:
                txt = re.sub(r"\*\*|`", "", m.group(2))
                rows.append({"id": m.group(1), "decision (first sentence)": (txt.split(". ")[0])[:190]})
    out = ["**How to re-run.** Open the notebook, set `RUN_PROFILE = \"full\"`, `N_STOCKS`, `SPEED` and `PART` in the first cell and run all; finished stages are skipped. Locally: "
           "`secprice run --profile full --stages S00-S24` with `SEC_USER_AGENT` set to 'Name email'. Tests: `pytest -m \"not network\"`.",
           "**File map.** Each stage writes to `<DATA_ROOT>/<profile>/<stage>/<hash>/`; reports are in `reports/` (this document as .md, .html, .pdf; figures in `reports/figures/`), "
           "provenance in `manifests/` (run manifest, environment manifest, frozen architecture and G4 hash, pre-registration hash, lockbox log).\n\n"
           + md_table(stages[["stage", "name", "main outputs"]], 30),
           "**Decision log (summary).** The full log with rationales is `docs/decision_log.md`.\n\n" + (md_table(pd.DataFrame(rows), 90) if rows else "_(decision log not bundled with this run)_")]
    return "\n\n".join(out)


def key_findings(facts, rc) -> list[str]:
    sk, lt = rc["skill"], rc["lifts"]
    out = []
    l6 = _skill_row(sk, "locked", "L6") if sk is not None and len(sk) else None
    if l6 is not None:
        out.append(f"Locked test (2023-2026, n = {int(l6.n):,}): L6 macro-F1 {l6.macro_f1:.3f}; real chance level {l6.chance_f1_shuffled:.3f}; the signal is in whether a stock moves "
                   f"(move AUC {l6.move_auc:.3f}) not which way (direction AUC {l6.direction_auc:.3f} {_ci(l6.direction_auc_lo, l6.direction_auc_hi)}).")
    if lt is not None and (lt.comparison == "L7 - L6").any():
        r = lt[lt.comparison == "L7 - L6"].iloc[0]
        out.append("Text adds no detectable value: paired macro-F1 lift of L7 over L6 is " + "; ".join(
            f"{BLOCK_LABEL[b]} {r[b]:+.3f} {_ci(r[b + '_lo'], r[b + '_hi'])}" for b in ("wf", "dev", "locked") if pd.notna(r.get(b))) + ".")
    return out


def claim_tail(rc) -> str:
    sk, lt = rc["skill"], rc["lifts"]
    if sk is None or sk.empty or lt is None or lt.empty or not (lt.comparison == "L7 - L6").any():
        return ""
    r = lt[lt.comparison == "L7 - L6"].iloc[0]
    l6 = _skill_row(sk, "locked", "L6")
    s = ""
    if pd.notna(r.get("locked")):
        s += f" On the locked {BLOCK_LABEL['locked'].split(' ')[1]} test the same comparison gives {r['locked']:+.3f} {_ci(r['locked_lo'], r['locked_hi'])}."
    if l6 is not None:
        s += (f" The structured models' skill is in whether a stock moves (move AUC {l6.move_auc:.3f}), not which way (direction AUC {l6.direction_auc:.3f} "
              f"{_ci(l6.direction_auc_lo, l6.direction_auc_hi)}).")
    return s



# ---------------------------------------------------------------------------------------------------- orchestrator
def _try(ctx, name, fn, default, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        ctx.warn(f"report part '{name}' not built: {type(e).__name__}: {str(e)[:160]}")
        return default


def relabel_stress_audit(leak: pd.DataFrame | None, rl: pd.DataFrame) -> pd.DataFrame | None:
    """Leakage item 4 says 'pass' because the stress folds exist; if no stress-fold scores were produced, say so."""
    if leak is None or not len(leak) or "item" not in leak:
        return leak
    scored = bool(rl is not None and len(rl) and rl.run_id.astype(str).str.contains("stress").any())
    if scored:
        return leak
    out = leak.copy()
    m = out["item"].astype(str) == "4"
    out.loc[m, "status"] = "built, not scored"
    out.loc[m, "detail"] = out.loc[m, "detail"].astype(str) + "; firm- and sector-held-out fold scores were not produced in this run (report-level relabel)"
    return out


def add_narrative(ctx, F, facts, opt, ev, usel, tg, rl) -> None:
    facts["_rl"], facts["_ev"] = rl, ev
    perm = opt(ctx, "S11", "permutation.csv", "csv")
    facts["_perm_null"] = float(perm[perm.kind == "permuted"].macro_f1.mean()) if perm is not None and len(perm) else None
    empty_stage = pd.DataFrame(columns=["stage", "name", "part", "needs", "main outputs", "hours", "result"])
    stages = _try(ctx, "stages", stage_table, empty_stage, ctx)
    empty_rc = {"skill": pd.DataFrame(), "shares": {}, "lifts": pd.DataFrame(), "repro": pd.DataFrame(), "head_l6": None, "l7": None}
    rc = _try(ctx, "context", results_in_context, empty_rc, ctx, facts)
    _try(ctx, "context-facts", apply_context_to_facts, None, facts, rc)
    h4 = facts.get("hypotheses_core")
    if h4 is not None and len(h4):
        x = h4[h4.hypothesis.astype(str).str.startswith("H4")]
        if len(x):
            facts["h4_summary"] = f"{x.value.min():+.3f} to {x.value.max():+.3f}; all CIs " + ("include 0" if ((x.ci_low <= 0) & (x.ci_high >= 0)).all() else "mostly exclude 0")
            facts["h4_verdict"] = "no detectable difference" if ((x.ci_low <= 0) & (x.ci_high >= 0)).all() else "mixed: see H4 table"
        st = h4[h4.hypothesis.astype(str).str.startswith("H5")]
        if len(st):
            r = st.iloc[0]
            facts["stack_summary"] = f"{r.value:+.3f} {_ci(r.ci_low, r.ci_high)}"
            facts["h5_verdict"] = "not supported: CI includes 0" if r.ci_low <= 0 <= r.ci_high else "see table"
    rg = _try(ctx, "rag", rag_facts, {"s16": {}, "cache_n": 0, "prevalence": pd.DataFrame(), "example": None, "gold": {}}, ctx)
    # figures
    _try(ctx, "fig-pipeline", draw_pipeline, None, ctx, F, stages)
    _try(ctx, "fig-arch", draw_arch_figs, None, ctx, F, facts, usel, rl)
    _try(ctx, "fig-rag", draw_rag, None, ctx, F, rg)
    _try(ctx, "fig-context", draw_context_figures, None, ctx, F, rc, facts)
    _try(ctx, "fig-resources", draw_resources, None, ctx, F, stages)
    _try(ctx, "fig-sector", draw_sector, None, ctx, F, usel)
    _try(ctx, "fig-prevalence", prevalence_figure, None, ctx, F, rg.get("prevalence"))
    N = _try(ctx, "compose", compose, {}, ctx, facts, usel, ev, tg, rc, rg, stages)
    N["context"] = _safe(ctx, "context-text", context_text, ctx, facts, rc)
    N["selection"] = _safe(ctx, "selection-text", selection_text, ctx, facts, rc)
    N["mcs"] = _safe(ctx, "mcs-text", mcs_text, ctx)
    N["answers"] = _safe(ctx, "answers", answers_text, ctx, facts, rc)
    N["issues"] = _safe(ctx, "issues", issues_text, ctx, facts, rc, rl)
    N["appendix"] = _safe(ctx, "appendix", appendix_text, ctx, stages)
    facts["narr"] = N
    facts["key_findings"] = _try(ctx, "key-findings", key_findings, [], facts, rc)
    facts["claim_tail"] = _try(ctx, "claim-tail", claim_tail, "", rc)
    facts["rc"] = rc

