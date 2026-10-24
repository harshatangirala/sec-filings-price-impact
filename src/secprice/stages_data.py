"""CPU data stages S00-S10 (SPEC.md §17.1)."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .config import config_hash, file_sha256, get, text_sha256
from .events.timestamps import acceptance_from_sources, derive_times, header_to_et, json_to_et, timing_policy_label
from .logging_utils import ErrorRateExceeded, ErrorTracker, StageIncomplete
from .runner import stage
from .storage import write_json

PERIODIC = {"10-K", "10-K405", "10-KT", "10-Q"}
WANTED = PERIODIC | {"8-K"}


def _mp_ctx():
    """spawn: CUDA may be initialised in the parent (Colab); children import secprice via PYTHONPATH."""
    import multiprocessing as mp
    src = str(Path(__file__).resolve().parents[1])
    pp = os.environ.get("PYTHONPATH", "")
    if src not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = src + (os.pathsep + pp if pp else "")
    return mp.get_context("spawn")


def _ts(s) -> pd.Timestamp:
    return pd.Timestamp(s)


# =====================================================================================================  S00
@stage("S00", "env / config / prereg check", "CPU", [], ["run_profile", "seed"],
       outputs=["environment_manifest.json", "config_resolved.json"],
       assertions=["config valid", "preregistration present", "full profile requires G2 (pilot done + owner opt-in)"],
       card="Environment and resolved configuration. The config hash stamps every artifact.")
def s00(ctx):
    from .config import repo_root
    cfg = ctx.cfg
    env = C.env_manifest()
    write_json(ctx.out / "environment_manifest.json", env)
    write_json(ctx.storage.manifests / "environment_manifest.json", env)
    write_json(ctx.out / "config_resolved.json", cfg)
    root = Path(cfg["_meta"]["repo_root"])
    prereg = root / "configs" / "preregistration.yaml"
    assert prereg.exists(), "configs/preregistration.yaml missing"
    if cfg["run_profile"] == "full":
        s18 = ctx.storage.data_root / "pilot" / "S18"          # pilot complete = development holdout scored (needs the GPU part)
        pilot_done = s18.exists() and any((p / "_SUCCESS").exists() for p in s18.glob("*"))
        if not bool(get(cfg, "full.owner_opt_in_full", False)):
            raise RuntimeError("Gate G2: set full.owner_opt_in_full: true in the owner settings to start a full run.")
        if get(cfg, "full.require_pilot_success", True) and not pilot_done:
            raise RuntimeError("Gate G2: a full run needs a completed pilot (pilot/S18 _SUCCESS: development holdout scored). "
                               "Run the pilot first, or set full.require_pilot_success: false to record an explicit owner override.")
        if not pilot_done:
            ctx.warn("G2 override: full run without a completed pilot (owner decision, logged)")
    if not os.environ.get(cfg["sec"]["user_agent_env"]) and not ctx.services.get("fixture_root"):
        ctx.warn(f"{cfg['sec']['user_agent_env']} not set: network stages S02-S06 will refuse to run")
    return {"config_hash": config_hash(cfg), "prereg_sha256": text_sha256(prereg), "owner_overrides": cfg["_meta"]["owner_overrides"],
            "gpu": env["gpu"]}


# =====================================================================================================  S01
@stage("S01", "Hugging Face dataset audit", "CPU", ["S00"], ["hf.repo", "hf.revision"],
       outputs=["hf_census.parquet", "hf_dataset_audit.md"], assertions=["no labels used"],
       card="Census of the HF 10-K corpus (metadata only). Defines the firm overlap with our OHLCV data; never a text source.")
def s01(ctx):
    from .audit.hf_audit import FORBIDDEN_KEYS, audit_markdown, build_census
    h = ctx.cfg["hf"]
    cached = ctx.storage.inputs / f"hf_census_{h['revision'][:12]}.parquet"
    if cached.exists():
        census = pd.read_parquet(cached)
        src = f"cached census {cached.name} (built by secprice.audit.hf_audit.build_census)"
    elif ctx.cfg["run_profile"] == "smoke" and not ctx.services.get("allow_hf_stream_in_smoke"):
        ctx.warn("smoke: no cached HF census in inputs; HF overlap requirement is skipped for smoke")
        census = pd.DataFrame(columns=["cik", "tickers", "form", "filing_date"])
        src = "none (smoke)"
    else:
        census = build_census(h["repo"], h["revision"], ctx.storage.hot_root / "hf_census_parts", ctx.log)
        src = "streamed raw JSONL shards"
    assert not (set(census.columns) & FORBIDDEN_KEYS), "labels must never be used"
    ctx.write(census, "hf_census.parquet")
    md = audit_markdown(census, h["repo"], h["revision"], f"Source of this census: {src}.") if len(census) else \
        "# Hugging Face SEC dataset audit (S01)\n\nNo census available in this smoke run."
    (ctx.out / "hf_dataset_audit.md").write_text(md, encoding="utf-8")
    (ctx.storage.reports / "hf_dataset_audit.md").write_text(md, encoding="utf-8")
    return {"firms": int(census.cik.nunique()) if len(census) else 0, "filings": int(len(census)), "source": src}


# =====================================================================================================  S02
@stage("S02", "Universe", "CPU", ["S01"], ["universe", "study", "hf.revision", "market.local_ohlcv_subdir", "seed"],
       outputs=["universe.parquet", "firms_per_year.csv", "submissions/"],
       assertions=[">= n_stocks unique firms (full) or failure report", "selection reads no price returns"],
       card="Firms common to our OHLCV files and the HF 10-K corpus, mapped to CIK; S&P-500 members first, then a seeded "
            "sector-stratified fill of non-members; every firm carries sp_bucket (SP500 / NonSP500). SURVIVOR-ONLY: the OHLCV "
            "set holds current listings only, so firms that delisted before 2026 are absent (survivorship-biased).")
def s02(ctx):
    from .market.prices import LocalTriframeProvider
    from .sec.submissions import fetch_submissions, fetch_ticker_map
    from .universe.build import firms_per_year, load_ff12, load_sp500, map_tickers, select_universe, sic_to_ff12, sp500_intervals
    cfg, U = ctx.cfg, ctx.cfg["universe"]
    st, en = _ts(cfg["study"]["start"]), _ts(cfg["study"]["end"])
    prov = LocalTriframeProvider(ctx.storage.inputs / cfg["market"]["local_ohlcv_subdir"])
    cov = prov.coverage()
    if cov.empty:
        raise RuntimeError(f"no OHLCV parquet files in {prov.folder}; copy final/1day/*.parquet there (see README)")
    client = C.sec_client(ctx)
    tmap = fetch_ticker_map(client)
    census = C.read(ctx, "S01", "hf_census.parquet")
    cand = map_tickers(cov, tmap, census if len(census) else pd.DataFrame({"cik": [], "tickers": []}))
    sp = load_sp500(U["sp500_commit"], ctx.storage.inputs / f"sp500_ticker_start_end_{U['sp500_commit'][:8]}.csv")
    ff = load_ff12(U["ff12_url"], ctx.storage.inputs / "Siccodes12.zip")
    hf_ciks = census.groupby(census.cik.astype(int)).size() if len(census) else pd.Series(dtype=int)
    allc = cand.ciks_all.map(json.loads)
    cand["in_hf"] = allc.map(lambda cs: any(c in hf_ciks.index for c in cs))
    cand["hf_10k_count"] = allc.map(lambda cs: int(sum(hf_ciks.get(c, 0) for c in cs)))
    ivs = [sp500_intervals(sp, t) for t in cand.ticker]
    cand["sp500_intervals"] = [json.dumps([[str(a.date()), None if pd.isna(b) else str(b.date())] for a, b in iv]) for iv in ivs]
    cand["ever_sp500"] = [any(a <= en and (pd.isna(b) or b >= st) for a, b in iv) for iv in ivs]
    cand["index_start"] = [min((a for a, b in iv), default=pd.NaT) for iv in ivs]
    cand["index_end"] = [max(((b if pd.notna(b) else pd.Timestamp("2099-12-31")) for a, b in iv), default=pd.NaT) for iv in ivs]
    need_hf = bool(U["require_hf_overlap"]) and len(census) > 0
    pre = cand[cand.cik.notna() & (cand.in_hf | (not need_hf))].copy()
    if U.get("fixed_tickers"):
        pre = pre[pre.ticker.isin([t.upper() for t in U["fixed_tickers"]])]
    sub_dir = ctx.out / "submissions"
    sub_dir.mkdir(exist_ok=True)
    metas = {}
    for cik in sorted({c for cs in pre.ciks_all.map(json.loads) for c in cs}):
        p = sub_dir / f"{cik:010d}.parquet"
        mp = sub_dir / f"{cik:010d}.meta.json"
        if p.exists() and mp.exists():
            metas[cik] = json.loads(mp.read_text())
            continue
        meta, f = fetch_submissions(client, cik)
        f.to_parquet(p, index=False)
        mp.write_text(json.dumps(meta, default=str))
        metas[cik] = meta
    # firm id = the CIK with the most periodic filings in the window; filer CIKs = every CIK with filings
    n_per_c = {}
    for c in metas:
        f = pd.read_parquet(sub_dir / f"{c:010d}.parquet", columns=["form", "filing_date"])
        w = f[(pd.to_datetime(f.filing_date) >= st - pd.DateOffset(years=4)) & (pd.to_datetime(f.filing_date) <= en)]
        n_per_c[c] = int(w.form.isin(WANTED).sum())
    firm, filers = [], []
    for cs, c0 in zip(cand.ciks_all.map(json.loads), cand.cik):
        have = [c for c in cs if n_per_c.get(c, 0) > 0]
        firm.append(max(have, key=lambda c: (n_per_c[c], c == c0)) if have else c0)
        filers.append(json.dumps(sorted(have)))
    cand["cik"] = pd.array(firm, dtype="Int64")
    cand["filer_ciks"] = filers
    rows = []
    for r in cand.itertuples(index=False):
        m = metas.get(int(r.cik)) if pd.notna(r.cik) else None
        others = [metas.get(c) or {} for c in json.loads(r.ciks_all) if c != (int(r.cik) if pd.notna(r.cik) else None)]
        fn = [x.get("name") for x in ((m or {}).get("formerNames") or [])] + [o.get("name") for o in others] + \
             [x.get("name") for o in others for x in (o.get("formerNames") or [])]
        rows.append({"sec_name": (m or {}).get("name"), "sic": (m or {}).get("sic"), "sic_desc": (m or {}).get("sicDescription"),
                     "former_names": json.dumps(sorted({x for x in fn if x})),
                     "sec_tickers": json.dumps((m or {}).get("tickers") or []), "fiscal_year_end": (m or {}).get("fiscalYearEnd"),
                     "entity_type": (m or {}).get("entityType"), "filer_category": (m or {}).get("category")})
    cand = pd.concat([cand.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    cand["ff12_industry"] = cand.sic.map(lambda s: sic_to_ff12(s, ff))
    n_per = {}
    for cik in metas:
        f = pd.read_parquet(sub_dir / f"{cik:010d}.parquet", columns=["form", "filing_date"])
        w = f[(pd.to_datetime(f.filing_date) >= st) & (pd.to_datetime(f.filing_date) <= en)]
        n_per[cik] = (int(w.form.isin(PERIODIC).sum()), int(w.form.isin({"20-F", "40-F", "6-K"}).sum()))
    fl = cand.filer_ciks.map(json.loads)
    cand["n_periodic_in_window"] = fl.map(lambda cs: sum(n_per.get(c, (0, 0))[0] for c in cs))
    cand["n_foreign_forms"] = fl.map(lambda cs: sum(n_per.get(c, (0, 0))[1] for c in cs))
    cand["delisted_flag"] = False
    reason = np.full(len(cand), None, dtype=object)
    reason[cand.cik.isna()] = "no_cik_mapping"
    if U.get("fixed_tickers"):
        reason[(reason == None) & ~cand.ticker.isin([t.upper() for t in U["fixed_tickers"]])] = "not_in_profile_ticker_list"  # noqa: E711
    reason[(reason == None) & need_hf & ~cand.in_hf] = "not_in_hf_10k_corpus"  # noqa: E711
    reason[(reason == None) & (cand.n_periodic_in_window == 0) & (cand.n_foreign_forms > 0)] = "foreign_filer_no_10k_10q"  # noqa: E711
    reason[(reason == None) & (cand.n_periodic_in_window == 0)] = "no_periodic_filings_in_window"  # noqa: E711
    reason[(reason == None) & ((cand.price_end < st) | (cand.price_start > en))] = "no_prices_in_window"  # noqa: E711
    cand["exclusion_reason"] = reason
    # one row per CIK: keep the ticker with the longest price history (share classes)
    cand["_len"] = (cand.price_end - cand.price_start).dt.days
    cand = cand.sort_values(["cik", "_len"], ascending=[True, False])
    dup = cand.cik.notna() & cand.duplicated("cik", keep="first")
    cand.loc[dup & cand.exclusion_reason.isna(), "exclusion_reason"] = "duplicate_share_class"
    cand["eligible"] = cand.exclusion_reason.isna()
    n = int(U["n_stocks"])
    sel = select_universe(cand.reset_index(drop=True), n, int(cfg["seed"]), U.get("fixed_tickers") or None,
                          balanced=cfg["run_profile"] == "pilot")
    sel["tickers_history"] = sel.apply(lambda r: json.dumps(sorted(set([r.ticker] + json.loads(r.sec_tickers or "[]")))), axis=1)
    sel = sel.drop(columns=["_len"])
    from .evaluation.buckets import firm_bucket
    sel["sp_bucket"] = firm_bucket(sel.ever_sp500)
    ns = int(sel.selected.sum())
    # survivorship, quantified: historical S&P 500 members in the window vs those our survivor-only data can cover
    in_win = sp[(sp.start_date <= en) & (sp.end_date.isna() | (sp.end_date >= st))]
    hist = set(in_win.ticker.str.upper())
    covered = set(sel[sel.eligible & sel.ever_sp500].ticker.str.upper())
    still = set(sp[sp.end_date.isna()].ticker.str.upper())
    cov = {"sp500_tickers_in_window": len(hist), "covered_by_eligible_universe": len(hist & covered),
           "coverage_share": round(len(hist & covered) / max(1, len(hist)), 3),
           "window_members_no_longer_in_index": len(hist - still),
           "selected_sp500": int((sel.selected & sel.ever_sp500).sum()), "selected_non_sp500": int((sel.selected & ~sel.ever_sp500).sum()),
           "eligible_sp500": int((sel.eligible & sel.ever_sp500).sum()), "eligible_non_sp500": int((sel.eligible & ~sel.ever_sp500).sum()),
           "note": "ticker-level match against fja05680 (ticker changes and the list's early-year gaps make this approximate); "
                   "members that delisted, merged or were acquired are missing from our survivor-only OHLCV data"}
    write_json(ctx.out / "sp500_coverage.json", cov)
    if ns < n:
        msg = (f"universe: {ns} firms selected < target {n} (eligible {int(sel.eligible.sum())}); reasons: "
               f"{sel.exclusion_reason.value_counts().to_dict()}")
        ctx.warn(msg)
        (ctx.out / "universe_failure_report.md").write_text(msg, encoding="utf-8")
    ctx.write(sel, "universe.parquet")
    fpy = firms_per_year(sel, cfg["study"]["start"], cfg["study"]["end"])
    fpy.to_csv(ctx.out / "firms_per_year.csv", index=False)
    return {"candidates": int(len(sel)), "eligible": int(sel.eligible.sum()), "selected": ns,
            "selected_sp500": cov["selected_sp500"], "selected_non_sp500": cov["selected_non_sp500"],
            "sp500_history_coverage": cov["coverage_share"], "sec_requests": client.stats}


# =====================================================================================================  S03
@stage("S03", "Calendar, prices, actions, factors", "CPU", ["S02"], ["market", "study"],
       outputs=["calendar.parquet", "prices.parquet", "corporate_actions.parquet", "factors.parquet", "price_coverage.csv"],
       assertions=["split/dividend unit tests pass", "explicit downloader parameters recorded"],
       card="Firm x session panel: split-adjusted OHLC from our local files (+ yfinance warm-up), raw close, total-return index.")
def s03(ctx):
    from .market.calendar import TradingCalendar
    from .market.factors import fetch_factors
    from .market.prices import LocalTriframeProvider, YFinanceProvider, build_series, splice
    cfg, M = ctx.cfg, ctx.cfg["market"]
    u = C.universe(ctx)
    end_cal = (_ts(cfg["study"]["end"]) + pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    cal = ctx.services.get("calendar") or TradingCalendar.nyse(M["warmup_start"], end_cal)
    ctx.write(cal.to_frame(), "calendar.parquet")
    local = LocalTriframeProvider(ctx.storage.inputs / M["local_ohlcv_subdir"])
    yfp = ctx.services.get("yf_provider") or YFinanceProvider(M["yfinance"], ctx.storage.inputs / "yf_frozen.parquet", ctx.log)
    frames, acts, covrows = [], [], []
    for r in u.itertuples(index=False):
        lb = local.load(r.ticker)
        first = lb.date.min() if len(lb) else _ts(end_cal)
        wb, wa = yfp.history(r.ticker, M["warmup_start"], (first + pd.Timedelta(days=60)).strftime("%Y-%m-%d"))
        _, all_acts = yfp.history(r.ticker, M["warmup_start"], end_cal)
        bars, seam = splice(lb, wb[wb.date < first + pd.Timedelta(days=60)] if len(wb) else wb, int(M["seam_check_days"]))
        s = build_series(bars, all_acts)
        s["ticker"], s["cik"] = r.ticker, int(r.cik)
        frames.append(s)
        if len(all_acts):
            acts.append(all_acts.assign(ticker=r.ticker, cik=int(r.cik)))
        sess = cal.sessions[(cal.sessions >= s.date.min()) & (cal.sessions <= s.date.max())] if len(s) else []
        covrows.append({"ticker": r.ticker, "cik": int(r.cik), "first": s.date.min() if len(s) else None,
                        "last": s.date.max() if len(s) else None, "n_bars": len(s),
                        "missing_sessions_in_range": int(len(set(sess) - set(s.date))) if len(s) else None,
                        "delisted_history": "not covered (survivors only)", **seam})
    sb, _ = yfp.history("SPY", M["warmup_start"], end_cal)
    _, sa = yfp.history("SPY", M["warmup_start"], end_cal)
    spy = build_series(sb, sa)
    spy["ticker"], spy["cik"] = "SPY", 0
    frames.append(spy)
    px = pd.concat(frames, ignore_index=True)
    px["source_version"] = np.where(px.source.astype(str).str.startswith("local"), local.version, yfp.version)
    px = px[["cik", "ticker", "date", "open", "high", "low", "close_raw", "close_split_adj", "total_return_index",
             "volume", "dividend", "source", "source_version"]]
    ctx.write(px, "prices.parquet")
    ca = pd.concat(acts, ignore_index=True) if acts else pd.DataFrame(columns=["date", "type", "value", "ticker", "cik"])
    ctx.write(ca, "corporate_actions.parquet")
    fac, fprov = (ctx.services["factors"], {"source": "fixture"}) if "factors" in ctx.services else \
        fetch_factors(M["factors"]["ff5_url"], M["factors"]["mom_url"], ctx.storage.inputs)
    ctx.write(fac, "factors.parquet")
    cov = pd.DataFrame(covrows)
    cov.to_csv(ctx.out / "price_coverage.csv", index=False)
    write_json(ctx.out / "price_params.json", {"yfinance_params": M["yfinance"], "yfinance_version": yfp.version,
                                               "local_version": local.version, "factors": fprov, "calendar": cal.source})
    return {"tickers": int(u.shape[0]), "price_rows": int(len(px)), "spy_rows": int(len(spy)),
            "factor_last_date": str(pd.to_datetime(fac.date).max().date()) if len(fac) else None}


# =====================================================================================================  S04
def _manifest(ctx, u) -> pd.DataFrame:
    cfg = ctx.cfg
    st, en = _ts(cfg["study"]["start"]), _ts(cfg["study"]["end"])
    t0 = st - pd.DateOffset(years=float(cfg["text"]["predecessor_lookback_years"]))
    a0 = st - pd.DateOffset(years=float(cfg["accounting"]["predecessor_lookback_years"]))
    sub_dir = ctx.dep("S02") / "submissions"
    frames = []
    for r in u.itertuples(index=False):
        for fc in (json.loads(r.filer_ciks) if isinstance(r.filer_ciks, str) else [int(r.cik)]) or [int(r.cik)]:
            p = sub_dir / f"{int(fc):010d}.parquet"
            if p.exists():
                f = pd.read_parquet(p)
                f["filer_cik"] = int(fc)
                f["cik"] = int(r.cik)
                frames.append(f)
    m = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    m["base_form"] = m.form.str.replace("/A", "", regex=False)
    m = m[m.base_form.isin(WANTED)].copy()
    m["is_amendment"] = m.form.str.endswith("/A")
    fd = pd.to_datetime(m.filing_date)
    m["in_study"] = (fd >= st) & (fd <= en)
    per = m.base_form.isin(PERIODIC)
    m["in_window"] = np.where(per, (fd >= t0) & (fd <= en), (fd >= a0) & (fd <= en))
    m = m[m.in_window].copy()
    m["items_list"] = m.items_raw.map(lambda s: [x.strip() for x in str(s).split(",") if x.strip()] if pd.notna(s) and str(s) else [])
    acc = m.acceptance_raw.map(lambda s: json_to_et(s) if pd.notna(s) and str(s) else pd.NaT)
    m["acceptance_json_et"] = acc
    m["json_time_is_placeholder"] = acc.map(lambda t: bool(pd.notna(t) and t.hour == 0 and t.minute == 0 and t.second == 0))
    tax = C.taxonomy(cfg)
    pre_8k = (~per) & ~m.in_study
    earn_like = m.items_list.map(lambda it: (not it) or any(tax.map_item(i, "2000-01-01") == "earnings" or str(i) in tax.legacy_earn
                                                            or tax.map_item(i, "2010-01-01") == "earnings" for i in it))
    m["download_task"] = ~m.is_amendment & (m.in_study | per | (pre_8k & earn_like))
    reason = np.where(m.is_amendment, "AMENDMENT_AUDIT_ONLY", np.where(m.download_task, "OK_TASK", "PRESTUDY_8K_NOT_EARNINGS"))
    m["audit_reason_code"] = reason
    if cfg["run_profile"] == "smoke":
        cap = int(get(cfg, "smoke.max_filings_per_firm_form", 10))
        m = m.sort_values("filing_date")
        spread = lambda g: g.iloc[np.unique(np.linspace(0, len(g) - 1, min(cap, len(g))).round().astype(int))]  # noqa: E731
        keep = pd.concat([spread(g) for _, g in m[m.in_study].groupby(["cik", "base_form"])]).index.union(
            m[~m.in_study].groupby(["cik", "base_form"]).tail(4).index)
        m = m.loc[keep]
    return m.sort_values(["cik", "filing_date", "accession"]).reset_index(drop=True)


def periodic_expectations(per: pd.DataFrame) -> pd.DataFrame:
    """Expected 1 x 10-K and 3 x 10-Q per firm x report year while the firm files; reason codes. 8-Ks are event-driven
    and never expected (a company-year without an 8-K is not missing data)."""
    rows = []
    for cik, g in per.groupby("cik"):
        y0, y1 = int(g.ry.min()), int(g.ry.max())
        for y in range(y0, y1 + 1):
            gy = g[g.ry == y]
            k, q = int(gy.base_form.str.startswith("10-K").sum()), int((gy.base_form == "10-Q").sum())
            code = "OK" if (k == 1 and q == 3) else ("MISSING_10K" if k == 0 else ("EXTRA_10K_POSSIBLE_TRANSITION" if k > 1 else
                                                                                    f"10Q_COUNT_{q}"))
            rows.append({"cik": cik, "year": y, "expected_10k": 1, "observed_10k": k, "expected_10q": 3, "observed_10q": q,
                         "reason_code": code})
    return pd.DataFrame(rows, columns=["cik", "year", "expected_10k", "observed_10k", "expected_10q", "observed_10q", "reason_code"])


@stage("S04", "EDGAR index + coverage audit", "CPU", ["S02"],
       ["study", "text.predecessor_lookback_years", "accounting.predecessor_lookback_years", "sec.header_check_sample",
        "timing", "smoke", "taxonomy_version"],
       outputs=["filing_manifest.parquet", "coverage_firm_year.csv", "ts_quality_by_year.csv", "header_check.csv",
                "data_coverage_summary.md"],
       assertions=["no 8-K expected", "timestamp-quality table"],
       card="Observed filings per firm with corrected acceptance times, periodic-filing expectations and reason codes.")
def s04(ctx):
    from .sec.archives import index_headers_url
    from .events.timestamps import parse_header_acceptance
    cfg = ctx.cfg
    u = C.universe(ctx)
    m = _manifest(ctx, u)
    # periodic expectations by firm x calendar year of report date (8-Ks are event-driven: no expectation)
    per = m[m.base_form.isin(PERIODIC) & ~m.is_amendment & m.in_study].copy()
    per["ry"] = pd.to_datetime(per.report_date.replace("", np.nan)).dt.year.fillna(pd.to_datetime(per.filing_date).dt.year)
    cov = periodic_expectations(per)
    assert "8-K" not in set(cov.columns), "8-Ks carry no expectation"
    cov.to_csv(ctx.out / "coverage_firm_year.csv", index=False)
    # header sample check of the JSON timestamp rule (periodic filings with index-headers pages)
    client = C.sec_client(ctx)
    rng = np.random.default_rng(int(cfg["seed"]))
    pool = m[m.base_form.isin(PERIODIC) & ~m.json_time_is_placeholder & m.acceptance_json_et.notna()]
    pool = pool[pd.to_datetime(pool.filing_date) >= "2016-01-01"]
    k = min(int(cfg["sec"]["header_check_sample"]), len(pool))
    chk = []
    for r in pool.iloc[rng.choice(len(pool), size=k, replace=False)].itertuples(index=False) if k else []:
        b = client.get(index_headers_url(r.filer_cik, r.accession))
        raw = parse_header_acceptance(b.decode("latin-1")) if b else None
        if raw:
            ht = header_to_et(raw)
            chk.append({"accession": r.accession, "json_et": r.acceptance_json_et, "header_et": ht,
                        "match": bool(abs((r.acceptance_json_et - ht).total_seconds()) < 1)})
    hc = pd.DataFrame(chk, columns=["accession", "json_et", "header_et", "match"])
    hc.to_csv(ctx.out / "header_check.csv", index=False)
    m["ts_quality_json"] = np.where(m.acceptance_json_et.isna(), "C", np.where(m.json_time_is_placeholder, "B", "A"))
    tq = m.assign(year=pd.to_datetime(m.filing_date).dt.year).groupby(["year", "ts_quality_json"]).size().unstack(fill_value=0)
    tq.to_csv(ctx.out / "ts_quality_by_year.csv")
    m["url"] = None
    m["parse_status"] = None
    m["checksum"] = None
    ctx.write(m, "filing_manifest.parquet")
    mm = (hc.match.mean() if len(hc) else None)
    md = ["# Data coverage summary (S04)", "",
          f"- Firms: {m.cik.nunique()}; filings in window: {len(m):,}; download tasks: {int(m.download_task.sum()):,}",
          f"- By form: {m.form.value_counts().to_dict()}",
          f"- Periodic firm-years: {len(cov):,}; reason codes: {cov.reason_code.value_counts().to_dict() if len(cov) else {}}",
          "- 8-K filings are event-driven: a company-year without an 8-K is not missing data (no expectation).",
          f"- JSON acceptance-time rule checked against {len(hc)} SGML headers: match rate {mm if mm is None else round(mm, 4)}",
          f"- Timestamp quality (JSON-derived) by year: see ts_quality_by_year.csv"]
    (ctx.out / "data_coverage_summary.md").write_text("\n".join(md), encoding="utf-8")
    (ctx.storage.reports / "data_coverage_summary.md").write_text("\n".join(md), encoding="utf-8")
    return {"filings": int(len(m)), "tasks": int(m.download_task.sum()), "header_checks": int(len(hc)),
            "header_match_rate": None if mm is None else float(mm)}


def teacher_extraction_will_run(cfg: dict, inputs: Path) -> bool:
    """The teacher extraction (budget.llm_teacher_filings) runs only after gold-val labels exist (G3b) or with the owner's
    exploratory override (D-037); otherwise only the gold-sample LLM prompts run."""
    g = Path(inputs) / "gold"
    has_gold = g.exists() and any(g.glob("gold_*_filings_template*.csv"))
    return bool(has_gold or get(cfg, "rag.run_extraction_without_gold", False))


def preflight_projection(cfg: dict, manifest: pd.DataFrame, data_root, hot_root) -> dict:
    """CPU-side preflight (hard rule 4, §13.3, G2): projected events, chunks, storage and GPU hours from the filing index,
    before the long download. Constants come from the pilot (configs/default.yaml budget.preflight_*)."""
    B = cfg["budget"]
    t = manifest[manifest.download_task]
    by = t.base_form.map(lambda f: "10-K" if str(f).startswith("10-K") else "10-Q" if str(f).startswith("10-Q") else "8-K")
    cpf = {k: float(v) for k, v in B["preflight_chunks_per_filing"].items()}
    chunks = float(sum(cpf.get(k, 10.0) * n for k, n in by.value_counts().items()))
    n_enc = len(cfg["models"]["encoders"])
    events = int(round(len(t) * 0.89))                       # events per filing on the pilot (same-day 8-Ks merge)
    emb_gb = events * 768 * 2 * 2 * n_enc / 1e9 * 1.1          # mean + max pooled float16 vectors, +10% index
    text_gb = len(t) * float(B["preflight_bytes_per_filing"]) / 1e9
    gpu_b = chunks * n_enc / float(B["preflight_assumed_chunks_per_s"]) / 3600
    teacher = teacher_extraction_will_run(cfg, Path(data_root) / cfg["paths"].get("inputs_dirname", "inputs"))
    prompts = (int(B["llm_teacher_filings"]) if teacher else 0) + int(B["llm_gold_filings"])
    gpu_c = prompts * float(B["preflight_llm_s_per_prompt"]) / 3600
    rep = {"profile": cfg["run_profile"], "firms": int(t.cik.nunique()), "filings_to_download": int(len(t)),
           "filings_by_form": by.value_counts().to_dict(), "est_events": events, "est_chunks": int(chunks),
           "est_storage_gb": round(emb_gb + text_gb, 2), "est_embedding_gb": round(emb_gb, 2), "est_text_gb": round(text_gb, 2),
           "est_gpu_hours_encoders": round(gpu_b, 2), "est_gpu_hours_llm": round(gpu_c, 2), "llm_prompts": prompts,
           "teacher_extraction_runs": teacher,
           "est_sec_download_hours": round(len(t) * 1.6 / float(cfg["sec"]["max_rps"]) / 3600, 1),
           "disk_free_gb": {"data_root": C.disk_free_gb(data_root), "hot_root": C.disk_free_gb(hot_root)},
           "budget": {k: B[k] for k in ("max_storage_gb", "max_gpu_hours_b", "max_gpu_hours_c")}}
    over = []
    if rep["est_storage_gb"] > float(B["max_storage_gb"]):
        over.append(f"storage {rep['est_storage_gb']} GB > max_storage_gb {B['max_storage_gb']}")
    if gpu_b > float(B["max_gpu_hours_b"]):
        over.append(f"encoder GPU hours {gpu_b:.1f} > max_gpu_hours_b {B['max_gpu_hours_b']}")
    if gpu_c > float(B["max_gpu_hours_c"]):
        over.append(f"LLM GPU hours {gpu_c:.1f} > max_gpu_hours_c {B['max_gpu_hours_c']}")
    rep["over_budget"] = over
    rep["passed"] = not over
    return rep


# =====================================================================================================  S05
@stage("S05", "Download + parse + taxonomy + anonymize", "CPU", ["S04"],
       ["sec.max_8k_bytes", "sec.max_doc_bytes", "text.max_chars_per_section", "text.sections", "smoke", "taxonomy_version",
        "budget.max_error_rate", "hf.qa_enabled", "hf.qa_sample_filings"],
       outputs=["sections/<cik>.parquet", "exhibits/<cik>.parquet", "ledger/<cik>.parquet", "sections_index.parquet",
                "parse_qa.csv"],
       assertions=["taxonomy version recorded", "OTHER_OR_UNMAPPED share reported", "error rate within budget"],
       card="Primary documents (10-K/10-Q) and full 8-K submissions downloaded under the fair-access rate limit, parsed into "
            "target sections with Exhibit-13 fallback, anonymized (firm names/tickers masked).")
def s05(ctx):
    from .parse.worker import run_firm, run_firm_safe
    from .sec.client import resolve_user_agent
    cfg, S = ctx.cfg, ctx.cfg["sec"]
    m = C.manifest(ctx)
    pf = preflight_projection(cfg, m, ctx.storage.data_root, ctx.storage.hot_root)
    write_json(ctx.out / "preflight_projection.json", pf)
    write_json(ctx.storage.manifests / "preflight_projection.json", pf)
    if cfg["run_profile"] == "full" and not pf["passed"]:
        raise RuntimeError(f"preflight (G2) refuses the full run before the download: {pf['over_budget']}; raise the budget "
                           "in configs/default.yaml only as an explicit owner decision")
    tasks = m[m.download_task]
    u = C.universe(ctx).set_index("cik")
    fixture = ctx.services.get("fixture_root")
    nproc = int(S.get("parse_workers") or 0) or max(1, (os.cpu_count() or 2) - 1)
    # one process per firm, each limited to max_rps / nproc (the total never exceeds sec.max_rps); downloads are
    # latency-bound, so more processes use more of the fair-access budget (RAM ~100 MB per worker)
    nproc = 1 if fixture else min(nproc, int(S.get("max_parse_workers") or 8), max(1, tasks.cik.nunique()))
    ua = "fixture" if fixture else resolve_user_agent(S["user_agent_env"])
    jobs = []
    for cik, g in tasks.groupby("cik"):
        ur = u.loc[cik] if cik in u.index else None
        names = [ur.sec_name] + json.loads(ur.former_names or "[]") if ur is not None else []
        tickers = json.loads(ur.tickers_history) if ur is not None else []
        jobs.append({"cik": int(cik), "out_dir": str(ctx.out), "fixture_root": fixture, "user_agent": ua,
                     "rps_per_proc": float(S["max_rps"]) / nproc, "cache_dir": str(ctx.storage.hot_root / "sec_cache"),
                     "cache_days": S["cache_days"], "cache_max_bytes": int(S.get("cache_max_bytes") or 0),
                     "retry_errors": bool(S.get("retry_failed_filings", True)),
                     "timeout_s": S["timeout_s"], "max_retries": S["max_retries"],
                     "backoff_base_s": S["backoff_base_s"], "blocked_403_limit": S.get("blocked_403_limit", 20),
                     "names": [n for n in names if n], "tickers": tickers, "max_chars": int(cfg["text"]["max_chars_per_section"]),
                     "max_8k_bytes": int(S["max_8k_bytes"]), "max_doc_bytes": int(S["max_doc_bytes"]),
                     "filings": g[["cik", "filer_cik", "accession", "form", "primary_doc", "filing_date"]].to_dict("records")})
    t0 = time.time()
    done, crashes = 0, []
    wall = float(S.get("s05_max_wall_hours") or 0) * 3600

    def _over():
        return wall > 0 and time.time() - t0 > wall

    if nproc == 1:
        for j in jobs:
            if _over():
                break
            r = run_firm_safe(j)
            done += 1
            if r.get("blocked"):
                raise RuntimeError(r["blocked"])
            if r.get("crash"):
                crashes.append(r)
    else:
        with ProcessPoolExecutor(max_workers=nproc, mp_context=_mp_ctx()) as ex:
            futs = {ex.submit(run_firm_safe, j): j["cik"] for j in jobs}
            for f in as_completed(futs):
                if f.cancelled():
                    continue
                r = f.result()
                done += 1
                if r.get("blocked"):
                    for ff in futs:
                        ff.cancel()
                    raise RuntimeError(r["blocked"])
                if r.get("crash"):
                    crashes.append(r)
                if _over():
                    for ff in futs:
                        ff.cancel()
                if done % 10 == 0 or done == len(jobs):
                    ctx.log.info(f"S05 firms {done}/{len(jobs)} in {time.time() - t0:.0f}s")
    n_ledger = len(list((ctx.out / "ledger").glob("*.parquet"))) if (ctx.out / "ledger").exists() else 0
    if n_ledger < len(jobs) and _over():
        raise StageIncomplete(f"S05 reached its {wall / 3600:.1f} h wall-clock budget with {n_ledger}/{len(jobs)} firms done; "
                              "run the same cell again to continue (finished firms are skipped).")
    if crashes:
        ctx.warn(f"{len(crashes)} firm jobs crashed: {[c['crash'][:120] for c in crashes[:5]]}")
    led = pd.concat([pd.read_parquet(p) for p in sorted((ctx.out / "ledger").glob("*.parquet"))], ignore_index=True)
    sec_idx = []
    for p in sorted((ctx.out / "sections").glob("*.parquet")):
        s = pd.read_parquet(p, columns=["accession", "cik", "form", "section_id", "n_chars", "n_tokens_est", "parse_method",
                                        "by_reference", "truncated", "mask_firm", "mask_ticker"])
        s["part"] = p.name
        sec_idx.append(s)
    sec_idx = pd.concat(sec_idx, ignore_index=True) if sec_idx else pd.DataFrame()
    ctx.write(led, "parse_ledger.parquet")
    ctx.write(sec_idx, "sections_index.parquet")
    # error-rate guard (download/parse errors only; 'no_sections' is reported, not an error)
    err = led.status.isin(["error", "http_404"]).mean() if len(led) else 0.0
    mm = m.merge(led[["accession", "status", "n_sections"]], on="accession", how="left")
    mm["year"] = pd.to_datetime(mm.filing_date).dt.year
    qa = mm[mm.download_task].groupby(["base_form", "year"]).agg(
        observed=("accession", "size"), downloaded_ok=("status", lambda s: int((s == "ok").sum())),
        with_sections=("n_sections", lambda s: int((s.fillna(0) > 0).sum()))).reset_index()
    if len(sec_idx):
        found = sec_idx.groupby(["form", "section_id"]).size().rename("n").reset_index()
        found.to_csv(ctx.out / "section_found_counts.csv", index=False)
    qa.to_csv(ctx.out / "parse_qa.csv", index=False)
    tax = C.taxonomy(cfg)
    k8 = m[(m.base_form == "8-K") & m.download_task]
    cats = [c for r in k8.itertuples(index=False) for c in tax.categories(r.items_list, r.filing_date)]
    other_share = (np.mean([c == tax.catch_all for c in cats]) if cats else None)
    hfqa = _hf_parser_qa(ctx, m, sec_idx)
    if err > float(cfg["budget"]["max_error_rate"]) and len(led) >= 50:
        raise ErrorRateExceeded(f"S05 error rate {err:.3f} > budget; see parse_ledger.parquet")
    return {"firms": len(jobs), "filings": int(len(led)), "status": led.status.value_counts().to_dict(), "hf_parser_qa": hfqa,
            "sections": int(len(sec_idx)), "error_rate": round(float(err), 4), "taxonomy_version": tax.version,
            "other_or_unmapped_share": other_share, "wall_s": round(time.time() - t0, 1)}


def _one_text_per_section(df: pd.DataFrame, col: str) -> pd.Series:
    """section_id -> the longest text (duplicates collapse to one string, missing text to '')."""
    if df.empty:
        return pd.Series(dtype=object)
    d = df.assign(_n=df[col].fillna("").astype(str).str.len()).sort_values("_n", ascending=False)
    return d.drop_duplicates("section_id").set_index("section_id")[col].fillna("").astype(str)


def _hf_parser_qa(ctx, m: pd.DataFrame, sec_idx: pd.DataFrame) -> dict:
    """Optional QA of our 10-K section parser against the HF corpus sections (same CIK + filing date), streaming at
    most hf.qa_max_shards shards. Agreement is reported as length ratio and token-set Jaccard per section."""
    from .audit.hf_audit import qa_sections_for
    h = ctx.cfg["hf"]
    if not h.get("qa_enabled", True) or ctx.cfg["run_profile"] == "smoke" or ctx.services.get("fixture_root"):
        return {"ran": False, "reason": "disabled for smoke/offline runs"}
    census = C.read(ctx, "S01", "hf_census.parquet")
    if census.empty or sec_idx.empty:
        return {"ran": False, "reason": "no census or sections"}
    k = m[(m.base_form == "10-K") & m.download_task][["cik", "accession", "filing_date"]]
    c = census.assign(cik_i=census.cik.astype(int))
    j = k.merge(c[["cik_i", "filing_date", "shard"]], left_on=["cik", "filing_date"], right_on=["cik_i", "filing_date"])
    if j.empty:
        return {"ran": False, "reason": "no matched 10-Ks"}
    top = j.shard.value_counts().index[: int(h.get("qa_max_shards", 2))]
    j = j[j.shard.isin(top)]
    j = j.sample(min(len(j), int(h["qa_sample_filings"])), random_state=0)
    try:
        hf = qa_sections_for(h["repo"], h["revision"], census[census.shard.isin(top)], j[["cik", "filing_date"]])
    except Exception as e:
        return {"ran": False, "reason": f"stream failed: {e}"}
    rows = []
    for r in j.itertuples(index=False):
        p = ctx.out / "sections" / f"{int(r.cik):010d}.parquet"
        if not p.exists():
            continue
        ours = pd.read_parquet(p, columns=["accession", "section_id", "text_raw"])
        ours = _one_text_per_section(ours[ours.accession == r.accession], "text_raw")
        # the HF corpus can hold two filings for one firm and date (e.g. a 10-K and its amendment): keep the longest
        theirs = _one_text_per_section(hf[(hf.cik == int(r.cik)) & (hf.filing_date == r.filing_date)], "hf_text")
        for sid in ("10K_1", "10K_1A", "10K_7", "10K_7A"):
            a, b = ours.get(sid, ""), theirs.get(sid, "")
            if not a and not b:
                continue
            ta, tb = set(a.lower().split()), set(b.lower().split())
            rows.append({"cik": int(r.cik), "accession": r.accession, "section_id": sid, "ours_chars": len(a), "hf_chars": len(b),
                         "len_ratio": len(a) / len(b) if b else None,
                         "token_jaccard": len(ta & tb) / len(ta | tb) if (ta | tb) else None})
    qa = pd.DataFrame(rows)
    qa.to_csv(ctx.out / "hf_parser_qa.csv", index=False)
    return {"ran": True, "filings": int(j.shape[0]), "median_token_jaccard_by_section":
            qa.groupby("section_id").token_jaccard.median().round(3).to_dict() if len(qa) else {}}


# =====================================================================================================  S06
@stage("S06", "XBRL + release EPS + SUE", "CPU", ["S05"], ["accounting", "timing", "sec.max_rps"],
       outputs=["xbrl_facts.parquet", "eps_release.parquet", "sue_by_accession.parquet", "sue_sources.parquet", "sue_coverage.csv"],
       assertions=["no later-cluster source", "ordering by public time"],
       card="Point-in-time accounting: XBRL facts with public times, as-first-reported quarterly EPS, release-exhibit EPS and SUE.")
def s06(ctx):
    from .features.accounting import attach_public_time, compute_sue_table, quarterly_eps
    from .parse.eps_release import USABLE_CONFIDENCE, infer_quarter_end, is_results_release, parse_release_eps
    from .sec.companyfacts import facts_frame, fetch_companyfacts
    cfg, A, T = ctx.cfg, ctx.cfg["accounting"], ctx.cfg["timing"]
    m = C.manifest(ctx)
    led = C.read(ctx, "S05", "parse_ledger.parquet")
    u = C.universe(ctx).set_index("cik")
    acc_t = _final_acceptance(m, led, cfg, ctx)
    times = derive_times(acc_t["acceptance_time"], T["public_lag_minutes"], T["decision_latency_minutes"])
    ft = pd.concat([acc_t[["accession", "cik", "form", "filing_date", "report_date"]].reset_index(drop=True), times], axis=1)
    client = C.sec_client(ctx)
    facts = []
    tags = list(A["eps_tags"]) + list(A["fact_tags"])
    for (cik, fc) in sorted(set(zip(m.cik.astype(int), m.filer_cik.astype(int)))):
        js = fetch_companyfacts(client, int(fc))
        ff = facts_frame(js, tags, [A["shares_tag"]])
        ff["cik"] = int(cik)                       # firm id (a predecessor/successor filer maps to the same firm)
        ff["filer_cik"] = int(fc)
        facts.append(ff)
    facts = pd.concat(facts, ignore_index=True) if facts else pd.DataFrame()
    facts = attach_public_time(facts, ft, T["public_lag_minutes"], T["date_only_assumed_time"]) if len(facts) else facts
    ctx.write(facts, "xbrl_facts.parquet")
    qe = quarterly_eps(facts) if len(facts) else pd.DataFrame(columns=["cik", "kind", "period_end", "qkey", "value",
                                                                       "source_accession", "public_available_time"])
    ctx.write(qe, "eps_quarterly_xbrl.parquet")
    # release EPS from EX-99 exhibits of earnings candidates
    tax = C.taxonomy(cfg)
    rel_rows = []
    k8 = m[(m.base_form == "8-K") & m.download_task]
    for cik, g in k8.groupby("cik"):
        p = ctx.dep("S05") / "exhibits" / f"{int(cik):010d}.parquet"
        if not p.exists():
            continue
        ex = pd.read_parquet(p)
        ex = ex[ex.exhibit_type.str.startswith("EX-99")]
        fye = u.loc[cik, "fiscal_year_end"] if cik in u.index else None
        for r in g.itertuples(index=False):
            modern, legacy = tax.earnings_candidate(r.items_list, r.filing_date)
            if not (modern or legacy or not r.items_list):
                continue
            exs = ex[ex.accession == r.accession]
            best, flag = None, False
            for e in exs.itertuples(index=False):
                if is_results_release(e.text):
                    flag = True
                    pr = parse_release_eps(e.text)
                    if pr["eps_diluted"] is not None or pr["eps_basic"] is not None:
                        best = pr
                        break
                    best = best or pr
            inf, inf_src = infer_quarter_end(r.filing_date, fye)
            pe, ps = (best or {}).get("period_end"), (best or {}).get("period_source")
            if pe and inf and abs((pd.Timestamp(pe) - pd.Timestamp(inf)).days) > 45:
                pe, ps = inf, "inferred_override_text_far"
            if not pe:
                pe, ps = inf, inf_src
            rel_rows.append({"accession": r.accession, "cik": int(cik), "is_results_release": flag,
                             "eps_diluted": (best or {}).get("eps_diluted"), "eps_basic": (best or {}).get("eps_basic"),
                             "parse_confidence": (best or {}).get("parse_confidence", "none"),
                             "source_line": (best or {}).get("source_line"), "period_end": pe, "period_source": ps,
                             "candidate_modern": modern, "candidate_legacy_equivalent": legacy})
    rel = pd.DataFrame(rel_rows, columns=["accession", "cik", "is_results_release", "eps_diluted", "eps_basic", "parse_confidence",
                                          "source_line", "period_end", "period_source", "candidate_modern",
                                          "candidate_legacy_equivalent"])
    rel = rel.merge(ft[["accession", "public_available_time", "decision_time"]], on="accession", how="left")
    from .parse.eps_release import flag_suspect_annual
    fye_map = {int(c): u.loc[c, "fiscal_year_end"] for c in u.index if "fiscal_year_end" in u.columns}
    rel = flag_suspect_annual(rel, fye_map)
    if len(rel) and rel.get("suspect_annual", pd.Series(dtype=bool)).any():
        ctx.warn(f"{int(rel.suspect_annual.sum())} release EPS values withheld as probable full-year figures (suspect_annual)")
    # QA: release EPS vs first-reported XBRL for the same quarter (QA only, never an input)
    from .features.accounting import _qkey
    rel["qkey"] = _qkey(pd.to_datetime(rel["period_end"])) if len(rel) else pd.Series(dtype="Int64")
    xd = qe[qe.kind == "diluted"].sort_values("public_available_time")[["cik", "qkey", "value"]].rename(columns={"value": "xbrl_eps"}) if len(qe) else \
        pd.DataFrame(columns=["cik", "qkey", "xbrl_eps"])
    rel = rel.merge(xd.drop_duplicates(["cik", "qkey"]), on=["cik", "qkey"], how="left").drop_duplicates("accession")
    rel["xbrl_match_flag"] = np.where(rel.xbrl_eps.notna() & rel.eps_diluted.notna(),
                                      (rel.eps_diluted - rel.xbrl_eps).abs() <= 0.015, None)
    ctx.write(rel, "eps_release.parquet")
    usable = rel[rel.is_results_release & rel.parse_confidence.isin(USABLE_CONFIDENCE) & rel.eps_diluted.notna()]
    rel_hist = pd.DataFrame({"cik": usable.cik, "qkey": usable.qkey, "value": usable.eps_diluted,
                             "source_accession": usable.accession, "public_available_time": usable.public_available_time})
    xh = qe[qe.kind == "diluted"][["cik", "qkey", "value", "source_accession", "public_available_time", "period_end"]] if len(qe) else rel_hist.iloc[:0]
    # SUE per candidate accession: verified-release candidates and periodic filings
    evs = []
    for r in usable.itertuples(index=False):
        evs.append({"accession": r.accession, "cik": r.cik, "kind": "release", "decision_time": r.decision_time,
                    "eps_q": r.eps_diluted, "qkey": r.qkey})
    perf = ft[ft.form.isin(PERIODIC)]
    for r in perf.itertuples(index=False):
        evs.append({"accession": r.accession, "cik": r.cik, "kind": "periodic", "decision_time": r.decision_time,
                    "eps_q": np.nan, "qkey": None})
    evdf = pd.DataFrame(evs)
    if len(evdf):
        evdf = evdf.drop_duplicates("accession")
        evdf["decision_time"] = pd.to_datetime(evdf["decision_time"], utc=True).dt.tz_convert("America/New_York")
        sue, src = compute_sue_table(evdf, xh, rel_hist, int(A["sue_min_diffs"]), int(A["sue_max_diffs"]))
    else:
        sue, src = pd.DataFrame(), pd.DataFrame(columns=["accession", "source_accession", "source_public_time", "source_kind"])
    if len(src) and len(evdf):
        dmap = evdf.set_index("accession")["decision_time"]
        bad = pd.to_datetime(src.source_public_time, utc=True) >= pd.to_datetime(src.accession.map(dmap), utc=True)
        assert not bad.any(), "SUE history must be public strictly before decision_time"
    ctx.write(sue, "sue_by_accession.parquet")
    ctx.write(src, "sue_sources.parquet")
    if len(sue):
        sc = sue.merge(ft[["accession", "filing_date"]], on="accession", how="left")
        sc["year"] = pd.to_datetime(sc.filing_date).dt.year
        sc["kind"] = np.where(sc.eps_q_source == "release_exhibit", "8-K release", "10-Q/10-K")
        cov = sc.groupby(["year", "kind"]).agg(n=("accession", "size"), sue_available=("sue", lambda s: int(s.notna().sum()))).reset_index()
        cov.to_csv(ctx.out / "sue_coverage.csv", index=False)
    match = rel.xbrl_match_flag.dropna()
    return {"facts": int(len(facts)), "xbrl_quarters": int(len(qe)), "release_candidates": int(len(rel)),
            "results_releases": int(rel.is_results_release.sum()) if len(rel) else 0,
            "release_eps_parsed_usable": int(len(usable)), "release_vs_xbrl_match_rate": float(match.mean()) if len(match) else None,
            "sue_available": int(sue.sue.notna().sum()) if len(sue) else 0}


def _final_acceptance(m: pd.DataFrame, led: pd.DataFrame, cfg: dict, ctx=None) -> pd.DataFrame:
    """acceptance_time per accession: SGML header when the download carried one, else the JSON rule (quality A when the
    rule's header checks pass, otherwise C), else filing date (quality B)."""
    hdr = led.set_index("accession")["header_acceptance_raw"].to_dict() if len(led) else {}
    # rule verification: every 8-K full-submission header vs the corrected JSON value (+ S04 sample)
    comp = []
    for r in m.itertuples(index=False):
        h = hdr.get(r.accession)
        if h and pd.notna(r.acceptance_json_et) and not r.json_time_is_placeholder:
            comp.append(abs((r.acceptance_json_et - header_to_et(h)).total_seconds()) < 1)
    s04 = []
    if ctx is not None:
        p = ctx.dep("S04") / "header_check.csv"
        if p.exists():
            hc = pd.read_csv(p)
            s04 = hc["match"].astype(bool).tolist() if len(hc) else []
    allc = comp + s04
    mismatch = 1 - float(np.mean(allc)) if allc else None
    ok = mismatch is None or mismatch <= float(cfg["timing"]["json_rule_max_mismatch"])
    rows = []
    for r in m.itertuples(index=False):
        t, q, src, _ = acceptance_from_sources(r.acceptance_raw, hdr.get(r.accession), r.filing_date,
                                               cfg["timing"]["date_only_assumed_time"], ok)
        rows.append((r.accession, r.cik, r.base_form, r.form, r.filing_date, r.report_date, t, q, src))
    out = pd.DataFrame(rows, columns=["accession", "cik", "form", "raw_form", "filing_date", "report_date", "acceptance_time",
                                      "ts_quality", "ts_source"])
    out.attrs["json_rule_mismatch"] = mismatch
    out.attrs["json_rule_n"] = len(allc)
    return out


# =====================================================================================================  S07
@stage("S07", "Events + clocks", "CPU", ["S06", "S03"], ["timing", "study", "targets.horizons", "contam_window_sessions",
                                                          "taxonomy_version"],
       outputs=["events.parquet", "usable_cutoff.json", "event_counts.csv", "filing_manifest_timed.parquet"],
       assertions=["four timestamps", "buckets", "earnings types", "overlaps", "event construction reads no price fields"],
       card="One row per event (periodic filing or same-day 8-K group) with acceptance / public / feature-ready / decision times.")
def s07(ctx):
    from .events.build import build_events, events_to_parquet_safe
    cfg = ctx.cfg
    m = C.manifest(ctx)
    led = C.read(ctx, "S05", "parse_ledger.parquet")
    acc = _final_acceptance(m, led, cfg, ctx)
    mm = m.drop(columns=[c for c in ("acceptance_time",) if c in m.columns]).merge(
        acc[["accession", "acceptance_time", "ts_quality", "ts_source"]], on="accession", how="left")
    mm["form"] = mm["base_form"]
    titles = led.set_index("accession")["header_item_titles"].to_dict() if len(led) else {}
    hitems = led.set_index("accession")["header_items"].to_dict() if len(led) else {}
    mm["item_titles"] = mm.accession.map(lambda a: [t for t in str(titles.get(a) or "").split(" || ") if t])
    mm["items_list"] = [it if len(it) else [x for x in str(hitems.get(a) or "").split(",") if x]
                        for it, a in zip(mm.items_list, mm.accession)]
    mm = mm[mm.download_task]
    rel = C.read(ctx, "S06", "eps_release.parquet")
    rel_info = rel[["accession", "is_results_release"]].assign(release_period_end=rel.period_end) if len(rel) else \
        pd.DataFrame(columns=["accession", "is_results_release", "release_period_end"])
    si = C.read(ctx, "S05", "sections_index.parquet")
    sav = si.groupby("accession")["section_id"].apply(list).rename("sections").reset_index() if len(si) else \
        pd.DataFrame(columns=["accession", "sections"])
    cal = C.calendar(ctx)
    ev = build_events(mm, cal, cfg, C.taxonomy(cfg), rel_info, sav)
    u = C.universe(ctx)
    ev = ev.merge(u[["cik", "ticker", "ff12_industry"]], on="cik", how="left")
    price_cols = [c for c in ev.columns if c in ("close", "open", "ret", "raw_ret")]
    assert not price_cols, "event construction must not read price fields"
    px = C.read(ctx, "S03", "prices.parquet", columns=["ticker", "date"])
    last_px = pd.to_datetime(px[px.ticker != "SPY"].date).max()
    last_doc = pd.to_datetime(mm[mm.in_study].filing_date).max()
    cutoff = min(last_px, last_doc, pd.Timestamp(cfg["study"]["end"]))
    partial = cutoff < pd.Timestamp(cfg["study"]["end"]) or pd.Timestamp.today() < pd.Timestamp(cfg["study"]["end"]) + pd.offsets.MonthEnd(0)
    ci = int(cal.sessions.searchsorted(cutoff, side="right") - 1)
    write_json(ctx.out / "usable_cutoff.json", {"usable_cutoff": str(cutoff.date()), "usable_cutoff_idx": ci,
                                                "last_price_date": str(last_px.date()), "last_document_date": str(last_doc.date()),
                                                "final_month_partial": bool(partial),
                                                "json_rule_mismatch": acc.attrs.get("json_rule_mismatch"),
                                                "json_rule_n": acc.attrs.get("json_rule_n"),
                                                "timing_policy": timing_policy_label(cfg)})
    ctx.write(events_to_parquet_safe(ev), "events.parquet")
    # SPEC.md §16 filing_manifest contract: acceptance time, timestamp quality and period of report are only known after the
    # SGML headers are read (S05, D-047), so the complete per-filing manifest is written here
    fm = mm.drop(columns=[c for c in ("items_list", "item_titles") if c in mm.columns]).merge(
        led[["accession", "url", "status", "checksum", "header_period", "header_acceptance_raw"]].rename(
            columns={"status": "parse_status_s05", "header_period": "period_of_report"}), on="accession", how="left",
        suffixes=("", "_s05"))
    fm["items_raw"] = mm["items_list"].map(lambda x: ",".join(map(str, x))).values
    fm["items_source"] = np.where(mm.accession.map(lambda a: bool(str(hitems.get(a) or ""))), "sgml_header_or_json", "json")
    fm["acceptance_raw"] = fm.get("header_acceptance_raw")
    for c in ("url", "checksum"):
        if f"{c}_s05" in fm:
            fm[c] = fm[f"{c}_s05"].where(fm[f"{c}_s05"].notna(), fm[c])
            fm = fm.drop(columns=[f"{c}_s05"])
    fm["parse_status"] = fm.pop("parse_status_s05")
    ctx.write(fm, "filing_manifest_timed.parquet")
    cnt = ev.groupby(["event_type", "session_bucket"]).size().unstack(fill_value=0)
    cnt.to_csv(ctx.out / "event_counts.csv")
    return {"events": int(len(ev)), "by_type": ev.event_type.value_counts().to_dict(),
            "earnings_types": ev.earnings_type.value_counts().to_dict(), "usable_cutoff": str(cutoff.date()),
            "json_rule_mismatch": acc.attrs.get("json_rule_mismatch")}


# =====================================================================================================  S08
@stage("S08", "Targets + labels", "CPU", ["S07"], ["targets", "labels", "timing"],
       outputs=["targets.parquet", "target_summary.csv"],
       assertions=["INTRA not T_info-eligible", "per-horizon censoring", "T_trade entry >= decision_time"],
       card="T_info (information response) and T_trade (tradable, primary) abnormal returns, labels, vol ratio, |AR|.")
def s08(ctx):
    from .targets.build import build_targets
    cfg = ctx.cfg
    ev = C.events(ctx)
    cal = C.calendar(ctx)
    panel, spy = C.panel_and_spy(ctx, cal)
    uc = C.usable_cutoff(ctx)
    tg = build_targets(ev, panel, spy, C.factors(ctx), cfg, int(uc["usable_cutoff_idx"]))
    # assertions
    intra = ev.loc[ev.session_bucket == "INTRA", "event_id"]
    ti = tg[(tg.target_def == "T_info") & tg.event_id.isin(intra)]
    assert ti.label3.isna().all(), "INTRA events must not carry a T_info label"
    e = ev.set_index("event_id")
    ent = e["ttrade_entry_idx"].astype(int)
    ok = ent < len(cal)
    open_t = pd.Series([cal.open_time(i) for i in ent[ok]], index=ent[ok].index)
    assert (open_t >= pd.to_datetime(e.loc[ok[ok].index, "decision_time"])).all(), "T_trade entry must be at/after decision"
    ctx.write(tg, "targets.parquet")
    sm = tg.groupby(["target_def", "horizon"]).agg(rows=("event_id", "size"), labelled=("label3", lambda s: int(s.notna().sum())),
                                                   censored=("censored", "sum"), up=("label3", lambda s: float((s == 2).mean())),
                                                   side=("label3", lambda s: float((s == 1).mean())),
                                                   down=("label3", lambda s: float((s == 0).mean())),
                                                   ar_mean=("ar_mm", "mean"), ar_std=("ar_mm", "std")).reset_index()
    sm.to_csv(ctx.out / "target_summary.csv", index=False)
    return {"rows": int(len(tg)), "primary_labelled": int(tg[(tg.target_def == cfg["targets"]["primary"]["target_def"])
                                                             & (tg.horizon == cfg["targets"]["primary"]["horizon"])].label3.notna().sum())}


# =====================================================================================================  S09
def _text_job(args):
    from .features.text import firm_text_features, load_lm, model_text
    cik, part, evs, lm_path, max_chars = args
    lm = load_lm(lm_path)
    if not Path(part).exists():
        return pd.DataFrame({"event_id": [e["event_id"] for e in evs]}), pd.DataFrame(columns=["event_id", "text"])
    s = pd.read_parquet(part, columns=["accession", "section_id", "text_raw", "text_masked"])
    raw, msk = {}, {}
    for r in s.itertuples(index=False):
        raw.setdefault(r.accession, {})[r.section_id] = r.text_raw
        msk.setdefault(r.accession, {})[r.section_id] = r.text_masked
    ev = pd.DataFrame(evs)
    f = firm_text_features(ev, raw, lm)
    txt = []
    for e in evs:
        secs = {}
        for a in e["accessions"]:
            for sid, t in (msk.get(a) or {}).items():
                secs[sid] = (secs[sid] + "\n\n" + t) if sid in secs else t
        txt.append((e["event_id"], model_text(secs, e["event_type"], max_chars)))
    return f, pd.DataFrame(txt, columns=["event_id", "text"])


def lm_path(ctx) -> Path:
    p = ctx.storage.inputs / "lexicons" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
    if not p.exists():
        import requests
        rost = C.roster(ctx.cfg)["lexical"]["loughran_mcdonald"]
        r = requests.get(rost["sheet_export_url"], timeout=120)
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
    pin = C.roster(ctx.cfg)["lexical"]["loughran_mcdonald"].get("sha256_verified_2026_10_23")
    got = file_sha256(p)
    if pin and got != pin and not ctx.services.get("fixture_root"):   # hard rule 12 (offline fixtures use a tiny list)
        raise RuntimeError(f"Loughran-McDonald dictionary sha256 {got} != pinned {pin} (configs/models.yaml); "
                           "verify the new file and update the pin as a logged decision")
    return p


@stage("S09", "Structured + lexical features", "CPU", ["S08"], ["text", "accounting", "labels"],
       outputs=["features.parquet", "feature_registry.csv", "event_text.parquet"],
       assertions=["availability <= decision_time for all"],
       card="Price, metadata, accounting (SUE, XBRL), lexical (LM), change and novelty features at decision time.")
def s09(ctx):
    from .features.accounting import pit_facts
    from .features.price import price_features
    from .features.registry import availability_violations, registry_frame, sue_rule_violations
    from .track_a.event_study import window_cars
    cfg = ctx.cfg
    ev = C.events(ctx)
    cal = C.calendar(ctx)
    panel, spy = C.panel_and_spy(ctx, cal)
    feats = [ev[["event_id"]].copy()]
    pf = price_features(ev, panel, spy)
    feats.append(pf)
    # ---- metadata
    md = pd.DataFrame({"event_id": ev.event_id})
    for t in ("periodic_10K", "periodic_10Q", "8K"):
        md[f"meta_type_{t}"] = (ev.event_type == t).astype(float)
    tax = C.taxonomy(cfg)
    cats = sorted(set(tax.modern.values()) | set(tax.legacy.values()) | {tax.catch_all})
    for c in cats:
        md[f"meta_cat_{c}"] = ev.categories.map(lambda x: float(c in x))
    md["meta_n_items"] = ev["items"].map(len).astype(float)
    md["meta_n_accessions"] = ev.accessions.map(len).astype(float)
    for sid in [s for v in cfg["text"]["sections"].values() for s in v]:
        md[f"meta_has_{sid}"] = ev.section_ids.map(lambda x: float(sid in x))
    for ind in sorted(ev.ff12_industry.fillna("Other").unique()):
        md[f"meta_ff12_{ind}"] = (ev.ff12_industry.fillna("Other") == ind).astype(float)
    md["meta_year"] = ev.year.astype(float)
    md["meta_month"] = pd.to_datetime(ev.decision_date).dt.month.astype(float)
    md["meta_dow"] = pd.to_datetime(ev.acceptance_time).dt.dayofweek.astype(float)
    for b in ("PRE", "INTRA", "POST", "NONTRADING"):
        md[f"meta_bucket_{b}"] = (ev.session_bucket == b).astype(float)
    for q in ("A", "B", "C"):
        md[f"meta_tsq_{q}"] = (ev.ts_quality == q).astype(float)
    for et in ("earnings_announcement_verified", "earnings_date_ambiguous", "periodic_filing_only"):
        md[f"meta_earn_{et}"] = (ev.earnings_type == et).astype(float)
    lag = (pd.to_datetime(ev.filing_date) - pd.to_datetime(ev.report_date.replace("", np.nan), errors="coerce")).dt.days
    md["meta_filing_lag_days"] = np.where(ev.event_type.str.startswith("periodic"), lag, np.nan)
    ev_sorted = ev.sort_values("decision_time")
    prev = ev_sorted.groupby(["cik", "event_type"])["decision_time"].shift(1)
    md["meta_days_since_prev_same"] = ((ev_sorted.decision_time - prev).dt.total_seconds() / 86400).reindex(ev.index)
    md["_avail_metadata"] = pd.to_datetime(ev.acceptance_time, utc=True)
    feats.append(md)
    # ---- accounting
    facts = C.read(ctx, "S06", "xbrl_facts.parquet")
    tags = [("Assets", "inst"), ("Liabilities", "inst"), ("StockholdersEquity", "inst"), ("NetIncomeLoss", "fy"),
            ("Revenues", "fy"), ("RevenueFromContractWithCustomerExcludingAssessedTax", "fy"), ("SalesRevenueNet", "fy"),
            ("EarningsPerShareDiluted", "q"), (cfg["accounting"]["shares_tag"], "inst")]
    if len(facts):
        pit = pit_facts(ev, facts, tags)
    else:
        pit = pd.DataFrame({"event_id": ev.event_id, "_avail_accounting": pd.NaT})
    ac = pd.DataFrame({"event_id": pit.event_id})
    g = lambda c: pit[c] if c in pit else np.nan  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        ac["acc_log_assets"] = np.log(g("Assets_inst"))
        ac["acc_leverage"] = g("Liabilities_inst") / g("Assets_inst")
        ac["acc_roa"] = g("NetIncomeLoss_fy") / g("Assets_inst")
        rev = pd.concat([pd.Series(g("Revenues_fy")), pd.Series(g("RevenueFromContractWithCustomerExcludingAssessedTax_fy")),
                         pd.Series(g("SalesRevenueNet_fy"))], axis=1).bfill(axis=1).iloc[:, 0] if len(pit) else np.nan
        ac["acc_log_revenue"] = np.log(rev)
        ac["acc_eps_q_latest"] = g("EarningsPerShareDiluted_q")
        sh = g(f"{cfg['accounting']['shares_tag']}_inst")
        cidx = cal.v_last_close_at_or_before(pd.DatetimeIndex(pd.to_datetime(ev.set_index("event_id").loc[pit.event_id, "decision_time"])).tz_convert("UTC").asi8)
        tick = ev.set_index("event_id").loc[pit.event_id, "ticker"].to_numpy()
        price = np.array([panel[t]["close_split_adj"][i] if panel.has(t) and 0 <= i < len(cal) else np.nan for t, i in zip(tick, cidx)])
        mcap = price * np.asarray(sh, float)
        ac["acc_log_mcap"] = np.log(mcap)
        ac["acc_book_to_market"] = np.asarray(g("StockholdersEquity_inst"), float) / mcap
    ac = ac.replace([np.inf, -np.inf], np.nan)
    ac["_avail_accounting"] = pit["_avail_accounting"].values
    sue = C.read(ctx, "S06", "sue_by_accession.parquet")
    srcs = C.read(ctx, "S06", "sue_sources.parquet")
    a2e = {a: e for e, accs in zip(ev.event_id, ev.accessions) for a in accs}
    if len(sue):
        sue["event_id"] = sue.accession.map(a2e)
        own = sue.dropna(subset=["event_id"]).sort_values("sue_reason").drop_duplicates("event_id")
        own = own[["event_id", "sue", "n_prior_diffs", "hist_max_public_time"]].rename(columns={"sue": "sue_event", "n_prior_diffs": "sue_n_diffs"})
        ac = ac.merge(own, on="event_id", how="left")
        hm = pd.to_datetime(ac.pop("hist_max_public_time"), utc=True)
        aa = pd.to_datetime(ac["_avail_accounting"], utc=True)
        ac["_avail_accounting"] = pd.concat([aa, hm], axis=1).max(axis=1)
        # last available SUE (from an earlier event) for every event
        evs = ev[["event_id", "cik", "decision_time"]].merge(own[["event_id", "sue_event"]], on="event_id", how="left")
        evs = evs.sort_values("decision_time")
        evs["sue_last"] = evs.groupby("cik")["sue_event"].transform(lambda s: s.shift(1).ffill())
        ac = ac.merge(evs[["event_id", "sue_last"]], on="event_id", how="left")
        sue_src = srcs.assign(event_id=srcs.accession.map(a2e)).dropna(subset=["event_id"])
    else:
        ac["sue_event"] = ac["sue_n_diffs"] = ac["sue_last"] = np.nan
        sue_src = pd.DataFrame(columns=["event_id", "source_accession", "source_public_time"])
    feats.append(ac)
    # ---- EAR of the most recent different-cluster verified release whose [-1,+1] window closed before decision
    rels = ev[ev.earnings_type == "earnings_announcement_verified"]
    ear = pd.DataFrame({"event_id": ev.event_id, "ear_last": np.nan, "ear_src_cluster": None, "_avail_ear": pd.NaT})
    if len(rels):
        uc = C.usable_cutoff(ctx)
        wc = window_cars(rels, panel, spy, None, cfg, int(uc["usable_cutoff_idx"]))
        rels = rels.merge(wc[["event_id", "car_m1_1"]], on="event_id", how="left")
        rels["win_close"] = [cal.close_time(min(i + 1, len(cal) - 1)) for i in rels.trackA_day0_idx]
        out = []
        for cik, g2 in ev.groupby("cik"):
            rr = rels[rels.cik == cik].sort_values("win_close")
            for r in g2.itertuples(index=False):
                c = rr[(rr.win_close < r.decision_time) & (rr.cluster_id != r.cluster_id)]
                if len(c):
                    x = c.iloc[-1]
                    out.append((r.event_id, x.car_m1_1, x.cluster_id, x.win_close))
        if out:
            e2 = pd.DataFrame(out, columns=["event_id", "ear_last", "ear_src_cluster", "_avail_ear"])
            ear = ear[["event_id"]].merge(e2, on="event_id", how="left")
    ear["_avail_ear"] = pd.to_datetime(ear["_avail_ear"], utc=True)
    feats.append(ear)
    # ---- lexical / change / novelty (per firm, parallel)
    lmp = str(lm_path(ctx))
    jobs = []
    sec_dir = ctx.dep("S05") / "sections"
    for cik, g3 in ev.groupby("cik"):
        evs = g3[["event_id", "event_type", "decision_time", "accessions", "public_available_time"]].to_dict("records")
        jobs.append((int(cik), str(sec_dir / f"{int(cik):010d}.parquet"), evs, lmp, int(cfg["text"]["model_text_chars"])))
    tf, tt = [], []
    nproc = 1 if ctx.services.get("fixture_root") else min(8, max(1, (os.cpu_count() or 2) - 1))
    if nproc == 1:
        res = [_text_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=nproc, mp_context=_mp_ctx()) as ex:
            res = list(ex.map(_text_job, jobs, chunksize=2))
    for f, t in res:
        tf.append(f)
        tt.append(t)
    txf = pd.concat(tf, ignore_index=True)
    txf["_avail_lexical"] = txf.event_id.map(ev.set_index("event_id")["public_available_time"]).pipe(pd.to_datetime, utc=True)
    feats.append(txf)
    F = feats[0]
    for f in feats[1:]:
        F = F.merge(f, on="event_id", how="left")
    # ---- registry + availability assertions (EAR source cluster check handled by the registry)
    dec = ev.set_index("event_id").loc[F.event_id, "decision_time"].reset_index(drop=True)
    viol = availability_violations(F, dec)
    assert len(viol) == 0, f"availability violations: {viol.head().to_dict('records')}"
    sv = sue_rule_violations(sue_src.rename(columns={"source_public_time": "source_public_time"}), ev)
    assert len(sv) == 0, f"SUE rule violations: {sv.head().to_dict('records')}"
    cols = [c for c in F.columns if c != "event_id" and not c.startswith("_") and c != "ear_src_cluster"]
    reg = registry_frame(cols)
    reg.to_csv(ctx.out / "feature_registry.csv", index=False)
    F["txt_has_text"] = F["txt_has_text"].fillna(False).astype(bool)
    ctx.write(F, "features.parquet")
    ctx.write(pd.concat(tt, ignore_index=True), "event_text.parquet")
    return {"features": len(cols), "events": int(len(F)), "families": reg.family.value_counts().to_dict(),
            "availability_violations": 0, "sue_rule_violations": 0}


# =====================================================================================================  S10
@stage("S10", "Splits + cohorts + leakage audit", "CPU", ["S09"], ["splits", "seed", "contam_window_sessions"],
       outputs=["splits.parquet", "blocks.csv", "fold_provenance.csv", "leakage_audit.csv", "planted_leaks.json"],
       assertions=["all assertions pass", "4 planted leaks caught"],
       card="Walk-forward folds (expanding), development holdout, locked test; purge + embargo; stress folds; leakage audit.")
def s10(ctx):
    from .rag.core import retrieval_guard
    from .splits.leakage import audit, planted_leaks
    from .splits.provenance import fold_labels, ledger_frame
    from .splits.walkforward import assign_roles, blocks_frame, make_blocks
    cfg = ctx.cfg
    ev = C.events(ctx)
    tg = C.read(ctx, "S08", "targets.parquet")
    F = C.read(ctx, "S09", "features.parquet")
    uc = C.usable_cutoff(ctx)
    blocks = make_blocks(cfg, pd.Timestamp(uc["usable_cutoff"]))
    H = int(cfg["splits"]["purge_horizon_sessions"])
    hz = max([h for h in cfg["targets"]["horizons"] if h <= H] or [max(cfg["targets"]["horizons"])])
    ex = tg[tg.target_def.isin(["T_info", "T_trade"]) & (tg.horizon == hz)].groupby("event_id")["exit_date"].max()
    sp = assign_roles(ev, blocks, cfg, ev.event_id.map(ex))
    ctx.write(sp, "splits.parquet")
    bf = blocks_frame(blocks)
    bf.to_csv(ctx.out / "blocks.csv", index=False)
    rost = C.roster(cfg)
    fl = fold_labels(rost, blocks)
    fl.to_csv(ctx.out / "fold_provenance.csv", index=False)
    ledger_frame(rost).to_csv(ctx.out / "model_provenance_ledger.csv", index=False)
    ledger_frame(rost).to_csv(ctx.storage.reports / "model_provenance_ledger.csv", index=False)
    sue_src = C.read(ctx, "S06", "sue_sources.parquet")
    a2e = {a: e for e, accs in zip(ev.event_id, ev.accessions) for a in accs}
    sue_src = sue_src.assign(event_id=sue_src.accession.map(a2e)).dropna(subset=["event_id"])
    pl = planted_leaks(F, ev, tg, sue_src, retrieval_guard)
    write_json(ctx.out / "planted_leaks.json", pl)
    si = C.read(ctx, "S05", "sections_index.parquet")
    extra = {"planted": pl, "retrieval_guard_ok": retrieval_guard(), "mask_firm_total": int(si.mask_firm.sum()) if len(si) else 0,
             "json_rule_mismatch": uc.get("json_rule_mismatch"), "ear_violations": 0, "sue_violations": 0}
    univ = C.universe(ctx)
    la = audit(ev, F, tg, sp, sue_src, univ, cfg, extra)
    la.to_csv(ctx.out / "leakage_audit.csv", index=False)
    la.to_csv(ctx.storage.reports / "leakage_audit.csv", index=False)
    fails = la[la.status == "fail"]
    assert all(p["caught"] for p in pl), f"planted leaks missed: {[p for p in pl if not p['caught']]}"
    assert fails.empty, f"leakage audit failures: {fails.to_dict('records')}"
    return {"folds": len(blocks["folds"]), "blocks": bf.assign(start=bf.start.astype(str), end_exclusive=bf.end_exclusive.astype(str)).to_dict("records"),
            "planted_leaks_caught": sum(p["caught"] for p in pl), "audit": la.status.value_counts().to_dict()}
