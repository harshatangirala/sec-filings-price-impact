"""Content for each flowchart step: what was done, why, key numbers, interactive evidence, report figures, guardrails, decisions.

Numbers are read from the snapshot (lib.data); the prose only describes what the pipeline does."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from . import charts as C
from . import data as D
from .flow import BY_ID, IDS
from .theme import PHASES, bullets_html, card, chip, esc, metric_cards, step_header, tint


def _plot(fig, key: str):
    if fig is None:
        st.info("This chart's input is not in the data snapshot.")
    else:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def _table(df: pd.DataFrame | None, key: str, height: int | None = None, **kw):
    if df is None or not len(df):
        st.info("Not available in this snapshot.")
    else:
        st.dataframe(df, width="stretch", hide_index=True, key=key, height=height or "auto", **kw)


# ============================================================================================ shared facts
def facts() -> dict:
    m = D.meta()
    e, mo = m.get("eda", {}), m.get("modelling", {})
    f = {"m": m, "e": e, "mo": mo, "cfg": m.get("config", {})}
    f["chance"] = D.chance_level()
    f["l6_lock"], f["l7_lock"], f["l1_lock"] = D.skill_row("locked", "L6"), D.skill_row("locked", "L7"), D.skill_row("locked", "L1")
    f["text_wf"], f["text_dev"], f["text_lock"] = (D.lift_row(p, "L7 - L6") for p in ("walk-forward", "development", "locked"))
    f["chg_lock"] = D.lift_row("locked", "L6 - L6_nochange")
    f["struct_lock"], f["struct_wf"], f["struct_dev"] = (D.lift_row(p, "L6 - L1") for p in ("locked", "walk-forward", "development"))
    f["sue"] = D.sue_test()
    f["g3"] = D.js("g3.json") or {}
    f["frozen"] = D.js("frozen_architecture.json") or {}
    lad = D.csv("ladder.csv")
    if lad is not None:
        s = lad[(lad.cohort == "common") & (lad.tuned == "tuned") & (lad.rung == "L6")]
        f["l6_wf_best"] = s.sort_values("mean_fold_score", ascending=False).iloc[0] if len(s) else None
    return f


def lift_text(r, nd=3) -> str:
    return "n/a" if r is None else f"{r.lift:+.{nd}f} {D.ci(r.ci_low, r.ci_high, nd)}"


def hypothesis_board(F: dict) -> pd.DataFrame:
    h4 = D.csv("h4.csv")
    dl = D.csv("dev_lifts.csv")
    stack = dl[dl.hypothesis == "H5"].iloc[0] if dl is not None and (dl.hypothesis == "H5").any() else None

    def v(r):
        return "not supported" if r is not None and r.ci_low <= 0 <= r.ci_high else ("supported" if r is not None else "n/a")
    rows = [
        ("H1", "Filing text adds out-of-time value beyond price, metadata, accounting", f"locked {lift_text(F['text_lock'])}", v(F["text_lock"])),
        ("H2", "Changes in filing language beat absolute sentiment", f"locked {lift_text(F['chg_lock'])}", v(F["chg_lock"])),
        ("H3", "Validated RAG extraction adds value beyond embeddings", "needs gold labels", "not testable"),
        ("H4", "Point-in-time (vintage) encoders differ from static ones",
         (f"{h4.diff_vintage_minus_static.min():+.3f} to {h4.diff_vintage_minus_static.max():+.3f}, all CIs include 0" if h4 is not None else "n/a"),
         "no difference" if h4 is not None and ((h4.ci_low <= 0) & (h4.ci_high >= 0)).all() else "see table"),
        ("H5", "A stacked ensemble beats the best single model", f"development {stack.value:+.3f} {D.ci(stack.ci_low, stack.ci_high)}" if stack is not None else "n/a",
         "not supported" if stack is not None and stack.ci_low <= 0 <= stack.ci_high else "n/a"),
        ("H6", "Richer RAG designs beat hybrid + rerank", "needs gold labels", "not testable"),
        ("H7", "Analogue retrieval or time-series models add value", "extension disabled", "not run"),
        ("H8", "Gains shrink out of sample (strict, costs, locked)", f"structured over price: walk-forward {lift_text(F['struct_wf'])}, locked {lift_text(F['struct_lock'])}",
         "gain persists; backtest shows no edge"),
    ]
    return pd.DataFrame(rows, columns=["hypothesis", "claim", "evidence", "verdict"])


VERDICT_COLORS = {"supported": "#22C55E", "not supported": "#EF4444", "no difference": "#EF4444", "not testable": "#94A3B8",
                  "not run": "#94A3B8", "gain persists; backtest shows no edge": "#F59E0B"}


def render_board(board: pd.DataFrame, cols: int = 4):
    rows = [board.iloc[i:i + cols] for i in range(0, len(board), cols)]
    for r in rows:
        cs = st.columns(cols)
        for c, (_, x) in zip(cs, r.iterrows()):
            col = VERDICT_COLORS.get(x.verdict, "#6366F1")
            c.markdown(f"<div class='verdict' style='background:{tint(col, .10)};border-left:6px solid {col}'>"
                       f"<div style='font-weight:800;font-size:1.05rem'>{esc(x.hypothesis)} "
                       f"<span class='chip' style='background:{col};color:white'>{esc(x.verdict)}</span></div>"
                       f"<div style='font-size:.88rem;margin:4px 0'>{esc(x.claim)}</div>"
                       f"<div class='small-muted'>{esc(x.evidence)}</div></div>", unsafe_allow_html=True)


# ============================================================================================ step specs
def spec(sid: str, F: dict) -> dict:
    e, mo, cfg = F["e"], F["mo"], F["cfg"]
    ev_types = e.get("events_by_type", {})
    S = {}

    # ------------------------------------------------------------------ 1 question
    S["question"] = dict(
        sub="Turn a vague idea ('do filings move prices?') into falsifiable, pre-registered tests that cannot be tuned after seeing the answer.",
        metrics=[("Hypotheses", "8", "H1-H8, frozen before the locked test"), ("Tracks", "3", "event study / prediction / content"),
                 ("Decisions logged", D.n(len(D.csv("decisions.csv")) if D.csv("decisions.csv") is not None else None), "every deviation documented"),
                 ("Primary target", "T_trade h=1", "next tradable session, 3 classes")],
        what=["Three separate tracks: <b>A</b> event study (association), <b>B</b> prospective prediction from information available at decision time, "
              "<b>C</b> content validity of extracted text against human labels (never against prices).",
              "Primary prediction target: the abnormal return of the first tradable session after a filing becomes public, in three classes "
              "(Down / Sideways / Up).",
              "Eight hypotheses were written down and hashed (pre-registration) before any locked-test result existed."],
        why=["Finance data is noisy and easy to over-fit: pre-registering the tests, the metric (macro-F1) and the decision gates stops the "
             "analysis from drifting towards whatever looks good.",
             "Separating association (Track A) from prediction (Track B) avoids the classic error of reading an event-study effect as a tradable signal."],
        evidence=lambda: (st.markdown("#### Hypothesis scoreboard (computed from the run)"), render_board(hypothesis_board(F))),
        figures=["A01"], decisions=["D-056", "D-057", "D-058", "D-074", "D-076"],
        guards=["Association is not causation; prediction is not profitability.", "Results are validation-selected, never claimed as a global optimum.",
                "The locked test can be scored once; a second scoring is relabelled as development data."])

    # ------------------------------------------------------------------ 2 collect
    def ev_collect():
        c1, c2 = st.columns([1.1, 1])
        with c1:
            _plot(C.filings_by_year(), "fby")
        with c2:
            _plot(C.sources_treemap(F["m"]), "tree")
        ds = (F["m"].get("data_sources") or {})
        src = pd.DataFrame([
            ("SEC EDGAR", "10-K, 10-Q, 8-K + EX-99 exhibits; acceptance times from the SGML header", f"{D.n(e.get('filings'))} filings", "public, <= 6 requests/s, User-Agent from env"),
            ("EDGAR XBRL company facts", "accounting values as first reported", f"{D.n(e.get('xbrl_facts'))} facts, {e.get('xbrl_tags')} tags", "public"),
            ("Earnings-release exhibits", "release EPS for the earnings surprise (SUE)", f"{D.n(e.get('results_releases'))} results releases", "parsed from EDGAR"),
            ("Hugging Face SEC corpus", "firm overlap + parser QA only (never study text)", str(ds.get("hf_dataset", "")), "public dataset, pinned revision"),
            ("Daily prices (OHLCV)", "returns, volatility, benchmark, backtest", f"{D.n(e.get('price_bars'))} bars, {e.get('price_first')} to {e.get('price_last')}", "local files + Yahoo Finance"),
            ("Fama-French / Carhart factors", "Carhart alpha, 12-industry map", "monthly/daily factor files", "Kenneth French data library"),
            ("S&P 500 membership history", "universe and index flag", str(ds.get("sp500_membership", "")), "GitHub list, pinned commit"),
            ("Loughran-McDonald dictionary", "finance tone words", "1993-2025 master dictionary", "academic use, sha256 pinned")],
            columns=["source", "used for", "volume in this run", "access"])
        _table(src, "src")
    S["collect"] = dict(
        sub="Gather every input a trader could have had at the time, from free and reproducible sources, with pinned versions.",
        metrics=[("SEC filings", D.n(e.get("filings")), "10-K, 10-Q, 8-K for 150 firms"), ("XBRL facts", D.n(e.get("xbrl_facts")), f"{e.get('xbrl_tags')} accounting tags"),
                 ("Results releases", D.n(e.get("results_releases")), "parsed EX-99 exhibits"), ("Daily price bars", D.n(e.get("price_bars")),
                 f"{(e.get('price_first') or '')[:4]}-{(e.get('price_last') or '')[:4]}")],
        what=["Built the filing list per firm from the SEC submissions index and downloaded every 10-K, 10-Q and 8-K (with exhibits) from EDGAR.",
              "Read the exact acceptance time from each filing's SGML header (the SEC JSON times follow two different conventions, decision D-047).",
              "Loaded XBRL company facts, daily prices (local files spliced with Yahoo Finance), Fama-French factors and the S&P 500 membership history."],
        why=["Point-in-time research needs the exact moment information became public, so timestamps come from the filing header, not from a later database.",
             "Every source is free, pinned to a version and re-downloadable, so the study can be reproduced."],
        evidence=ev_collect, figures=["A02", "F01", "F02"], decisions=["D-003", "D-047", "D-068"],
        guards=["SEC fair-access policy: identifying User-Agent from the environment, at most 6 requests per second, back-off on errors.",
                "No news or social-media text anywhere (hard rule 1): SEC filings only.",
                "The Hugging Face corpus is used only to choose firms and check the parser, never as study text."])

    # ------------------------------------------------------------------ 3 EDA
    def ev_eda():
        t1, t2, t3, t4 = st.tabs(["Firms", "Events over time", "8-K content", "Timing"])
        with t1:
            c1, c2 = st.columns([1, 1.25])
            with c1:
                _plot(C.sectors_bar(), "sect")
            with c2:
                u = D.csv("universe.csv")
                q = st.text_input("Search a ticker or company", "", key="tick_q")
                if u is not None:
                    v = u[["ticker", "sec_name", "ff12_industry", "sic_desc", "price_start", "n_bars"]]
                    if q:
                        v = v[v.ticker.str.contains(q.upper(), na=False) | v.sec_name.str.upper().str.contains(q.upper(), na=False)]
                    _table(v.rename(columns={"sec_name": "company", "ff12_industry": "industry", "sic_desc": "SIC description",
                                             "price_start": "prices from", "n_bars": "daily bars"}), "unitab", height=340)
            _plot(C.events_per_firm(), "epf")
        with t2:
            _plot(C.events_by_year(), "eby")
            _plot(C.firms_per_year(), "fpy")
        with t3:
            k = st.slider("How many categories", 5, 25, 15, key="cat_k")
            _plot(C.categories_8k(k), "cat8k")
        with t4:
            c1, c2 = st.columns(2)
            with c1:
                _plot(C.session_donut(), "sess")
            with c2:
                _plot(C.ts_quality_by_year(), "tsq")
    tq = e.get("ts_quality", {})
    S["eda"] = dict(
        sub="Know the data before modelling it: who is in the sample, what they file, and when.",
        metrics=[("Firms", D.n(e.get("firms")), f"of {e.get('eligible_firms')} eligible ({e.get('candidate_firms')} candidates)"),
                 ("Events", D.n(e.get("events")), f"8-K {D.n(ev_types.get('8K'))}, 10-Q {D.n(ev_types.get('periodic_10Q'))}, 10-K {D.n(ev_types.get('periodic_10K'))}"),
                 ("Exact timestamps", D.pct(tq.get("A", 0) / max(1, sum(tq.values()))), "quality A from the SEC header"),
                 ("Clean events", D.pct(e.get("clean_share_h60")), "no other same-firm event within 60 sessions")],
        what=[f"Selected {e.get('firms')} firms that were S&P 500 members at some point in 2000-2026, seeded and stratified by Fama-French industry "
              f"from {e.get('eligible_firms')} eligible firms (prices available and at least one 10-K in the Hugging Face corpus).",
              f"Built {D.n(e.get('events'))} events from {e.get('event_first')} to {e.get('event_last')}; same-firm same-day 8-Ks are merged into one event.",
              "Profiled filing types, 8-K item categories, the time of day of filings, timestamp quality and the overlap between events."],
        why=["Most filings arrive after the close (POST) or before the open (PRE), which decides when a trade could happen.",
             "8-Ks dominate the event count, so the event mix (earnings vs other news) shapes what any model can learn."],
        evidence=ev_eda, figures=["F03", "F04", "F05", "F08", "R05"], decisions=["D-056", "D-057", "D-076"],
        guards=["Survivor-only universe: all 150 firms are listed today, which biases drift and backtest results (stated everywhere).",
                "Firms were selected by rule (seed + industry strata), never by later price behaviour (hard rule 9)."])

    # ------------------------------------------------------------------ 4 parse
    hq = D.csv("hf_parser_qa.csv")
    jac = float(hq[hq.hf_chars > 0].token_jaccard.median()) if hq is not None and len(hq) else None

    def ev_parse():
        c1, c2 = st.columns(2)
        with c1:
            _plot(C.coverage_by_form(), "cov")
        with c2:
            _plot(C.hf_qa_box(), "hfqa")
        _table(D.csv("section_found_counts.csv"), "sfc")
    S["parse"] = dict(
        sub="Turn raw HTML / SGML filings into clean text split into the item sections that carry the information.",
        metrics=[("Filings processed", D.n(e.get("filings")), "full submissions"), ("Parsed OK", D.pct((e.get("filings_parsed_ok") or 0) / max(1, e.get("filings") or 1)),
                 f"{D.n(e.get('filings_parsed_ok'))} filings"), ("Tables flattened", D.n(e.get("tables_flattened")), "kept as 'cell | cell' rows"),
                 ("Parser vs HF corpus", D.f3(jac), "median word overlap, 10-K sections")],
        what=["HTML and SGML converted to plain text; tables flattened so numeric parsers (release EPS) can read them; page numbers and table borders removed.",
              "Item sections located by item number plus title keywords; when the table of contents repeats a heading, the candidate with the longest span wins.",
              "Sections incorporated by reference are flagged and recovered from the EX-13 exhibit where possible."],
        why=["Only some sections carry new information (MD&A, risk factors, 8-K items, press releases); whole filings are mostly boilerplate.",
             "Checking our 10-K sections against an independent corpus measures parser quality instead of assuming it."],
        evidence=ev_parse, figures=["F01", "F02"], decisions=["D-047", "D-068"],
        guards=["Per-filing retry and a compressed on-disk cache; a parse ledger records the status of every filing.",
                "Regular expressions were tested against adversarial inputs after one caused catastrophic backtracking (D-068)."])

    # ------------------------------------------------------------------ 5 NLP
    tx = cfg.get("text") or {}

    def ev_nlp():
        c1, c2 = st.columns([1, 1.2])
        with c1:
            _plot(C.text_funnel(F["m"]), "fun")
        with c2:
            _plot(C.sections_bar(), "secbar")
        secs = D.csv("sections.csv")
        opts = list(secs.section_id) if secs is not None else []
        pick = st.multiselect("Compare section lengths", opts, default=[o for o in ("10K_7", "10Q_MDA", "8K_EX99", "8K_ITEMS") if o in opts],
                              format_func=lambda s: C.SECTION_NAMES.get(s, s), key="seclen")
        _plot(C.section_length_hist(pick), "seclenh")
        steps = pd.DataFrame([
            ("1 Clean", "HTML/SGML to text, tables flattened, page debris removed"),
            ("2 Section", "keep 10-K Items 1/1A/7/7A, 10-Q MD&A / market risk / risk factors, 8-K items + EX-99"),
            ("3 Mask", "replace the firm's own names, former names and tickers (firm-memorisation control)"),
            ("4 Filter", f"drop numeric-heavy lines; cap sections at {D.n(tx.get('max_chars_per_section'))} characters, model text at {D.n(tx.get('model_text_chars'))}"),
            ("5 Chunk", f"{tx.get('chunk_words')}-word windows with {tx.get('chunk_overlap_words')}-word overlap, at most {tx.get('max_chunks_per_section')} per section"),
            ("6 Tokenise", f"each encoder's own tokenizer, {(cfg.get('models') or {}).get('max_seq_len')} tokens, length-sorted batches"),
            ("7 Pool", "chunk vectors -> mean/max per section -> chunk-weighted event vector; change = cosine to the previous same-type filing"),
            ("8 Classic NLP", "Loughran-McDonald tone, filing-change similarity (fixed hashing, no learned vocabulary), novelty; TF-IDF+SVD fitted per training fold")],
            columns=["step", "what happens"])
        _table(steps, "nlpsteps")
    S["nlp"] = dict(
        sub="Prepare text identically for every model so differences between models cannot come from different inputs.",
        metrics=[("Sections kept", D.n(e.get("sections")), "9 section types"), ("Firm-name masks", D.n(e.get("firm_name_masks")), f"+ {D.n(e.get('ticker_masks'))} ticker masks"),
                 ("Text chunks", D.n(e.get("chunks_est")), f"{tx.get('chunk_words')} words, {tx.get('chunk_overlap_words')} overlap"),
                 ("Tokens", D.n(e.get("tokens_est")), "estimated, all chunks")],
        what=["Masked each firm's own names and tickers so a model cannot simply memorise which company it is reading about.",
              "Dropped numeric-heavy lines from narrative text (the numbers are used separately as accounting features) and capped section length.",
              "Cut sections into overlapping 350-word chunks shared by every encoder and by the retriever.",
              "Built classic NLP features too: Loughran-McDonald tone, change versus the firm's previous filing, novelty, and TF-IDF + SVD."],
        why=["Memorising firms is a form of leakage: a model could learn 'AAPL tends to go up' instead of reading the text.",
             "Nothing in the text pipeline is fitted on future data: hashing replaces a learned vocabulary, and TF-IDF + SVD is refitted inside every training fold."],
        evidence=ev_nlp, figures=["A02"], decisions=["D-068"],
        guards=["Same chunks for all encoders (fair comparison).", "Statistics such as IDF are fitted on training rows of each fold only (leakage item 5)."])

    # ------------------------------------------------------------------ 6 events
    lab = D.csv("labels_by_year.csv")
    shares = lab.groupby("label").events.sum() / lab.events.sum() if lab is not None else pd.Series(dtype=float)
    tm = cfg.get("timing") or {}

    def ev_events():
        _plot(C.clock_diagram(F["m"]), "clock")
        c1, c2 = st.columns(2)
        with c1:
            _plot(C.labels_by_year(), "lby")
        with c2:
            _plot(C.ar_hist(), "arh")
        ts = D.csv("target_summary.csv")
        if ts is not None:
            _table(ts[ts.target_def.isin(["T_trade", "T_info"])].round(4), "tsum")
    S["events"] = dict(
        sub="Decide exactly when each filing became usable, what a trader could have done, and what counts as Up, Sideways or Down.",
        metrics=[("Public lag (assumed)", f"+{tm.get('public_lag_minutes')} min", "after EDGAR acceptance"), ("Decision latency", f"+{tm.get('decision_latency_minutes')} min", "time to act"),
                 ("Label threshold", f"k = {(cfg.get('labels') or {}).get('k')}", "x the stock's own sigma x sqrt(h)"),
                 ("Class balance", " / ".join(f"{k[0]} {v:.0%}" for k, v in shares.items()) if len(shares) else "n/a", "Down / Sideways / Up")],
        what=["Four timestamps per event: acceptance, assumed public time, feature-ready time and decision time.",
              "T_trade (primary target): enter at the first market open at or after the decision time, exit at the close of the h-th session (for h = 1 the same session: an open-to-close return); the abnormal "
              "return subtracts beta x SPY on exactly the same interval.",
              "Labels: Up or Down when the abnormal return is beyond 0.5 x the stock's own volatility x sqrt(horizon), otherwise Sideways.",
              "Overlap flags mark events with another same-firm event nearby; a 'clean' cohort excludes them."],
        why=["Using the close-to-close return around a filing would include price moves that happened before anyone could act.",
             "A stock-specific threshold makes the three classes comparable across quiet and volatile stocks."],
        evidence=ev_events, figures=["F06", "F07", "F09", "F10"], decisions=["D-003", "D-047", "D-071", "D-073"],
        guards=["Same-day 8-K groups are clocked at their LAST acceptance; a full-scale run caught a look-ahead leak here (D-071).",
                "Stock and benchmark returns always cover identical intervals; INTRA events are excluded from the information-response target.",
                "Public availability is an assumption (acceptance + 5 min), stated as such."])

    # ------------------------------------------------------------------ 7 features
    fam = e.get("feature_families") or {}

    def ev_features():
        c1, c2 = st.columns([1.15, 1])
        with c1:
            _plot(C.feature_sunburst(), "sun")
        with c2:
            fr = D.csv("feature_registry.csv")
            if fr is not None:
                g = fr.groupby("family").agg(features=("name", "size"), examples=("name", lambda s: ", ".join(list(s)[:4])),
                                             availability=("availability_rule", "first")).reset_index()
                _table(g, "famtab")
            _plot(C.sue_coverage(), "suecov")
        fr = D.csv("feature_registry.csv")
        if fr is not None:
            with st.expander("Browse all registered features"):
                famsel = st.multiselect("Families", sorted(fr.family.unique()), default=sorted(fr.family.unique()), key="famsel")
                _table(fr[fr.family.isin(famsel)], "fr_all", height=320)
    S["features"] = dict(
        sub="Describe each event with information that was public at decision time, and prove it with an assertion on every row.",
        metrics=[("Registered features", D.n(sum(fam.values()) if fam else None), f"{len(fam)} families"),
                 ("Availability violations", D.n(e.get("availability_violations")), "latest input after decision time"),
                 ("SUE-rule violations", D.n(e.get("sue_rule_violations")), "surprise from a later filing"),
                 ("SUE available", D.n(e.get("sue_available")), f"of {D.n(e.get('sue_filings'))} filings")],
        what=["Price (returns, momentum, volatility, liquidity, beta, market state), metadata (event type, 8-K categories, sections present, industry, calendar, "
              "timing), accounting (assets, leverage, ROA, revenue, EPS, market cap, book-to-market, SUE), lexical tone, filing change and novelty.",
              "Every family carries an availability timestamp; the rule 'latest input <= decision time' is asserted for all events.",
              "Earnings surprise (SUE) uses only the event's own filing or earlier public filings, with values as first reported."],
        why=["Look-ahead bias is the most common reason backtests look better than reality; asserting availability per row makes it impossible to miss.",
             "As-first-reported accounting values avoid using restated numbers that were not known at the time (fix D-077)."],
        evidence=ev_features, figures=["F15"], decisions=["D-072", "D-077"],
        guards=["The EAR (earnings-announcement return) is never a feature for its own event (hard rule 7).",
                "A cross-run consistency check found and fixed a stale-accounting bug before the final run (D-077)."])

    # ------------------------------------------------------------------ 8 splits
    ac = e.get("audit_counts") or {}

    def ev_splits():
        _plot(C.split_timeline(), "split")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            la = D.csv("leakage_audit.csv")
            if la is not None:
                stat = st.multiselect("Show audit items with status", sorted(la.status.unique()), default=sorted(la.status.unique()), key="lastat")
                _table(la[la.status.isin(stat)], "la", height=360)
        with c2:
            pl = D.js("planted_leaks.json") or []
            st.markdown("**Planted leaks (the audit must catch all four)**")
            for p in pl:
                ok = p.get("caught")
                st.markdown(chip(("CAUGHT  " if ok else "MISSED  ") + p.get("leak", "").replace("_", " "), "#22C55E" if ok else "#EF4444", solid=True),
                            unsafe_allow_html=True)
            _plot(C.provenance_heatmap(), "prov")
    sp = cfg.get("splits") or {}
    S["splits"] = dict(
        sub="Evaluate the way the model would really be used: train on the past, test on the future, and prove nothing leaked.",
        metrics=[("Walk-forward folds", "6", "2008-2019, two years each"), ("Embargo", f"{sp.get('embargo_days')} days", "plus purge of overlapping windows"),
                 ("Audit items passed", D.n(sum(v for k, v in ac.items() if str(k).startswith("pass"))), f"of {sum(ac.values()) if ac else 'n/a'}; others informational"),
                 ("Planted leaks caught", f"{e.get('planted_leaks_caught')}/4", "self-test of the audit")],
        what=["Expanding walk-forward validation (train on everything before each 2-year block), then a 2020-2022 development holdout, then a 2023-2026 "
              "locked test scored exactly once after the final model was frozen.",
              "Purge: training events whose outcome window reaches into the next block are removed; a 90-day embargo separates blocks.",
              "A 24-item leakage checklist with assertions, plus four deliberately planted leaks that the audit must catch."],
        why=["Random train/test splits leak the future in time-series data; the report's random-split demo (figure F38) shows how they flatter a model.",
             "An audit that has never caught a planted leak cannot be trusted to catch a real one."],
        evidence=ev_splits, figures=["F17", "F37", "F38", "F39"], decisions=["D-073"],
        guards=["Firm- and sector-held-out stress folds are built but were not scored in this run (stated in the report).",
                "Every model is labelled strict or retrospective per fold depending on its training cutoff."])

    # ------------------------------------------------------------------ 9 baselines
    lad = D.csv("ladder.csv")
    b6 = F.get("l6_wf_best")

    def ev_baselines():
        heads = sorted(lad[lad.cohort == "common"]["head"].dropna().unique()) if lad is not None else []
        heads = [h for h in heads if h != "prior"]
        sel = st.pills("Model heads", heads, selection_mode="multi", default=heads, key="heads")
        _plot(C.ladder_chart("common", list(sel or heads) + ["prior"]), "ladder")
        t1, t2, t3, t4 = st.tabs(["Fold stability", "Tuning", "Fit & learning", "Sanity checks"])
        with t1:
            if lad is not None:
                runs = sorted(lad[(lad.cohort == "common") & (lad.tuned == "tuned")].run_id)
                pick = st.multiselect("Runs", runs, default=[r for r in runs if "|L6|" in r or "|L1|enet" in r][:4], key="foldruns")
                _plot(C.ladder_folds(pick), "folds")
        with t2:
            _plot(C.tuning_dumbbell(), "tune")
            tri = D.csv("optuna_trials.csv")
            if tri is not None:
                params = [c for c in tri.columns if c.startswith("params_") and tri[c].notna().sum() > 5]
                p = st.selectbox("Hyperparameter", params, format_func=lambda s: s.replace("params_", ""), key="optp")
                _plot(C.optuna_scatter(p), "opt")
        with t3:
            c1, c2 = st.columns(2)
            with c1:
                _plot(C.fit_scatter(), "fit")
            with c2:
                _plot(C.learning_curve(), "lc")
        with t4:
            _plot(C.permutation_hist(), "perm")
            adv = D.csv("adversarial.csv")
            if adv is not None:
                st.caption("Adversarial validation: can a classifier tell training rows from validation rows? AUC near 1 means the periods differ "
                           "(covariate shift); it is not by itself evidence of leakage.")
                _table(adv[["fold", "auc", "n_train", "n_val"]].round(3), "adv")
    S["baselines"] = dict(
        sub="Climb an ablation ladder: add one kind of information at a time and measure what each adds, on identical events.",
        metrics=[("Best strict model", f"L6 / {b6['head']}" if b6 is not None else "n/a", "all structured features"),
                 ("Its macro-F1", D.f3(b6.macro_f1) if b6 is not None else "n/a", f"CI {D.ci(b6.ci_low, b6.ci_high, signed=False)}" if b6 is not None else ""),
                 ("Real chance level", D.f3(F["chance"]), f"majority baseline {D.f3(mo.get('majority_baseline_f1'))}"),
                 ("Configurations tried", D.n(e.get("configs_tried_s11")), "counted for multiple-testing corrections")],
        what=["Rungs: L0 class prior, L1 price, L2 metadata, L3 accounting, L4 text-derived features, L6 = all structured features together.",
              "Three model heads per rung: LightGBM and CatBoost (gradient-boosted trees) and an elastic-net logistic regression.",
              "Hyperparameters tuned with Optuna (TPE sampler, median pruner) on the last three walk-forward folds, then refitted on every fold.",
              "Class-weighted loss, probabilities corrected back to true frequencies, decision rule argmax p(c)/prior(c)."],
        why=["Macro-F1 weights Down, Sideways and Up equally, so a model cannot look good by always predicting the biggest class.",
             "Trees capture interactions in tabular data; the elastic net is a convex reference whose optimum is certified."],
        evidence=ev_baselines, figures=["F18", "F20", "F21", "F22", "F23", "FX1"], decisions=["D-021"],
        guards=["All preprocessing (TF-IDF, scalers, imputers, calibration) is fitted on training rows of each fold only.",
                "Small search budgets: several rungs share the same tuned settings (Optuna's first 10 candidates are random with a fixed seed).",
                "A permutation test and a random-split demo show the pipeline is not scoring by accident."])

    # ------------------------------------------------------------------ 10 embeddings
    pf, s14 = D.js("preflight.json") or {}, D.js("s14.json") or {}

    def ev_embed():
        c1, c2 = st.columns([1, 1.2])
        with c1:
            _plot(C.encoder_time(s14), "enct")
        with c2:
            led = D.csv("model_ledger.csv")
            if led is not None:
                used = set(((cfg.get("speed") or {}).get("fast") or {}).get("encoders") or [])
                led = led.assign(**{"used in this run": led.key.map(lambda k: "yes" if k in used else ("no (fast mode)" if k in ("sec_bert", "finbert_tone") else "-"))})
                _table(led[["key", "model_id", "stated_cutoff", "license", "used in this run"]], "ledger")
        _plot(C.provenance_heatmap(), "prov2")
    S["embed"] = dict(
        sub="Turn every chunk of filing text into vectors with two encoders, one of which never saw the future.",
        metrics=[("Chunks embedded", D.n(pf.get("est_chunks")), "per encoder (estimate)"), ("Encoders", "2", "ChronoBERT vintages + BGE-base"),
                 ("GPU time (S14)", f"{(s14.get('wall_s') or 0) / 3600:.1f} h", "Colab GPU"), ("Benchmark", f"{pf.get('bench_chunks_per_s', 0):.0f} chunks/s",
                 f"projected {pf.get('projected_gpu_hours_encoders', 0):.1f} GPU-h")],
        what=["ChronoBERT: 26 yearly vintages; an event in year Y is embedded with the vintage trained only on text up to the end of year Y-1.",
              "BGE-base: a strong general sentence encoder that also powers the dense retriever in the RAG system.",
              "A GPU preflight (S13) benchmarked throughput and refused to start if the projected GPU hours exceeded the budget."],
        why=["A model trained on text written after the event can 'know' the outcome; ChronoBERT's vintages make a strict, look-ahead-free comparison possible (H4).",
             "Fast mode (decision D-074) dropped SEC-BERT and FinBERT to fit the Colab time budget: a time decision, not a measured result."],
        evidence=ev_embed, figures=["A03"], decisions=["D-074", "D-075"],
        guards=["Encoders with unknown training cutoffs (BGE) appear only on the retrospective leaderboard.",
                "The benchmark warms up before timing (fix D-075, found on the first Colab attempt)."])

    # ------------------------------------------------------------------ 11 text models
    g3 = F["g3"]
    lift = (g3.get("lift") or {})
    h4 = D.csv("h4.csv")
    fts = D.js("finetune_status.json") or {}

    def ev_text():
        _plot(C.emb_vs_l6(), "embl6")
        c1, c2 = st.columns(2)
        with c1:
            if lift:
                _plot(C.forest([("best L7 - L6 (walk-forward)", lift["value"], lift["ci_low"], lift["ci_high"], PHASES["Text & GPU"])],
                               "Gate G3: text lift needed for fine-tuning", threshold=0.01, h=220), "g3")
        with c2:
            if h4 is not None:
                rows = [(f"{r.rung}/{r['head']}: ChronoBERT - {r.static}", r.diff_vintage_minus_static, r.ci_low, r.ci_high, PHASES["Text & GPU"])
                        for _, r in h4.iterrows()]
                _plot(C.forest(rows, "H4: point-in-time vintage vs static encoder", h=260), "h4")
        _table(D.csv("emb_ladder_summary.csv"), "embtab")
    S["textmodels"] = dict(
        sub="Ask the central question: does adding filing text to everything else improve out-of-time predictions?",
        metrics=[("G3 text lift", f"{lift.get('value', float('nan')):+.4f}", f"CI {D.ci(lift.get('ci_low', 0), lift.get('ci_high', 0), 4)}"),
                 ("Gate threshold", "+0.01", "and CI above 0"), ("G3 result", "passed" if g3.get("passed") else "not passed", f"p = {lift.get('p_boot', float('nan')):.2f}"),
                 ("Fine-tuning", "ran" if fts.get("ran") else "not run", "only allowed if G3 passes")],
        what=["L5 = embeddings alone; L7 = everything in L6 plus embedding components; both tuned with Optuna for each encoder.",
              "Gate G3 compares the best L7 with L6 on identical events with a paired month-block bootstrap.",
              "H4 compares the point-in-time ChronoBERT with the static BGE encoder on the same rows and heads.",
              "Fine-tuning the top-2 encoders was pre-registered to run only if G3 passed; it did not, so no encoder was fine-tuned."],
        why=["A gate stops expensive fine-tuning (and the many extra configurations it brings) when there is no evidence of signal to amplify.",
             "Paired comparisons on identical events remove noise from different samples."],
        evidence=ev_text, figures=["F19", "F20"], decisions=["D-074"],
        guards=["Text conclusions refer to ChronoBERT and BGE only (fast mode).", "Every comparison uses the common cohort of events eligible for all models."])

    # ------------------------------------------------------------------ 12 RAG
    rag = D.js("rag.json") or {}
    s16 = rag.get("s16") or {}
    qb = (rag.get("question_bank") or {}).get("fields") or {}
    rc = rag.get("rag_config") or {}

    def ev_rag():
        fi = D.figure_index().get("A04")
        if fi and fi.get("local"):
            st.image(fi["local"], width="stretch")
        c1, c2 = st.columns([1, 1.1])
        with c1:
            _table(pd.DataFrame([(k, v.get("question"), ", ".join(v.get("directions") or [])) for k, v in qb.items()],
                                columns=["field", "question asked", "allowed answers"]), "qb", height=420)
        with c2:
            _plot(C.rag_prevalence(), "prev")
        ex = rag.get("examples") or []
        if ex:
            st.markdown("**Browse real model output** (unvalidated: there are no gold labels to score it)")
            i = st.selectbox("Example filing", range(len(ex)), format_func=lambda k: f"example {k + 1}", key="ragex")
            rows = [(f, o.get("direction"), o.get("certainty"), (o.get("evidence") or [{}])[0].get("quote", "")[:260])
                    for f, o in ex[i].items() if isinstance(o, dict) and o.get("present")]
            _table(pd.DataFrame(rows, columns=["field", "answer", "certainty", "evidence quote"]), "ragrows")
    S["rag"] = dict(
        sub="Extract structured, evidence-cited events from filings with a local LLM, guarded so it cannot invent facts.",
        metrics=[("Question fields", D.n(len(qb)), "guidance, impairment, M&A, ..."), ("Filings pre-filled", D.n(s16.get("prefilled_events")), "development / validation only"),
                 ("Seconds per filing", f"{s16.get('seconds_per_filing', 0):.0f} s", "Qwen2.5-7B on GPU"),
                 ("Gate G3b", "not passed", "no human gold labels yet")],
        what=[f"Retrieval: BM25 (k1 {rc.get('bm25_k1')}, b {rc.get('bm25_b')}) + dense BGE, fused with reciprocal rank (k {rc.get('rrf_k')}), "
              f"cross-encoder rerank of the top {rc.get('rerank_top_n')}, top {rc.get('top_k')} passages to the generator.",
              "Generation: Qwen2.5-7B-Instruct, run locally with JSON-schema-constrained decoding, answers 12 questions with quotes and chunk ids.",
              "Guardrails: schema validation + one retry; a number must appear in its cited evidence; every quote must be a substring of its chunk, "
              "else the field is nulled; citation offsets verified; responses cached by model revision, prompt and passages.",
              "The output pre-fills annotation templates for humans; scoring it needs those human gold labels."],
        why=["Free-text LLM answers can hallucinate; constrained decoding plus verbatim-quote checks make every extracted fact auditable.",
             "A historical-evidence path only ever sees the firm's earlier filings, so retrieval cannot leak the future (self-test + planted leak)."],
        evidence=ev_rag, figures=["A04", "R06"], decisions=[],
        guards=["Gold-test filings were never sent to the model.", "Without gold labels: no retrieval tournament, no extraction F1, no student model, no L8 (H3, H6 untested)."])

    # ------------------------------------------------------------------ 13 selection
    fz = F["frozen"]
    pm = fz.get("primary_model") or {}
    l6l, l7l = F["l6_lock"], F["l7_lock"]

    def ev_sel():
        t1, t2, t3, t4 = st.tabs(["Scores vs chance", "Move vs direction", "Lifts by period", "Details"])
        with t1:
            _plot(C.chance_chart(), "chance")
            _plot(C.finalists_chart(), "final")
        with t2:
            _plot(C.skill_chart(), "skill")
            mdl = st.segmented_control("Confusion matrix for", ["L6", "L7", "L1", "L6_nochange"], default="L6", key="cmm")
            _plot(C.confusion(mdl or "L6"), "cm")
        with t3:
            per = st.pills("Periods", ["walk-forward", "development", "locked"], selection_mode="multi",
                           default=["walk-forward", "development", "locked"], key="lper")
            _plot(C.lifts_forest(list(per) if per else None), "lifts")
        with t4:
            rep = D.csv("reproducibility.csv")
            st.markdown("**Why L6 and L7 are a tie.** The rule froze the model with the highest walk-forward score: "
                        f"{pm.get('name')} ({pm.get('head')}/{pm.get('enc')}) at {pm.get('wf_score', float('nan')):.4f}. Scored inside the same stage, "
                        f"L6 reaches {mo.get('s15_l6_best_mean_fold', float('nan')):.4f}, and re-scoring the same model on two machines changes its "
                        "predictions (table below), so the gap is below the noise.")
            _table(rep.round(4) if rep is not None else None, "rep")
            mp = D.js("mcs_pbo.json") or {}
            st.markdown(f"**Model Confidence Set** (development log-loss) keeps: {', '.join((mp.get('mcs') or {}).get('included') or [])}. "
                        f"**PBO** = {(mp.get('pbo') or {}).get('pbo')} over {(mp.get('pbo') or {}).get('n_configs')} configurations (diagnostic only).")
            lk = D.js("lockbox_log.json") or {}
            st.markdown(f"**Lockbox log:** scored at {lk.get('scored_at', 'n/a')[:19]} UTC, reused = {lk.get('reused')}, previous = {lk.get('previous')}.")
    S["selection"] = dict(
        sub="Choose the final model on data it has never seen, freeze it, then score the locked 2023-2026 period exactly once.",
        metrics=[("Locked L6 macro-F1", D.f3(l6l.macro_f1) if l6l is not None else "n/a", f"chance {D.f3(F['chance'])}, n = {D.n(l6l.n) if l6l is not None else 'n/a'}"),
                 ("Text lift (locked)", f"{F['text_lock'].lift:+.3f}" if F["text_lock"] is not None else "n/a", D.ci(F['text_lock'].ci_low, F['text_lock'].ci_high) if F["text_lock"] is not None else ""),
                 ("Move AUC", D.f3(l6l.move_auc) if l6l is not None else "n/a", "will the stock move? (L6, locked)"),
                 ("Direction AUC", D.f3(l6l.direction_auc) if l6l is not None else "n/a", f"which way? CI {D.ci(l6l.direction_auc_lo, l6l.direction_auc_hi, signed=False)}" if l6l is not None else "")],
        what=["Finalists (L6, L6 without change features, L1, L7) retrained on all pre-2020 data and scored on the 2020-2022 development holdout.",
              "Gate G4: the chosen architecture, hyperparameters and thresholds were frozen and hashed; only then was the locked test unlocked and scored once.",
              "Beyond macro-F1, the predictions were split into 'will it move' and 'which way' skill, and compared with the real chance level."],
        why=["Scoring the test period once, after a hash-locked freeze, is what makes the final number an honest out-of-sample estimate.",
             "Macro-F1 above chance can come from predicting volatility rather than direction; separating the two shows what the model really knows."],
        evidence=ev_sel, figures=["F24", "F25", "F26", "F27", "F31", "F40", "R01", "R02", "R03"], decisions=["D-078"],
        guards=["The locked test is now spent: any re-run can only treat 2023-2026 as development data.",
                "L7 (frozen by the rule) is on the retrospective leaderboard; L6 is the best strict model and the two are statistically tied."])

    # ------------------------------------------------------------------ 14 event study
    ta = D.csv("track_a_groups.csv")
    k8 = ta[(ta.group_by == "event_type") & (ta.group == "8K") & (ta.event_set == "all") & (ta.measure == "car_m1_1")] if ta is not None else None
    k8 = k8.iloc[0] if k8 is not None and len(k8) else None
    sue = F["sue"]

    def ev_es():
        c1, c2 = st.columns([1, 2.2])
        with c1:
            groups = sorted(D.csv("car_paths.csv").group.unique()) if D.csv("car_paths.csv") is not None else []
            gsel = st.multiselect("Groups", groups, default=[g for g in ("8K", "periodic_10Q", "periodic_10K") if g in groups], key="cargr")
            es = st.radio("Events", ["all", "clean"], horizontal=True, key="cares")
        with c2:
            _plot(C.car_paths(gsel, es), "car")
        c1, c2 = st.columns(2)
        with c1:
            _plot(C.sue_quintiles(), "sueq")
        with c2:
            h = D.csv("hypothesis_tests.csv")
            if h is not None:
                _table(h[[c for c in ("test", "estimate", "ci_low", "ci_high", "p_adj_bh", "n", "verdict") if c in h]].round(4), "ht", height=360)
        if ta is not None:
            with st.expander("All event-study results (CARs by group, window and event set)"):
                gb = st.selectbox("Group by", sorted(ta.group_by.unique()), key="tagb")
                _table(ta[ta.group_by == gb][["group", "event_set", "measure", "mean", "t", "p_adj_bh", "n", "reject_bh"]].round(5), "tagrp", height=320)
        tp = D.csv("topics.csv")
        if tp is not None:
            with st.expander("Topics found in training-period text (and their average CARs)"):
                _table(tp, "topics")
    S["eventstudy"] = dict(
        sub="Measure how prices react around filings and whether the post-earnings-announcement drift (PEAD) shows up: association only.",
        metrics=[("8-K CAR[-1,+1]", f"{k8['mean'] * 100:+.2f}%" if k8 is not None else "n/a", f"t = {k8.t:.1f}, BH-adjusted p = {k8.p_adj_bh:.3f}" if k8 is not None else ""),
                 ("SUE drift Q5-Q1", f"{sue.estimate * 100:+.2f}%" if sue is not None else "n/a", f"[{sue.ci_low * 100:+.2f}%, {sue.ci_high * 100:+.2f}%], days +2 to +60" if sue is not None else ""),
                 ("Verdict", str(sue.verdict) if sue is not None else "n/a", "strength and robustness, not direction"),
                 ("Releases tested", D.n(sue.n) if sue is not None else "n/a", "verified earnings releases with SUE")],
        what=["Cumulative abnormal returns (market model, estimated on days -250 to -30) around day 0, by filing type, 8-K category and session.",
              "Standard errors clustered by firm and by date; Benjamini-Hochberg false-discovery control within pre-declared families.",
              "PEAD test: sort earnings releases into SUE quintiles (prior-quarter breakpoints) and compare days +2 to +60."],
        why=["Classic PEAD says good surprises keep drifting up; here the high-surprise quintile UNDERPERFORMED, a reversal.",
             "In a survivor-only universe, firms that kept falling after bad news were delisted and are missing, which pushes low-SUE returns up; "
             "these data cannot separate that bias from a genuine reversal."],
        evidence=ev_es, figures=["F11", "F12", "F13", "F14", "F15", "F16"], decisions=["D-057", "D-073"],
        guards=["'Supported' needs FDR significance, a consistent sign in 2 of 3 sub-periods, clean-event, cost and lag robustness; it says nothing about direction.",
                "Not a consensus-based PEAD replication: no analyst forecasts are used."])

    # ------------------------------------------------------------------ 15 backtest
    bt = D.csv("backtest.csv")
    bsp = D.js("backtest_spec.json") or {}

    def _bt(block, leg, lag=0):
        if bt is None:
            return None
        x = bt[(bt.sp_bucket == "Combined") & (bt.block == block) & (bt.leg == leg) & (bt.lag == lag)].sort_values("cost_bps")
        return x.iloc[0] if len(x) else None
    lo, lms, ls = _bt("locked", "long_only"), _bt("locked", "long_minus_spy_matched"), _bt("locked", "long_short_hypothetical")

    def ev_bt():
        c1, c2, c3 = st.columns(3)
        block = c1.segmented_control("Period", ["dev", "locked"], default="locked", key="btb")
        lag = c2.segmented_control("Lag (sessions)", [0, 1], default=0, key="btl")
        cost = c3.segmented_control("Cost (bps per side)", [10, 20], default=10, key="btc")
        legs = st.pills("Lines", ["long_ret", "spy_matched", "long_excess_spy", "ls_ret"], selection_mode="multi",
                        default=["long_ret", "spy_matched", "long_excess_spy"], key="btlegs",
                        format_func=lambda s: {"long_ret": "long book", "spy_matched": "SPY same days", "long_excess_spy": "long minus SPY",
                                               "ls_ret": "long-short"}[s])
        _plot(C.backtest_cum(block or "locked", lag if lag is not None else 0, cost or 10, list(legs or [])), "btcum")
        if bt is not None:
            x = bt[(bt.sp_bucket == "Combined") & (bt.block == (block or "locked")) & (bt.lag == (lag or 0)) & (bt.cost_bps == (cost or 10))]
            _table(x[["leg", "ann_return", "ann_vol", "sharpe", "deflated_sharpe_prob", "max_drawdown", "carhart_alpha_ann", "carhart_alpha_t"]].round(4), "bttab")
    S["backtest"] = dict(
        sub="Check economic relevance: would trading the frozen model's signal have earned anything after costs?",
        metrics=[("Long book (locked)", D.pct(lo.ann_return) if lo is not None else "n/a", f"per year, Sharpe {lo.sharpe:.2f}" if lo is not None else ""),
                 ("Long minus SPY", D.pct(lms.ann_return) if lms is not None else "n/a", "per year, locked, 10 bps"),
                 ("Long-short", D.pct(ls.ann_return) if ls is not None else "n/a", "hypothetical, locked"),
                 ("Signal model", f"{(bsp.get('signal_model') or {}).get('name', 'n/a')}", f"horizon {bsp.get('horizon')} sessions")],
        what=["Signal = P(Up) - P(Down) from the frozen model; long the top 20%, short the bottom 20% (thresholds from training-period predictions only).",
              "Entry at the first open after the decision time; costs of 10 and 20 bps per side; a one-session lag as a robustness check; borrow cost on shorts.",
              "Reported against SPY on the same days, with Carhart alpha and a deflated Sharpe ratio that accounts for the configurations tried."],
        why=["A classifier that beats chance can still be worthless to trade; a cost-aware backtest is the honest test of 'prediction is not profitability'."],
        evidence=ev_bt, figures=["F30"], decisions=[],
        guards=["Shorts are hypothetical (no borrow availability claim); capacity is not modelled; survivor-only universe has no delisting returns.",
                "Positive development-period numbers did not survive on the locked test."])

    # ------------------------------------------------------------------ 16 findings
    def ev_find():
        st.markdown("#### Hypothesis scoreboard")
        render_board(hypothesis_board(F))
        c1, c2 = st.columns(2)
        with c1:
            card("Smallest defensible claim",
                 f"<p>On 150 surviving S&amp;P 500 firms, under point-in-time controls, adding filing-text embeddings to price, metadata and accounting "
                 f"features gave no out-of-sample gain (locked 2023-2026: {lift_text(F['text_lock'])}). The models beat chance only by predicting "
                 f"<b>whether</b> a stock moves (AUC {D.f3(l6l.move_auc) if l6l is not None else 'n/a'}), not <b>which way</b> "
                 f"(AUC {D.f3(l6l.direction_auc) if l6l is not None else 'n/a'}).</p>", PHASES["Findings"])
            card("Known issues (disclosed)", bullets_html([
                "Tree models are not bit-reproducible across machines (re-scored predictions differ for up to ~20% of events).",
                "The frozen L7 beat L6 only on scores from two different runs; inside one run L6 is ahead: a tie.",
                "Stress folds built but not scored; Track C needs gold labels; fast mode used 2 of 4 encoders.",
                "Survivor-only universe, daily bars, 150 firms."], PHASES["Evaluation"]), PHASES["Evaluation"])
        with c2:
            card("Next steps (major project, 500 firms)", bullets_html([
                "Re-run the 500-firm study on the corrected accounting code (S&amp;P vs non-S&amp;P buckets, more power).",
                "Deterministic LightGBM; compare finalists inside one run with a strict-model tie-break.",
                "Score stress folds; add locked-test lifts to the pipeline.", "Annotate the gold set (templates are pre-filled) for Track C.",
                "Add delisted firms to remove survivorship bias."], PHASES["Features"]), PHASES["Features"])
            _plot(C.stage_hours(), "hours")
        pdf = D.report_pdf()
        if pdf:
            st.download_button("Download the full report (PDF)", pdf, "SEC_filings_price_impact_report.pdf", "application/pdf", key="dlpdf")
    S["findings"] = dict(
        sub="What the evidence supports, what it does not, and what would make the answer stronger.",
        metrics=[("Text adds value?", "No", f"locked {lift_text(F['text_lock'])}"), ("Beats chance?", "Yes, modestly",
                 f"{D.f3(l6l.macro_f1) if l6l is not None else 'n/a'} vs {D.f3(F['chance'])}"),
                 ("Predicts direction?", "No", f"AUC {D.f3(l6l.direction_auc) if l6l is not None else 'n/a'}"),
                 ("PEAD", "Reversal", f"{sue.estimate * 100:+.2f}% (survivor caveat)" if sue is not None else "n/a")],
        what=["A complete, reproducible point-in-time study of 60k SEC filing events with an honest null result for text.",
              "Structured information (event type, timing, size, price state) predicts how much a stock will react, not which way.",
              "Earnings surprises are followed by a reversal in this survivor-only sample; the backtest shows no tradable edge."],
        why=["A credible null is a valid result: it was obtained with a pre-registered design, a single-use locked test and an audited pipeline."],
        evidence=ev_find, figures=["R01", "R02", "R03", "R04"], decisions=["D-076", "D-077", "D-078"],
        guards=["Association is not causation; prediction is not profitability.", "All results are conditioned on a survivor-only universe."])
    return S[sid]


# ============================================================================================ renderer
def render(sid: str, F: dict) -> None:
    sp = spec(sid, F)
    i = IDS.index(sid)
    _, title, summary, phase, stages, icon = BY_ID[sid]
    col = PHASES[phase]
    step_header(i + 1, f"{icon}  {title}", sp["sub"], col, [phase, f"stages {stages}"])
    metric_cards(sp["metrics"], col)
    t1, t2, t3, t4 = st.tabs(["📝 What we did & why", "📊 Interactive evidence", "🖼️ Report figures", "🛡️ Guardrails & decisions"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            card("What we did", bullets_html(sp["what"], col), col)
        with c2:
            card("Why it matters", bullets_html(sp["why"], col), col)
    with t2:
        sp["evidence"]()
    with t3:
        fi = D.figure_index()
        figs = [fi.get(f) for f in sp["figures"] if fi.get(f)]
        if not figs:
            st.info("No report figure for this step.")
        for f in figs:
            with st.expander(f"{f['id']}  {f['title']}", expanded=len(figs) <= 2):
                if f.get("local"):
                    st.image(f["local"], width="stretch")
                    st.caption(f.get("caption", ""))
                else:
                    st.warning(f"{f['id']}: {f.get('status')}")
    with t4:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            card("Guardrails", bullets_html(sp["guards"], col), col)
        with c2:
            dec = D.csv("decisions.csv")
            if dec is not None and sp["decisions"]:
                d = dec[dec.id.isin(sp["decisions"])][["id", "summary"]]
                st.markdown("**Decisions behind this step** (from the decision log)")
                _table(d, f"dec_{sid}")
            st_ = D.csv("stages.csv")
            if st_ is not None:
                ids = [x.strip() for x in stages.replace("prereg", "").split(",") if x.strip().startswith("S")]
                expand = []
                for x in ids:
                    if "-" in x:
                        a, b = x.split("-")
                        expand += [f"S{k:02d}" for k in range(int(a[1:]), int(b[1:]) + 1)]
                    else:
                        expand.append(x)
                s = st_[st_.stage.isin(expand)][["stage", "name", "part", "outputs", "hours", "result"]]
                if len(s):
                    st.markdown("**Pipeline stages and outputs**")
                    _table(s.assign(hours=s.hours.map(lambda h: "n/a" if pd.isna(h) else f"{h * 60:.1f} min")), f"stg_{sid}")
