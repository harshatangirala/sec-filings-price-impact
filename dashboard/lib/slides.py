"""The presentation deck: one dict per slide (kicker, title, lead, colour, bullets, visual, speaker notes), built from the snapshot."""
from __future__ import annotations

import base64

import streamlit as st

from . import charts as C
from . import data as D
from .flow import flowchart
from .steps import facts, hypothesis_board, lift_text, render_board
from .theme import PHASES, chip, esc


def _img(fid: str):
    def draw():
        f = D.figure_index().get(fid)
        if f and f.get("local"):
            st.image(f["local"], width="stretch")
        else:
            st.info(f"Figure {fid} is not in the snapshot.")
    draw.fig_id = fid
    return draw


def _fig(builder, *a, **k):
    def draw():
        fig = builder(*a, **k)
        if fig is None:
            st.info("Chart input not in the snapshot.")
        else:
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"slide_{builder.__name__}_{id(draw)}")
    draw.builder = (builder, a, k)
    return draw


def _bignums(items):
    def draw():
        cols = st.columns(len(items))
        for c, (v, lab, col) in zip(cols, items):
            c.markdown(f"<div style='text-align:center;padding:14px 4px;border-radius:14px;background:white;border:1px solid #E2E8F0;"
                       f"border-bottom:5px solid {col}'><div class='bignum' style='color:{col}'>{esc(v)}</div><div class='bignum-l'>{esc(lab)}</div></div>",
                       unsafe_allow_html=True)
    draw.bignums = items
    return draw


def build() -> list[dict]:
    F = facts()
    e, mo, cfg = F["e"], F["mo"], F["cfg"]
    l6, l7 = F["l6_lock"], F["l7_lock"]
    sue = F["sue"]
    g3 = (F["g3"] or {}).get("lift") or {}
    rag = D.js("rag.json") or {}
    s16 = rag.get("s16") or {}
    ev_types = e.get("events_by_type", {})
    b6 = F.get("l6_wf_best")
    S = []

    def add(**k):
        S.append(k)

    add(kicker="Point-in-time NLP study · 150 S&P 500 stocks · 2000-2026", title="Do SEC filings predict the next stock-price move?",
        lead="An end-to-end, leakage-controlled pipeline from raw EDGAR filings to a single-use locked test.", color=PHASES["Design"], layout="title",
        bullets=[], visual=_bignums([(D.n(e.get("firms")), "S&P 500 firms", PHASES["Data"]), (D.n(e.get("filings")), "SEC filings", PHASES["Pre-processing"]),
                                     (D.n(e.get("events")), "events", PHASES["Features"]), ("25", "pipeline stages", PHASES["Modelling"]),
                                     (D.n(e.get("chunks_est")), "text chunks", PHASES["Text & GPU"])]),
        notes="Open with the question, then the scale: 150 firms, 64k filings, 60k events, 26 years. Stress that everything was evaluated as a trader "
              "would have experienced it, at the moment each filing became public.")
    add(kicker="Motivation", title="The question and how we kept ourselves honest", color=PHASES["Design"],
        lead="Three tracks that are never mixed, eight pre-registered hypotheses, one locked test.",
        bullets=["<b>Track A</b>: how do prices react around filings? (association only)",
                 "<b>Track B</b>: can information available at decision time predict the next session's abnormal return (Down / Sideways / Up)?",
                 "<b>Track C</b>: does extracted content match human labels? (built; needs gold labels)",
                 "Pre-registered metric (macro-F1), gates (G3, G3b, G4) and a locked 2023-2026 test scored exactly once."],
        visual=_fig(C.clock_diagram, F["m"]),
        notes="Pre-registration and the locked test are what make the final numbers credible. The clock shows the four timestamps: features may only "
              "use what is public by the decision time.")
    add(kicker="Step 2 · Data", title="Free, reproducible, point-in-time data", color=PHASES["Data"],
        lead="Every source is pinned to a version and re-downloadable.",
        bullets=[f"<b>{D.n(e.get('filings'))}</b> SEC filings (10-K, 10-Q, 8-K + exhibits) from EDGAR, acceptance times from the SGML header",
                 f"<b>{D.n(e.get('xbrl_facts'))}</b> XBRL accounting facts, <b>{D.n(e.get('results_releases'))}</b> parsed earnings releases",
                 f"<b>{D.n(e.get('price_bars'))}</b> daily price bars ({(e.get('price_first') or '')[:4]}-{(e.get('price_last') or '')[:4]}), Fama-French factors, S&P 500 history",
                 "No news or social media: SEC filings are the only text"],
        visual=_fig(C.filings_by_year), notes="Point out the 8-K volume and that acceptance times come from the filing header, not from the SEC JSON "
                                                "(which mixes two time conventions, decision D-047).")
    add(kicker="Step 3 · Universe & EDA", title="150 firms, 60k events: know the data first", color=PHASES["Data"],
        lead=f"{D.n(ev_types.get('8K'))} 8-K, {D.n(ev_types.get('periodic_10Q'))} 10-Q and {D.n(ev_types.get('periodic_10K'))} 10-K events.",
        bullets=[f"Seeded, industry-stratified draw of S&P 500 members from {e.get('eligible_firms')} eligible firms",
                 "Most filings arrive outside market hours (after the close or before the open)",
                 f"{D.pct((e.get('ts_quality') or {}).get('A', 0) / max(1, sum((e.get('ts_quality') or {1: 1}).values())))} of events have an exact timestamp",
                 "<b>Survivor-only</b>: every firm is listed today, a bias we state everywhere"],
        visual=_fig(C.events_by_year), notes="The event mix is dominated by 8-Ks; earnings-related events are where most price reaction happens.")
    add(kicker="Architecture", title="A 25-stage pipeline, CPU + GPU, fully resumable", color=PHASES["Modelling"],
        lead="Each stage writes to a hash-named folder; finished stages are skipped on re-run.",
        bullets=["Part 1 (CPU): data, events, features, baselines, event study", "Part 2 (GPU, Colab): embeddings, text models, RAG extraction",
                 "Part 3 (CPU): development holdout, freeze, locked test, backtest, report", "110 automated tests; every assumption is an assertion"],
        visual=lambda: st.plotly_chart(flowchart(None), width="stretch", config={"displaylogo": False, "staticPlot": True}, key="deck_flow"),
        notes="Show the flow: the same order as the interactive flowchart window.")
    add(kicker="Steps 4-5 · Parsing & NLP", title="From raw HTML to model-ready text", color=PHASES["Pre-processing"],
        lead=f"{D.n(e.get('sections'))} item sections, ~{D.n(e.get('chunks_est'))} chunks, ~{D.n(e.get('tokens_est'))} tokens.",
        bullets=["Item sections only: 10-K Items 1/1A/7/7A, 10-Q MD&A and risk sections, 8-K items and press releases",
                 f"Firm names and tickers masked ({D.n(e.get('firm_name_masks'))} + {D.n(e.get('ticker_masks'))} replacements) so models cannot memorise companies",
                 "350-word overlapping chunks shared by every encoder and the retriever",
                 "No vocabulary learned from the future: hashing for change features, TF-IDF refitted inside each training fold"],
        visual=_fig(C.text_funnel, F["m"]), notes="Masking is a leakage control: without it a model could learn firm identity instead of content.")
    add(kicker="Steps 6-7 · Targets & features", title="A tradable target and 128 point-in-time features", color=PHASES["Features"],
        lead="Enter at the first open after the decision time; label Up / Down beyond 0.5 x the stock's own volatility.",
        bullets=["Abnormal return = stock minus beta x SPY on exactly the same interval",
                 "Price, metadata, accounting (incl. earnings surprise), lexical tone, filing change and novelty features",
                 f"Availability asserted on every row: <b>{D.n(e.get('availability_violations'))}</b> violations",
                 "Two look-ahead leaks and a stale-accounting bug were caught and fixed before the final run"],
        visual=_fig(C.labels_by_year), notes="Mention D-071 (same-day 8-K groups) and D-077 (accounting as first reported) as examples of the audit working.")
    add(kicker="Step 8 · Evaluation design", title="Time-ordered splits and an audited pipeline", color=PHASES["Features"],
        lead="Six walk-forward folds, a development holdout, a locked test; purge + 90-day embargo.",
        bullets=["24-item leakage checklist, each with an assertion", f"<b>{e.get('planted_leaks_caught')}/4</b> deliberately planted leaks caught by the audit",
                 "Strict vs retrospective leaderboards by each model's training cutoff", "Locked test unlocked once, after a hash-locked freeze (G4)"],
        visual=_fig(C.split_timeline), notes="Random splits would leak the future; the walk-forward design trains only on the past.")
    add(kicker="Step 9 · Models", title="An ablation ladder on identical events", color=PHASES["Modelling"],
        lead="Each rung adds one kind of information; LightGBM, CatBoost and elastic net; Optuna tuning.",
        bullets=[f"Best strict model: L6 / {b6['head'] if b6 is not None else 'n/a'}, walk-forward macro-F1 {D.f3(b6.macro_f1) if b6 is not None else 'n/a'}",
                 f"Real chance level about {D.f3(F['chance'])} (not the {D.f3(mo.get('majority_baseline_f1'))} majority baseline)",
                 "Metadata (event type, timing, sections present) is the strongest single block; accounting alone is the weakest",
                 f"{D.n(e.get('configs_tried_s11'))} configurations counted for multiple-testing corrections"],
        visual=_fig(C.ladder_chart, "common", None), notes="Read the bars against the dashed chance line.")
    add(kicker="Steps 10-11 · Text & GPU", title="Does filing text add anything?", color=PHASES["Text & GPU"],
        lead="ChronoBERT (26 point-in-time vintages) and BGE embeddings of ~900k chunks on a Colab GPU.",
        bullets=[f"Gate G3: text lift {g3.get('value', float('nan')):+.4f} {D.ci(g3.get('ci_low', 0), g3.get('ci_high', 0), 4)}: below +0.01, so no fine-tuning",
                 "Point-in-time and static encoders perform the same (H4)", "Text alone (L5) is weaker than the structured features",
                 "Fast mode: 2 of 4 encoders, chosen for the time budget"],
        visual=_fig(C.emb_vs_l6), notes="The gate is pre-registered: we do not fine-tune when there is no evidence of a signal to amplify.")
    add(kicker="Step 12 · RAG", title="Evidence-grounded extraction with a local LLM", color=PHASES["RAG"],
        lead="Hybrid BM25 + dense retrieval, rerank, Qwen2.5-7B with JSON-constrained decoding and guardrails.",
        bullets=["12 questions: guidance, results, impairment, restructuring, M&A, dividends, buybacks, litigation, ...",
                 "Every quote must be a verbatim substring of its cited chunk, numbers must appear in the evidence",
                 f"Pre-filled {D.n(s16.get('prefilled_events'))} filings for human annotation ({s16.get('seconds_per_filing', 0):.0f} s each)",
                 "Scoring it needs human gold labels: G3b, student model and L8 are the next step"],
        visual=_img("A04"), notes="Be clear that the RAG output is unvalidated until gold labels exist.")
    add(kicker="Step 13 · Results", title="Accuracy in context: modest, real, stable", color=PHASES["Evaluation"],
        lead=f"Locked 2023-2026 test (n = {D.n(l6.n) if l6 is not None else 'n/a'}): L6 macro-F1 {D.f3(l6.macro_f1) if l6 is not None else 'n/a'} vs chance {D.f3(F['chance'])}.",
        bullets=["Scores hold from walk-forward (2008-2019) to development (2020-22) to locked (2023-26)",
                 "The majority-class baseline (0.19) and permutation null (0.25) understate chance; shuffled predictions give ~0.33",
                 "L6 (strict) and L7 (with text) are statistically tied"],
        visual=_fig(C.chance_chart), notes="This is the slide most people misread: compare to 0.33, not 0.19.")
    add(kicker="Step 13 · Results", title="The model knows WHETHER a stock moves, not WHICH WAY", color=PHASES["Evaluation"],
        lead=f"Move AUC {D.f3(l6.move_auc) if l6 is not None else 'n/a'} vs direction AUC {D.f3(l6.direction_auc) if l6 is not None else 'n/a'} on the locked test.",
        bullets=["Telling movers from quiet stocks is a volatility / event-type effect", "Among stocks that moved, up vs down is a coin flip",
                 "So the classifier cannot be turned into a directional trading signal"],
        visual=_fig(C.skill_chart), notes="Direction AUC CIs include 0.5 in the development and locked periods.")
    add(kicker="Step 13 · Results", title="Text adds no detectable value", color=PHASES["Evaluation"],
        lead=f"L7 - L6 on the locked test: {lift_text(F['text_lock'])}.",
        bullets=[f"Walk-forward {lift_text(F['text_wf'])}", f"Development {lift_text(F['text_dev'])}", f"Locked {lift_text(F['text_lock'])}",
                 f"By contrast, structured features beat price-only: locked {lift_text(F['struct_lock'])}"],
        visual=_fig(C.lifts_forest, None), notes="A credible null with tight intervals is a valid finding.")
    add(kicker="Step 14 · Event study", title="Prices react to filings; earnings drift reverses", color=PHASES["Event study"],
        lead=(f"SUE quintile spread (Q5-Q1, days +2 to +60): {sue.estimate * 100:+.2f}% [{sue.ci_low * 100:+.2f}%, {sue.ci_high * 100:+.2f}%]" if sue is not None else ""),
        bullets=["Small, significant abnormal returns around 8-Ks (two-way clustered errors, FDR control)",
                 "High-surprise firms UNDERPERFORM low-surprise firms: the opposite of classic PEAD",
                 "Survivor-only universe pushes low-SUE returns up; we cannot separate that bias from a real reversal"],
        visual=_fig(C.sue_quintiles), notes="Association only. Do not present this as a trading strategy.")
    bt = D.csv("backtest.csv")
    lms = None
    if bt is not None:
        x = bt[(bt.sp_bucket == "Combined") & (bt.block == "locked") & (bt.leg == "long_minus_spy_matched") & (bt.lag == 0)].sort_values("cost_bps")
        lms = x.iloc[0] if len(x) else None
    add(kicker="Step 15 · Backtest", title="Prediction is not profitability", color=PHASES["Event study"],
        lead=(f"Locked test, long book minus SPY: {lms.ann_return * 100:+.1f}% a year at 10 bps." if lms is not None else ""),
        bullets=["Long the top 20% of P(Up) - P(Down), short the bottom 20%; thresholds from training data only",
                 "Costs, a one-session lag, borrow costs, Carhart alpha and a deflated Sharpe ratio",
                 "Development-period gains did not survive on the locked test"],
        visual=_fig(C.backtest_cum, "locked", 0, 10, ["long_ret", "spy_matched", "long_excess_spy"]), notes="An economic-relevance check, not an investable strategy.")
    add(kicker="Honesty", title="Known issues and limitations", color=PHASES["Findings"],
        lead="Disclosed in the report; none changes the conclusions.",
        bullets=["Survivor-only universe, daily bars, 150 firms", "Tree models are not bit-reproducible across machines (~20% of predicted classes differ on re-scoring)",
                 "The frozen L7 won only on scores from two different runs; L6 and L7 are tied",
                 "Stress folds not scored; Track C needs gold labels; fast mode used 2 of 4 encoders"],
        visual=lambda: render_board(hypothesis_board(F), cols=2), notes="Showing the limitations is part of the result.")
    add(kicker="Conclusion", title="What we can defend, and what comes next", color=PHASES["Findings"],
        lead="A credible null for text; a magnitude-not-direction signal; a survivorship-sensitive drift reversal.",
        bullets=["<b>Smallest defensible claim:</b> on 150 surviving S&P 500 firms, filing-text embeddings add no out-of-sample value to price, metadata "
                 f"and accounting features (locked {lift_text(F['text_lock'])})",
                 "Next: 500-firm study (S&P vs non-S&P), deterministic training, gold labels for Track C, delisted firms",
                 "Everything is reproducible: one notebook, pinned data, 25 resumable stages, 110 tests"],
        visual=_bignums([(f"{F['text_lock'].lift:+.3f}" if F["text_lock"] is not None else "n/a", "text lift (locked)", PHASES["Evaluation"]),
                         (D.f3(l6.move_auc) if l6 is not None else "n/a", "move AUC", PHASES["Modelling"]),
                         (D.f3(l6.direction_auc) if l6 is not None else "n/a", "direction AUC", PHASES["RAG"]),
                         (f"{sue.estimate * 100:+.1f}%" if sue is not None else "n/a", "SUE drift Q5-Q1", PHASES["Event study"])]),
        notes="Close on the claim and the roadmap.")
    for i, s in enumerate(S):
        s["n"] = i + 1
    return S


# ------------------------------------------------------------------ standalone HTML handout (download)
def handout_html(slides: list[dict]) -> str:
    import plotly.io as pio
    parts, first = [], True
    for s in slides:
        vis = ""
        v = s.get("visual")
        if hasattr(v, "builder"):
            b, a, k = v.builder
            fig = b(*a, **k)
            if fig is not None:
                vis = pio.to_html(fig, include_plotlyjs="cdn" if first else False, full_html=False)
                first = False
        elif hasattr(v, "fig_id"):
            f = D.figure_index().get(v.fig_id)
            if f and f.get("local"):
                vis = f"<img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(open(f['local'], 'rb').read()).decode()}'/>"
        elif hasattr(v, "bignums"):
            vis = "<div style='display:flex;gap:12px;flex-wrap:wrap'>" + "".join(
                f"<div style='flex:1;min-width:130px;text-align:center;border:1px solid #e2e8f0;border-bottom:5px solid {c};border-radius:12px;padding:10px'>"
                f"<div style='font-size:2rem;font-weight:800;color:{c}'>{esc(x)}</div><div style='font-size:.8rem;color:#64748b'>{esc(l)}</div></div>"
                for x, l, c in v.bignums) + "</div>"
        bl = "".join(f"<li>{b}</li>" for b in s.get("bullets", []))
        parts.append(f"<section style='page-break-after:always;margin:0 0 40px 0;border-radius:18px;overflow:hidden;border:1px solid #e2e8f0'>"
                     f"<div style='background:{s['color']};color:white;padding:18px 26px'><div style='font-size:.8rem;letter-spacing:.1em;text-transform:uppercase'>"
                     f"{esc(s['kicker'])} · slide {s['n']}</div><h2 style='margin:4px 0'>{esc(s['title'])}</h2><div>{esc(s.get('lead', ''))}</div></div>"
                     f"<div style='padding:16px 26px'><ul style='font-size:1.02rem;line-height:1.5'>{bl}</ul>{vis}"
                     f"<p style='color:#92400e;background:#fffbeb;padding:8px 12px;border-radius:8px'><b>Notes:</b> {esc(s.get('notes', ''))}</p></div></section>")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>SEC filings study slides</title>"
            "<style>body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;color:#0f172a}</style></head><body>"
            + "".join(parts) + "</body></html>")
