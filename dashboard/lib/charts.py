"""Plotly figures for the dashboard. Every builder reads the snapshot and returns a Figure, or None when its input is missing."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import data as D
from .theme import CLASS_COLORS, MODEL_COLORS, PALETTE, PERIOD_COLORS, PHASES, tint

SECTION_NAMES = {"10K_1": "10-K Item 1 Business", "10K_1A": "10-K Item 1A Risk factors", "10K_7": "10-K Item 7 MD&A",
                 "10K_7A": "10-K Item 7A Market risk", "10Q_MDA": "10-Q MD&A", "10Q_MR": "10-Q Market risk", "10Q_RF": "10-Q Risk factors",
                 "8K_ITEMS": "8-K item text", "8K_EX99": "8-K EX-99 exhibit"}
EVENT_NAMES = {"8K": "8-K", "periodic_10Q": "10-Q", "periodic_10K": "10-K"}


def _ref(fig: go.Figure, y: float, name: str, color: str, dash: str = "dash", **kw) -> None:
    """Horizontal reference line whose label lives in the legend (never on top of the data)."""
    fig.add_hline(y=y, line=dict(color=color, dash=dash, width=2), **kw)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash, width=2), name=name, hoverinfo="skip"),
                  **({"row": kw["row"], "col": kw["col"]} if "row" in kw else {}))


def _h(fig: go.Figure, h: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(height=h, title=title)
    return fig


# =========================================================================================== data collection / EDA
def filings_by_year():
    d = D.csv("filings_by_year.csv")
    if d is None:
        return None
    fig = px.area(d, x="year", y="filings", color="base_form", color_discrete_sequence=[PHASES["Data"], "#6366F1", "#14B8A6", "#F59E0B"],
                  labels={"base_form": "form", "filings": "filings"})
    return _h(fig, 360, "Filings downloaded per year, by form")


def sources_treemap(meta: dict):
    e = meta.get("eda", {})
    cov = D.csv("coverage_by_form.csv")
    rows = [("SEC EDGAR", "Filings", None)]
    if cov is not None:
        for r in cov.itertuples():
            rows.append((f"{r.base_form} filings", "SEC EDGAR", int(r.observed)))
    rows += [("XBRL", "Filings", None), ("XBRL facts", "XBRL", e.get("xbrl_facts")),
             ("Release exhibits", "XBRL", e.get("release_rows")), ("Prices", "Filings", None),
             ("Daily price bars", "Prices", e.get("price_bars"))]
    df = pd.DataFrame(rows, columns=["label", "parent", "value"])
    df = df[df.label != "Filings"]
    df.loc[df.parent == "Filings", "parent"] = ""
    vals = df.value.fillna(0).astype(float)
    # parents carry the sum of their children so the treemap is consistent
    for p in ("SEC EDGAR", "XBRL", "Prices"):
        vals[df.label == p] = vals[df.parent == p].sum()
    fig = go.Figure(go.Treemap(labels=df.label, parents=df.parent, values=np.log10(vals.clip(lower=1)) + 0.01, branchvalues="total",
                               customdata=vals, hovertemplate="%{label}: %{customdata:,.0f} records<extra></extra>",
                               marker=dict(colors=[PHASES["Data"], "#38BDF8", "#7DD3FC", "#BAE6FD", "#BAE6FD", PHASES["Pre-processing"],
                                                   "#5EEAD4", "#99F6E4", PHASES["Event study"], "#FDBA74"][:len(df)]),
                               textinfo="label"))
    return _h(fig, 330, "Raw data volume by source (tile area ~ log of record count; hover for counts)")


def sectors_bar():
    u = D.csv("universe.csv")
    if u is None:
        return None
    c = u.ff12_industry.value_counts().sort_values()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", text=c.values, textposition="outside",
                           marker=dict(color=c.values, colorscale=[[0, "#BAE6FD"], [1, "#4338CA"]])))
    fig.update_layout(xaxis_title="firms", yaxis_title=None)
    return _h(fig, 380, "150 firms by Fama-French 12 industry")


def events_by_year():
    d = D.csv("events_by_year.csv")
    if d is None:
        return None
    d = d.assign(type=d.event_type.map(EVENT_NAMES))
    fig = px.bar(d, x="year", y="events", color="type", color_discrete_map={"8-K": PHASES["Data"], "10-Q": "#6366F1", "10-K": PHASES["Event study"]})
    return _h(fig, 360, "Events per year by filing type")


def categories_8k(top: int = 15):
    d = D.csv("categories_8k.csv")
    if d is None:
        return None
    d = d.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=d.events, y=d.categories.str.replace("_", " "), orientation="h", marker_color=PHASES["Data"],
                           text=d.events, textposition="outside"))
    return _h(fig, 430, f"Top {top} 8-K categories (an 8-K can carry several)")


def session_donut():
    d = D.csv("session_buckets.csv")
    if d is None:
        return None
    g = d.groupby("session_bucket").events.sum().reset_index()
    fig = go.Figure(go.Pie(labels=g.session_bucket, values=g.events, hole=.55,
                           marker=dict(colors=["#6366F1", "#F59E0B", "#0EA5E9", "#94A3B8"])))
    fig.update_layout(annotations=[dict(text="when filed", x=.5, y=.5, showarrow=False, font_size=13)])
    return _h(fig, 330, "When filings arrive (PRE = before the open, INTRA = in session, POST = after the close)")


def ts_quality_by_year():
    d = D.csv("ts_quality_by_year.csv")
    if d is None:
        return None
    d = d.assign(share=d.events / d.groupby("year").events.transform("sum"))
    fig = px.bar(d, x="year", y="share", color="ts_quality", color_discrete_map={"A": "#22C55E", "B": "#F59E0B", "C": "#EF4444"})
    fig.update_layout(yaxis_tickformat=".0%")
    return _h(fig, 330, "Timestamp quality by year (A = exact from the SEC header)")


def events_per_firm():
    d = D.csv("events_by_firm.csv")
    if d is None:
        return None
    fig = px.box(d, x="ff12_industry", y="events", points="all", hover_name="ticker", color="ff12_industry",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False, xaxis_title=None)
    return _h(fig, 380, "Events per firm, by industry (hover a dot for the ticker)")


def firms_per_year():
    d = D.csv("firms_per_year.csv")
    if d is None:
        return None
    fig = go.Figure()
    for col, c in (("firms_in_sp500", PHASES["Evaluation"]), ("unique_firms", "#94A3B8")):
        if col in d:
            fig.add_trace(go.Scatter(x=d.year, y=d[col], name=col.replace("_", " "), mode="lines+markers", line=dict(color=c, width=3)))
    return _h(fig, 300, "Firms per year (all 150 survive to 2026: a survivor-only universe)")


# =========================================================================================== parsing / NLP
def coverage_by_form():
    d = D.csv("coverage_by_form.csv")
    if d is None:
        return None
    fig = go.Figure([go.Bar(name="downloaded", x=d.base_form, y=d.observed, marker_color="#BAE6FD"),
                     go.Bar(name="parsed into sections", x=d.base_form, y=d.parsed, marker_color=PHASES["Pre-processing"],
                            text=[D.pct(x) for x in d.parsed_share], textposition="outside")])
    fig.update_layout(barmode="overlay")
    return _h(fig, 320, "Parse success by form")


def hf_qa_box():
    d = D.csv("hf_parser_qa.csv")
    if d is None:
        return None
    d = d[d.hf_chars > 0]
    fig = px.box(d, x="section_id", y="token_jaccard", color="section_id", color_discrete_sequence=PALETTE, points=False)
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.02], xaxis_title=None)
    return _h(fig, 320, "Parser QA: word overlap of our 10-K sections with the Hugging Face corpus (1 = identical)")


def text_funnel(meta: dict):
    e = meta.get("eda", {})
    stages = [("Filings downloaded", e.get("filings")), ("Parsed OK", e.get("filings_parsed_ok")), ("Item sections kept", e.get("sections")),
              ("Text chunks (350 words)", e.get("chunks_est"))]
    stages = [(a, b) for a, b in stages if b]
    fig = go.Figure(go.Funnel(y=[a for a, _ in stages], x=[b for _, b in stages], textinfo="value",
                              marker=dict(color=[PHASES["Data"], "#38BDF8", PHASES["Pre-processing"], "#5EEAD4"])))
    return _h(fig, 300, "From filings to model-ready text")


def sections_bar():
    d = D.csv("sections.csv")
    if d is None:
        return None
    d = d.assign(name=d.section_id.map(SECTION_NAMES).fillna(d.section_id))
    fig = go.Figure([go.Bar(x=d.name, y=d.sections, name="sections", marker_color=PHASES["Pre-processing"]),
                     go.Scatter(x=d.name, y=d.median_tokens, name="median tokens", yaxis="y2", mode="markers+lines",
                                marker=dict(size=10, color=PHASES["Text & GPU"]), line=dict(dash="dot"))])
    fig.update_layout(yaxis=dict(title="sections"), yaxis2=dict(title="median tokens", overlaying="y", side="right", showgrid=False),
                      xaxis_tickangle=-25)
    return _h(fig, 380, "Sections kept and their typical length")


def section_length_hist(sections: list[str]):
    d = D.csv("section_length_hist.csv")
    if d is None:
        return None
    fig = go.Figure()
    for i, s in enumerate(sections):
        g = d[d.section_id == s]
        fig.add_trace(go.Scatter(x=10 ** g.log10_tokens, y=g.sections, name=SECTION_NAMES.get(s, s), mode="lines", fill="tozeroy",
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)]), opacity=.7))
    fig.update_layout(xaxis_type="log", xaxis_title="estimated tokens per section (log scale)", yaxis_title="sections")
    return _h(fig, 340, "How long are the sections?")


# =========================================================================================== events and labels
def clock_diagram(meta: dict):
    t = (meta.get("config") or {}).get("timing") or {}
    lag, lat = t.get("public_lag_minutes", 5), t.get("decision_latency_minutes", 30)
    pts = [(0, "EDGAR acceptance", "#0EA5E9"), (lag, f"public (+{lag} min, assumed)", "#6366F1"),
           (lag + lat, f"decision time (+{lat} min)", "#EF4444"), (lag + lat + 60, "T_trade entry: next market open", "#22C55E")]
    fig = go.Figure()
    fig.add_shape(type="line", x0=-40, x1=150, y0=0, y1=0, line=dict(color="#94A3B8", width=3))
    fig.add_shape(type="rect", x0=-40, x1=lag + lat, y0=-.18, y1=.18, fillcolor=tint("#22C55E", .12), line_width=0)
    fig.add_annotation(x=-20, y=.32, text="features may use only what is known here", showarrow=False, font=dict(color="#15803D"))
    for i, (x, lab, c) in enumerate(pts):
        fig.add_trace(go.Scatter(x=[x], y=[0], mode="markers", marker=dict(size=18, color=c, line=dict(color="white", width=2)),
                                 name=lab, hovertemplate=lab + "<extra></extra>"))
        fig.add_annotation(x=x, y=(-.5 if i % 2 else .55), text=f"<b>{lab}</b>", showarrow=True, arrowhead=0, ay=(25 if i % 2 else -25),
                           ax=0, font=dict(color=c, size=12))
    fig.add_annotation(x=130, y=-.55, text="exit: close of session entry + h - 1", showarrow=False, font=dict(color="#64748B"))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False, range=[-45, 155]), yaxis=dict(visible=False, range=[-1, 1]))
    return _h(fig, 260, "The event clock (schematic; minutes after acceptance)")


def labels_by_year():
    d = D.csv("labels_by_year.csv")
    if d is None:
        return None
    d = d.assign(share=d.events / d.groupby("year").events.transform("sum"))
    fig = px.bar(d, x="year", y="share", color="label", color_discrete_map=CLASS_COLORS, category_orders={"label": ["Down", "Sideways", "Up"]})
    fig.update_layout(yaxis_tickformat=".0%")
    return _h(fig, 330, "Primary label balance by year (T_trade, next session)")


def ar_hist():
    d = D.csv("ar_hist.csv")
    if d is None:
        return None
    fig = go.Figure(go.Bar(x=d.abnormal_return_pct, y=d.events, marker_color=np.where(d.abnormal_return_pct < 0, CLASS_COLORS["Down"], CLASS_COLORS["Up"])))
    fig.update_layout(xaxis_title="next-session abnormal return, % (clipped at +/-10%)", yaxis_title="events", bargap=0.02)
    return _h(fig, 320, "Distribution of the target (labels use a stock-specific threshold k x sigma)")


# =========================================================================================== features / splits
def feature_sunburst():
    d = D.csv("feature_registry.csv")
    if d is None:
        return None
    fam_c = {"price": "#0EA5E9", "metadata": "#6366F1", "accounting": "#14B8A6", "lexical": "#22C55E", "change": "#F59E0B",
             "novelty": "#EC4899", "ear": "#94A3B8"}
    fig = px.sunburst(d, path=["family", "name"], color="family", color_discrete_map=fam_c)
    fig.update_traces(insidetextorientation="radial")
    return _h(fig, 480, "128 engineered features by family (click a family to zoom)")


def sue_coverage():
    d = D.csv("sue_coverage.csv")
    if d is None:
        return None
    d = d.assign(share=d.sue_available / d.n.replace(0, np.nan))
    fig = px.line(d, x="year", y="share", color="kind", markers=True, color_discrete_sequence=[PHASES["Features"], PHASES["Text & GPU"]])
    fig.update_layout(yaxis_tickformat=".0%")
    return _h(fig, 300, "Share of filings with an earnings surprise (SUE) available")


def split_timeline():
    b = D.csv("blocks.csv")
    if b is None:
        return None
    col = {"walk_forward_val": PHASES["Modelling"], "development_holdout": PHASES["Text & GPU"], "locked_test": PHASES["Evaluation"]}
    lab = {"walk_forward_val": "walk-forward validation", "development_holdout": "development holdout", "locked_test": "locked test (scored once)"}
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[(pd.Timestamp(b.start.min()) - pd.Timestamp("2000-01-01")).days], base=["2000-01-01"], y=["training (expanding)"],
                         orientation="h", marker_color="#CBD5E1", name="first training window", hovertemplate="2000 onwards: training data<extra></extra>"))
    for r in b.itertuples():
        dur = (pd.Timestamp(r.end_exclusive) - pd.Timestamp(r.start)).days
        fig.add_trace(go.Bar(x=[dur], base=[r.start], y=[lab[r.kind]], orientation="h", marker_color=col[r.kind], name=r.block,
                             text=f"{r.block}<br>{r.events:,} ev", textposition="inside", insidetextanchor="middle",
                             hovertemplate=f"{r.block}: {r.start} to {r.end_exclusive}<br>{r.events:,} events<extra></extra>"))
    fig.update_layout(barmode="overlay", showlegend=False, xaxis=dict(type="date"), yaxis=dict(categoryorder="array",
                      categoryarray=["locked test (scored once)", "development holdout", "walk-forward validation", "training (expanding)"]))
    return _h(fig, 300, "Time-ordered splits: 6 walk-forward folds, a development holdout, a locked test (90-day embargo between blocks)")


def provenance_heatmap():
    d = D.csv("fold_provenance.csv")
    if d is None:
        return None
    p = d.assign(v=(d.leaderboard == "strict").astype(int)).pivot_table(index="model", columns="block", values="v", aggfunc="first")
    order = [c for c in ("wf0", "wf1", "wf2", "wf3", "wf4", "wf5", "dev", "locked") if c in p.columns]
    p = p[order]
    fig = go.Figure(go.Heatmap(z=p.values, x=p.columns, y=p.index, colorscale=[[0, "#FECACA"], [1, "#BBF7D0"]], showscale=False,
                               text=np.where(p.values == 1, "strict", "retro"), texttemplate="%{text}", xgap=2, ygap=2))
    return _h(fig, 360, "Leaderboard per model and block: strict (training cutoff before the events) vs retrospective")


# =========================================================================================== models
def ladder_chart(cohort: str = "common", heads: list[str] | None = None):
    d = D.csv("ladder.csv")
    if d is None:
        return None
    d = d[(d.cohort == cohort) & ((d.tuned == "tuned") | (d.rung == "L0"))]
    if heads:
        d = d[d["head"].isin(heads) | (d.rung == "L0")]
    order = [r for r in ("L0", "L1", "L2", "L3", "L4", "L4_levels", "L4_change", "L5", "L6_nochange", "L6", "L7") if r in set(d.rung)]
    fig = go.Figure()
    for i, (h, g) in enumerate(d.groupby("head")):
        g = g.set_index("rung").reindex(order).dropna(subset=["macro_f1"])
        lab = g.index if "encoder" not in g or (g.encoder.fillna("-") == "-").all() else [f"{r} ({e})" if e not in ("-", "n/a") else r
                                                                                         for r, e in zip(g.index, g.encoder.fillna("-"))]
        fig.add_trace(go.Bar(name=h, x=list(lab), y=g.macro_f1, error_y=dict(type="data", symmetric=False, array=g.ci_high - g.macro_f1,
                                                                             arrayminus=g.macro_f1 - g.ci_low),
                             marker_color=PALETTE[i % len(PALETTE)],
                             hovertemplate="%{x} / " + h + "<br>macro-F1 %{y:.3f}<extra></extra>"))
    ch = D.chance_level()
    if ch:
        _ref(fig, ch, f"real chance level {ch:.3f}", "#0F172A")
    fig.update_layout(barmode="group", yaxis_title="pooled walk-forward macro-F1", yaxis_range=[0.15, 0.45])
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    return _h(fig, 420, "Ablation ladder: each rung adds information (95% month-block bootstrap CIs)")


def ladder_folds(run_ids: list[str]):
    d = D.csv("ladder_folds.csv")
    if d is None or not run_ids:
        return None
    d = d[d.run_id.isin(run_ids)].sort_values("block")
    fig = px.line(d, x="block", y="macro_f1", color="run_id", markers=True, color_discrete_sequence=PALETTE)
    fig.update_layout(yaxis_title="macro-F1 per validation fold", legend=dict(orientation="v", y=1, x=1.01, xanchor="left"))
    return _h(fig, 360, "Stability across the six walk-forward folds (2008-2019, two years each)")


def tuning_dumbbell():
    d = D.csv("tuning.csv")
    if d is None:
        return None
    d = d.dropna(subset=["default_mean_f1", "tuned_mean_f1"])
    d = d.assign(label=d.rung + " / " + d["head"] + np.where(d.encoder.fillna("-") != "-", " / " + d.encoder.fillna(""), ""))
    d = d.sort_values("tuned_mean_f1")
    fig = go.Figure()
    for r in d.itertuples():
        fig.add_shape(type="line", x0=r.default_mean_f1, x1=r.tuned_mean_f1, y0=r.label, y1=r.label, line=dict(color="#CBD5E1", width=3))
    fig.add_trace(go.Scatter(x=d.default_mean_f1, y=d.label, mode="markers", name="default", marker=dict(color="#94A3B8", size=10)))
    fig.add_trace(go.Scatter(x=d.tuned_mean_f1, y=d.label, mode="markers", name="tuned (Optuna)", marker=dict(color=PHASES["Modelling"], size=11)))
    fig.update_layout(xaxis_title="mean walk-forward macro-F1")
    return _h(fig, max(380, 18 * len(d)), "Default vs tuned hyperparameters")


def fit_scatter():
    d = D.csv("fit_diagnosis.csv")
    if d is None:
        return None
    fig = px.scatter(d, x="mean_train", y="mean_val", color="verdict", symbol="head", hover_name="run_id", size_max=14,
                     color_discrete_map={"good fit": "#22C55E", "overfit": "#EF4444", "unstable": "#F59E0B", "overfit, unstable": "#B91C1C"})
    lo = float(min(d.mean_train.min(), d.mean_val.min())) - .02
    hi = float(max(d.mean_train.max(), d.mean_val.max())) + .02
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#94A3B8", dash="dot"))
    fig.update_traces(marker=dict(size=12, line=dict(color="white", width=1)))
    fig.update_layout(xaxis_title="training macro-F1", yaxis_title="validation macro-F1")
    return _h(fig, 380, "Fit diagnosis: far below the diagonal = overfitting")


def learning_curve():
    d = D.csv("learning_curves.csv")
    if d is None:
        return None
    fig = go.Figure([go.Scatter(x=d.n_train, y=d.train_macro_f1, name="training", mode="lines+markers", line=dict(color="#94A3B8", width=3)),
                     go.Scatter(x=d.n_train, y=d.val_macro_f1, name="validation", mode="lines+markers", line=dict(color=PHASES["Modelling"], width=3))])
    fig.update_layout(xaxis_title="training events", yaxis_title="macro-F1")
    return _h(fig, 320, "Learning curve: training vs validation macro-F1 as the training set grows")


def permutation_hist():
    d = D.csv("permutation.csv")
    if d is None:
        return None
    obs = d[d.kind == "observed"].macro_f1.iloc[0]
    null = d[d.kind == "permuted"].macro_f1
    fig = go.Figure(go.Histogram(x=null, nbinsx=12, marker_color="#CBD5E1", name="shuffled labels"))
    fig.add_vline(x=obs, line=dict(color=PHASES["Evaluation"], width=3), annotation_text=f"observed {obs:.3f}")
    fig.update_layout(xaxis_title="macro-F1", yaxis_title="runs", showlegend=False)
    return _h(fig, 300, "Sanity check: the pipeline scores far lower when labels are shuffled")


def optuna_scatter(param: str):
    d = D.csv("optuna_trials.csv")
    if d is None or param not in d:
        return None
    x = d.dropna(subset=[param, "value"])
    fig = px.scatter(x, x=param, y="value", color="head" if "head" in x else None, symbol="rung" if "rung" in x else None,
                     color_discrete_sequence=PALETTE, hover_data=[c for c in ("rung", "head", "number") if c in x])
    fig.update_layout(yaxis_title="objective (mean macro-F1, tuning folds)", xaxis_title=param.replace("params_", ""))
    return _h(fig, 340, "Optuna trials")


# =========================================================================================== text / GPU
def encoder_time(s14: dict):
    rows = [(k, v.get("seconds", 0) / 60, v.get("events")) for k, v in (s14 or {}).items() if isinstance(v, dict) and "seconds" in v]
    if not rows:
        return None
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], marker_color=[PHASES["Text & GPU"], PHASES["RAG"]][:len(rows)],
                           text=[f"{r[1]:.0f} min" for r in rows], textposition="outside"))
    fig.update_layout(yaxis_title="minutes on the Colab GPU")
    return _h(fig, 300, "Embedding time per encoder")


def emb_vs_l6():
    e, lad = D.csv("emb_ladder_summary.csv"), D.csv("ladder.csv")
    if e is None:
        return None
    fig = go.Figure()
    for i, (rung, g) in enumerate(e.groupby("rung")):
        fig.add_trace(go.Bar(name=f"{rung} ({'embeddings only' if rung == 'L5' else 'L6 + embeddings'})", x=g.encoder + " / " + g["head"],
                             y=g.tuned_mean_f1, marker_color=[PHASES["Text & GPU"], PHASES["RAG"]][i % 2]))
    if lad is not None:
        l6 = lad[(lad.cohort == "common_emb") & (lad.rung == "L6")]
        if len(l6):
            _ref(fig, float(l6.mean_fold_score.max()), f"best L6 without text {l6.mean_fold_score.max():.3f}", PHASES["Modelling"])
    fig.update_layout(barmode="group", yaxis_title="mean walk-forward macro-F1", yaxis_range=[0.25, 0.42])
    return _h(fig, 360, "Text embeddings alone (L5) and added to everything else (L7)")


def forest(rows: list[tuple[str, float, float, float, str]], title: str, ref: float = 0.0, threshold: float | None = None, h: int = 300):
    if not rows:
        return None
    fig = go.Figure()
    for lab, v, lo, hi, c in rows:
        fig.add_trace(go.Scatter(x=[v], y=[lab], mode="markers", marker=dict(size=13, color=c),
                                 error_x=dict(type="data", symmetric=False, array=[hi - v], arrayminus=[v - lo], color=c, thickness=2.5),
                                 hovertemplate=f"{lab}<br>{v:+.4f} [{lo:+.4f}, {hi:+.4f}]<extra></extra>", showlegend=False))
    fig.add_vline(x=ref, line=dict(color="#0F172A", width=1.5))
    if threshold is not None:
        fig.add_vline(x=threshold, line=dict(color=PHASES["Features"], dash="dash"), annotation_text=f"gate +{threshold}")
    fig.update_layout(xaxis_title="difference in macro-F1 (95% CI)", yaxis=dict(autorange="reversed"))
    return _h(fig, h, title)


# =========================================================================================== RAG
def rag_prevalence():
    d = D.csv("rag_prevalence_rows.csv")
    if d is None:
        return None
    g = d.groupby("field").agg(present=("present", "mean")).reset_index().sort_values("present")
    fig = go.Figure(go.Bar(x=g.present, y=g.field.str.replace("_", " "), orientation="h", marker_color=PHASES["RAG"],
                           text=[D.pct(x) for x in g.present], textposition="outside"))
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="share of pre-filled filings where the field was extracted (unvalidated)")
    return _h(fig, 380, "What Qwen extracted from 149 development filings")


# =========================================================================================== evaluation
def finalists_chart():
    rows = []
    for tag, name in (("dev_results.csv", "development 2020-22"), ("locked_results.csv", "locked 2023-26")):
        d = D.csv(tag)
        if d is not None:
            d = d[~d.run_id.str.contains("clean_test") & ~d.run_id.str.contains("STACK")]
            for r in d.itertuples():
                rows.append({"model": r.rung, "period": name, "macro_f1": r.macro_f1, "lo": r.ci_low, "hi": r.ci_high,
                             "leaderboard": r.leaderboard})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    fig = go.Figure()
    for i, (p, g) in enumerate(d.groupby("period")):
        fig.add_trace(go.Bar(name=p, x=g.model, y=g.macro_f1, marker_color=[PHASES["Text & GPU"], PHASES["Evaluation"]][i],
                             error_y=dict(type="data", symmetric=False, array=g.hi - g.macro_f1, arrayminus=g.macro_f1 - g.lo)))
    ch = D.chance_level()
    if ch:
        _ref(fig, ch, f"real chance level {ch:.3f}", "#0F172A")
    fig.update_layout(barmode="group", yaxis_range=[0.2, 0.45], yaxis_title="macro-F1")
    return _h(fig, 360, "Finalists on the development holdout and the locked test")


def chance_chart():
    s = D.csv("skill.csv")
    m = D.meta().get("modelling", {})
    if s is None:
        return None
    fig = go.Figure()
    for mdl in ("L1", "L6", "L7"):
        g = s[s.model == mdl]
        fig.add_trace(go.Bar(name=mdl, x=g.period, y=g.macro_f1, marker_color=MODEL_COLORS[mdl], text=[f"{v:.3f}" for v in g.macro_f1],
                             textposition="outside"))
    ch = D.chance_level()
    for y, lab, c, dash in ((ch, "real chance (shuffled own predictions)", "#0F172A", "dash"),
                            (m.get("permutation_null_mean"), "permutation null", "#94A3B8", "dashdot"),
                            (m.get("majority_baseline_f1"), "majority-class baseline", "#CBD5E1", "dot")):
        if y:
            _ref(fig, y, f"{lab} {y:.3f}", c, dash)
    fig.update_layout(barmode="group", yaxis_range=[0.15, 0.45], yaxis_title="macro-F1", margin=dict(b=70))
    return _h(fig, 430, "Read the scores against the real chance level (~0.33), not the majority baseline")


def skill_chart():
    s = D.csv("skill.csv")
    if s is None:
        return None
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Will the stock MOVE?", "Which WAY? (among movers)"), shared_yaxes=True)
    for j, col in enumerate(("move_auc", "direction_auc")):
        for mdl in ("L1", "L6", "L7"):
            g = s[s.model == mdl]
            fig.add_trace(go.Bar(name=mdl, x=g.period, y=g[col], marker_color=MODEL_COLORS[mdl], showlegend=(j == 0),
                                 error_y=dict(type="data", symmetric=False, array=g[col + "_hi"] - g[col], arrayminus=g[col] - g[col + "_lo"])),
                          row=1, col=j + 1)
        fig.add_hline(y=0.5, line=dict(color="#0F172A", dash="dash", width=2), row=1, col=j + 1)
    fig.update_layout(barmode="group", yaxis_range=[0.4, 0.72], yaxis_title="AUC (0.5 = no information)")
    return _h(fig, 400, None)


def lifts_forest(periods: list[str] | None = None):
    d = D.csv("lifts.csv")
    if d is None:
        return None
    rows = []
    for r in d.itertuples():
        if periods and r.period not in periods:
            continue
        rows.append((f"{r.comparison} | {r.period}", r.lift, r.ci_low, r.ci_high, PERIOD_COLORS.get(r.period, "#6366F1")))
    return forest(rows, "Paired lifts on identical events, by period", h=430)


def confusion(model: str):
    d = D.csv("locked_confusion.csv")
    if d is None:
        return None
    g = d[d.model == model].pivot(index="true", columns="pred", values="events").reindex(index=["Down", "Sideways", "Up"],
                                                                                         columns=["Down", "Sideways", "Up"])
    share = g.div(g.sum(axis=1), axis=0)
    fig = go.Figure(go.Heatmap(z=share.values, x=[f"pred {c}" for c in share.columns], y=[f"true {i}" for i in share.index],
                               colorscale=[[0, "#EEF2FF"], [1, "#4338CA"]], zmin=0, zmax=.6,
                               text=[[f"{share.values[i, j]:.0%}<br>({int(g.values[i, j]):,})" for j in range(3)] for i in range(3)],
                               texttemplate="%{text}", showscale=False))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _h(fig, 330, f"Locked-test confusion matrix, {model} (row shares)")


# =========================================================================================== event study / backtest
def car_paths(groups: list[str], event_set: str = "all"):
    d = D.csv("car_paths.csv")
    if d is None:
        return None
    d = d[(d.event_set == event_set) & d.group.isin(groups)]
    fig = go.Figure()
    for i, (g, x) in enumerate(d.groupby("group")):
        fig.add_trace(go.Scatter(x=x.rel_day, y=x.caar * 100, name=EVENT_NAMES.get(g, g), mode="lines",
                                 line=dict(width=3, color=PALETTE[i % len(PALETTE)])))
    fig.add_vline(x=0, line=dict(color="#0F172A", dash="dot"))
    fig.add_hline(y=0, line=dict(color="#94A3B8"))
    fig.update_layout(xaxis_title="trading days around the filing (day 0)", yaxis_title="cumulative abnormal return, %")
    return _h(fig, 380, "Average cumulative abnormal return around filings (association, not causation)")


def sue_quintiles():
    d = D.csv("sue_quintiles.csv")
    if d is None:
        return None
    c = ["#EF4444", "#F97316", "#94A3B8", "#22C55E", "#15803D"]
    fig = go.Figure(go.Bar(x=[f"Q{int(q)}" + (" (most negative surprise)" if q == 1 else " (most positive)" if q == 5 else "") for q in d.q],
                           y=d["mean"] * 100, marker_color=c[:len(d)],
                           error_y=dict(type="data", array=1.96 * d["sem"] * 100), text=[f"{v * 100:+.2f}%" for v in d["mean"]],
                           textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#0F172A"))
    fig.update_layout(yaxis_title="mean CAR days +2 to +60, %")
    return _h(fig, 360, "Post-earnings drift by earnings-surprise quintile: a reversal, not classic drift")


def backtest_cum(block: str, lag: int, cost: int, legs: list[str]):
    d = D.parquet("backtest_daily.parquet")
    if d is None:
        return None
    x = d[(d.block == block) & (d.lag == lag) & (d.cost_bps == cost)].sort_values("date")
    names = {"long_ret": "long book", "spy_matched": "SPY on the same days", "ls_ret": "long-short (hypothetical)", "long_excess_spy": "long minus SPY"}
    cols = {"long_ret": PHASES["Features"], "spy_matched": "#94A3B8", "ls_ret": PHASES["RAG"], "long_excess_spy": PHASES["Modelling"]}
    fig = go.Figure()
    for leg in legs:
        if leg in x:
            fig.add_trace(go.Scatter(x=pd.to_datetime(x.date), y=(1 + x[leg].fillna(0)).cumprod() - 1, name=names[leg],
                                     line=dict(width=3, color=cols[leg])))
    fig.add_hline(y=0, line=dict(color="#0F172A", width=1))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="cumulative return")
    return _h(fig, 380, f"Backtest, {block} block, lag {lag}, {cost} bps per side (economic-relevance check only)")


def stage_hours():
    d = D.csv("stages.csv")
    if d is None:
        return None
    d = d.dropna(subset=["hours"])
    col = {"CPU": PHASES["Data"], "GPU": PHASES["Text & GPU"], "CPU-GPU": PHASES["RAG"]}
    fig = go.Figure(go.Bar(x=d.stage, y=d.hours * 60, marker_color=[col.get(p, "#94A3B8") for p in d.part],
                           hovertext=d.name + " (" + d.part + ")", hovertemplate="%{x} %{hovertext}<br>%{y:.1f} min<extra></extra>"))
    fig.update_layout(yaxis_title="minutes (wall clock)", yaxis_type="log")
    return _h(fig, 320, "Compute time per stage (blue CPU, amber GPU, pink CPU/GPU; log scale)")


def params_text(js: str) -> str:
    try:
        return ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in json.loads(js).items())
    except Exception:
        return str(js)
