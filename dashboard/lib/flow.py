"""The project flowchart: 16 steps from the research question to the findings, drawn with Plotly so a click selects a step."""
from __future__ import annotations

import plotly.graph_objects as go

from .theme import PHASES, tint

# (id, title, one-line summary, phase, stages, icon)
STEPS = [
    ("question", "Research question & design", "Do SEC filings predict the next price reaction? Pre-registered hypotheses, 3 tracks", "Design", "S00, prereg", "🎯"),
    ("collect", "Data collection", "EDGAR filings, XBRL, prices, factors, S&P 500 history, tone dictionary", "Data", "S01-S04", "🗄️"),
    ("eda", "Universe & EDA", "150 S&P 500 firms, 60k events: who files what, when", "Data", "S02, S07", "🔎"),
    ("parse", "Download & parsing", "64k filings to clean text; item sections; parser QA", "Pre-processing", "S05", "📄"),
    ("nlp", "NLP pre-processing", "Masking, numeric filter, chunking, tokenisation", "Pre-processing", "S05, S14", "🧹"),
    ("events", "Events, clocks & labels", "4 timestamps per event; tradable target; Up / Sideways / Down", "Features", "S07-S08", "⏱️"),
    ("features", "Feature engineering", "128 point-in-time features: price, metadata, accounting, text", "Features", "S06, S09", "🧮"),
    ("splits", "Splits & leakage control", "Walk-forward, holdout, locked test; 24-item audit, planted leaks", "Features", "S10", "🛡️"),
    ("baselines", "Baseline models & tuning", "Ablation ladder L0-L6; LightGBM, CatBoost, elastic net; Optuna", "Modelling", "S11", "📈"),
    ("embed", "Text embeddings (GPU)", "ChronoBERT yearly vintages + BGE on 900k chunks", "Text & GPU", "S13-S14", "🧠"),
    ("textmodels", "Text models & fine-tuning gate", "L5 / L7 heads, H4 vintage test, gate G3 decides fine-tuning", "Text & GPU", "S15", "🔬"),
    ("rag", "RAG extraction", "Hybrid retrieval + rerank + Qwen2.5-7B with guardrails", "RAG", "S16-S17", "🧩"),
    ("selection", "Model selection & locked test", "Development holdout, G4 freeze, locked test scored once", "Evaluation", "S18-S19", "🔒"),
    ("eventstudy", "Event study & PEAD", "Abnormal returns around filings; post-earnings drift", "Event study", "S20-S21", "📊"),
    ("backtest", "Backtest", "Does the signal earn anything after costs?", "Event study", "S22", "💹"),
    ("findings", "Findings & next steps", "Answers, smallest defensible claim, known issues", "Findings", "S23-S24", "🏁"),
]
IDS = [s[0] for s in STEPS]
BY_ID = {s[0]: s for s in STEPS}

COLS, W, H, GX, GY = 4, 2.55, 1.25, 0.55, 0.75


def _pos(i: int) -> tuple[float, float]:
    r, c = divmod(i, COLS)
    if r % 2:
        c = COLS - 1 - c                                   # serpentine: odd rows run right to left
    return c * (W + GX), -r * (H + GY)


def _rounded(x0, y0, x1, y1, r=0.18) -> str:
    return (f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} "
            f"L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - r} L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z")


def flowchart(selected: str | None, phase_filter: list[str] | None = None, done: dict | None = None) -> go.Figure:
    fig = go.Figure()
    centers = {}
    for i, (sid, title, summary, phase, stages, icon) in enumerate(STEPS):
        x, y = _pos(i)
        cx, cy = x + W / 2, y + H / 2
        centers[sid] = (cx, cy)
        c = PHASES[phase]
        dim = bool(phase_filter) and phase not in phase_filter
        sel = sid == selected
        fill = c if sel else tint(c, 0.10 if dim else 0.22)
        fig.add_shape(type="path", path=_rounded(x, y, x + W, y + H), fillcolor=fill,
                      line=dict(color=c if not dim else tint(c, .35), width=4 if sel else 2), layer="below")
        tc = "white" if sel else ("#94A3B8" if dim else "#0F172A")
        fig.add_annotation(x=x + 0.14, y=y + H - 0.2, text=f"<b>{i + 1:02d}</b>", showarrow=False, xanchor="left",
                           font=dict(size=12, color="white" if sel else c))
        fig.add_annotation(x=x + W - 0.14, y=y + H - 0.2, text=icon, showarrow=False, xanchor="right", font=dict(size=16))
        fig.add_annotation(x=cx, y=cy + 0.02, text=f"<b>{title}</b>", showarrow=False, font=dict(size=13, color=tc))
        badge = (done or {}).get(sid, "")
        if sid == "eventstudy":
            badge = (badge + "  ·  " if badge else "") + "runs in parallel after 08"
        fig.add_annotation(x=cx, y=y + 0.22, text=f"{stages}{('  ·  ' + badge) if badge else ''}", showarrow=False,
                           font=dict(size=10, color="rgba(255,255,255,.9)" if sel else ("#CBD5E1" if dim else "#475569")))
    # arrows between consecutive steps
    for a, b in zip(IDS[:-1], IDS[1:]):
        (ax, ay), (bx, by) = centers[a], centers[b]
        if abs(ay - by) < 1e-6:                            # same row
            sx = 1 if bx > ax else -1
            fig.add_annotation(x=bx - sx * W / 2, y=by, ax=ax + sx * W / 2, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.2, arrowcolor="#64748B", text="")
        else:                                              # row change: drop down on the same column
            fig.add_annotation(x=bx, y=by + H / 2, ax=ax, ay=ay - H / 2, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.2, arrowcolor="#64748B", text="")
    # invisible-ish click targets
    xs = [centers[s][0] for s in IDS]
    ys = [centers[s][1] for s in IDS]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", customdata=IDS,
                             marker=dict(size=95, symbol="square", color="rgba(255,255,255,0.01)", line=dict(width=0)),
                             hovertext=[f"<b>{s[1]}</b><br>{s[2]}<br><i>{s[3]} · {s[4]}</i>" for s in STEPS],
                             hovertemplate="%{hovertext}<extra>click to open</extra>", showlegend=False))
    span_x = COLS * (W + GX)
    rows = (len(STEPS) + COLS - 1) // COLS
    fig.update_layout(height=150 * rows + 60, margin=dict(l=4, r=4, t=8, b=4), clickmode="event+select", dragmode=False,
                      xaxis=dict(visible=False, range=[-0.2, span_x - GX + 0.2], fixedrange=True),
                      yaxis=dict(visible=False, range=[-(rows - 1) * (H + GY) - 0.25, H + 0.25], fixedrange=True, scaleanchor=None),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(bgcolor="white", font_size=12))
    return fig
