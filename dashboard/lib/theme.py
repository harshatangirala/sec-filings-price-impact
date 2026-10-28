"""Colours, CSS and small HTML building blocks. One colour per project phase, used everywhere (flowchart, cards, slides)."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PHASES = {
    "Design": "#6366F1",
    "Data": "#0EA5E9",
    "Pre-processing": "#14B8A6",
    "Features": "#22C55E",
    "Modelling": "#8B5CF6",
    "Text & GPU": "#F59E0B",
    "RAG": "#EC4899",
    "Evaluation": "#EF4444",
    "Event study": "#F97316",
    "Findings": "#334155",
}
CLASS_COLORS = {"Down": "#EF4444", "Sideways": "#94A3B8", "Up": "#22C55E"}
MODEL_COLORS = {"L0": "#CBD5E1", "L1": "#94A3B8", "L2": "#0EA5E9", "L3": "#14B8A6", "L4": "#22C55E", "L5": "#F59E0B",
                "L6": "#6366F1", "L6_nochange": "#A5B4FC", "L7": "#EC4899", "L4_levels": "#86EFAC", "L4_change": "#4ADE80",
                "STACK": "#F97316"}
PERIOD_COLORS = {"walk-forward": "#6366F1", "development": "#F59E0B", "locked": "#EF4444"}
PALETTE = ["#6366F1", "#0EA5E9", "#14B8A6", "#22C55E", "#F59E0B", "#EC4899", "#EF4444", "#F97316", "#8B5CF6", "#334155"]


def tint(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0F172A"),
        colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=48, b=70), hoverlabel=dict(font_size=12),
        xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"), yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(size=15)))
    pio.templates["secprice"] = t
    pio.templates.default = "plotly_white+secprice"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1400px; }
.hero { border-radius: 18px; padding: 26px 30px; color: white; margin-bottom: 14px;
        background: linear-gradient(120deg, #4338CA 0%, #6366F1 35%, #0EA5E9 70%, #14B8A6 100%);
        box-shadow: 0 10px 30px rgba(67,56,202,.25); }
.hero h1 { color: white; font-size: 2.0rem; font-weight: 800; margin: 0 0 4px 0; line-height: 1.15; }
.hero p { color: rgba(255,255,255,.92); margin: 0; font-size: 1.02rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 16px; }
.kpi { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); border-radius: 12px; padding: 10px 12px; }
.kpi .v { font-size: 1.35rem; font-weight: 800; color: white; }
.kpi .l { font-size: .78rem; color: rgba(255,255,255,.88); text-transform: uppercase; letter-spacing: .04em; }
.chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 0 6px 6px 0; }
.card { background: white; border-radius: 14px; padding: 16px 18px; border: 1px solid #E2E8F0; box-shadow: 0 2px 10px rgba(15,23,42,.05);
        margin-bottom: 12px; }
.card h4 { margin: 0 0 6px 0; font-size: 1.0rem; }
.card p, .card li { font-size: .93rem; color: #334155; }
.stephead { border-radius: 16px; padding: 18px 22px; color: white; margin: 6px 0 12px 0; }
.stephead h2 { color: white; margin: 0; font-size: 1.55rem; font-weight: 800; }
.stephead .sub { color: rgba(255,255,255,.92); font-size: .98rem; margin-top: 4px; }
.mcard { border-radius: 12px; padding: 10px 14px; background: white; border: 1px solid #E2E8F0; border-left: 6px solid #6366F1; height: 100%; }
.mcard .v { font-size: 1.3rem; font-weight: 800; color: #0F172A; }
.mcard .l { font-size: .78rem; color: #64748B; text-transform: uppercase; letter-spacing: .03em; }
.mcard .h { font-size: .78rem; color: #475569; margin-top: 2px; }
.verdict { border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; color: #0F172A; border: 1px solid #E2E8F0; }
.slide { border-radius: 22px; background: white; box-shadow: 0 18px 50px rgba(15,23,42,.12); border: 1px solid #E2E8F0;
         padding: 0; overflow: hidden; margin-bottom: 10px; }
.slide .band { padding: 22px 34px 18px 34px; color: white; }
.slide .band .kicker { font-size: .82rem; text-transform: uppercase; letter-spacing: .12em; opacity: .92; font-weight: 700; }
.slide .band h1 { color: white; font-size: 2.05rem; margin: 4px 0 0 0; font-weight: 800; line-height: 1.15; }
.slide .band .lead { color: rgba(255,255,255,.95); font-size: 1.05rem; margin-top: 8px; }
.slide .body { padding: 20px 34px 24px 34px; }
.bul { font-size: 1.05rem; color: #1E293B; line-height: 1.55; margin: 0 0 10px 0; padding-left: 22px; position: relative; }
.bul:before { content: ""; position: absolute; left: 0; top: .55em; width: 10px; height: 10px; border-radius: 3px; background: var(--c, #6366F1); }
.bignum { font-size: 2.6rem; font-weight: 800; line-height: 1.05; }
.bignum-l { font-size: .85rem; color: #64748B; text-transform: uppercase; letter-spacing: .05em; }
.notes { background: #FFFBEB; border: 1px dashed #F59E0B; border-radius: 12px; padding: 12px 16px; color: #78350F; font-size: .93rem; }
.thumb { border-radius: 12px; padding: 12px 14px; color: white; min-height: 92px; margin-bottom: 8px; font-weight: 700; }
.thumb small { display: block; opacity: .9; font-weight: 500; }
.legend-row span { display: inline-block; margin: 0 12px 6px 0; font-size: .85rem; }
.legend-row i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.small-muted { color: #64748B; font-size: .85rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def chip(text: str, color: str, solid: bool = False) -> str:
    bg, fg = (color, "white") if solid else (tint(color, 0.14), color)
    return f"<span class='chip' style='background:{bg};color:{fg};border:1px solid {tint(color, 0.35)}'>{esc(text)}</span>"


def hero(title: str, subtitle: str, kpis: list[tuple[str, str]]) -> None:
    k = "".join(f"<div class='kpi'><div class='v'>{esc(v)}</div><div class='l'>{esc(l)}</div></div>" for l, v in kpis)
    st.markdown(f"<div class='hero'><h1>{esc(title)}</h1><p>{esc(subtitle)}</p><div class='kpis'>{k}</div></div>", unsafe_allow_html=True)


def step_header(num: int, title: str, sub: str, color: str, chips: list[str]) -> None:
    c = "".join(f"<span class='chip' style='background:rgba(255,255,255,.18);color:white;border:1px solid rgba(255,255,255,.35)'>{esc(x)}</span>" for x in chips)
    st.markdown(f"<div class='stephead' style='background:linear-gradient(115deg,{color} 0%,{tint(color, .78)} 100%)'>"
                f"<div style='font-size:.8rem;letter-spacing:.12em;opacity:.9;font-weight:700'>STEP {num}</div><h2>{esc(title)}</h2>"
                f"<div class='sub'>{esc(sub)}</div><div style='margin-top:10px'>{c}</div></div>", unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str, str]], color: str, cols: int = 4) -> None:
    """items: (label, value, help line)."""
    if not items:
        return
    rows = [items[i:i + cols] for i in range(0, len(items), cols)]
    for row in rows:
        cs = st.columns(cols)
        for c, (lab, val, hlp) in zip(cs, row):
            c.markdown(f"<div class='mcard' style='border-left-color:{color}'><div class='l'>{esc(lab)}</div><div class='v'>{esc(val)}</div>"
                       f"<div class='h'>{esc(hlp)}</div></div>", unsafe_allow_html=True)
        st.write("")


def card(title: str, body_html: str, color: str = "#6366F1") -> None:
    st.markdown(f"<div class='card' style='border-top:4px solid {color}'><h4>{esc(title)}</h4>{body_html}</div>", unsafe_allow_html=True)


def bullets_html(items: list[str], color: str) -> str:
    return "".join(f"<div class='bul' style='--c:{color}'>{x}</div>" for x in items)


def phase_legend() -> None:
    st.markdown("<div class='legend-row'>" + "".join(f"<span><i style='background:{c}'></i>{esc(p)}</span>" for p, c in PHASES.items())
                + "</div>", unsafe_allow_html=True)
