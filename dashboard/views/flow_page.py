"""Window 1: the interactive project flowchart. Click a step (or use the tour buttons) to open everything about it."""
from __future__ import annotations

import streamlit as st

from lib import data as D
from lib import steps
from lib.flow import BY_ID, IDS, STEPS, flowchart
from lib.theme import PHASES, hero, phase_legend

F = steps.facts()
e = F["e"]
l6 = F["l6_lock"]

hero("SEC filings → price impact: the project, end to end",
     "Click any step of the flowchart (or take the guided tour) to see what was done, why, the numbers, interactive evidence and the report figures.",
     [("S&P 500 firms", D.n(e.get("firms"))), ("SEC filings", D.n(e.get("filings"))), ("events", D.n(e.get("events"))),
      ("years", f"{(e.get('event_first') or '')[:4]}-{(e.get('event_last') or '')[:4]}"),
      (f"locked macro-F1 (chance {D.f3(F['chance'])})", D.f3(l6.macro_f1) if l6 is not None else "n/a"),
      ("text lift (locked)", f"{F['text_lock'].lift:+.3f}" if F["text_lock"] is not None else "n/a")])

if "step" not in st.session_state:
    st.session_state.step = IDS[0]


def _go(delta: int):
    i = IDS.index(st.session_state.step)
    st.session_state.step = IDS[(i + delta) % len(IDS)]


def _jump():
    st.session_state.step = st.session_state.jump


st.session_state.jump = st.session_state.step          # keep the jump box in sync with clicks and the tour buttons
phases = st.pills("Highlight phases", list(PHASES), selection_mode="multi", key="phase_filter")
c2, c3, c4 = st.columns([3, .7, .7], vertical_alignment="bottom")
with c2:
    st.selectbox("Jump to a step", IDS, key="jump", on_change=_jump,
                 format_func=lambda s: f"{IDS.index(s) + 1:02d} · {BY_ID[s][1]}")
c3.button("◀ Prev", on_click=_go, args=(-1,), width="stretch", key="prev_step")
c4.button("Next ▶", on_click=_go, args=(1,), width="stretch", key="next_step")

# status badge per step from the stage results
stg = D.csv("stages.csv")
badges = {}
if stg is not None:
    res = dict(zip(stg.stage, stg.result))
    for sid, _, _, _, stages, _ in STEPS:
        ids = []
        for x in [p.strip() for p in stages.split(",")]:
            if x.startswith("S") and "-" in x:
                a, b = x.split("-")
                ids += [f"S{k:02d}" for k in range(int(a[1:]), int(b[1:]) + 1)]
            elif x.startswith("S"):
                ids.append(x)
        r = [res.get(s, "") for s in ids]
        badges[sid] = "✔ done" if r and all(v == "done" for v in r) else ("◐ partly (needs gold labels)" if any("not" in v or "ran" in v for v in r) else "")

fig = flowchart(st.session_state.step, list(phases or []), badges)
ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="flowchart",
                     config={"displayModeBar": False, "scrollZoom": False})
pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) else (ev.selection.points if ev else [])
if pts:
    cd = pts[0].get("customdata")
    cid = cd[0] if isinstance(cd, (list, tuple)) else cd
    if cid in IDS and cid != st.session_state.get("_last_click"):
        st.session_state._last_click = cid
        st.session_state.step = cid
        st.rerun()
phase_legend()
st.caption("Tip: hover a box for its summary, click it to open the step below. Colours mark the project phase. The event study (step 14, "
           "Track A) runs in parallel once the splits (step 08) exist.")
st.divider()
steps.render(st.session_state.step, F)
