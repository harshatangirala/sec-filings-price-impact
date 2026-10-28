"""Window 2: the presentation slide deck (navigation, overview grid, speaker notes, HTML handout download)."""
from __future__ import annotations

import streamlit as st

from lib import slides as SL
from lib.theme import bullets_html, esc

DECK = SL.build()
N = len(DECK)
if "slide" not in st.session_state:
    st.session_state.slide = 1


def _go(delta: int):
    st.session_state.slide = min(N, max(1, st.session_state.slide + delta))


def _set(i: int):
    st.session_state.slide = i
    st.session_state.overview = False


top = st.columns([.7, .7, 3.2, 1.1, 1.1, 1.3], vertical_alignment="center")
top[0].button("◀ Prev", on_click=_go, args=(-1,), width="stretch", disabled=st.session_state.slide <= 1, key="deck_prev")
top[1].button("Next ▶", on_click=_go, args=(1,), width="stretch", disabled=st.session_state.slide >= N, key="deck_next")
with top[2]:
    st.select_slider("Slide", options=list(range(1, N + 1)), key="slide", label_visibility="collapsed",
                     format_func=lambda i: f"{i} · {DECK[i - 1]['title'][:38]}")
overview = top[3].toggle("Overview", key="overview")
notes = top[4].toggle("Notes", key="notes", value=True)
with top[5]:
    if st.button("⬇ Handout (HTML)", width="stretch", key="mk_handout"):
        st.session_state.handout = SL.handout_html(DECK)
    if st.session_state.get("handout"):
        st.download_button("Save handout", st.session_state.handout, "sec_filings_study_slides.html", "text/html", width="stretch", key="dl_handout")

st.progress(st.session_state.slide / N, text=f"Slide {st.session_state.slide} of {N}")

if overview:
    cols = st.columns(4)
    for i, s in enumerate(DECK):
        with cols[i % 4]:
            st.markdown(f"<div class='thumb' style='background:linear-gradient(135deg,{s['color']},{s['color']}CC)'>"
                        f"<small>{s['n']:02d} · {esc(s['kicker'])}</small>{esc(s['title'])}</div>", unsafe_allow_html=True)
            st.button("Open", key=f"open_{i}", on_click=_set, args=(i + 1,), width="stretch")
else:
    s = DECK[st.session_state.slide - 1]
    st.markdown(f"<div class='slide'><div class='band' style='background:linear-gradient(115deg,{s['color']} 0%,{s['color']}C0 100%)'>"
                f"<div class='kicker'>{esc(s['kicker'])} · {s['n']}/{N}</div><h1>{esc(s['title'])}</h1>"
                f"<div class='lead'>{esc(s.get('lead', ''))}</div></div></div>", unsafe_allow_html=True)
    with st.container(border=True):
        if s.get("layout") == "title" or not s.get("bullets"):
            s["visual"]()
            if s.get("bullets"):
                st.markdown(bullets_html(s["bullets"], s["color"]), unsafe_allow_html=True)
        else:
            c1, c2 = st.columns([1, 1.35], gap="large")
            with c1:
                st.markdown(bullets_html(s["bullets"], s["color"]), unsafe_allow_html=True)
            with c2:
                s["visual"]()
    if notes:
        st.markdown(f"<div class='notes'><b>Speaker notes.</b> {esc(s.get('notes', ''))}</div>", unsafe_allow_html=True)
