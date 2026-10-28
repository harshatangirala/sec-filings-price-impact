"""SEC filings -> price impact: interactive project dashboard.

    streamlit run dashboard/app.py

Two windows: an interactive flowchart of the whole project (data collection to findings) and a presentation slide deck.
All numbers come from dashboard/data (built from a finished run by dashboard/build_snapshot.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lib import data as D  # noqa: E402
from lib.theme import PHASES, inject_css, register_plotly_template  # noqa: E402

st.set_page_config(page_title="SEC Filings → Price Impact", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
register_plotly_template()
inject_css()

with st.sidebar:
    m = D.meta()
    e = m.get("eda", {})
    st.markdown("### 📈 SEC filings → price impact")
    st.caption("Point-in-time NLP study of 10-K, 10-Q and 8-K filings for 150 S&P 500 stocks, 2000-2026.")
    st.markdown("**Data snapshot**")
    st.caption(f"Built {str(m.get('built_utc', 'n/a'))[:16]} UTC from the finished run  \nconfig `{m.get('config_hash', 'n/a')}` · "
               f"code `{str(m.get('git_commit', 'n/a'))[:20]}`")
    st.markdown("**Phases**")
    st.markdown("".join(f"<div style='margin:2px 0'><span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:{c};"
                        f"margin-right:6px'></span>{p}</div>" for p, c in PHASES.items()), unsafe_allow_html=True)
    pdf = D.report_pdf()
    if pdf:
        st.download_button("📄 Full report (PDF)", pdf, "SEC_filings_price_impact_report.pdf", "application/pdf", width="stretch", key="side_pdf")
    st.caption("Association is not causation; prediction is not profitability.")

pages = [st.Page(str(HERE / "views" / "flow_page.py"), title="Project Flow", icon="🗺️", default=True),
         st.Page(str(HERE / "views" / "deck_page.py"), title="Slide Deck", icon="🎞️")]
st.navigation(pages, position="top").run()
