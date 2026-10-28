"""Headless test of the Streamlit dashboard: both windows, every flowchart step, every slide, overview and notes.
Skipped when Streamlit or the data snapshot is not available."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = Path(__file__).resolve().parents[2] / "dashboard"
pytestmark = pytest.mark.skipif(not (DASH / "data" / "meta.json").exists(), reason="dashboard data snapshot not built")
sys.path.insert(0, str(DASH))


def _errors(at):
    return [str(x.value) for x in at.exception]


def _app():
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, _errors(at)
    return at


def test_flow_window_renders_every_step():
    from lib.flow import IDS
    at = _app()
    for k, sid in enumerate(IDS):
        at.session_state["step"] = sid
        at.run()
        assert not at.exception, (sid, _errors(at))
        assert any(f"STEP {k + 1}<" in md.value for md in at.markdown), sid


def test_flow_tour_buttons_move_between_steps():
    at = _app()
    at.button(key="next_step").click().run()
    assert at.session_state["step"] == "collect" and not at.exception
    at.button(key="prev_step").click().run()
    assert at.session_state["step"] == "question" and not at.exception


def test_deck_window_renders_every_slide_and_overview():
    from lib.slides import build
    at = _app()
    at.switch_page(str(DASH / "views" / "deck_page.py")).run()
    assert not at.exception, _errors(at)
    n = len(build())
    assert n >= 15
    for i in range(1, n + 1):
        at.session_state["slide"] = i
        at.run()
        assert not at.exception, (i, _errors(at))
    assert at.button(key="deck_next").disabled                       # last slide reached
    at.session_state["overview"] = True
    at.run()
    assert not at.exception, _errors(at)


def test_handout_is_self_contained_html():
    from lib.slides import build, handout_html
    html = handout_html(build())
    assert html.startswith("<!doctype html>") and html.count("<section") >= 15 and "plotly" in html
