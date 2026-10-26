"""Report analysis added after the 150-firm Colab run: chance level, move-vs-direction skill, paired lifts, figure slots."""
from pathlib import Path

import numpy as np
import pandas as pd

from secprice.evaluation.context import agreement, class_shares, paired_lifts, skill_decomposition
from secprice.report.build import build_html, md_table, resolve_slots_md
from secprice.report.narrative import relabel_stress_audit


def _synthetic(n=6000, move_signal=1.5, dir_signal=0.0, seed=0):
    """Three balanced-ish classes; probabilities that carry information about moving (and optionally about direction)."""
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1, 2], size=n, p=[0.32, 0.36, 0.32])
    moved = (y != 1).astype(float)
    side = 1 / (1 + np.exp(move_signal * (moved - 0.5) + rng.normal(0, 1, n)))        # lower P(side) for movers
    up_share = 0.5 + dir_signal * ((y == 2).astype(float) - 0.5) * moved + rng.normal(0, 0.15, n)
    up_share = np.clip(up_share, 0.02, 0.98)
    p = np.column_stack([(1 - side) * (1 - up_share), side, (1 - side) * up_share])
    d = pd.date_range("2015-01-01", periods=n, freq="6h")
    return pd.DataFrame({"y": y, "yhat": p.argmax(1), "p_down": p[:, 0], "p_side": p[:, 1], "p_up": p[:, 2], "date": d,
                         "event_id": [f"e{i}" for i in range(n)]})


def test_move_signal_without_direction_is_reported_as_such():
    s = skill_decomposition(_synthetic(move_signal=1.5, dir_signal=0.0), n_boot=60, seed=1, n_shuffle=20)
    assert s["move_auc"] > 0.62                                        # it knows WHETHER the stock moves
    assert abs(s["direction_auc"] - 0.5) < 0.035, s                    # but not which way (sampling error of this synthetic draw ~0.01)
    assert s["direction_auc_lo"] < s["direction_auc"] < s["direction_auc_hi"]
    assert abs(s["chance_f1_shuffled"] - 1 / 3) < 0.03                 # chance macro-F1 of 3 balanced classes is ~0.33, not 0.19


def test_direction_signal_is_detected():
    s = skill_decomposition(_synthetic(move_signal=0.0, dir_signal=0.6), n_boot=60, seed=1, n_shuffle=10)
    assert s["direction_auc_lo"] > 0.55, s


def test_agreement_and_class_shares():
    p = _synthetic(n=500)
    a = agreement(p, p)
    assert a["same_class_share"] == 1.0 and a["mean_abs_prob_diff"] == 0.0
    q = p.copy()
    q.loc[q.index[:50], "yhat"] = (q.loc[q.index[:50], "yhat"] + 1) % 3
    assert abs(agreement(p, q)["same_class_share"] - 0.9) < 1e-9
    sh = class_shares(p)
    assert abs(sh["down"] + sh["sideways"] + sh["up"] - 1) < 1e-9 and sh["always_sideways_accuracy"] == sh["sideways"]


def test_paired_lift_of_a_model_with_itself_is_zero():
    p = _synthetic(n=800)
    r = paired_lifts({"A": p, "B": p}, [("A", "B", "A - B")], n_boot=50, seed=0)
    assert len(r) == 1 and abs(r.lift.iloc[0]) < 1e-12 and r.ci_low.iloc[0] <= 0 <= r.ci_high.iloc[0]
    assert paired_lifts({"A": p}, [("A", "B", "A - B")], 10, 0).empty          # a missing partner is skipped, not invented


def test_stress_item_relabelled_only_when_unscored():
    leak = pd.DataFrame({"item": [3, 4], "check": ["a", "b"], "status": ["pass", "pass"], "detail": ["x", "stress_firm folds: 5"]})
    out = relabel_stress_audit(leak, pd.DataFrame({"run_id": ["common|L6|enet|-|T_trade|h1|tuned"]}))
    assert out.loc[out.item == 4, "status"].iloc[0] == "built, not scored" and out.loc[out.item == 3, "status"].iloc[0] == "pass"
    scored = relabel_stress_audit(leak, pd.DataFrame({"run_id": ["stress_firm|L6|enet"]}))
    assert scored.loc[scored.item == 4, "status"].iloc[0] == "pass"


def test_figure_slots_resolve_in_markdown_and_html(tmp_path: Path):
    png = tmp_path / "F99.png"
    png.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"))
    figs = [{"id": "F99", "title": "demo", "path": str(png), "caption": "how to read", "status": "ok"},
            {"id": "F98", "title": "missing", "path": None, "caption": "", "status": "not produced: needs gold labels"}]
    md = "## Section\n\n[[FIG:F99]]\n\n[[FIG:F98]]\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = resolve_slots_md(md, figs)
    assert "![F99 demo](figures/F99.png)" in out and "F98 missing: not produced: needs gold labels" in out and "[[FIG" not in out
    page = build_html(md, figs)
    assert "data:image/png;base64," in page and "not produced: needs gold labels" in page and "[[FIG" not in page
    assert "<table>" in page
    assert md_table(pd.DataFrame({"x": [1, 2]})).startswith("| x |")
