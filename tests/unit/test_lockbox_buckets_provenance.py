"""Lockbox single use (§6.1, M12), index buckets (D-056), provenance stamps (hard rule 15), backtest exposure (§11.7)."""
import json

import numpy as np
import pandas as pd
import pytest

from secprice.backtest.engine import exposure_stats
from secprice.evaluation.buckets import composition, event_buckets, expand, pooled_vs_bucket, score_predictions
from secprice.stages_model import lockbox_guard


def _cfg(unlock=True, reuse=False):
    return {"lockbox": {"unlock_locked_test": unlock, "acknowledge_reuse": reuse}}


def test_lockbox_single_use(tmp_path):
    frozen = {"g4_hash": "abc"}
    with pytest.raises(RuntimeError, match="sealed"):
        lockbox_guard(_cfg(unlock=False), tmp_path, frozen)
    with pytest.raises(RuntimeError, match="G4 hash"):
        lockbox_guard(_cfg(), tmp_path, frozen)                     # no G4 freeze yet
    (tmp_path / "g4_hash.txt").write_text("other")
    with pytest.raises(RuntimeError, match="G4 hash"):
        lockbox_guard(_cfg(), tmp_path, frozen)                     # frozen architecture changed after G4
    (tmp_path / "g4_hash.txt").write_text("abc")
    assert lockbox_guard(_cfg(), tmp_path, frozen) is False         # first and only scoring
    (tmp_path / "lockbox_log.json").write_text(json.dumps({"scored_at": "2026-10-23"}))
    with pytest.raises(RuntimeError, match="already scored"):
        lockbox_guard(_cfg(), tmp_path, frozen)                     # second scoring refused
    assert lockbox_guard(_cfg(reuse=True), tmp_path, frozen) is True   # explicit reuse -> relabelled as development


def _universe():
    return pd.DataFrame({"cik": [1, 2, 3], "ever_sp500": [True, True, False], "sp_bucket": ["SP500", "SP500", "NonSP500"],
                         "sp500_intervals": [json.dumps([["2010-01-01", None]]), json.dumps([["2010-01-01", "2022-12-31"]]),
                                             json.dumps([])]})


def _events():
    d = pd.to_datetime(["2022-06-01", "2023-06-01", "2023-06-01", "2022-06-01"])
    return pd.DataFrame({"event_id": ["a", "b", "c", "d"], "cik": [1, 2, 3, 2], "decision_date": d,
                         "event_type": ["8K", "8K", "periodic_10Q", "periodic_10K"], "earnings_type": [None] * 4,
                         "ts_quality": ["A"] * 4})


def test_event_buckets_firm_level_and_point_in_time():
    bk = event_buckets(_events(), _universe()).set_index("event_id")
    assert bk.sp_bucket.to_dict() == {"a": "SP500", "b": "SP500", "c": "NonSP500", "d": "SP500"}
    # firm 2 left the index on 2022-12-31: its 2023 event is in the S&P bucket (firm level) but not a member at the event
    assert bk.sp500_member_at_event.to_dict() == {"a": True, "b": False, "c": False, "d": True}
    x = expand(_events(), bk.reset_index())
    assert len(x) == 8 and (x.sp_bucket == "Combined").sum() == 4
    comp = composition(_events(), bk.reset_index(), _universe()).set_index("sp_bucket")
    assert comp.loc["Combined", "events"] == 4 and comp.loc["SP500", "firms"] == 2 and comp.loc["NonSP500", "events"] == 1
    assert comp.loc["SP500", "member_at_event_share"] == pytest.approx(2 / 3)


def test_bucket_scores_and_pooled_comparison():
    rng = np.random.default_rng(0)
    n = 300
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "cik": rng.integers(1, 4, n)})
    bk = pd.DataFrame({"event_id": ev.event_id, "cik": ev.cik, "sp_bucket": np.where(ev.cik == 3, "NonSP500", "SP500")})
    y = rng.integers(0, 3, n)
    p = rng.dirichlet([1, 1, 1], n)
    dates = pd.date_range("2015-01-01", periods=n, freq="W")
    preds = pd.DataFrame({"run_id": "r", "event_id": ev.event_id, "y": y, "yhat": p.argmax(1), "date": dates,
                          "p_down": p[:, 0], "p_side": p[:, 1], "p_up": p[:, 2]})
    sc = score_predictions(preds, bk, 50, 0)
    assert list(sc.sp_bucket) == ["Combined", "SP500", "NonSP500"]
    assert sc.set_index("sp_bucket").n_events.to_dict() == {"Combined": n, "SP500": int((ev.cik != 3).sum()),
                                                            "NonSP500": int((ev.cik == 3).sum())}
    same = pooled_vs_bucket(preds, preds, 50, 0)                    # identical predictions -> zero difference
    assert same["pooled_minus_bucket_macro_f1"] == 0 and same["n_events"] == n


def test_exposure_stats_counts_cash_days_and_turnover():
    d = pd.DataFrame({"n_long": [0, 2, 12, 12], "n_short": [0, 3, 0, 11], "turnover_long": [0, 0.5, 0.2, 0.0],
                      "turnover_short": [0, 0, 0, 0.4]})
    e = exposure_stats(d, 10)
    assert e["share_days_long_invested"] == 0.5 and e["share_days_short_in_cash"] == 0.25
    assert e["avg_names_long"] == pytest.approx(26 / 3) and e["turnover_short_daily"] == pytest.approx(0.1)


def test_git_commit_and_code_hash_are_stamped():
    from secprice.storage import code_hash, git_commit
    c = git_commit()
    assert c and (len(c.split("-")[0]) == 40 or c.startswith("embedded:"))
    assert len(code_hash()) == 12
