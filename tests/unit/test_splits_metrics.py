"""Retrieval (#13), splits (#14), planted leaks (#15), cohorts (#16), metrics (#17), bootstrap (#18), resilience (#19),
config (#20); gold-set integrity (addendum P1.6); backtest entry rules."""
import json

import numpy as np
import pandas as pd
import pytest

from secprice.config import config_hash, load_config
from secprice.evaluation.bootstrap import macro_f1_ci, paired_macro_f1_diff
from secprice.evaluation.metrics import classification_metrics, f1_from_cm, regression_metrics
from secprice.rag.core import historical_candidates, retrieval_guard
from secprice.splits.walkforward import assign_roles, make_blocks


def test_historical_retrieval_never_current_or_future():
    assert retrieval_guard()
    now = pd.Timestamp("2015-06-01 12:00", tz="America/New_York")
    ch = pd.DataFrame({"chunk_id": ["a", "b"], "cik": [1, 1], "accession": ["x", "y"],
                       "public_time": [now, now - pd.Timedelta(seconds=1)], "text": ["", ""]})
    assert historical_candidates(ch, 1, now, []).chunk_id.tolist() == ["b"]       # strict '<'


def _cfg():
    return load_config("smoke")


def test_walk_forward_purge_embargo_and_date_disjoint():
    cfg = _cfg()
    cfg["splits"].update({"min_train_years": 1, "val_block_years": 1, "dev_start": "2023-01-01", "locked_test_start": "2023-07-01",
                          "embargo_days": 30})
    cfg["study"]["start"] = "2021-01-01"
    b = make_blocks(cfg, pd.Timestamp("2023-12-31"))
    assert [x.start for x in b["folds"]] == [pd.Timestamp("2022-01-01")]
    d = pd.to_datetime(["2021-03-01", "2021-12-10", "2021-12-28", "2022-02-01", "2023-02-01"])
    ev = pd.DataFrame({"event_id": list("abcde"), "decision_date": d, "cik": [1, 2, 1, 2, 1], "ff12_industry": "Other"})
    exits = pd.Series(pd.to_datetime(["2021-04-01", "2021-12-20", "2022-01-20", "2022-03-01", "2023-03-01"]))
    sp = assign_roles(ev, b, cfg, exits)
    wf = sp[sp.scheme == "walk_forward"].set_index("event_id").role
    assert wf["a"] == "train" and wf["b"] == "embargo" and wf["c"] == "purged" and wf["d"] == "val"
    tr_dates = set(ev[ev.event_id.isin(wf[wf == "train"].index)].decision_date.dt.date)
    va_dates = set(ev[ev.event_id.isin(wf[wf == "val"].index)].decision_date.dt.date)
    assert not tr_dates & va_dates


def test_planted_leaks_all_caught():
    from secprice.splits.leakage import planted_leaks
    et = "America/New_York"
    ev = pd.DataFrame({"event_id": ["e1", "e2"], "cik": [1, 1], "accessions": [["a1"], ["a2"]],
                       "decision_time": [pd.Timestamp("2020-01-10 17:00", tz=et), pd.Timestamp("2020-02-10 17:00", tz=et)],
                       "public_available_time": [pd.Timestamp("2020-01-10 16:35", tz=et), pd.Timestamp("2020-02-10 16:35", tz=et)],
                       "cluster_id": ["C1", "C1"], "cluster_role": ["release", "periodic_after_release"],
                       "earnings_type": ["earnings_announcement_verified", None]})
    tg = pd.DataFrame({"event_id": ["e1", "e2"], "target_def": "T_trade", "horizon": 5,
                       "exit_date": pd.to_datetime(["2020-01-17", "2020-02-18"])})
    F = pd.DataFrame({"event_id": ["e1", "e2"]})
    res = planted_leaks(F, ev, tg, pd.DataFrame(columns=["event_id", "source_accession", "source_public_time"]), retrieval_guard)
    assert len(res) == 4 and all(r["caught"] for r in res), res


def test_common_cohort_identical_across_models():
    from secprice.models.ladder import Ladder
    cfg = _cfg()
    ev = pd.DataFrame({"event_id": list("abcd"), "decision_date": pd.date_range("2021-01-01", periods=4), "cik": 1})
    F = pd.DataFrame({"event_id": list("abcd"), "px_ret_20": [0.1, np.nan, 0.2, 0.3], "px_vol_20": [0.1, 0.1, 0.2, 0.3],
                      "acc_log_assets": [1, 1, np.nan, 1], "txt_has_text": [True, True, True, False], "overlap_flag_h20": False})
    ev["overlap_flag_h20"] = False
    lad = Ladder(cfg, ev, F, pd.DataFrame(columns=["event_id", "target_def", "horizon", "label3", "weight_uniq"]),
                 pd.DataFrame(columns=["event_id", "scheme", "fold", "role"]))
    ids = pd.Index(list("abcd"))
    m1 = lad.cohort(["L1", "L3", "L4", "L6"], [None], ids)
    m2 = lad.cohort(["L6", "L1", "L4", "L3"], [None], ids)
    assert m1.tolist() == m2.tolist() == [True, False, False, False]


def test_metrics_hand_computed_and_nulls():
    y = np.array([0, 0, 1, 1, 2, 2])
    yhat = np.array([0, 1, 1, 1, 2, 0])
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (y, yhat), 1)
    f = f1_from_cm(cm)
    # class0 P=1/2 R=1/2 F=.5 ; class1 P=2/3 R=1 F=.8 ; class2 P=1 R=1/2 F=2/3
    assert f["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert f["accuracy"] == pytest.approx(4 / 6)
    P = np.eye(3)[yhat] * 0.8 + 0.2 / 3
    m = classification_metrics(y, P, np.ones(3) / 3, yhat)
    assert m["macro_f1"] == pytest.approx(f["macro_f1"])
    m2 = classification_metrics(np.array([0, 0, 1]), np.tile([0.5, 0.3, 0.2], (3, 1)), np.ones(3) / 3)
    assert m2["auc_up"] is None and m2["auc_macro_ovr"] is None            # class absent -> null, never 0
    r = regression_metrics(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4]), hist_mean=np.array([2.5] * 4))
    assert r["r2"] == pytest.approx(1.0) and r["oos_r2_vs_hist_mean"] == pytest.approx(1.0)


def test_block_bootstrap_resamples_months():
    d = pd.to_datetime(["2020-01-05"] * 50 + ["2020-02-05"] * 50)
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    ci = macro_f1_ci(y, y, d, n_boot=200, seed=0)
    assert ci["n_months"] == 2 and ci["value"] == pytest.approx(1.0)
    diff = paired_macro_f1_diff(y, y, y, d, n_boot=100)
    assert diff["value"] == 0 and diff["ci_low"] == diff["ci_high"] == 0


def test_resume_does_not_duplicate(tmp_path):
    from secprice.parse.worker import run_firm
    (tmp_path / "ledger").mkdir()
    pd.DataFrame({"accession": ["x"]}).to_parquet(tmp_path / "ledger" / "0000000001.parquet")
    out = run_firm({"cik": 1, "out_dir": str(tmp_path), "filings": []})
    assert out["skipped"] is True


def test_checkpoint_reload_reproduces_predictions(tmp_path):
    import pickle
    from secprice.models.heads import Head
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    h = Head("lightgbm", {"num_leaves": 8}, seed=1).fit(X[:200], y[:200], np.ones(200), X[200:250], y[200:250], np.ones(50))
    p1 = h.predict_proba(X[250:])
    pickle.dump(h, open(tmp_path / "h.pkl", "wb"))
    h2 = pickle.load(open(tmp_path / "h.pkl", "rb"))
    np.testing.assert_allclose(p1, h2.predict_proba(X[250:]))


def test_config_validation_and_hash_stability():
    a = load_config("smoke")
    b = load_config("smoke")
    assert config_hash(a) == config_hash(b)
    c = load_config("smoke", {"labels": {"k": 0.25}})
    assert config_hash(c) != config_hash(a)
    assert c["_meta"]["owner_overrides"][0]["key"] == "labels.k"
    with pytest.raises(ValueError):
        load_config("smoke", {"sec": {"max_rps": 20}})
    with pytest.raises(ValueError):
        load_config("smoke", {"study": {"start": "2024-01-01", "end": "2023-01-01"}})


def test_gold_partitions_disjoint_and_test_blind(tmp_path):
    from secprice.config import load_yaml, repo_root
    from secprice.gold.tools import GoldTestLock, export_templates, sample_gold
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(60)], "decision_date": pd.date_range("2010-01-01", periods=60, freq="W"),
                       "year": 2010, "event_type": "8K", "categories": [["earnings"]] * 60, "ff12_industry": "Other",
                       "accessions": [[f"a{i}"] for i in range(60)], "forms": [["8-K"]] * 60, "filing_date": "2010-01-01"})
    s = sample_gold(ev, pd.Timestamp("2020-01-01"), {"gold_dev": 5, "gold_val": 10, "gold_test": 10}, 0)
    assert not s.event_id.duplicated().any() and set(s.gold_split) == {"gold_dev", "gold_val", "gold_test"}
    qb = load_yaml(repo_root() / "configs" / "question_bank_v1.yaml")
    pre = {e: {f: {"present": True} for f in qb["fields"]} for e in s.event_id}
    export_templates(s, qb, None, tmp_path, prefill=pre)
    t = pd.read_csv(tmp_path / "gold_gold_test_filings_template.csv").fillna("")
    assert (t.llm_prefill_json == "").all()                                     # gold-test labelled blind
    v = pd.read_csv(tmp_path / "gold_gold_val_filings_template.csv").fillna("")
    assert (v.llm_prefill_json != "").all()
    lock = GoldTestLock(tmp_path / "lock.json")
    lock.consume("h")
    with pytest.raises(RuntimeError):
        lock.consume("h")


def test_backtest_entry_and_one_position_per_firm():
    from secprice.backtest.engine import positions
    sig = pd.DataFrame({"event_id": ["a", "b", "c"], "ticker": ["X", "X", "Y"], "entry_idx": [10, 12, 10], "score": [0.9, 0.95, -0.9]})
    p = positions(sig, -0.5, 0.5, 5).set_index("event_id")
    assert p.loc["a", "exit"] == 12 and bool(p.loc["a", "exit_at_open"])           # replaced at the new entry open
    assert p.loc["b", "entry"] == 12 and p.loc["c", "side"] == -1
    p1 = positions(sig, -0.5, 0.5, 5, lag=1).set_index("event_id")
    assert p1.loc["c", "entry"] == 11
