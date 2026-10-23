"""SPEC.md §18.1 #8 (SUE) and #9 (EAR); addendum P1.4 acceptance test (3 / 4 / 8 seasonal differences)."""
import numpy as np
import pandas as pd
import pytest

from secprice.features.accounting import _qkey, compute_sue_table, quarterly_eps, sue_for
from secprice.features.registry import ear_rule_violations, sue_rule_violations

ET = "America/New_York"


def _hist(n_quarters, start_q=2000 * 4):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"qkey": np.arange(start_q, start_q + n_quarters), "value": 1 + rng.normal(0, 0.1, n_quarters)})


@pytest.mark.parametrize("n_prior,expect_ok,expect_n", [(4 + 3, False, 3), (4 + 4, True, 4), (12, True, 8)])
def test_sue_seasonal_difference_counts(n_prior, expect_ok, expect_n):
    h = _hist(n_prior)
    qk = int(h.qkey.max()) + 1
    r = sue_for(1.5, qk, h, 4, 8)
    assert r["n_prior_diffs"] == expect_n                 # the number of prior observations is always recorded
    assert np.isfinite(r["sue"]) == expect_ok
    if not expect_ok:
        assert r["sue_reason"] == "only_3_prior_diffs"     # unavailable, never fabricated or silently shortened


def test_ytd_to_quarterly_and_first_reported():
    pub = lambda s: pd.Timestamp(s).tz_localize(ET)  # noqa: E731
    f = pd.DataFrame([
        # Q1-Q3 direct quarterly facts, FY + 9M YTD in the 10-K -> Q4 derived
        ("EarningsPerShareDiluted", "USD/shares", 1.0, "2020-01-01", "2020-03-31", "a1", pub("2020-05-01 16:00")),
        ("EarningsPerShareDiluted", "USD/shares", 1.1, "2020-04-01", "2020-06-30", "a2", pub("2020-08-01 16:00")),
        ("EarningsPerShareDiluted", "USD/shares", 1.2, "2020-07-01", "2020-09-30", "a3", pub("2020-11-01 16:00")),
        ("EarningsPerShareDiluted", "USD/shares", 3.3, "2020-01-01", "2020-09-30", "a3", pub("2020-11-01 16:00")),
        ("EarningsPerShareDiluted", "USD/shares", 4.6, "2020-01-01", "2020-12-31", "a4", pub("2021-02-20 16:00")),
        # a later restatement of Q1 must not replace the first-reported value
        ("EarningsPerShareDiluted", "USD/shares", 0.7, "2020-01-01", "2020-03-31", "a9", pub("2021-05-01 16:00")),
    ], columns=["tag", "unit", "value", "period_start", "period_end", "accession", "public_available_time"])
    f["cik"], f["form"] = 1, "10-Q"
    q = quarterly_eps(f)
    q1 = q[q.period_end == pd.Timestamp("2020-03-31")]
    assert q1.value.iloc[0] == pytest.approx(1.0) and q1.source_accession.iloc[0] == "a1"
    q4 = q[q.period_end == pd.Timestamp("2020-12-31")]
    assert q4.value.iloc[0] == pytest.approx(1.3) and q4.derivation.iloc[0] == "fy_minus_9m"


def _xbrl_hist(cik=1):
    rows = []
    for k in range(12):
        pe = pd.Timestamp("2017-03-31") + pd.offsets.QuarterEnd(k)
        rows.append({"cik": cik, "qkey": int(_qkey(pd.Series([pe])).iloc[0]), "value": 1 + 0.05 * k,
                     "source_accession": f"x{k}", "public_available_time": (pe + pd.Timedelta(days=40)).tz_localize(ET)})
    return pd.DataFrame(rows)


def test_release_eps_path_and_later_cluster_rejection():
    xh = _xbrl_hist()
    pe = pd.Timestamp("2020-03-31")
    qk = int(_qkey(pd.Series([pe])).iloc[0])
    dec = pd.Timestamp("2020-04-20 17:00").tz_localize(ET)
    # a later 10-Q of the same cluster reports the same quarter AFTER the release: must not enter
    later = pd.DataFrame([{"cik": 1, "qkey": qk, "value": 9.9, "source_accession": "later10q",
                           "public_available_time": pd.Timestamp("2020-05-05 16:00").tz_localize(ET)}])
    ev = pd.DataFrame([{"accession": "rel", "cik": 1, "kind": "release", "decision_time": dec, "eps_q": 1.9, "qkey": qk}])
    sue, src = compute_sue_table(ev, pd.concat([xh, later]), xh.iloc[:0], 4, 8)
    assert sue.eps_q_source.iloc[0] == "release_exhibit" and sue.eps_q.iloc[0] == pytest.approx(1.9)
    assert "later10q" not in set(src.source_accession)
    assert (pd.to_datetime(src.source_public_time) < dec).all()


def test_registry_rejects_later_cluster_sue_source():
    events = pd.DataFrame([
        {"event_id": "e_rel", "accessions": ["rel"], "cluster_id": "C1", "decision_time": pd.Timestamp("2020-04-20 17:00", tz=ET),
         "earnings_type": "earnings_announcement_verified"},
        {"event_id": "e_q", "accessions": ["q10"], "cluster_id": "C1", "decision_time": pd.Timestamp("2020-05-05 17:00", tz=ET),
         "earnings_type": None}])
    bad = pd.DataFrame([{"event_id": "e_rel", "source_accession": "q10", "source_public_time": pd.Timestamp("2020-05-05 16:35", tz=ET)}])
    assert len(sue_rule_violations(bad, events)) == 1


def test_ear_same_cluster_rejected_later_cluster_accepted_after_window():
    events = pd.DataFrame([{"event_id": "e1", "cluster_id": "C1", "decision_time": pd.Timestamp("2020-05-01 09:00", tz=ET)}])
    same = pd.DataFrame([{"event_id": "e1", "ear_src_cluster": "C1", "_avail_ear": pd.Timestamp("2020-04-01", tz="UTC")}])
    assert len(ear_rule_violations(same, events)) == 1
    open_window = pd.DataFrame([{"event_id": "e1", "ear_src_cluster": "C0", "_avail_ear": pd.Timestamp("2020-05-02", tz="UTC")}])
    assert len(ear_rule_violations(open_window, events)) == 1
    ok = pd.DataFrame([{"event_id": "e1", "ear_src_cluster": "C0", "_avail_ear": pd.Timestamp("2020-04-02", tz="UTC")}])
    assert len(ear_rule_violations(ok, events)) == 0


def test_release_eps_parser_formats():
    from secprice.parse.eps_release import parse_release_eps
    assert parse_release_eps("Net income of $106 million, or $.26 per diluted share.")["eps_diluted"] == pytest.approx(0.26)
    assert parse_release_eps("Earnings per share (assuming dilution) were $2.09, a decrease")["eps_diluted"] == pytest.approx(2.09)
    t = "Earnings per share:\nDiluted earnings per share | $ | 0.97 | $ | 0.67"
    r = parse_release_eps(t)
    assert r["eps_diluted"] == pytest.approx(0.97) and r["parse_confidence"] == "high"
    assert parse_release_eps("Diluted earnings per share, excluding specified items, were $1.30")["eps_diluted"] is None


def test_full_year_eps_in_q4_release_is_withheld():
    """Found on real 1999-2003 releases (AXP $5.42 and XOM $3.23 are full-year figures; the quarters were ~$1.4 and $1.01)."""
    from secprice.parse.eps_release import flag_suspect_annual
    t = pd.to_datetime(["1999-04-20", "1999-07-20", "1999-10-20", "2000-01-25", "2000-04-20"], utc=True)
    rel = pd.DataFrame({"accession": list("abcde"), "cik": 1, "is_results_release": True, "eps_diluted": [1.30, 1.41, 1.42, 5.42, 1.44],
                        "parse_confidence": "medium", "period_end": ["1999-03-31", "1999-06-30", "1999-09-30", "1999-12-31", "2000-03-31"],
                        "public_available_time": t})
    out = flag_suspect_annual(rel, {1: "1231"})
    assert out.set_index("accession").suspect_annual.to_dict() == {"a": False, "b": False, "c": False, "d": True, "e": False}
    assert out.set_index("accession").loc["d", "parse_confidence"] == "suspect_annual"
    # a non-Q4 quarter with a big jump is kept (the check applies at the fiscal year end only); too little history is kept
    rel2 = rel.copy(); rel2.loc[3, "period_end"] = "2000-02-15"
    assert not flag_suspect_annual(rel2, {1: "1231"}).suspect_annual.any()
    assert not flag_suspect_annual(rel.iloc[3:], {1: "1231"}).suspect_annual.any()
