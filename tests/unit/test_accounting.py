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


def test_release_eps_losses_text_tables_undifferentiated_and_multiyear():
    """Real 2004-2011 release formats (early-era run, D-066): loss wording, plain-text tables, EPS without 'diluted',
    five-year summary tables."""
    from secprice.parse.eps_release import USABLE_CONFIDENCE, _text_table_row, parse_release_eps as p
    r = p("For the second quarter the company reported a net loss of $25.3 million, or $0.80 per diluted share, compared "
          "with net income of $0.40 per diluted share a year ago.")
    assert r["eps_diluted"] == -0.80                                  # sign from 'net loss'; '25.3' is not a sentence end
    assert p("Diluted earnings per share were $0.20, compared with a loss of $0.05 per share.")["eps_diluted"] == 0.20
    assert p("Second quarter net loss per diluted share was $0.31.")["eps_diluted"] == -0.31
    assert _text_table_row("Diluted $ 0.08 $ (0.05)") == "Diluted | 0.08 | (0.05)"
    assert _text_table_row("Diluted (0.04) 0.27") == "Diluted | (0.04) | 0.27"   # parenthesis stays with the value
    assert _text_table_row("Diluted 16,983 15,198") is None                      # share counts are not EPS
    t = "Three months ended September 30, 2004\n(In thousands, except per share data)\nEarnings per share\nDiluted $ 0.08 $ (0.05)\n" \
        "Weighted average shares\nDiluted 16,953 15,182"
    assert p(t)["eps_diluted"] == 0.08 and p(t)["parse_confidence"] == "medium_table"
    u = p("Third quarter earnings per share were $0.39, up from $0.33.")
    assert u["eps_diluted"] == 0.39 and u["parse_confidence"] == "medium_undiff" and "medium_undiff" in USABLE_CONFIDENCE
    assert p("We expect earnings per share for the third quarter to come in between $0.55 to $0.59.")["eps_diluted"] is None
    five = "\n".join(["Selected financial data", "2010", "", "2009", "", "2008", "", "2007", "", "2006", "",
                      "Net income per common share - assuming dilution (dollars) | 6.22 | 3.98 | 8.69"])
    assert p(five)["eps_diluted"] is None                             # a multi-year summary, not the quarter


def test_suspect_annual_handles_february_fiscal_year_end_in_leap_years():
    """Regression (full run): FYE '0229' / period 2008-02-29 crashed the year shift."""
    from secprice.parse.eps_release import flag_suspect_annual
    t = pd.to_datetime(["2007-06-10", "2007-09-10", "2007-12-10", "2008-04-01"], utc=True)
    rel = pd.DataFrame({"accession": list("abcd"), "cik": 7, "is_results_release": True, "eps_diluted": [0.5, 0.52, 0.55, 2.1],
                        "parse_confidence": "medium", "period_end": ["2007-05-31", "2007-08-31", "2007-11-30", "2008-02-29"],
                        "public_available_time": t})
    out = flag_suspect_annual(rel, {7: "0229"}).set_index("accession")
    assert out.loc["d", "suspect_annual"] and not out.loc["a":"c", "suspect_annual"].any()


def test_release_cluster_exclusions_cover_same_day_and_later_periodic_filings():
    """D-072 (full run): a 10-Q filed minutes after the earnings 8-K on the same day was public before the release's
    decision time and entered its SUE history; every periodic filing on/after the release within 120 days is excluded."""
    import pandas as pd
    from secprice.stages_data import release_cluster_exclusions
    f = pd.DataFrame({"cik": [1, 1, 1, 1, 2], "accession": ["r1", "q_same_day", "q_later", "q_prior", "q_other_firm"],
                      "form": ["8-K", "10-Q", "10-K", "10-Q", "10-Q"],
                      "filing_date": ["2012-04-20", "2012-04-20", "2012-09-30", "2012-01-30", "2012-04-20"]})
    ex = release_cluster_exclusions(f, {"r1"})
    assert ex == {"r1": {"q_same_day"}}                    # q_later is beyond 120 days; q_prior is earlier; other firm excluded
    assert release_cluster_exclusions(f, {"r1"}, window_days=180) == {"r1": {"q_same_day", "q_later"}}
