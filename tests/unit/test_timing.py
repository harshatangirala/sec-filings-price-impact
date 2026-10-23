"""SPEC.md §18.1 #1-#4 and addendum P0.1 / P0.3 acceptance tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secprice.events.timestamps import (acceptance_from_sources, derive_times, header_to_et, json_to_et,
                                        parse_header_acceptance)
from secprice.market.calendar import TradingCalendar

FIX = Path(__file__).resolve().parents[1] / "fixtures"
ET = "America/New_York"


@pytest.fixture(scope="module")
def cal():
    return TradingCalendar.nyse("2019-01-01", "2026-12-31")


def T(s):
    return pd.Timestamp(s).tz_localize(ET)


# ---------------------------------------------------------------- #1 session buckets
def test_buckets_boundaries(cal):
    assert cal.bucket(T("2024-03-05 09:29:59")) == "PRE"
    assert cal.bucket(T("2024-03-05 09:30:00")) == "INTRA"
    assert cal.bucket(T("2024-03-05 15:59:59")) == "INTRA"
    assert cal.bucket(T("2024-03-05 16:00:00")) == "POST"
    assert cal.bucket(T("2024-03-09 12:00:00")) == "NONTRADING"           # Saturday
    assert cal.bucket(T("2024-07-04 10:00:00")) == "NONTRADING"           # holiday


def test_buckets_early_close_and_dst(cal):
    assert cal.is_early_close(cal.index_of_date("2024-11-29"))            # day after Thanksgiving, 13:00 close
    assert cal.bucket(T("2024-11-29 13:30:00")) == "POST"
    assert cal.bucket(T("2024-11-29 12:59:00")) == "INTRA"
    # DST changes: 09:30 ET open on both sides of the switch
    assert cal.bucket(T("2024-03-08 09:31:00")) == "INTRA" and cal.bucket(T("2024-03-11 09:31:00")) == "INTRA"
    assert cal.bucket(T("2024-11-01 09:29:00")) == "PRE" and cal.bucket(T("2024-11-04 09:29:00")) == "PRE"


# ---------------------------------------------------------------- #2 timestamps
def test_sgml_header_fixture_and_json_rule():
    hdr = (FIX / "aapl_8k_20260730_header.html").read_text(encoding="utf-8")
    raw = parse_header_acceptance(hdr)
    assert raw == "20260730163028"
    # the submissions JSON reported this filing as 2026-07-31T00:30:28.000Z (verified 2026-10-23)
    assert json_to_et("2026-07-31T00:30:28.000Z") == header_to_et(raw)
    assert json_to_et("2020-02-27T16:14:21.000Z") == T("2020-02-27 06:14:21")   # EST case
    full = (FIX / "aapl_8k_2002_full.txt").read_text(encoding="utf-8", errors="replace")
    assert header_to_et(parse_header_acceptance(full)) == json_to_et("2002-08-14T00:47:30.000Z")


def test_public_decision_and_quality_b():
    t, q, src, date_only = acceptance_from_sources("2001-02-12T10:00:00.000Z", None, "2001-02-12")
    assert q == "B" and date_only and t == T("2001-02-12 17:30:00")         # midnight placeholder -> assumed after close
    t2, q2, _, _ = acceptance_from_sources("2026-07-31T00:30:28.000Z", None, "2026-07-30")
    assert q2 == "A" and t2 == T("2026-07-30 16:30:28")
    t3, q3, _, _ = acceptance_from_sources("2026-07-31T00:30:28.000Z", None, "2026-07-30", json_rule_ok=False)
    assert q3 == "C"
    d = derive_times(pd.Series([t2]), 5, 30)
    assert d.public_available_time.iloc[0] == T("2026-07-30 16:35:28")
    assert d.decision_time.iloc[0] == T("2026-07-30 17:05:28")
    assert d.public_time_source.iloc[0] == "assumed_lag"                       # addendum P0.3: labelled as an assumption


# ---------------------------------------------------------------- #3 / #4 T_info and T_trade clocks
CASES = {  # acceptance, bucket, expected T_info base date, T_info exit date (h=1), T_trade entry date
    "PRE": ("2024-03-05 07:00:00", "2024-03-04", "2024-03-05", "2024-03-05"),
    "LATE_PRE": ("2024-03-05 09:10:00", "2024-03-04", "2024-03-05", "2024-03-06"),   # decision 09:45 > open -> next day
    "INTRA": ("2024-03-05 11:00:00", None, None, "2024-03-06"),
    "POST": ("2024-03-05 16:30:00", "2024-03-05", "2024-03-06", "2024-03-06"),
    "NONTRADING": ("2024-03-09 12:00:00", "2024-03-08", "2024-03-11", "2024-03-11"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_clocks(cal, case):
    acc, base_d, exit_d, entry_d = CASES[case]
    a = T(acc)
    tt = derive_times(pd.Series([a]), 5, 30).iloc[0]
    b = cal.last_close_at_or_before(a)
    d0 = cal.first_close_after(tt.public_available_time)
    e = cal.first_open_at_or_after(tt.decision_time)
    assert str(cal.date(e).date()) == entry_d
    assert cal.open_time(e) >= tt.decision_time                                  # T_trade entry never before decision
    if base_d:
        assert str(cal.date(b).date()) == base_d and str(cal.date(d0).date()) == exit_d
        assert cal.close_time(b) <= a                                              # base close precedes acceptance


def test_ttrade_no_component_before_decision(cal):
    """Addendum P0.1 acceptance test: with synthetic PRE, POST, NONTRADING and INTRA events no return component used
    as a Track B (T_trade) target occurs before decision time: the first price used is the entry open."""
    for acc, *_ in CASES.values():
        tt = derive_times(pd.Series([T(acc)]), 5, 30).iloc[0]
        e = cal.first_open_at_or_after(tt.decision_time)
        first_price_time = cal.open_time(e)
        assert first_price_time >= tt.decision_time
