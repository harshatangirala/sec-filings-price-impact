"""SPEC.md §18.1 #5-#7, #10 and addendum P0.2 acceptance tests (exact returns, interval-matched benchmark)."""
import numpy as np
import pandas as pd
import pytest

from secprice.market.prices import build_series
from secprice.market.returns import cc_return, oc_return
from secprice.targets.build import Estimator, label3, uniqueness_weights


def _bars(closes, opens=None, dates=None):
    n = len(closes)
    d = pd.date_range("2024-01-01", periods=n, freq="B") if dates is None else dates
    o = closes if opens is None else opens
    return pd.DataFrame({"date": d, "open": o, "high": np.maximum(o, closes), "low": np.minimum(o, closes),
                         "close": closes, "volume": 1e6, "source": "t"})


def test_split_raw_vs_adjusted_and_tri():
    # raw prices 100 -> 102 -> (2:1 split) 51.5 -> 52 ; split-adjusted basis = 50, 51, 51.5, 52
    adj = np.array([50.0, 51.0, 51.5, 52.0])
    d = pd.date_range("2024-01-01", periods=4, freq="B")
    acts = pd.DataFrame({"date": [d[2]], "type": ["split"], "value": [2.0]})
    s = build_series(_bars(adj, dates=d), acts)
    np.testing.assert_allclose(s.close_raw.values, [100, 102, 51.5, 52])
    np.testing.assert_allclose(s.total_return_index.values[-1] / s.total_return_index.values[0], 52 / 50)


def test_dividend_total_return_and_ex_date_rule():
    c = np.array([100.0, 101.0, 99.0, 100.0])        # ex-date at index 2 with a 2.0 dividend
    o = np.array([100.0, 100.5, 99.5, 99.0])
    d = pd.date_range("2024-01-01", periods=4, freq="B")
    s = build_series(_bars(c, o, d), pd.DataFrame({"date": [d[2]], "type": ["dividend"], "value": [2.0]}))
    tri = s.total_return_index.values
    assert cc_return(tri, 0, 3) == pytest.approx((101 / 100) * ((99 + 2) / 101) * (100 / 99) - 1)
    # bought at the open of the day before the ex-date: entitled to the dividend
    assert oc_return(o, c, tri, 1, 3) == pytest.approx((101 / 100.5) * ((99 + 2) / 101) * (100 / 99) - 1)
    # bought at the open OF the ex-date: not entitled
    assert oc_return(o, c, tri, 2, 3) == pytest.approx((99 / 99.5) * (100 / 99) - 1)
    # overnight gap before entry is excluded from an open->close holding return
    assert oc_return(o, c, tri, 3, 3) == pytest.approx(100 / 99 - 1)


def test_benchmark_interval_matches_stock_interval():
    """Stock and benchmark legs use the same formula on the same (entry open -> exit close) interval."""
    c = np.array([10.0, 11.0, 12.0, 13.0])
    o = np.array([9.5, 10.5, 11.5, 12.5])
    tri = c / c[0]
    for e, x in ((1, 1), (1, 3), (2, 3)):
        assert oc_return(o, c, tri, e, x) == pytest.approx(c[x] / o[e] - 1)


def test_market_model_known_beta():
    rng = np.random.default_rng(0)
    rm = rng.normal(0, 0.01, 400)
    r = 0.0002 + 1.5 * rm + rng.normal(0, 0.002, 400)
    cnt, a, b, s = Estimator(r, rm).window(np.array([50]), np.array([349]))
    assert cnt[0] == 300 and b[0] == pytest.approx(1.5, abs=0.05) and s[0] == pytest.approx(0.002, rel=0.2)


def test_carhart_on_synthetic():
    from secprice.targets.build import _carhart_car
    rng = np.random.default_rng(1)
    n = 400
    f = rng.normal(0, 0.01, (n, 4))
    rf = np.full(n, 0.0001)
    r = rf + f @ np.array([1.1, 0.3, -0.2, 0.1]) + rng.normal(0, 0.001, n)
    car = _carhart_car(r, f, rf, np.array([360]), np.array([365]), np.array([50]), np.array([330]), 120)
    assert abs(car[0]) < 0.01


def test_k_band_labels():
    ar = np.array([0.03, -0.03, 0.001, np.nan])
    sig = np.array([0.02, 0.02, 0.02, 0.02])
    lab = label3(ar, sig, 1, 0.5)                      # threshold 0.01
    assert lab[0] == 2 and lab[1] == 0 and lab[2] == 1 and np.isnan(lab[3])
    lab4 = label3(np.array([0.015]), np.array([0.02]), 4, 0.5)   # threshold 0.5*0.02*2 = 0.02 -> sideways
    assert lab4[0] == 1


def test_uniqueness_weights_overlaps():
    w = uniqueness_weights(np.array([0, 0, 10]), np.array([4, 4, 12]))
    assert w[0] == pytest.approx(0.5) and w[2] == pytest.approx(1.0)


def test_overlap_flags_and_clean_cohort():
    from secprice.events.build import _overlaps
    ev = pd.DataFrame({"cik": [1, 1, 1, 2], "trackA_day0_idx": [10, 12, 40, 11]})
    out = _overlaps(ev, {"targets": {"horizons": [1, 5]}, "contam_window_sessions": 5})
    assert out.overlap_flag_h5.tolist() == [True, False, False, False]
    assert out.overlap_flag_h1.tolist() == [False, False, False, False]
