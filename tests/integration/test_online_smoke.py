"""Online smoke (SPEC.md §18.2): the smoke chain S00 -> S08 against the live SEC and our price inputs for one firm.
Marked `network` and skipped unless SECPRICE_RUN_NETWORK=1, SEC_USER_AGENT and SECPRICE_NETWORK_INPUTS (folder with the
owner's inputs: ohlcv_1day/, yf_frozen.parquet, hf_census_*.parquet, sp500/Siccodes/factor caches) are set."""
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from secprice import pipeline as P

pytestmark = pytest.mark.network
NEED = os.environ.get("SECPRICE_RUN_NETWORK") == "1" and os.environ.get("SEC_USER_AGENT") and os.environ.get("SECPRICE_NETWORK_INPUTS")


@pytest.mark.skipif(not NEED, reason="set SECPRICE_RUN_NETWORK=1, SEC_USER_AGENT and SECPRICE_NETWORK_INPUTS to run")
def test_online_smoke_one_firm(tmp_path):
    src = Path(os.environ["SECPRICE_NETWORK_INPUTS"])
    inp = tmp_path / "inputs"
    (inp / "ohlcv_1day").mkdir(parents=True)
    for f in src.glob("*.*"):
        if f.is_file():
            shutil.copy2(f, inp / f.name)
    shutil.copy2(src / "ohlcv_1day" / "AAPL.parquet", inp / "ohlcv_1day" / "AAPL.parquet")
    if (src / "lexicons").exists():
        shutil.copytree(src / "lexicons", inp / "lexicons")
    ov = {"universe": {"n_stocks": 1, "fixed_tickers": ["AAPL"]}, "study": {"start": "2022-07-01", "end": "2023-06-30"},
          "splits": {"dev_start": "2023-03-01", "locked_test_start": "2023-05-01"},
          "market": {"warmup_start": "2021-06-01"}, "smoke": {"max_filings_per_firm_form": 4}, "hf": {"qa_enabled": False}}
    res = P.run(["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"], "smoke", ov, data_root=tmp_path,
                hot_root=tmp_path / "_hot")
    assert all(v == "done" for v in res.values()), res
    r = P.make_runner("smoke", ov, tmp_path, tmp_path / "_hot")
    led = pd.read_parquet(tmp_path / "smoke" / "S05" / r.stage_hash("S05") / "parse_ledger.parquet")
    assert len(led) > 0 and (led.status == "ok").mean() > 0.9
    assert led.header_acceptance_raw.notna().all(), "every filing's SGML acceptance time must be read (D-047)"
    ev = pd.read_parquet(tmp_path / "smoke" / "S07" / r.stage_hash("S07") / "events.parquet")
    assert len(ev) > 0 and (ev.ts_quality == "A").all()
