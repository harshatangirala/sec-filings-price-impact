"""S05 stops cleanly at its wall-clock budget (StageIncomplete) and a second run resumes and completes."""
import pytest

from secprice import pipeline as P
from secprice.logging_utils import StageIncomplete
from secprice.testing.fixtures import make_world, offline_overrides


def test_s05_wall_budget_then_resume(tmp_path):
    w = make_world(tmp_path)
    services = {"fixture_root": w["sec"], "factors": w["factors"]}
    P.run(["S00", "S01", "S02", "S03", "S04"], "smoke", offline_overrides(), data_root=tmp_path, hot_root=tmp_path / "_hot",
          services=services)
    ov = offline_overrides()
    ov["sec"] = {**ov.get("sec", {}), "s05_max_wall_hours": 1e-9}
    with pytest.raises(StageIncomplete):
        P.run(["S05"], "smoke", ov, data_root=tmp_path, hot_root=tmp_path / "_hot", services=services)
    res = P.run(["S05"], "smoke", offline_overrides(), data_root=tmp_path, hot_root=tmp_path / "_hot", services=services)
    assert res["S05"] == "done"


def test_s05_rerun_retries_only_failed_filings(tmp_path):
    """A re-run re-fetches only filings whose first attempt ended in a transient error; other rows are untouched."""
    import pandas as pd
    w = make_world(tmp_path)
    services = {"fixture_root": w["sec"], "factors": w["factors"]}
    P.run(["S00", "S01", "S02", "S03", "S04", "S05"], "smoke", offline_overrides(), data_root=tmp_path, hot_root=tmp_path / "_hot",
          services=services)
    r = P.make_runner("smoke", offline_overrides(), tmp_path, tmp_path / "_hot")
    led_p = sorted((tmp_path / "smoke" / "S05" / r.stage_hash("S05") / "ledger").glob("*.parquet"))[0]
    led = pd.read_parquet(led_p)
    victim = led.accession.iloc[0]
    before_other = led[led.accession != victim].reset_index(drop=True)
    led.loc[led.accession == victim, ["status", "error"]] = ["error", "RuntimeError: HTTP 503 after 6 attempts"]
    led.to_parquet(led_p, index=False)
    P.run(["S05"], "smoke", offline_overrides(), data_root=tmp_path, hot_root=tmp_path / "_hot", services=services, force=True)
    after = pd.read_parquet(led_p)
    assert after.loc[after.accession == victim, "status"].iloc[0] == "ok"
    assert len(after) == len(led) and after.accession.is_unique
    pd.testing.assert_frame_equal(after[after.accession != victim].reset_index(drop=True)[before_other.columns], before_other)
