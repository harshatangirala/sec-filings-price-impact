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
