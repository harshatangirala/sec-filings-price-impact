"""Offline integration smoke (SPEC.md §18.2, gate G1): S00 -> S12 and S20 -> S24 on synthetic fixtures, CPU only,
no network, no 7B model (addendum P1.5). TF-IDF stands in for embeddings."""
import json
from pathlib import Path

import pandas as pd
import pytest

from secprice import pipeline as P
from secprice.testing.fixtures import make_world, offline_overrides


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline")
    w = make_world(root)
    return root, w


def test_offline_smoke_end_to_end(world):
    root, w = world
    services = {"fixture_root": w["sec"], "factors": w["factors"]}
    stages = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S20", "S21", "S24"]
    res = P.run(stages, "smoke", offline_overrides(), data_root=root, hot_root=root / "_hot", services=services)
    assert all(v in ("done", "skipped") for v in res.values()), res
    rep = root / "smoke" / "reports" / "results_summary.md"
    assert rep.exists() and "Smallest defensible claim" in rep.read_text(encoding="utf-8")
    assert (root / "smoke" / "reports" / "results_report.html").exists()
    la = pd.read_csv(root / "smoke" / "reports" / "leakage_audit.csv")
    assert (la.status != "fail").all()
    st = P.status("smoke", offline_overrides(), data_root=root)
    assert set(st[st.stage.isin(stages)].status) == {"done"}
