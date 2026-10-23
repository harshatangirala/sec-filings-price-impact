"""Hard rule 12 / 15: every model revision is pinned; the Colab install pins equal requirements.lock."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_all_model_revisions_pinned():
    m = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    chrono = m["encoders"]["chrono_bert"]
    assert set(map(str, chrono["vintages"])) == set(chrono["revisions"]), "every ChronoBERT vintage needs a revision"
    sha = re.compile(r"^[0-9a-f]{40}$")
    assert all(sha.match(str(r)) for r in chrono["revisions"].values())
    for sec in ("encoders", "reranker", "llm", "judge", "student", "ner"):
        for name, spec in m[sec].items():
            if "model_id" in spec:
                assert sha.match(str(spec.get("revision", ""))), f"{name} lacks a pinned revision"


def test_notebook_pins_match_lock_file():
    # repository-only check: the notebook builder and the lock file are not part of the notebook-embedded package
    if not (ROOT / "tools" / "build_notebook.py").exists() or not (ROOT / "requirements.lock").exists():
        pytest.skip("repository-only check (builder / lock file not embedded in the notebook)")
    lock = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            k, v = line.strip().split("==")
            lock[k.lower().replace("_", "-")] = v
    src = (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")
    block = src[src.index("PINS = {"):src.index("}", src.index("PINS = {"))]
    pins = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', block))
    assert pins, "PINS block not found"
    for k, v in pins.items():
        assert lock.get(k.lower().replace("_", "-")) == v, f"{k}: notebook {v} vs lock {lock.get(k)}"
