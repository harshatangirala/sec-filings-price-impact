"""Programmatic API used by the CLI and the notebook (SPEC.md §17.2)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from . import stages_data, stages_model, stages_report  # noqa: F401  (register stages)
from .config import load_config, text_sha256
from .runner import REGISTRY, Runner, _stage_key
from .storage import Storage, write_json

def quiet() -> None:
    """Silence known, understood library warnings so real messages stay visible (Colab output)."""
    import warnings
    for cat in (FutureWarning, DeprecationWarning, PendingDeprecationWarning):
        warnings.filterwarnings("ignore", category=cat)
    warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")
    warnings.filterwarnings("ignore", message=".*An input array is constant.*")
    warnings.filterwarnings("ignore", message=".*divide by zero.*")
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
    warnings.filterwarnings("ignore", message=".*covariance of constraints does not have full rank.*")
    warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
    warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0.*")
    try:
        from scipy.stats import ConstantInputWarning
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
    except Exception:
        pass
    try:
        from statsmodels.tools.sm_exceptions import ValueWarning
        warnings.filterwarnings("ignore", category=ValueWarning)
    except Exception:
        pass


quiet()

CPU_PART_1 = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S20", "S21"]
GPU_PART = ["S13", "S14", "S15", "S16", "S17"]
CPU_PART_2 = ["S18", "S19", "S22", "S23", "S24"]


def expand(stages) -> list[str]:
    """'S00', ['S01','S03'], 'S00-S11' -> ordered list."""
    if isinstance(stages, str):
        stages = [stages]
    out = []
    keys = sorted(REGISTRY, key=_stage_key)
    for s in stages:
        if "-" in s:
            a, b = s.split("-")
            out += [k for k in keys if _stage_key(a) <= _stage_key(k) <= _stage_key(b)]
        else:
            out.append(s)
    return out


def make_runner(profile: str | None = None, overrides: dict | None = None, data_root=None, hot_root=None,
                services: dict | None = None) -> Runner:
    cfg = load_config(profile, overrides)
    return Runner(cfg, Storage(cfg, data_root, hot_root), services or {})


def run(stages, profile=None, overrides=None, force=False, data_root=None, hot_root=None, services=None, with_deps=False) -> dict:
    r = make_runner(profile, overrides, data_root, hot_root, services)
    return r.run(expand(stages), force=force, with_deps=with_deps)


def status(profile=None, overrides=None, data_root=None) -> pd.DataFrame:
    return pd.DataFrame(make_runner(profile, overrides, data_root).status())


def prereg_freeze(profile=None, overrides=None, data_root=None, force=False) -> str:
    r = make_runner(profile, overrides, data_root)
    root = Path(r.cfg["_meta"]["repo_root"])
    h = text_sha256(root / "configs" / "preregistration.yaml")
    p = r.storage.manifests / "prereg_hash.txt"
    if p.exists() and p.read_text().strip() != h:
        if not force:
            raise RuntimeError(f"preregistration already frozen with a different hash ({p.read_text().strip()}); "
                               "a change is a protocol deviation: re-run with force=True to record it")
        dev = r.storage.manifests / "prereg_deviations.json"
        log = json.loads(dev.read_text()) if dev.exists() else []
        log.append({"old": p.read_text().strip(), "new": h, "at": pd.Timestamp.utcnow().isoformat()})
        write_json(dev, log)
    p.write_text(h)
    return h


def freeze(profile=None, overrides=None, data_root=None) -> str:
    """G4 freeze is performed by S18; this returns the current G4 hash."""
    r = make_runner(profile, overrides, data_root)
    p = r.storage.manifests / "g4_hash.txt"
    if not p.exists():
        raise RuntimeError("no G4 freeze yet: run S18")
    return p.read_text().strip()


def unlock_test(profile=None, overrides=None, data_root=None, acknowledge_reuse=False, services=None) -> dict:
    ov = dict(overrides or {})
    ov.setdefault("lockbox", {})
    ov["lockbox"] = {**ov["lockbox"], "unlock_locked_test": True, "acknowledge_reuse": bool(acknowledge_reuse)}
    return run(["S19"], profile, ov, force=True, data_root=data_root, services=services)


def preflight(profile=None, overrides=None, data_root=None, hot_root=None) -> dict:
    """CPU preflight (G2) from the S04 filing index: projected events, chunks, storage and GPU hours vs the budget."""
    from .stages_data import preflight_projection
    r = make_runner(profile, overrides, data_root, hot_root)
    p = r.storage.profile_root / "S04" / r.stage_hash("S04") / "filing_manifest.parquet"
    if not p.exists():
        raise RuntimeError("run S04 first: the preflight projects from the filing index")
    return preflight_projection(r.cfg, pd.read_parquet(p), r.storage.data_root, r.storage.hot_root)


def gold(action: str, profile=None, overrides=None, data_root=None, hot_root=None) -> dict:
    """`secprice gold export|import|agreement` (SPEC.md §17.2, §12).
    export: (re)write the stratified sample, CSV templates and annotation guide (S12).
    import: validate the completed files in DATA_ROOT/inputs/gold against the frozen sample, then import them (S12).
    agreement: Cohen's kappa per field on items labelled by two annotators."""
    from . import common as C
    from .gold.tools import cohen_kappa, import_gold
    r = make_runner(profile, overrides, data_root, hot_root)
    sdir = r.storage.profile_root / "S12" / r.stage_hash("S12")
    if action == "export":
        run(["S12"], profile, overrides, force=True, data_root=data_root, hot_root=hot_root)
        return {"templates": sorted(str(p) for p in (sdir / "templates").glob("*"))}
    if not (sdir / "gold_sample.parquet").exists():
        raise RuntimeError("run `secprice gold export` first: the frozen gold sample does not exist yet")
    if action == "import":
        g, errs = import_gold(r.storage.inputs / "gold", pd.read_parquet(sdir / "gold_sample.parquet"), C.question_bank(r.cfg))
        if len(g):
            run(["S12"], profile, overrides, force=True, data_root=data_root, hot_root=hot_root)
        st = json.loads((sdir / "gold_status.json").read_text()) if (sdir / "gold_status.json").exists() else {}
        return {"valid_rows": int(len(g)), "errors": errs, "status": st}
    if action == "agreement":
        p = sdir / "gold_labels.parquet"
        if not p.exists():
            return {"note": "no imported labels; run `secprice gold import` first"}
        g = pd.read_parquet(p)
        out = []
        for f, gf in g.groupby("field"):
            n_ann = gf.groupby("event_id").annotator.nunique()
            both = gf[gf.event_id.isin(n_ann[n_ann > 1].index)]
            if both.annotator.nunique() < 2:
                continue
            a, b = sorted(both.annotator.unique())[:2]
            x = both[both.annotator == a].drop_duplicates("event_id").set_index("event_id").present_b
            y = both[both.annotator == b].drop_duplicates("event_id").set_index("event_id").present_b
            j = x.index.intersection(y.index)
            out.append({"field": f, "n_items": int(len(j)), "kappa_present": cohen_kappa(x.loc[j], y.loc[j])})
        return {"agreement": out or "no items labelled by two annotators (single-rater, provisional)"}
    raise ValueError(f"unknown gold action {action!r}")


def sync(direction: str, src_root, dst_root, profile: str, stages: list[str] | None = None) -> list[str]:
    """Copy stage artifacts (and inputs) between DATA_ROOTs (local <-> Drive)."""
    src, dst = Path(src_root), Path(dst_root)
    if direction == "pull":
        src, dst = dst, src
    copied = []
    for s in stages or sorted(REGISTRY, key=_stage_key):
        a = src / profile / s
        if a.exists():
            shutil.copytree(a, dst / profile / s, dirs_exist_ok=True)
            copied.append(s)
    for sub in ("manifests", "reports"):
        a = src / profile / sub
        if a.exists():
            shutil.copytree(a, dst / profile / sub, dirs_exist_ok=True)
    return copied
