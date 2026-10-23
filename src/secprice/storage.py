"""Artifact storage: DATA_ROOT/<profile>/<stage>/<stage_hash>/, manifests, _SUCCESS markers, atomic writes
(SPEC.md §16, §17.3)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .config import canonical_json, config_hash, file_sha256, get

_CODE_HASH: str | None = None


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    """Git commit of the code that produced an artifact, with a '-dirty' suffix when the package, configs or tests have
    uncommitted changes. In notebook-embedded mode (no .git) the notebook's build stamp (the commit of the embedded code,
    itself '-dirty' if the build had uncommitted code) plus the source hash stand in."""
    env = os.environ.get("SECPRICE_BUILD_COMMIT")
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=here)
        if out.returncode == 0 and out.stdout.strip():
            st = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "tests"], capture_output=True,
                                text=True, timeout=10, cwd=here.parents[1])
            dirty = st.returncode == 0 and bool(st.stdout.strip())
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return f"embedded:{env or 'unknown'}:src-{code_hash()}"


def code_hash() -> str:
    """Hash of the package source with line endings normalised (identical code hashes identically on Windows and Colab)."""
    global _CODE_HASH
    if _CODE_HASH is None:
        h = hashlib.sha256()
        base = Path(__file__).resolve().parent
        for p in sorted(base.rglob("*.py")):
            h.update(p.relative_to(base).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        _CODE_HASH = h.hexdigest()[:12]
    return _CODE_HASH


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=2, default=str, sort_keys=False))


def read_json(path: Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _to_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            sample = out[c].dropna().head(50)
            if len(sample) and sample.map(lambda v: isinstance(v, (list, dict, tuple, set))).any():
                out[c] = out[c].map(lambda v: None if v is None else json.dumps(v, default=str)
                                    if isinstance(v, (list, dict, tuple, set)) else v)
    return out


class Storage:
    def __init__(self, cfg: dict, data_root: str | os.PathLike | None = None, hot_root: str | os.PathLike | None = None):
        self.cfg = cfg
        dr = data_root or os.environ.get(get(cfg, "paths.data_root_env", "DATA_ROOT"))
        if not dr:
            dr = Path.cwd() / "data"
        self.data_root = Path(dr)
        hr = hot_root or os.environ.get(get(cfg, "paths.hot_root_env", "HOT_ROOT")) or (self.data_root / "_hot")
        self.hot_root = Path(hr)
        self.profile = cfg["run_profile"]
        self.profile_root = self.data_root / self.profile
        self.inputs = self.data_root / get(cfg, "paths.inputs_dirname", "inputs")
        for p in (self.profile_root, self.hot_root, self.inputs, self.profile_root / "logs",
                  self.profile_root / "manifests", self.profile_root / "reports"):
            p.mkdir(parents=True, exist_ok=True)
        self.global_config_hash = config_hash(cfg)

    # ---------------- paths
    def stage_dir(self, stage_id: str, stage_hash: str) -> Path:
        p = self.profile_root / stage_id / stage_hash
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports(self) -> Path:
        return self.profile_root / "reports"

    @property
    def manifests(self) -> Path:
        return self.profile_root / "manifests"

    @property
    def figures(self) -> Path:
        p = self.profile_root / "reports" / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------- artifacts
    def manifest_payload(self, stage_id: str, stage_hash: str, rows: int | None, started: str,
                         inputs: dict | None = None, warnings: list | None = None, extra: dict | None = None) -> dict:
        return {
            "stage": stage_id, "stage_hash": stage_hash, "config_hash": self.global_config_hash,
            "git_commit": git_commit(), "code_hash": code_hash(), "package_version": __version__,
            "input_artifact_hashes": inputs or {}, "row_count": rows, "start_time": started,
            "end_time": utcnow(), "warnings": warnings or [], **(extra or {}),
        }

    def write_parquet(self, df: pd.DataFrame, path: Path, stage_id: str, stage_hash: str, started: str,
                      inputs: dict | None = None, warnings: list | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".parquet")
        os.close(fd)
        try:
            _to_parquet_safe(df).to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        man = self.manifest_payload(stage_id, stage_hash, len(df), started, inputs, warnings)
        man["sha256"] = file_sha256(path)
        write_json(path.with_suffix(path.suffix + ".manifest.json"), man)
        return path

    @staticmethod
    def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(path, columns=columns)

    def success_path(self, stage_id: str, stage_hash: str) -> Path:
        return self.profile_root / stage_id / stage_hash / "_SUCCESS"

    def mark_success(self, stage_id: str, stage_hash: str, info: dict) -> None:
        d = self.stage_dir(stage_id, stage_hash)
        outputs = {}
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != "_SUCCESS" and not p.name.endswith(".manifest.json") and p.stat().st_size < 4e9:
                outputs[p.relative_to(d).as_posix()] = p.stat().st_size
        write_json(d / "_SUCCESS", {"stage": stage_id, "stage_hash": stage_hash, "finished": utcnow(),
                                    "config_hash": self.global_config_hash, "outputs": outputs, **info})

    def is_success(self, stage_id: str, stage_hash: str) -> bool:
        return self.success_path(stage_id, stage_hash).exists()

    def other_successes(self, stage_id: str, stage_hash: str) -> list[str]:
        base = self.profile_root / stage_id
        if not base.exists():
            return []
        return [p.name for p in base.iterdir() if p.is_dir() and p.name != stage_hash and (p / "_SUCCESS").exists()]

    def clear_stage(self, stage_id: str, stage_hash: str) -> None:
        d = self.profile_root / stage_id / stage_hash
        if d.exists():
            shutil.rmtree(d)


def df_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes())
    return h.hexdigest()[:16]


def json_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:16]


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
