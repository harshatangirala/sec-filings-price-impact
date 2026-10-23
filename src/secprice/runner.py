"""Stage registry, dependency resolution, skip-if-done, status (SPEC.md §17)."""
from __future__ import annotations

import json
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import config_hash, subset_hash
from .logging_utils import get_logger
from .storage import Storage, utcnow, write_json


@dataclass
class StageSpec:
    id: str
    name: str
    part: str                      # CPU / GPU / CPU-GPU
    deps: list[str]
    config_keys: list[str]
    func: Callable
    outputs: list[str] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    card: str = ""                 # how to read the outputs (stage card)


REGISTRY: dict[str, StageSpec] = {}


def stage(id: str, name: str, part: str, deps: list[str], config_keys: list[str], outputs=None,
          assertions=None, card: str = ""):
    def deco(fn):
        REGISTRY[id] = StageSpec(id, name, part, list(deps), list(config_keys), fn, outputs or [], assertions or [], card)
        return fn
    return deco


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class Context:
    """Passed to each stage function: config, storage, logger, resolved dependency dirs."""

    def __init__(self, runner: "Runner", spec: StageSpec, stage_hash: str):
        self.runner = runner
        self.cfg = runner.cfg
        self.storage: Storage = runner.storage
        self.spec = spec
        self.stage_id = spec.id
        self.stage_hash = stage_hash
        self.out = runner.storage.stage_dir(spec.id, stage_hash)
        self.log = get_logger(spec.id, runner.storage.profile_root / "logs")
        self.started = utcnow()
        self.warnings: list[str] = []
        self.services = runner.services

    def dep(self, stage_id: str) -> Path:
        h = self.runner.stage_hash(stage_id)
        p = self.storage.profile_root / stage_id / h
        if not (p / "_SUCCESS").exists():
            raise RuntimeError(f"dependency {stage_id} has no _SUCCESS for hash {h}; run it first")
        return p

    def dep_or_none(self, stage_id: str) -> Path | None:
        try:
            return self.dep(stage_id)
        except Exception:
            return None

    def write(self, df, name: str, inputs: dict | None = None) -> Path:
        return self.storage.write_parquet(df, self.out / name, self.stage_id, self.stage_hash, self.started,
                                          inputs, self.warnings)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log.warning(msg)


class Runner:
    def __init__(self, cfg: dict, storage: Storage | None = None, services: dict | None = None):
        self.cfg = cfg
        self.storage = storage or Storage(cfg)
        self.services = services or {}
        self._hash_cache: dict[str, str] = {}
        self.log = get_logger("runner", self.storage.profile_root / "logs")
        self.timings_path = self.storage.manifests / "run_manifest.json"

    def stage_hash(self, stage_id: str) -> str:
        if stage_id in self._hash_cache:
            return self._hash_cache[stage_id]
        spec = REGISTRY[stage_id]
        dep_hashes = {d: self.stage_hash(d) for d in spec.deps}
        h = subset_hash(self.cfg, spec.config_keys, extra={"deps": dep_hashes, "stage": stage_id})
        self._hash_cache[stage_id] = h
        return h

    def order(self, stage_ids: list[str]) -> list[str]:
        seen: list[str] = []

        def visit(s: str):
            if s in seen:
                return
            for d in REGISTRY[s].deps:
                visit(d)
            seen.append(s)

        for s in stage_ids:
            visit(s)
        return seen

    def status(self) -> list[dict]:
        rows = []
        for sid in sorted(REGISTRY, key=_stage_key):
            h = self.stage_hash(sid)
            if self.storage.is_success(sid, h):
                st = "done"
            elif self.storage.other_successes(sid, h):
                st = "stale"
            else:
                st = "pending"
            rows.append({"stage": sid, "name": REGISTRY[sid].name, "part": REGISTRY[sid].part, "hash": h, "status": st})
        return rows

    def run(self, stage_ids: list[str], force: bool | list[str] = False, with_deps: bool = False) -> dict:
        todo = self.order(stage_ids) if with_deps else list(stage_ids)
        results = {}
        manifest = _load_json(self.timings_path) or {"stages": {}}
        manifest.update({"profile": self.cfg["run_profile"], "config_hash": config_hash(self.cfg)})
        for sid in todo:
            spec = REGISTRY[sid]
            h = self.stage_hash(sid)
            forced = force is True or (isinstance(force, list) and sid in force)
            if self.storage.is_success(sid, h) and not forced:
                self.log.info(f"{sid} {spec.name}: done (hash {h}), skipped")
                results[sid] = "skipped"
                continue
            for d in spec.deps:
                if not self.storage.is_success(d, self.stage_hash(d)):
                    raise RuntimeError(f"{sid} needs {d} first (no _SUCCESS); run {d} or pass with_deps=True")
            set_seeds(int(self.cfg["seed"]))
            ctx = Context(self, spec, h)
            self.log.info(f"=== {sid} {spec.name} [{spec.part}] hash {h} ===")
            t0 = time.time()
            gpu_t0 = _gpu_time()
            try:
                info = spec.func(ctx) or {}
            except Exception as e:
                self.log.error(f"{sid} failed: {e}\n{traceback.format_exc()}")
                manifest["stages"][sid] = {"status": "failed", "error": str(e)[:500], "at": utcnow()}
                write_json(self.timings_path, manifest)
                raise
            wall = time.time() - t0
            info = {"wall_s": round(wall, 1), "gpu_active_s": _gpu_time() - gpu_t0, "warnings": ctx.warnings, **info}
            self.storage.mark_success(sid, h, info)
            manifest["stages"][sid] = {"status": "done", "hash": h, "wall_s": round(wall, 1), "at": utcnow(),
                                       "summary": {k: v for k, v in info.items() if k != "warnings"}}
            write_json(self.timings_path, manifest)
            _write_stage_card(self.storage, spec, h, info)
            self.log.info(f"{sid} done in {wall:.1f}s")
            results[sid] = "done"
        return results


def _gpu_time() -> float:
    # GPU-active time is approximated by wall time while CUDA is in use; not a credit measurement (SPEC.md §13.3).
    try:
        import torch
        if torch.cuda.is_available():
            return round(time.time(), 1)
    except Exception:
        pass
    return 0.0


def _load_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _stage_key(s: str):
    return (0, int(s[1:])) if s[1:].isdigit() else (1, s)


def _write_stage_card(storage: Storage, spec: StageSpec, h: str, info: dict) -> None:
    """docs/stage_cards/<stage>.md equivalent written next to the artifacts (purpose, outputs, assertions, how to read)."""
    lines = [f"# {spec.id} {spec.name}", "", f"- Part: {spec.part}", f"- Depends on: {', '.join(spec.deps) or 'none'}",
             f"- Stage hash: `{h}`", f"- Outputs: {', '.join(spec.outputs) or 'see folder'}",
             f"- Assertions: {'; '.join(spec.assertions) or 'none'}", "", "## How to read", spec.card or "-", "",
             "## Run summary", "```json", json.dumps({k: v for k, v in info.items()}, indent=2, default=str)[:6000], "```"]
    d = storage.profile_root / "stage_cards"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{spec.id}.md").write_text("\n".join(lines), encoding="utf-8")
