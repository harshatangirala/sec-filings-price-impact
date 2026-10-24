"""Configuration: load default.yaml, overlay profile and owner overrides, validate, hash (SPEC.md §15)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROFILES = ("smoke", "pilot", "full")
# Sections excluded from the config hash: paths and anything environment-specific (SPEC.md §15).
_HASH_EXCLUDE = {"paths", "_meta"}


def repo_root() -> Path:
    env = os.environ.get("SECPRICE_REPO")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(profile: str | None = None, overrides: dict | None = None, root: Path | None = None) -> dict:
    root = Path(root) if root else repo_root()
    cfg = load_yaml(root / "configs" / "default.yaml")
    profile = profile or cfg.get("run_profile", "smoke")
    if profile not in PROFILES:
        raise ValueError(f"run_profile must be one of {PROFILES}, got {profile!r}")
    prof_path = root / "configs" / "profiles" / f"{profile}.yaml"
    prof = load_yaml(prof_path) if prof_path.exists() else {}
    cfg = deep_merge(cfg, prof)
    cfg["run_profile"] = profile
    applied = []
    if overrides:
        flat_before = _flatten(cfg)
        cfg = deep_merge(cfg, overrides)
        for k, v in _flatten(overrides).items():
            applied.append({"key": k, "before": flat_before.get(k), "after": v})
    cfg["_meta"] = {"repo_root": str(root), "profile_file": str(prof_path), "owner_overrides": applied}
    if is_fast(cfg):
        cfg["models"]["encoders"] = list(cfg["speed"]["fast"]["encoders"])
    validate(cfg)
    return cfg


class ConfigError(ValueError):
    pass


def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def is_fast(cfg: dict) -> bool:
    """speed.mode=fast applies to the full profile only (smoke / pilot keep their own small settings; D-074)."""
    return (cfg.get("speed") or {}).get("mode") == "fast" and cfg.get("run_profile") == "full"


def gpu_heads(cfg: dict) -> list[str]:
    """Heads used by the GPU part (S15): the spec heads, or the fast subset (D-074)."""
    return list(cfg["speed"]["fast"]["heads"]) if is_fast(cfg) else list(cfg["tuning"]["heads"])


def gpu_trials(cfg: dict) -> dict:
    return dict(cfg["speed"]["fast"]["trials"]) if is_fast(cfg) else dict(cfg["tuning"]["trials"])


def gpu_finetune_allowed(cfg: dict) -> bool:
    return bool(cfg["speed"]["fast"]["finetune"]) if is_fast(cfg) else True


def validate(cfg: dict) -> None:
    """Types, ranges and required sections (SPEC.md §15). Secrets are checked only when a stage needs them."""
    for sec in ["run_profile", "seed", "paths", "sec", "universe", "study", "market", "timing", "text",
                "taxonomy_version", "targets", "labels", "splits", "models", "rag", "distill", "tuning",
                "gates", "audit", "bootstrap", "fdr", "trivial_effect_bps", "backtest", "budget", "extensions"]:
        _req(sec in cfg, f"config section missing: {sec}")
    _req(cfg["run_profile"] in PROFILES, "run_profile invalid")
    _req(isinstance(cfg["seed"], int), "seed must be int")
    sp = cfg.get("speed") or {}
    _req(sp.get("mode", "full") in ("full", "fast"), "speed.mode must be 'full' or 'fast'")
    if sp.get("mode") == "fast":
        _req(set(sp["fast"]["heads"]) <= set(cfg["tuning"]["heads"]), "speed.fast.heads must be a subset of tuning.heads")
        _req(all(e in cfg["models"]["encoders"] or e in ("sec_bert", "finbert_tone", "chrono_bert", "bge_base") for e in sp["fast"]["encoders"]),
             "speed.fast.encoders must be core roster encoders")
    s = cfg["sec"]
    _req(0 < float(s["max_rps"]) <= 10, "sec.max_rps must be in (0, 10] (SEC ceiling 10)")
    _req(int(s["max_retries"]) >= 1, "sec.max_retries >= 1")
    st, en = date.fromisoformat(cfg["study"]["start"]), date.fromisoformat(cfg["study"]["end"])
    _req(st < en, "study.start must precede study.end")
    _req(date.fromisoformat(cfg["market"]["warmup_start"]) < st, "market.warmup_start must precede study.start")
    _req(int(cfg["universe"]["n_stocks"]) > 0, "universe.n_stocks > 0")
    hz = cfg["targets"]["horizons"]
    _req(all(isinstance(h, int) and h > 0 for h in hz), "targets.horizons must be positive ints")
    _req(cfg["targets"]["primary"]["horizon"] in hz, "primary horizon must be in targets.horizons")
    _req(cfg["targets"]["primary"]["target_def"] in ("T_trade", "T_info"), "primary target_def invalid")
    _req(float(cfg["labels"]["k"]) > 0, "labels.k > 0")
    sp = cfg["splits"]
    ds, lt = date.fromisoformat(sp["dev_start"]), date.fromisoformat(sp["locked_test_start"])
    _req(st < ds <= lt, "splits: study.start < dev_start <= locked_test_start")
    _req(float(sp["embargo_days"]) >= 0, "splits.embargo_days >= 0")
    _req(int(cfg["contam_window_sessions"]) >= 1, "contam_window_sessions >= 1")
    _req(int(cfg["bootstrap"]["n"]) >= 10, "bootstrap.n >= 10")
    _req(0 < float(cfg["fdr"]["q"]) < 1, "fdr.q in (0,1)")
    for k, v in cfg["extensions"].items():
        _req(isinstance(v, bool), f"extensions.{k} must be bool")
    for k in ("public_lag_minutes", "decision_latency_minutes"):
        _req(float(cfg["timing"][k]) >= 0, f"timing.{k} >= 0")


def hashable_view(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in _HASH_EXCLUDE}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    """Hash of the resolved config, excluding paths and secrets. Stable across key order."""
    return hashlib.sha256(canonical_json(hashable_view(cfg)).encode()).hexdigest()[:16]


def subset_hash(cfg: dict, keys: list[str], extra: Any = None) -> str:
    sub = {k: get(cfg, k) for k in sorted(keys)}
    payload = {"sub": sub, "extra": extra, "profile": cfg.get("run_profile")}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


def load_side_config(cfg: dict, name: str) -> dict:
    root = Path(cfg.get("_meta", {}).get("repo_root") or repo_root())
    return load_yaml(root / "configs" / name)


def text_sha256(path: Path) -> str:
    """Hash of a text file's normalized content (LF line endings, trailing whitespace trimmed), so the repository copy
    and the notebook-embedded copy of the pre-registration hash identically."""
    t = Path(path).read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10)).rstrip() + chr(10)
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
