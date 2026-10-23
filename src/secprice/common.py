"""Shared helpers for stages: artifact loading, clients, calendar/panel construction."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_side_config
from .events.build import events_from_parquet
from .market.calendar import TradingCalendar
from .market.prices import PricePanel


def sec_client(ctx):
    """Real SEC client (User-Agent from env) or the fixture client when services['fixture_root'] is set."""
    from .sec.client import FixtureSECClient, SECClient
    fr = ctx.services.get("fixture_root")
    if fr:
        return FixtureSECClient(fr)
    return SECClient.from_config(ctx.cfg, ctx.storage.hot_root / "sec_cache", ctx.log)


def roster(cfg) -> dict:
    return load_side_config(cfg, "models.yaml")


def taxonomy(cfg):
    from .parse.taxonomy import Taxonomy8K
    return Taxonomy8K(load_side_config(cfg, f"{cfg['taxonomy_version']}.yaml"))


def question_bank(cfg) -> dict:
    return load_side_config(cfg, "question_bank_v1.yaml")


def read(ctx, stage: str, name: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(ctx.dep(stage) / name, columns=columns)


def read_json(ctx, stage: str, name: str):
    return json.loads((ctx.dep(stage) / name).read_text(encoding="utf-8"))


def decode_lists(df: pd.DataFrame, cols) -> pd.DataFrame:
    """List columns are stored as JSON strings in Parquet (storage._to_parquet_safe); decode them on read."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: json.loads(v) if isinstance(v, str) and v[:1] == "[" else
                              (list(v) if isinstance(v, (list, tuple, np.ndarray)) else []))
    return df


def manifest(ctx) -> pd.DataFrame:
    return decode_lists(read(ctx, "S04", "filing_manifest.parquet"), ["items_list"])


def calendar(ctx) -> TradingCalendar:
    return TradingCalendar.from_frame(read(ctx, "S03", "calendar.parquet"))


def panel_and_spy(ctx, cal: TradingCalendar):
    px = read(ctx, "S03", "prices.parquet")
    panel = PricePanel(px[px.ticker != "SPY"], cal)
    spy = PricePanel(px[px.ticker == "SPY"], cal)["SPY"]
    return panel, spy


def universe(ctx, selected_only: bool = True) -> pd.DataFrame:
    u = read(ctx, "S02", "universe.parquet")
    return u[u.selected] if selected_only else u


def events(ctx) -> pd.DataFrame:
    ev = events_from_parquet(read(ctx, "S07", "events.parquet"))
    ev["decision_date"] = pd.to_datetime(ev["decision_time"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return ev


def event_buckets(ctx) -> pd.DataFrame:
    """event_id -> sp_bucket (S&P 500 vs non-S&P 500, firm level) and point-in-time membership (D-056)."""
    from .evaluation.buckets import event_buckets as _eb
    return _eb(events(ctx), universe(ctx))


def factors(ctx) -> pd.DataFrame:
    return read(ctx, "S03", "factors.parquet")


def usable_cutoff(ctx) -> dict:
    return read_json(ctx, "S07", "usable_cutoff.json")


def env_manifest() -> dict:
    import importlib
    pk = {}
    for m in ["numpy", "pandas", "pyarrow", "sklearn", "scipy", "lightgbm", "catboost", "optuna", "statsmodels", "arch",
              "torch", "transformers", "sentence_transformers", "pandas_market_calendars", "yfinance", "lxml", "yaml",
              "matplotlib", "reportlab", "lmformatenforcer", "huggingface_hub", "jsonschema"]:
        try:
            x = importlib.import_module(m)
            pk[m] = getattr(x, "__version__", "?")
        except Exception:
            pk[m] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
                   "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}
    except Exception:
        pass
    return {"python": sys.version, "platform": platform.platform(), "packages": pk, "gpu": gpu,
            "cpu_count": os.cpu_count(), "in_colab": "google.colab" in sys.modules}


def disk_free_gb(path: Path) -> float | None:
    import shutil
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except Exception:
        return None
