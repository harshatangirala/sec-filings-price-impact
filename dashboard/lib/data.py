"""Cached loaders for the snapshot in dashboard/data (built by build_snapshot.py). Missing files return None, never a guess."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame | None:
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def js(name: str):
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def parquet(name: str) -> pd.DataFrame | None:
    p = DATA / name
    return pd.read_parquet(p) if p.exists() else None


def meta() -> dict:
    return js("meta.json") or {}


@st.cache_data(show_spinner=False)
def figure_index() -> dict:
    p = DATA / "figures" / "figure_index.json"
    if not p.exists():
        return {}
    out = {}
    for f in json.loads(p.read_text(encoding="utf-8")):
        local = DATA / "figures" / f"{f['id']}.png"
        out[f["id"]] = {**f, "local": str(local) if local.exists() and f.get("status") == "ok" else None}
    return out


def report_pdf() -> bytes | None:
    p = DATA / "results_report.pdf"
    return p.read_bytes() if p.exists() else None


def report_md() -> str | None:
    p = DATA / "results_summary.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


# ------------------------------------------------------------------ formatting
def n(x, nd=0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def pct(x, nd=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}%"


def ci(lo, hi, nd=3, signed=True) -> str:
    s = "+" if signed else ""
    return f"[{lo:{s}.{nd}f}, {hi:{s}.{nd}f}]"


# ------------------------------------------------------------------ frequently used facts (all from the snapshot)
def skill_row(period: str, model: str):
    s = csv("skill.csv")
    if s is None:
        return None
    x = s[(s.period == period) & (s.model == model)]
    return None if x.empty else x.iloc[0]


def lift_row(period: str, prefix: str):
    s = csv("lifts.csv")
    if s is None:
        return None
    x = s[(s.period == period) & s.comparison.str.startswith(prefix)]
    return None if x.empty else x.iloc[0]


def chance_level() -> float | None:
    s = csv("skill.csv")
    if s is None:
        return None
    x = s[s.model.isin(["L6", "L7"])]
    return float(x.chance_f1_shuffled.mean()) if len(x) else None


def sue_test():
    h = csv("hypothesis_tests.csv")
    if h is None:
        return None
    x = h[h.test.str.startswith("SUE-conditioned post-earnings drift (Q5-Q1, SUE_release)")]
    return None if x.empty else x.iloc[0]
