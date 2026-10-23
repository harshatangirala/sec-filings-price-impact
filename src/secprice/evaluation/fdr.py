"""Benjamini-Hochberg FDR within pre-declared families (SPEC.md §3 fdr.q)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bh(pvals, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvals, float)
    out_adj = np.full(len(p), np.nan)
    rej = np.zeros(len(p), bool)
    ok = np.isfinite(p)
    if not ok.any():
        return out_adj, rej
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    res = np.empty(m)
    res[order] = adj
    out_adj[ok] = res
    rej[ok] = res <= q
    return out_adj, rej


def apply_families(df: pd.DataFrame, family_col: str = "fdr_family", p_col: str = "p_value", q: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    df["p_adj_bh"] = np.nan
    df["reject_bh"] = False
    for fam, g in df.groupby(family_col):
        adj, rej = bh(g[p_col].values, q)
        df.loc[g.index, "p_adj_bh"] = adj
        df.loc[g.index, "reject_bh"] = rej
    return df
