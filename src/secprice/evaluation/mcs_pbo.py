"""Model Confidence Set and PBO (CSCV) - diagnostics only; they never replace or rescue walk-forward and
locked-test evidence (SPEC.md §9.2). Each diagnostic records which loss streams and configurations entered it."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, reps: int = 1000, seed: int = 0) -> dict:
    """losses: rows = time blocks (months), columns = models; lower is better."""
    L = losses.dropna(axis=0, how="any")
    if L.shape[0] < 6 or L.shape[1] < 2:
        return {"included": list(L.columns), "pvalues": {}, "note": "too few blocks/models for MCS", "n_blocks": int(L.shape[0])}
    try:
        from arch.bootstrap import MCS
        m = MCS(L.values, size=size, reps=reps, block_size=max(1, int(np.sqrt(L.shape[0]))), seed=seed)
        m.compute()
        inc = [L.columns[i] for i in m.included]
        pv = {L.columns[int(i)]: float(p) for i, p in zip(m.pvalues.index, m.pvalues.iloc[:, 0])}
        return {"included": inc, "pvalues": pv, "n_blocks": int(L.shape[0]), "method": "arch.bootstrap.MCS (Hansen-Lunde-Nason)"}
    except Exception as e:
        return {"included": list(L.columns), "pvalues": {}, "note": f"MCS unavailable: {e}", "n_blocks": int(L.shape[0])}


def pbo_cscv(perf: pd.DataFrame, n_splits: int = 8) -> dict:
    """perf: rows = time blocks, columns = configurations; higher is better. Bailey et al. CSCV."""
    P = perf.dropna(axis=1, how="any")
    T, N = P.shape
    if N < 2 or T < n_splits:
        return {"pbo": None, "note": f"need >=2 configs and >= {n_splits} blocks (have {N}, {T})", "n_configs": int(N)}
    groups = np.array_split(np.arange(T), n_splits)
    logits = []
    for comb in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([groups[i] for i in comb])
        oos_idx = np.concatenate([groups[i] for i in range(n_splits) if i not in comb])
        is_perf = P.iloc[is_idx].mean()
        oos_perf = P.iloc[oos_idx].mean()
        best = is_perf.idxmax()
        rank = (oos_perf.rank(pct=True)[best] * N) / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": int(len(logits)), "n_configs": int(N),
            "n_blocks": int(T), "logit_median": float(np.median(logits))}


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, skew: float = 0.0, kurt: float = 3.0,
                    sr_var_trials: float | None = None) -> float | None:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado) of a per-period Sharpe `sr` given the number of trials."""
    from scipy.stats import norm
    if n_obs < 3 or n_trials < 1 or not np.isfinite(sr):
        return None
    v = sr_var_trials if sr_var_trials is not None else 1.0 / max(n_obs - 1, 1)
    e = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - e) * norm.ppf(1 - 1.0 / n_trials) + e * norm.ppf(1 - 1.0 / (n_trials * np.e))) if n_trials > 1 else 0.0
    den = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / den))
