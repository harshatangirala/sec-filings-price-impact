"""Core metrics (SPEC.md §9.2). Every metric returns None with a reason when it is inapplicable (never 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = (0, 1, 2)          # Down, Sideways, Up
CLASS_NAMES = {0: "down", 1: "sideways", 2: "up"}


def _safe(fn, *a, **k):
    try:
        v = fn(*a, **k)
        return float(v) if v is not None and np.isfinite(v) else None
    except Exception:
        return None


def confusion(y: np.ndarray, yhat: np.ndarray, k: int = 3) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y.astype(int), yhat.astype(int)), 1)
    return cm


def f1_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(0).astype(float)
    true = cm.sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / pred, np.nan)
        r = np.where(true > 0, tp / true, np.nan)
        f = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    f = np.where(true > 0, np.nan_to_num(f), np.nan)
    present = true > 0
    macro = float(np.nanmean(f[present])) if present.any() else None
    weighted = float(np.nansum(f[present] * true[present]) / true[present].sum()) if present.any() else None
    return {"precision": p, "recall": r, "f1": f, "macro_f1": macro, "weighted_f1": weighted,
            "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else None,
            "balanced_accuracy": float(np.nanmean(r[present])) if present.any() else None}


def ece_top(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> float | None:
    if len(y) == 0:
        return None
    conf = proba.max(1)
    pred = proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def reliability(proba: np.ndarray, y: np.ndarray, cls: int, bins: int = 10) -> pd.DataFrame:
    p = proba[:, cls]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "mean_pred": float(p[m].mean()), "frac_pos": float((y[m] == cls).mean()),
                         "n": int(m.sum())})
    return pd.DataFrame(rows)


def decide(proba: np.ndarray, prior: np.ndarray, n_train: int | None = None) -> np.ndarray:
    """Cost-balanced decision rule: argmax p(c|x) / prior(c), prior from the training fold.

    The prior is Laplace-smoothed and a class absent from training is never predicted (otherwise a near-zero prior
    would make that class win every row)."""
    prior = np.asarray(prior, float)
    n = n_train if n_train else 1000
    pr = (prior * n + 1.0) / (n + len(prior))
    score = proba / pr
    score[:, prior <= 0] = -np.inf
    return np.argmax(score, axis=1)


def classification_metrics(y: np.ndarray, proba: np.ndarray, prior: np.ndarray, yhat: np.ndarray | None = None,
                           coverage_levels=(1.0, 0.5, 0.25, 0.1), ece_max: float = 0.10) -> dict:
    from sklearn.metrics import (average_precision_score, cohen_kappa_score, log_loss, matthews_corrcoef,
                                 roc_auc_score)
    y = np.asarray(y).astype(int)
    proba = np.clip(np.asarray(proba, float), 1e-9, 1)
    proba = proba / proba.sum(1, keepdims=True)
    yhat = decide(proba, prior) if yhat is None else yhat
    out: dict = {"n": int(len(y))}
    if len(y) == 0:
        out["null_reason"] = "no_rows"
        return out
    cm = confusion(y, yhat)
    f = f1_from_cm(cm)
    out.update({k: f[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")})
    for c in CLASSES:
        nm = CLASS_NAMES[c]
        out[f"precision_{nm}"] = None if np.isnan(f["precision"][c]) else float(f["precision"][c])
        out[f"recall_{nm}"] = None if np.isnan(f["recall"][c]) else float(f["recall"][c])
        out[f"f1_{nm}"] = None if np.isnan(f["f1"][c]) else float(f["f1"][c])
        pos = (y == c)
        out[f"auc_{nm}"] = _safe(roc_auc_score, pos, proba[:, c]) if 0 < pos.sum() < len(y) else None
        out[f"prauc_{nm}"] = _safe(average_precision_score, pos, proba[:, c]) if pos.any() else None
    present = [c for c in CLASSES if 0 < (y == c).sum() < len(y)]
    out["auc_macro_ovr"] = (float(np.mean([out[f"auc_{CLASS_NAMES[c]}"] for c in present]))
                            if len(present) == 3 and all(out[f"auc_{CLASS_NAMES[c]}"] is not None for c in present) else None)
    if out["auc_macro_ovr"] is None:
        out["auc_macro_ovr_null_reason"] = "a class is absent"
    pr = [out[f"prauc_{CLASS_NAMES[c]}"] for c in CLASSES if out[f"prauc_{CLASS_NAMES[c]}"] is not None]
    out["prauc_macro"] = float(np.mean(pr)) if len(pr) == 3 else None
    out["kappa"] = _safe(cohen_kappa_score, y, yhat)
    out["mcc"] = _safe(matthews_corrcoef, y, yhat)
    out["logloss"] = _safe(log_loss, y, proba, labels=list(CLASSES))
    pri = np.tile(np.clip(prior, 1e-9, 1), (len(y), 1))
    out["logloss_prior"] = _safe(log_loss, y, pri, labels=list(CLASSES))
    onehot = np.eye(3)[y]
    out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    out["ece"] = ece_top(proba, y)
    out["guardrail_logloss_ok"] = (out["logloss"] is not None and out["logloss_prior"] is not None
                                   and out["logloss"] <= out["logloss_prior"])
    out["guardrail_ece_ok"] = out["ece"] is not None and out["ece"] <= ece_max     # pre-registered ece_max (0.10)
    out["cm"] = cm.tolist()
    conf = proba.max(1)
    order = np.argsort(-conf, kind="stable")
    for cov in coverage_levels:
        m = order[: max(1, int(round(cov * len(y))))]
        cmc = confusion(y[m], yhat[m])
        fc = f1_from_cm(cmc)
        out[f"acc_at_{int(cov * 100)}"] = fc["accuracy"]
        out[f"macro_f1_at_{int(cov * 100)}"] = fc["macro_f1"]
    return out


def regression_metrics(y: np.ndarray, yhat: np.ndarray, hist_mean: np.ndarray | None = None,
                       dates: pd.Series | None = None, kind: str = "return") -> dict:
    from scipy.stats import spearmanr
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    out = {"n": int(ok.sum())}
    if len(y) < 3:
        out["null_reason"] = "fewer_than_3_rows"
        return out
    ss_res = np.sum((y - yhat) ** 2)
    out["r2"] = float(1 - ss_res / np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else None
    if hist_mean is not None:
        hm = np.asarray(hist_mean, float)[ok]
        den = np.sum((y - hm) ** 2)
        out["oos_r2_vs_hist_mean"] = float(1 - ss_res / den) if den > 0 else None
    out["mae"] = float(np.mean(np.abs(y - yhat)))
    out["rmse"] = float(np.sqrt(np.mean((y - yhat) ** 2)))
    out["spearman"] = _safe(lambda: spearmanr(y, yhat).statistic)
    out["directional_accuracy"] = float(np.mean(np.sign(y) == np.sign(yhat))) if kind == "return" else None
    if kind == "vol_ratio":
        rv, fv = np.exp(2 * y), np.exp(2 * yhat)
        q = rv / fv - np.log(rv / fv) - 1
        out["qlike"] = float(np.mean(q))
    if dates is not None:
        d = pd.to_datetime(pd.Series(dates).reset_index(drop=True)[ok]).dt.to_period("M")
        df = pd.DataFrame({"m": d.values, "y": y, "p": yhat})
        ics = df.groupby("m").apply(lambda g: spearmanr(g.y, g.p).statistic if len(g) >= 10 and g.p.nunique() > 1 else np.nan,
                                    include_groups=False).dropna()
        out["rank_ic_mean"] = float(ics.mean()) if len(ics) else None
        out["rank_ic_months"] = int(len(ics))
        if len(ics) >= 6 and ics.std() > 0:
            out["rank_ic_icir"] = float(ics.mean() / ics.std())
            out["rank_ic_t"] = float(ics.mean() / ics.std() * np.sqrt(len(ics)))
        else:
            out["rank_ic_icir"] = out["rank_ic_t"] = None
    return out


def power_label(n: int, ci_width: float | None = None, wide: float = 0.10) -> str:
    if n < 30:
        return "warning (n<30)"
    if n < 100:
        return "exploratory (30<=n<100)"
    if ci_width is not None and ci_width > wide:
        return "inconclusive (wide CI)"
    return "adequate"
