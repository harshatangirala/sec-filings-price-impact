"""Results in context (report section 'How good is the accuracy, really?'): what a macro-F1 means relative to chance, and
whether a three-class model predicts WHETHER a stock moves (magnitude) or WHICH WAY (direction). Pure functions on stored
predictions (columns y, yhat, p_down, p_side, p_up, date); nothing is fitted and nothing here selects a model.

Class coding (SPEC.md section 5.5): 0 = Down, 1 = Sideways, 2 = Up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

PCOLS = ["p_down", "p_side", "p_up"]


def _month_groups(dates) -> list[np.ndarray]:
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    code = (d.dt.year * 12 + d.dt.month).to_numpy()
    _, inv = np.unique(code, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, cuts)


def _auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    if y_bin.min() == y_bin.max():
        return float("nan")
    return float(roc_auc_score(y_bin, score))


def skill_decomposition(p: pd.DataFrame, n_boot: int = 300, seed: int = 0, n_shuffle: int = 100) -> dict:
    """Macro-F1 against its real chance level, and the move / direction split of the probabilities.

    chance_f1_shuffled : macro-F1 of the model's OWN predicted classes after shuffling them against the true labels (keeps the
                         predicted class mix, destroys any information): the honest chance level for this model, about 0.33
                         for three roughly balanced classes (the majority-class baseline of 0.19 is not a fair chance level).
    move_auc           : AUC of P(not Sideways) for 'the stock moved' (true class != Sideways).
    direction_auc      : among events that did move, AUC of P(Up) - P(Down) for 'moved Up' (0.5 = no directional information).
    Confidence intervals come from a month-block bootstrap (events in the same month are correlated).
    """
    y = p["y"].astype(int).to_numpy()
    yhat = p["yhat"].astype(int).to_numpy()
    P = p[PCOLS].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    moved = y != 1
    out = {"n": int(len(p)), "macro_f1": float(f1_score(y, yhat, average="macro", labels=[0, 1, 2], zero_division=0))}
    sh = [f1_score(y, rng.permutation(yhat), average="macro", labels=[0, 1, 2], zero_division=0) for _ in range(n_shuffle)]
    out["chance_f1_shuffled"] = float(np.mean(sh))
    out["chance_f1_shuffled_p95"] = float(np.percentile(sh, 95))
    out["move_auc"] = _auc(moved.astype(int), 1.0 - P[:, 1])
    out["direction_auc"] = _auc((y[moved] == 2).astype(int), (P[:, 2] - P[:, 0])[moved])
    out["n_moved"] = int(moved.sum())
    groups = _month_groups(p["date"].to_numpy())
    mv, dr = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), size=len(groups))])
        yy, PP = y[idx], P[idx]
        m = yy != 1
        mv.append(_auc(m.astype(int), 1.0 - PP[:, 1]))
        dr.append(_auc((yy[m] == 2).astype(int), (PP[:, 2] - PP[:, 0])[m]))
    for key, arr in (("move_auc", mv), ("direction_auc", dr)):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        out[key + "_lo"] = float(np.percentile(a, 2.5)) if len(a) else float("nan")
        out[key + "_hi"] = float(np.percentile(a, 97.5)) if len(a) else float("nan")
    return out


def class_shares(p: pd.DataFrame) -> dict:
    y = p["y"].astype(int).to_numpy()
    s = np.bincount(y, minlength=3) / max(1, len(y))
    return {"down": float(s[0]), "sideways": float(s[1]), "up": float(s[2]),
            "always_sideways_accuracy": float(s[1]), "majority_class_accuracy": float(s.max())}


def agreement(pa: pd.DataFrame, pb: pd.DataFrame) -> dict:
    """Same model re-scored in two runs (e.g. S11 on the CPU stage and S15 on the GPU stage, same rows and folds):
    share of identical predicted classes, mean absolute probability difference and the macro-F1 difference."""
    a, b = pa.set_index("event_id"), pb.set_index("event_id")
    j = a.index.intersection(b.index)
    if len(j) == 0:
        return {"n": 0}
    ya, yb = a.loc[j, "yhat"].astype(int).to_numpy(), b.loc[j, "yhat"].astype(int).to_numpy()
    y = a.loc[j, "y"].astype(int).to_numpy()
    return {"n": int(len(j)), "same_class_share": float(np.mean(ya == yb)),
            "mean_abs_prob_diff": float(np.abs(a.loc[j, PCOLS].to_numpy() - b.loc[j, PCOLS].to_numpy()).mean()),
            "max_abs_prob_diff": float(np.abs(a.loc[j, PCOLS].to_numpy() - b.loc[j, PCOLS].to_numpy()).max()),
            "macro_f1_a": float(f1_score(y, ya, average="macro", labels=[0, 1, 2], zero_division=0)),
            "macro_f1_b": float(f1_score(y, yb, average="macro", labels=[0, 1, 2], zero_division=0))}


def paired_lifts(preds: dict[str, pd.DataFrame], pairs: list[tuple[str, str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    """Paired month-block-bootstrap lift (a - b) of macro-F1 on identical events, for each (a, b, label) whose two
    prediction sets exist."""
    from .bootstrap import paired_macro_f1_diff
    rows = []
    for a, b, label in pairs:
        if a not in preds or b not in preds:
            continue
        pa, pb = preds[a].set_index("event_id"), preds[b].set_index("event_id")
        j = pa.index.intersection(pb.index)
        if len(j) < 30:
            continue
        d = paired_macro_f1_diff(pa.loc[j, "y"].to_numpy(), pa.loc[j, "yhat"].to_numpy(), pb.loc[j, "yhat"].to_numpy(),
                                 pa.loc[j, "date"].to_numpy(), n_boot, seed)
        rows.append({"comparison": label, "a": a, "b": b, "lift": d["value"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                     "p_boot": d["p_boot"], "n_events": int(len(j)), "n_months": d.get("n_months")})
    return pd.DataFrame(rows)
