"""Model provenance: per-fold strict vs retrospective labels (SPEC.md §6.4)."""
from __future__ import annotations

import pandas as pd


def model_cutoffs(roster: dict) -> dict:
    """{model_key: cutoff Timestamp | 'vintage' | 'folds' | None(unknown)}"""
    out = {}
    for grp in ("encoders", "reranker", "llm", "judge", "student", "ner"):
        for k, v in (roster.get(grp) or {}).items():
            c = v.get("stated_cutoff", "unknown")
            out[k] = "vintage" if c == "vintage" else (None if c in (None, "unknown") else pd.Timestamp(c))
    for k in (roster.get("tabular") or {}):
        out[k] = "folds"
    out["lexical"] = "folds"
    out["tfidf"] = "folds"
    return out


def label(cutoff, block_start: pd.Timestamp) -> str:
    if cutoff in ("vintage", "folds"):
        return "strict"
    if cutoff is None:
        return "retrospective"
    return "strict" if cutoff < block_start else "retrospective"


def ledger_frame(roster: dict) -> pd.DataFrame:
    rows = []
    for grp in ("encoders", "reranker", "llm", "judge", "student", "ner"):
        for k, v in (roster.get(grp) or {}).items():
            rows.append({"key": k, "group": grp, "model_id": v.get("model_id") or v.get("model_id_template"),
                         "revision": v.get("revision") or "per-vintage (see models.yaml)", "card_url": v.get("card_url"),
                         "license": v.get("license"), "training_corpus": v.get("training_corpus"),
                         "stated_cutoff": v.get("stated_cutoff"), "fine_tune_data": v.get("fine_tune_data"),
                         "task": v.get("task"), "labels": str(v.get("labels"))})
    for k in ("lightgbm", "catboost", "enet"):
        rows.append({"key": k, "group": "tabular", "model_id": k, "revision": "library version in environment manifest",
                     "stated_cutoff": "trained on our folds only", "task": "head"})
    return pd.DataFrame(rows)


def fold_labels(roster: dict, blocks: dict) -> pd.DataFrame:
    cut = model_cutoffs(roster)
    rows = []
    allb = [(b.name, b.start) for b in blocks["folds"]] + [("dev", blocks["dev"].start), ("locked", blocks["locked"].start)]
    for m, c in cut.items():
        for name, st in allb:
            rows.append({"model": m, "block": name, "block_start": st, "leaderboard": label(c, st)})
    return pd.DataFrame(rows)
