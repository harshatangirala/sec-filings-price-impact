"""S10 splits: expanding walk-forward folds -> development holdout -> locked test, with purge + embargo,
stress folds and the pedagogical random-split demo (SPEC.md §6.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Block:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp        # exclusive


def _add_years(t: pd.Timestamp, years: float) -> pd.Timestamp:
    whole = int(years)
    frac = years - whole
    out = t + pd.DateOffset(years=whole)
    if frac:
        out = out + pd.DateOffset(months=int(round(frac * 12)))
    return out


def make_blocks(cfg: dict, usable_cutoff: pd.Timestamp) -> dict:
    sp = cfg["splits"]
    start = pd.Timestamp(cfg["study"]["start"])
    dev_start, lock_start = pd.Timestamp(sp["dev_start"]), pd.Timestamp(sp["locked_test_start"])
    vals = []
    v0 = _add_years(start, float(sp["min_train_years"]))
    i = 0
    while v0 < dev_start:
        v1 = min(_add_years(v0, float(sp["val_block_years"])), dev_start)
        vals.append(Block(f"wf{i}", v0, v1))
        v0, i = v1, i + 1
    return {"folds": vals, "dev": Block("dev", dev_start, lock_start),
            "locked": Block("locked", lock_start, usable_cutoff + pd.Timedelta(days=1)), "start": start}


def assign_roles(ev: pd.DataFrame, blocks: dict, cfg: dict, exit_dates: pd.Series) -> pd.DataFrame:
    """ev: event_id, decision_date (naive date), cik, ff12_industry. exit_dates: max target exit date at the purge
    horizon (T_info / T_trade), aligned to ev. Returns long table: event_id, scheme, fold, role."""
    emb = pd.Timedelta(days=float(cfg["splits"]["embargo_days"]))
    d = pd.to_datetime(ev["decision_date"])
    ex = pd.to_datetime(exit_dates).fillna(pd.Timestamp.max.normalize())
    out = []

    def scheme(name: str, fold: int, test: Block, role_name: str):
        role = np.full(len(ev), "unused", dtype=object)
        before = d < test.start
        purged = before & (ex >= test.start)
        embargo = before & ~purged & (d >= test.start - emb)
        role[before & ~purged & ~embargo] = "train"
        role[purged] = "purged"
        role[embargo] = "embargo"
        role[(d >= test.start) & (d < test.end)] = role_name
        out.append(pd.DataFrame({"event_id": ev["event_id"].values, "scheme": name, "fold": fold, "role": role}))

    for i, b in enumerate(blocks["folds"]):
        scheme("walk_forward", i, b, "val")
    scheme("dev_holdout", 0, blocks["dev"], "dev")
    scheme("locked_test", 0, blocks["locked"], "locked")
    # stress tests on the pre-holdout period (firm- and sector-held-out); time overlap allowed: stress only
    pre = (d < blocks["dev"].start).values
    k = int(cfg["splits"].get("stress_folds", 5))
    rng = np.random.default_rng(int(cfg["seed"]))
    firms = np.array(sorted(ev["cik"].unique()))
    rng.shuffle(firms)
    fmap = {c: i % k for i, c in enumerate(firms)}
    fold_f = ev["cik"].map(fmap).to_numpy()
    for j in range(k):
        role = np.where(~pre, "unused", np.where(fold_f == j, "test", "train"))
        out.append(pd.DataFrame({"event_id": ev["event_id"].values, "scheme": "stress_firm", "fold": j, "role": role}))
    for j, ind in enumerate(sorted(ev["ff12_industry"].fillna("Other").unique())):
        role = np.where(~pre, "unused", np.where(ev["ff12_industry"].fillna("Other").to_numpy() == ind, "test", "train"))
        out.append(pd.DataFrame({"event_id": ev["event_id"].values, "scheme": "stress_sector", "fold": j, "role": role,
                                 "group": ind}))
    rnd = rng.random(len(ev))
    role = np.where(~pre, "unused", np.where(rnd < 0.2, "test", "train"))
    out.append(pd.DataFrame({"event_id": ev["event_id"].values, "scheme": "random_demo", "fold": 0, "role": role}))
    return pd.concat(out, ignore_index=True)


def blocks_frame(blocks: dict) -> pd.DataFrame:
    rows = [{"block": b.name, "kind": "walk_forward_val", "start": b.start, "end_exclusive": b.end} for b in blocks["folds"]]
    rows += [{"block": "dev", "kind": "development_holdout", "start": blocks["dev"].start, "end_exclusive": blocks["dev"].end},
             {"block": "locked", "kind": "locked_test", "start": blocks["locked"].start, "end_exclusive": blocks["locked"].end}]
    return pd.DataFrame(rows)
