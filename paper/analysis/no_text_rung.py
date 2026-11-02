"""A rung the pipeline never ran: price + metadata + accounting, with no text features of any kind.

L6 already contains word counts, language change, novelty and TF-IDF, so L7 - L6 only tests language-model embeddings on top of
classical text. This script fits the numbers-only rung with L6's chosen settings on the same folds, and compares it with the stored
L6 and L7 predictions on identical events. Written after the locked test: the locked-period rows are exploratory, and nothing
here selected or changed a frozen model.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import secprice.models.ladder as LD
from secprice.models.ladder import Ladder

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables"
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
SEED, B = 20261102, 2000
PC = ["p_down", "p_side", "p_up"]

LD.RUNGS["NUM"] = ["price", "metadata", "accounting"]
LD.RUNGS["NUM_TXT"] = ["price", "metadata", "accounting", "lexical", "change", "novelty"]   # word features without TF-IDF

cfg = json.load(open(ROOT / "dashboard" / "data" / "meta.json"))["config"]
cfg.setdefault("seed", 42)
ev = pd.read_parquet(g("S07", "events.parquet"))
ev["decision_date"] = pd.to_datetime(ev["decision_time"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
F = pd.read_parquet(g("S09", "features.parquet"))
tg = pd.read_parquet(g("S08", "targets.parquet"), columns=["event_id", "target_def", "horizon", "label3", "weight_uniq"], filters=[("target_def", "=", "T_trade"), ("horizon", "=", 1)])
sp = pd.read_parquet(g("S10", "splits.parquet"))
sp = sp[sp.scheme.isin(["walk_forward", "dev_holdout", "locked_test"])]
lad = Ladder(cfg, ev, F, tg, sp, None, {}, None, None, None)
lab = lad.labelled("T_trade", 1)
params = json.load(open(g("S11", "best_params.json")))["params"]
keep = ["event_id", "y", "date"] + PC + ["yhat"]
p11 = pd.read_parquet(g("S11", "predictions.parquet"), columns=keep + ["run_id"], filters=[("run_id", "in", ["common|L6|enet|-|T_trade|h1|tuned", "common|L6|lightgbm|-|T_trade|h1|tuned", "common|L4|enet|-|T_trade|h1|tuned", "common|L1|enet|-|T_trade|h1|tuned"])])
p15 = pd.read_parquet(g("S15", "predictions.parquet"), columns=keep + ["run_id"], filters=[("run_id", "=", "common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned")])
dev = pd.read_parquet(g("S18", "dev_predictions.parquet"), columns=keep + ["name"]); lock = pd.read_parquet(g("S19", "locked_predictions.parquet"), columns=keep + ["name"])
STORED = {"Walk-forward 2008-2019": {"L6": p11[p11.run_id == "common|L6|enet|-|T_trade|h1|tuned"], "L7": p15, "L1": p11[p11.run_id == "common|L1|enet|-|T_trade|h1|tuned"]},
          "Development 2020-2022": {n: dev[dev.name == n] for n in ("L6", "L7", "L1")}, "Locked test 2023-2026": {n: lock[lock.name == n] for n in ("L6", "L7", "L1")}}
SCHEME = {"Walk-forward 2008-2019": "walk_forward", "Development 2020-2022": "dev_holdout", "Locked test 2023-2026": "locked_test"}


def f1(y, h):
    v = []
    for c in (0, 1, 2):
        tp = ((y == c) & (h == c)).sum(); fp = ((y != c) & (h == c)).sum(); fn = ((y == c) & (h != c)).sum()
        v.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(v))


def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    return (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def met(y, h, P):
    m = y != 1
    return dict(macro_f1=f1(y, h), logloss=float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))), move_auc=auc(m.astype(int), 1 - P[:, 1]), direction_auc=auc((y[m] == 2).astype(int), (P[:, 2] - P[:, 0])[m]))


def pair(a, b, label, per):
    x = a[["event_id", "y", "date", "yhat"] + PC].merge(b[["event_id", "yhat"] + PC], on="event_id", suffixes=("_a", "_b")).dropna(subset=["y"])
    y = x.y.astype(int).to_numpy(); ha, hb = x.yhat_a.to_numpy().astype(int), x.yhat_b.to_numpy().astype(int)
    Pa, Pb = x[[c + "_a" for c in PC]].to_numpy(float), x[[c + "_b" for c in PC]].to_numpy(float)
    mon = pd.to_datetime(x.date).dt.to_period("M").astype(str).to_numpy(); u, inv = np.unique(mon, return_inverse=True); gr = [np.where(inv == i)[0] for i in range(len(u))]
    rng = np.random.default_rng(SEED); pt_a, pt_b = met(y, ha, Pa), met(y, hb, Pb); dr = {k: [] for k in pt_a}
    for _ in range(B):
        i = np.concatenate([gr[k] for k in rng.integers(0, len(gr), len(gr))]); ma, mb = met(y[i], ha[i], Pa[i]), met(y[i], hb[i], Pb[i])
        for k in dr: dr[k].append(ma[k] - mb[k])
    return [dict(period=per, comparison=label, metric=k, a=pt_a[k], b=pt_b[k], lift=pt_a[k] - pt_b[k], lo=np.nanpercentile(dr[k], 2.5), hi=np.nanpercentile(dr[k], 97.5), n=len(x)) for k in dr]


rows, levels = [], []
for per, sch in SCHEME.items():
    st = STORED[per]; ids = pd.Index(st["L6"].dropna(subset=["y"]).event_id.unique())
    ids_all = pd.Index(lab.index[lab.index.isin(ids) | True])
    cohort = lab.index[lab.index.isin(F[F.event_id.isin(st["L6"].event_id) | True].event_id)]
    # the training rows are the common cohort's training rows; the evaluation rows are exactly the stored L6 events
    allowed = pd.Index(sorted(set(pd.concat([v.event_id for v in STORED["Walk-forward 2008-2019"].values()]).unique()) | set(ids)))
    base = lab.index[lad.cohort(["L1", "L2", "L3", "L4", "L6"], [None], lab.index)]
    for name, rung, head, pk in (("Numbers only, elastic net", "NUM", "enet", "L6|enet"), ("Numbers only, LightGBM", "NUM", "lightgbm", "L6|lightgbm"), ("Numbers + word features, elastic net", "NUM_TXT", "enet", "L6|enet")):
        r = lad.run(rung, head, params.get(pk), lab, base, sch, "cls", None, keep=True)
        pr = r["preds"]; pr = pr[pr.event_id.isin(ids)]
        pr.to_parquet(OUT / f"_pred_{rung}_{head}_{sch}.parquet")
        y, h, P = pr.y.astype(int).to_numpy(), pr.yhat.to_numpy().astype(int), pr[PC].to_numpy(float)
        levels.append(dict(period=per, model=name, n=len(pr), **met(y, h, P)))
        print(per, name, len(pr), {k: round(v, 4) for k, v in met(y, h, P).items()}, flush=True)
        if name.endswith("elastic net") and rung == "NUM":
            rows += pair(st["L6"], pr, "All classical text (L6 - numbers only, elastic net)", per)
            rows += pair(st["L7"], pr, "Classical text + embeddings (L7 - numbers only)", per)
            rows += pair(pr, st["L1"], "Metadata and accounting (numbers only - L1)", per)
        if name == "Numbers only, LightGBM":
            rows += pair(st["L7"], pr, "Classical text + embeddings (L7 - numbers only, both LightGBM)", per)
        if name.startswith("Numbers + word"):
            rows += pair(st["L6"], pr, "TF-IDF factors (L6 - numbers + word features)", per)
pd.DataFrame(rows).to_csv(OUT / "no_text_lifts.csv", index=False); pd.DataFrame(levels).to_csv(OUT / "no_text_levels.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 70)
print(pd.DataFrame(levels).round(4).to_string()); print(pd.DataFrame(rows).query("metric in ['macro_f1','direction_auc']").round(4).to_string())
for f in OUT.glob("_pred_*.parquet"):
    f.unlink()
