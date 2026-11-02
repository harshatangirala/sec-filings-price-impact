"""Is the larger text lift on pre-open filings a text effect, or a change of model class (L6 elastic net versus L7 LightGBM)?

Compares the same head with and without text vectors, by session bucket, on the walk-forward folds (the only period where the
same-head L6 model was scored). Exploratory. Reads stored predictions only.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
OUT = Path(__file__).resolve().parents[1] / "tables"
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
SEED, B = 20261102, 2000


def f1(y, h):
    v = []
    for c in (0, 1, 2):
        tp = ((y == c) & (h == c)).sum(); fp = ((y != c) & (h == c)).sum(); fn = ((y == c) & (h != c)).sum()
        v.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(v))


cols = ["event_id", "y", "date", "yhat", "run_id"]
p15 = pd.read_parquet(g("S15", "predictions.parquet"), columns=cols)
p11 = pd.read_parquet(g("S11", "predictions.parquet"), columns=cols, filters=[("run_id", "=", "common|L6|enet|-|T_trade|h1|tuned")])
b = pd.read_parquet(g("S20", "track_a_cars.parquet"), columns=["event_id", "session_bucket"])
get = lambda r: p15[p15.run_id == r].dropna(subset=["y"]).merge(b, on="event_id")
L6g, L7b, L7c = get("common_emb|L6|lightgbm|-|T_trade|h1|tuned"), get("common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned"), get("common_emb|L7|lightgbm|chrono_bert|T_trade|h1|tuned")
L6e = p11.dropna(subset=["y"]).merge(b, on="event_id")
rows = []
for lab, a, c in (("LightGBM+BGE vs LightGBM", L7b, L6g), ("LightGBM+ChronoBERT vs LightGBM", L7c, L6g), ("LightGBM vs elastic net, no text", L6g, L6e), ("LightGBM+BGE vs elastic net (the frozen pair)", L7b, L6e)):
    for bk, nm in (("PRE", "Before the open"), ("INTRA", "During the session"), ("POST", "After the close")):
        x = a[a.session_bucket == bk].merge(c[["event_id", "yhat"]], on="event_id", suffixes=("_a", "_b"))
        y, ha, hb = x.y.astype(int).to_numpy(), x.yhat_a.to_numpy(), x.yhat_b.to_numpy()
        m = pd.to_datetime(x.date).dt.to_period("M").astype(str).to_numpy(); u, inv = np.unique(m, return_inverse=True); gr = [np.where(inv == i)[0] for i in range(len(u))]
        rng = np.random.default_rng(SEED); dr = []
        for _ in range(B):
            i = np.concatenate([gr[k] for k in rng.integers(0, len(gr), len(gr))]); dr.append(f1(y[i], ha[i]) - f1(y[i], hb[i]))
        rows.append(dict(comparison=lab, bucket=nm, lift=f1(y, ha) - f1(y, hb), lo=np.percentile(dr, 2.5), hi=np.percentile(dr, 97.5), n=len(x)))
d = pd.DataFrame(rows); d.to_csv(OUT / "same_head_by_session.csv", index=False); print(d.round(4).to_string())
