"""Writes the out-of-sample predictions the paper's paired tests use to paper/replication, so they can be checked without the run folder."""
import glob
import os
from pathlib import Path

import pandas as pd

R = os.environ.get("RUN_ROOT", r"C:\\Users\\harsh\\Downloads\\sec_report_mini\\full")
OUT = Path(__file__).resolve().parents[1] / "replication"
OUT.mkdir(exist_ok=True)
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
keep = ["event_id", "y", "date", "p_down", "p_side", "p_up", "yhat"]
runs = {"common|L1|enet|-|T_trade|h1|tuned": "L1", "common|L6|enet|-|T_trade|h1|tuned": "L6", "common|L6_nochange|enet|-|T_trade|h1|tuned": "L6_nochange"}
p11 = pd.read_parquet(g("S11", "predictions.parquet"), columns=keep + ["run_id"], filters=[("run_id", "in", list(runs))]).assign(name=lambda d: d.run_id.map(runs), period="walk_forward")
e = {"common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned": "L7", "common_emb|L7|lightgbm|chrono_bert|T_trade|h1|tuned": "L7_lightgbm_chrono", "common_emb|L6|lightgbm|-|T_trade|h1|tuned": "L6_lightgbm"}
p15 = pd.read_parquet(g("S15", "predictions.parquet"), columns=keep + ["run_id"], filters=[("run_id", "in", list(e))]).assign(name=lambda d: d.run_id.map(e), period="walk_forward")
dv = pd.read_parquet(g("S18", "dev_predictions.parquet"), columns=keep + ["name"]).assign(period="development")
lk = pd.read_parquet(g("S19", "locked_predictions.parquet"), columns=keep + ["name"]).assign(period="locked")
x = pd.concat([p11.drop(columns="run_id"), p15.drop(columns="run_id"), dv, lk]).dropna(subset=["y"])
for c in ("p_down", "p_side", "p_up"):
    x[c] = x[c].astype("float32")
x.to_parquet(OUT / "predictions.parquet", compression="zstd", index=False)
print(x.groupby(["period", "name"]).size().to_string()); print(round((OUT / "predictions.parquet").stat().st_size / 1e6, 1), "MB")
