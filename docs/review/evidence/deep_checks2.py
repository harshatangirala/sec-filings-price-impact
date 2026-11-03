"""Follow-ups to deep_checks.py: (a) direction skill by year, (b) a one-feature volatility baseline for magnitude,
(c) what the 'accounting availability' filter really requires, (d) the calendar-year feature. Stored outputs only."""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
OUT = Path(__file__).resolve().parent
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)
F = {}

feat = pd.read_parquet(g("S09", "features.parquet"))
ev = pd.read_parquet(g("S07", "events.parquet"), columns=["event_id", "decision_time", "earnings_type", "session_bucket"])
ev["dec"] = pd.to_datetime(ev.decision_time, utc=True).dt.tz_convert("America/New_York"); ev["year"] = ev.dec.dt.year
tg = pd.read_parquet(g("S08", "targets.parquet"), columns=["event_id", "target_def", "horizon", "ar_mm"], filters=[("target_def", "=", "T_trade"), ("horizon", "=", 1)]).dropna(subset=["ar_mm"])

# (c) the ladder's availability rule: any non-null column whose name starts with acc_ or sue_
acc_cols = [c for c in feat.columns if c.startswith(("acc_", "sue_"))]
F["accounting_columns_in_availability_rule"] = acc_cols
f = feat.merge(ev[["event_id", "year"]], on="event_id")
f["has_accounting_rule"] = f[acc_cols].notna().any(axis=1)
f["has_acc_only"] = f[[c for c in acc_cols if c.startswith("acc_")]].notna().any(axis=1)
f["acc_log_assets_present"] = f.acc_log_assets.notna()
cov = f.groupby("year")[["has_accounting_rule", "has_acc_only", "acc_log_assets_present"]].mean().round(3); cov.to_csv(OUT / "accounting_rule_vs_content.csv"); print(cov.loc[2000:2012])
F["share_has_accounting_rule_2000_2008"] = float(f[f.year <= 2008].has_accounting_rule.mean())
F["share_with_actual_acc_values_2000_2008"] = float(f[f.year <= 2008].has_acc_only.mean())
F["feature_meta_year_present"] = "meta_year" in feat.columns
F["calendar_features"] = [c for c in feat.columns if c.startswith("meta_") and any(k in c for k in ("year", "month", "weekday", "dow"))]

# (b) volatility baselines versus the model's magnitude score on the same events
lock = pd.read_parquet(g("S19", "locked_predictions.parquet")); dev = pd.read_parquet(g("S18", "dev_predictions.parquet"))
rows = []
for per, d in (("development", dev), ("locked", lock)):
    base = d[d.name == "L6"][["event_id"]].merge(tg, on="event_id").merge(feat[["event_id", "px_idio_vol", "px_vol_60", "px_vol_20", "px_amihud_20", "px_dist_52w_high"]], on="event_id").merge(ev[["event_id", "earnings_type", "session_bucket"]], on="event_id")
    for name in ("L1", "L6", "L6_nochange", "L7"):
        x = d[d.name == name].merge(tg, on="event_id"); x["mv"] = 1 - x.p_side
        rows.append(dict(period=per, score=f"model {name}: 1 - P(Sideways)", rank_IC_vs_abs_AR=spearmanr(x.mv, x.ar_mm.abs())[0], n=len(x)))
    for c in ("px_idio_vol", "px_vol_60", "px_vol_20"):
        rows.append(dict(period=per, score=f"single feature {c}", rank_IC_vs_abs_AR=spearmanr(base[c], base.ar_mm.abs(), nan_policy="omit")[0], n=int(base[c].notna().sum())))
    # event-type x session x volatility baseline: rank of the product of group mean |AR| (from training years only) and px_idio_vol
    tr = feat[["event_id", "px_idio_vol"]].merge(tg, on="event_id").merge(ev[["event_id", "earnings_type", "session_bucket", "year"]], on="event_id")
    tr = tr[tr.year < (2020 if per == "development" else 2023)]
    gm = tr.assign(a=tr.ar_mm.abs()).groupby(["earnings_type", "session_bucket"]).a.mean().rename("gm").reset_index()
    base = base.merge(gm, on=["earnings_type", "session_bucket"], how="left")
    rows.append(dict(period=per, score="baseline: group mean |AR| (train years) x px_idio_vol", rank_IC_vs_abs_AR=spearmanr(base.gm * base.px_idio_vol, base.ar_mm.abs(), nan_policy="omit")[0], n=len(base)))
    rows.append(dict(period=per, score="baseline: group mean |AR| (train years) only", rank_IC_vs_abs_AR=spearmanr(base.gm, base.ar_mm.abs(), nan_policy="omit")[0], n=len(base)))
mag = pd.DataFrame(rows); mag.to_csv(OUT / "magnitude_baselines.csv", index=False); print(mag.round(4).to_string())

# (a) direction rank IC by calendar year, L6 walk-forward (elastic net), development and locked
p11 = pd.read_parquet(g("S11", "predictions.parquet"), columns=["event_id", "y", "date", "p_down", "p_side", "p_up", "run_id"], filters=[("run_id", "in", ["common|L6|enet|-|T_trade|h1|tuned", "common|L1|enet|-|T_trade|h1|tuned"])])
p15 = pd.read_parquet(g("S15", "predictions.parquet"), columns=["event_id", "y", "date", "p_down", "p_side", "p_up", "run_id"], filters=[("run_id", "=", "common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned")])
allp = pd.concat([p11.assign(model=p11.run_id.str.split("|").str[1]), p15.assign(model="L7"),
                  dev.assign(model=dev.name, run_id="dev"), lock.assign(model=lock.name, run_id="locked")]).drop_duplicates(["event_id", "model", "run_id"])
allp = allp[allp.model.isin(["L6", "L7", "L1"])].merge(tg, on="event_id").merge(ev[["event_id", "year"]], on="event_id")
allp["s"] = allp.p_up - allp.p_down
rng = np.random.default_rng(7)
yrows = []
for (model, yr), x in allp.groupby(["model", "year"]):
    if len(x) < 300:
        continue
    ic = spearmanr(x.s, x.ar_mm)[0]; r1, r2 = x.s.rank().to_numpy(), x.ar_mm.rank().to_numpy()
    bs = [np.corrcoef(r1[i], r2[i])[0, 1] for i in (rng.integers(0, len(x), len(x)) for _ in range(200))]
    yrows.append(dict(model=model, year=int(yr), n=len(x), direction_rank_IC=ic, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), source=("walk-forward" if yr <= 2019 else "development" if yr <= 2022 else "locked")))
yd = pd.DataFrame(yrows); yd.to_csv(OUT / "direction_ic_by_year.csv", index=False)
print(yd[yd.model == "L6"].round(4).to_string())
for lab, a, b in (("2008-2012", 2008, 2012), ("2013-2019", 2013, 2019), ("2020-2026", 2020, 2026)):
    x = allp[(allp.model == "L6") & allp.year.between(a, b)]
    ic = spearmanr(x.s, x.ar_mm)[0]; F[f"L6_direction_IC_{lab}"] = [round(float(ic), 4), len(x)]
print({k: v for k, v in F.items() if k.startswith("L6_")})
json.dump(F, open(OUT / "deep_checks2.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in F.items() if not k.startswith("L6_")}, indent=1, default=str)[:1800])
