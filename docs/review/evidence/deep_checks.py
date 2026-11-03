"""Second-pass empirical checks for the quant review (docs/review/12_*). Stored outputs only; refits nothing.

    RUN_ROOT=<finished run folder> python docs/review/evidence/deep_checks.py
"""
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
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 80)
F = {}

# ------------------------------------------------------------------ 1. price data quality
px = pd.read_parquet(g("S03", "prices.parquet"))
px = px[px.ticker != "SPY"].sort_values(["ticker", "date"])
px["ret"] = px.groupby("ticker").total_return_index.pct_change()
cc = px.groupby("ticker").close_split_adj.pct_change()
F["bars"] = int(len(px))
F["bars_open_gt_high_or_lt_low"] = int(((px.open > px.high * 1.0001) | (px.open < px.low * 0.9999)).sum())
F["bars_high_lt_low"] = int((px.high < px.low).sum())
F["bars_nonpositive_price"] = int(((px.open <= 0) | (px.close_split_adj <= 0)).sum())
F["bars_zero_volume"] = int((px.volume.fillna(0) == 0).sum())
F["bars_open_equals_close_share"] = float((px.open == px.close_split_adj).mean())
F["abs_daily_move_gt_40pct"] = int((cc.abs() > 0.40).sum())
F["abs_daily_move_gt_25pct"] = int((cc.abs() > 0.25).sum())
big = px.assign(cc=cc)[cc.abs() > 0.40][["ticker", "date", "close_split_adj", "cc", "source"]].sort_values("cc")
big.to_csv(OUT / "price_large_moves.csv", index=False)
F["large_moves_by_source"] = big.source.value_counts().to_dict()
# stale prices: 5+ consecutive identical closes
same = (px.groupby("ticker").close_split_adj.diff() == 0).astype(int)
runs = same.groupby([px.ticker, (same == 0).cumsum()]).cumsum()
F["stale_runs_ge5_days"] = int((runs == 5).sum())
F["bars_by_source"] = px.source.value_counts().to_dict()
cov = px.groupby("ticker").agg(first=("date", "min"), last=("date", "max"), n=("date", "size"))
cov["span_days"] = (cov["last"] - cov["first"]).dt.days
cov["expected_sessions"] = cov.span_days * 252 / 365.25
cov["completeness"] = cov.n / cov.expected_sessions
F["firm_price_completeness_quantiles"] = cov.completeness.quantile([0, .05, .5, .95, 1]).round(3).to_dict()
F["firms_with_completeness_lt_0.97"] = int((cov.completeness < 0.97).sum())

# ------------------------------------------------------------------ 2. events and targets
ev = pd.read_parquet(g("S07", "events.parquet"), columns=["event_id", "ticker", "cik", "event_type", "decision_time", "acceptance_time", "ts_quality", "session_bucket", "earnings_type", "ttrade_entry_idx", "trackA_day0_idx"])
ev["dec"] = pd.to_datetime(ev.decision_time, utc=True).dt.tz_convert("America/New_York")
ev["year"] = ev.dec.dt.year
F["events"] = int(len(ev)); F["duplicate_event_ids"] = int(ev.event_id.duplicated().sum())
F["decision_on_weekend"] = int((ev.dec.dt.weekday >= 5).sum())
F["events_per_decision_date_top"] = {str(k): int(v) for k, v in ev.dec.dt.date.value_counts().head(5).items()}
F["share_events_on_top_20_dates"] = float(ev.dec.dt.date.value_counts().head(20).sum() / len(ev))
F["share_events_on_top_100_dates"] = float(ev.dec.dt.date.value_counts().head(100).sum() / len(ev))
F["n_distinct_decision_dates"] = int(ev.dec.dt.date.nunique())
F["top10_firms_share_of_events"] = float(ev.ticker.value_counts().head(10).sum() / len(ev))
ts = ev.groupby("year").ts_quality.value_counts(normalize=True).unstack().fillna(0).round(3); ts.to_csv(OUT / "timestamp_quality_by_year.csv")
print(ts.loc[2000:2008])

tg = pd.read_parquet(g("S08", "targets.parquet"), columns=["event_id", "target_def", "horizon", "ar_mm", "sigma", "beta", "n_est", "weight_uniq", "null_reason", "label3", "raw_ret"], filters=[("target_def", "=", "T_trade"), ("horizon", "=", 1)])
F["ttrade_null_reasons"] = {str(k): int(v) for k, v in tg.null_reason.fillna("ok").value_counts().items()}
t = tg.dropna(subset=["ar_mm"]).merge(ev[["event_id", "ticker", "dec", "year", "session_bucket", "earnings_type"]], on="event_id")
F["ar_abs_gt_20pct"] = int((t.ar_mm.abs() > 0.20).sum()); F["ar_abs_gt_50pct"] = int((t.ar_mm.abs() > 0.50).sum())
t.reindex(t.ar_mm.abs().sort_values(ascending=False).index).head(25)[["ticker", "dec", "ar_mm", "raw_ret", "sigma", "beta"]].to_csv(OUT / "largest_abnormal_returns.csv", index=False)
F["beta_quantiles"] = t.beta.quantile([.01, .05, .5, .95, .99]).round(2).to_dict()
F["beta_lt_0_share"] = float((t.beta < 0).mean())
F["sigma_quantiles"] = t.sigma.quantile([.01, .5, .99]).round(4).to_dict()
F["n_est_lt_250_share"] = float((t.n_est < 200).mean())
F["uniqueness_weight_mean"] = float(tg.weight_uniq.mean()); F["effective_events_by_uniqueness"] = float(tg.weight_uniq.sum())
yr = t.groupby("year").agg(n=("ar_mm", "size"), mean_abs=("ar_mm", lambda s: s.abs().mean()), sd=("ar_mm", "std"), down=("label3", lambda s: (s == 0).mean()), up=("label3", lambda s: (s == 2).mean())); yr.to_csv(OUT / "target_by_year.csv"); print(yr.round(4))

# ------------------------------------------------------------------ 3. feature coverage by year and univariate leakage screen
feat = pd.read_parquet(g("S09", "features.parquet"))
f = feat.merge(ev[["event_id", "year"]], on="event_id")
cov_y = f.groupby("year").agg(acc_any=("acc_log_assets", lambda s: s.notna().mean()), sue=("sue_event", lambda s: s.notna().mean()), ear=("ear_last", lambda s: s.notna().mean()), chg=("chg_cosine", lambda s: s.notna().mean()))
cov_y.round(3).to_csv(OUT / "feature_coverage_by_year.csv"); print(cov_y.round(3).loc[2000:2012])
num = [c for c in feat.columns if c != "event_id" and not c.startswith("_") and pd.api.types.is_numeric_dtype(feat[c])]
m = f.merge(t[["event_id", "ar_mm"]], on="event_id")
rows = []
for c in num:
    z = m[[c, "ar_mm"]].dropna()
    if len(z) > 2000 and z[c].nunique() > 2:
        rows.append(dict(feature=c, rho_direction=spearmanr(z[c], z.ar_mm)[0], rho_magnitude=spearmanr(z[c], z.ar_mm.abs())[0], n=len(z)))
u = pd.DataFrame(rows); u.to_csv(OUT / "feature_univariate_screen.csv", index=False)
print(u.reindex(u.rho_direction.abs().sort_values(ascending=False).index).head(8).round(3).to_string())
print(u.reindex(u.rho_magnitude.abs().sort_values(ascending=False).index).head(8).round(3).to_string())
F["max_abs_univariate_rho_direction"] = float(u.rho_direction.abs().max()); F["max_abs_univariate_rho_magnitude"] = float(u.rho_magnitude.abs().max())
F["top_direction_features"] = u.reindex(u.rho_direction.abs().sort_values(ascending=False).index).head(5)[["feature", "rho_direction"]].round(4).values.tolist()
F["top_magnitude_features"] = u.reindex(u.rho_magnitude.abs().sort_values(ascending=False).index).head(8)[["feature", "rho_magnitude"]].round(4).values.tolist()

# ------------------------------------------------------------------ 4. measured design effect for the rank IC (replaces the illustrative n_eff)
lock = pd.read_parquet(g("S19", "locked_predictions.parquet")); dev = pd.read_parquet(g("S18", "dev_predictions.parquet"))
p11 = pd.read_parquet(g("S11", "predictions.parquet"), columns=["event_id", "y", "date", "p_down", "p_side", "p_up", "run_id"], filters=[("run_id", "=", "common|L6|enet|-|T_trade|h1|tuned")])
rng = np.random.default_rng(20261103)
rows = []
for per, d in (("development", dev[dev.name == "L6"]), ("locked", lock[lock.name == "L6"]), ("walk-forward", p11)):
    x = d.merge(t[["event_id", "ar_mm", "ticker"]], on="event_id").dropna(subset=["ar_mm"])
    x["s"] = x.p_up - x.p_down; x["mon"] = pd.to_datetime(x.date).dt.to_period("M").astype(str)
    ic = spearmanr(x.s, x.ar_mm)[0]; n = len(x)
    # month-block and firm-block bootstrap of the rank IC (ranks recomputed on the pooled sample once; resample blocks of pre-ranked data)
    rs, ra = x.s.rank().to_numpy(), x.ar_mm.rank().to_numpy()
    def ic_of(idx):
        a, b = rs[idx], ra[idx]; return np.corrcoef(a, b)[0, 1]
    out = {}
    for name, key in (("month", x.mon), ("firm", x.ticker)):
        codes, uniq = pd.factorize(key); groups = [np.where(codes == i)[0] for i in range(len(uniq))]
        bs = [ic_of(np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])) for _ in range(300)]
        out[name] = np.std(bs)
    se_iid = 1 / np.sqrt(n)
    rows.append(dict(period=per, n=n, rank_IC=ic, se_iid=se_iid, se_month_block=out["month"], se_firm_block=out["firm"], design_effect_month=(out["month"] / se_iid) ** 2, design_effect_firm=(out["firm"] / se_iid) ** 2,
                     n_eff_month=n / max(1, (out["month"] / se_iid) ** 2), n_eff_firm=n / max(1, (out["firm"] / se_iid) ** 2), mde_t3_month=3 * out["month"], mde_t3_firm=3 * out["firm"]))
de = pd.DataFrame(rows); de.to_csv(OUT / "design_effect_ic.csv", index=False); print(de.round(4).to_string())

json.dump(F, open(OUT / "deep_checks.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in F.items() if k not in ("top_magnitude_features",)}, indent=1, default=str)[:6000])
