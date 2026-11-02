"""Facts used by the 2026-11-03 quant review (docs/review). Reads stored stage outputs only; refits nothing.

    RUN_ROOT=<finished run folder> python docs/review/evidence/review_facts.py
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
pd.set_option("display.width", 200, "display.max_columns", 40)
F = {}

feat = pd.read_parquet(g("S09", "features.parquet"))
cars = pd.read_parquet(g("S20", "track_a_cars.parquet"), columns=["event_id", "event_type", "session_bucket", "earnings_type", "decision_date", "cik", "sigma"])
tg = pd.read_parquet(g("S08", "targets.parquet"), columns=["event_id", "target_def", "horizon", "ar_mm", "sigma", "raw_ret", "label3", "eligible"])
ev = pd.read_parquet(g("S07", "events.parquet"), columns=["event_id", "ticker", "categories", "items", "decision_time"])

# ------------------------------------------------------------------ 1. feature quality
reg = pd.read_csv(Path(r"C:\Users\harsh\Downloads\sec-filings-price-impact\dashboard\data\feature_registry.csv"))
cols = [c for c in feat.columns if not c.startswith("_") and c != "event_id"]
miss = feat[cols].isna().mean()
fam = reg.set_index("name").family
F["n_features_total"] = len(cols)
F["missing_by_family"] = {k: round(float(miss[[c for c in cols if fam.get(c) == k]].mean()), 3) for k in sorted(set(fam.dropna()))}
F["dup_px_rev_1m_px_ret_20"] = bool(np.allclose(feat.px_rev_1m, feat.px_ret_20, equal_nan=True))
num = feat[cols].select_dtypes("number")
F["zero_variance_features"] = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
c = num.corr(method="spearman").abs(); np.fill_diagonal(c.values, 0)
F["n_feature_pairs_abs_spearman_gt_0.95"] = int((np.triu(c.values, 1) > 0.95).sum())
F["top_correlated_pairs"] = [(a, b, round(float(c.loc[a, b]), 3)) for a, b in [(i, j) for i in c.index for j in c.columns if i < j] if c.loc[a, b] > 0.97][:12]
F["sue_event_available_share"] = float(feat.sue_event.notna().mean())
F["acc_features_available_share"] = float(feat[[x for x in cols if x.startswith("acc_")]].notna().any(axis=1).mean())
F["txt_has_text_share"] = float(feat.txt_has_text.fillna(False).mean())

# ------------------------------------------------------------------ 2. target geometry
tt = tg[(tg.target_def == "T_trade") & (tg.horizon == 1)].dropna(subset=["ar_mm"]).merge(cars[["event_id", "event_type", "session_bucket", "earnings_type"]], on="event_id")
ti = tg[(tg.target_def == "T_info") & (tg.horizon == 1)].dropna(subset=["ar_mm"]).merge(cars[["event_id", "session_bucket"]], on="event_id")
F["ttrade_sd_over_sigma"] = float((tt.ar_mm.std() / tt.sigma.mean()))
z = tt.ar_mm / tt.sigma
F["ttrade_share_abs_gt_0.5sigma"] = float((z.abs() > 0.5).mean())
from scipy.stats import norm
F["normal_share_abs_gt_0.5sigma_if_sd_equals_sigma"] = float(2 * (1 - norm.cdf(0.5)))
F["ttrade_median_abs_z"] = float(z.abs().median())
by = []
for b in ("PRE", "INTRA", "POST"):
    a = tt[tt.session_bucket == b].ar_mm.abs().mean()
    i = ti[ti.session_bucket == b].ar_mm.abs().mean() if b != "INTRA" else np.nan
    by.append(dict(session=b, mean_abs_Ttrade=a, mean_abs_Tinfo=i, ratio_trade_over_info=a / i if i == i else np.nan, n_trade=int((tt.session_bucket == b).sum())))
by = pd.DataFrame(by); by.to_csv(OUT / "reaction_before_entry.csv", index=False); print(by.round(4).to_string())
e = tt[tt.earnings_type == "earnings_announcement_verified"]; ne = tt[tt.earnings_type != "earnings_announcement_verified"]
F["mean_abs_ar_earnings_vs_other"] = [float(e.ar_mm.abs().mean()), float(ne.ar_mm.abs().mean())]

# ------------------------------------------------------------------ 3. rank IC of the stored predictions
rows = []
lock = pd.read_parquet(g("S19", "locked_predictions.parquet")); dev = pd.read_parquet(g("S18", "dev_predictions.parquet"))
for per, d in (("development", dev), ("locked", lock)):
    for name, x in d.groupby("name"):
        x = x.merge(tt[["event_id", "ar_mm", "earnings_type", "session_bucket"]], on="event_id").dropna(subset=["ar_mm"])
        x["s"] = x.p_up - x.p_down; x["mv"] = 1 - x.p_side
        mo = x.date.astype(str).str[:7]
        ic_m = x.groupby(mo).apply(lambda g_: spearmanr(g_.s, g_.ar_mm)[0] if len(g_) > 20 else np.nan).dropna()
        rows.append(dict(period=per, model=name, n=len(x), rank_IC_direction=spearmanr(x.s, x.ar_mm)[0], monthly_IC_mean=ic_m.mean(), monthly_IC_t=ic_m.mean() / (ic_m.std() / np.sqrt(len(ic_m))), n_months=len(ic_m),
                         rank_IC_move_vs_abs=spearmanr(x.mv, x.ar_mm.abs())[0],
                         IC_earnings=spearmanr(x[x.earnings_type == "earnings_announcement_verified"].s, x[x.earnings_type == "earnings_announcement_verified"].ar_mm)[0]))
ic = pd.DataFrame(rows); ic.to_csv(OUT / "rank_ic.csv", index=False); print(ic.round(4).to_string())

# ------------------------------------------------------------------ 4. statistical power (what a null can and cannot rule out)
pw = []
for per, n, months in (("locked", 8199, 43), ("development", 7453, 36), ("walk-forward", 28533, 144)):
    for rho_c in (0.0, 0.3, 0.6):          # how much of the sampling variance is lost to within-month clustering (illustrative)
        neff = n / (1 + rho_c * (n / months - 1) * 0.02)
        pw.append(dict(period=per, n=n, n_eff_assumed=int(neff), ic_detectable_t2=2 / np.sqrt(neff), ic_detectable_t3=3 / np.sqrt(neff)))
pw = pd.DataFrame(pw); pw.to_csv(OUT / "power_ic.csv", index=False); print(pw.round(4).to_string())
# fundamental law: IR = IC * sqrt(breadth)
yrs = (pd.to_datetime(cars.decision_date).dt.year.value_counts().sort_index())
F["events_per_year_recent"] = {int(k): int(v) for k, v in yrs.loc[2019:2025].items()}
breadth = float(yrs.loc[2019:2025].mean())
F["fundamental_law_gross_IR"] = {f"IC={ic_:.3f}": round(ic_ * np.sqrt(breadth), 2) for ic_ in (0.01, 0.02, 0.03, 0.05)}
F["breadth_per_year"] = breadth
F["breadth_if_universe_x6"] = breadth * 6

# ------------------------------------------------------------------ 5. event mix
mix = cars.groupby(["event_type", "earnings_type"]).size().rename("n").reset_index(); mix.to_csv(OUT / "event_mix.csv", index=False)
cats = ev.categories.explode().value_counts().head(15); F["top_8k_categories"] = {str(k): int(v) for k, v in cats.items()}
per_firm = cars.groupby("cik").size(); F["events_per_firm_quantiles"] = per_firm.quantile([0, .1, .5, .9, 1]).round(0).to_dict()
F["firms"] = int(cars.cik.nunique())

# ------------------------------------------------------------------ 6. sensitivity targets that exist but were never reported
t2 = pd.read_parquet(g("S08", "targets.parquet"), columns=["target_def", "horizon", "ar_mm", "label3"])
F["target_defs_stored"] = {k: int(v) for k, v in t2.groupby("target_def").size().items()}

json.dump(F, open(OUT / "review_facts.json", "w"), indent=1, default=str)
print(json.dumps(F, indent=1, default=str))
