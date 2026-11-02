"""All numbers in the paper that the pipeline's stages do not already report.

Reads stored stage outputs only (predictions, CARs, features, targets). Nothing here refits a model, so the locked test
stays spent. Writes CSV tables and facts.json next to this repository's paper/ folder.

    RUN_ROOT=<finished run folder> python paper/analysis/make_tables.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from secprice.track_a.drift import prior_quarter_quintiles, spread_test
from secprice.evaluation.buckets import event_buckets

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(exist_ok=True)
SEED, B = 20261102, 2000
PCOLS = ["p_down", "p_side", "p_up"]
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
facts = {}


def auc(y, s):
    """Rank-based AUC (fast enough for the bootstrap)."""
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_f1(y, yhat):
    fs = []
    for c in (0, 1, 2):
        tp = np.sum((y == c) & (yhat == c)); fp = np.sum((y != c) & (yhat == c)); fn = np.sum((y == c) & (yhat != c))
        fs.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(fs))


def metrics(y, yhat, P, full=True):
    out = {"macro_f1": macro_f1(y, yhat)}
    if full:
        moved = y != 1
        out["accuracy"] = float(np.mean(y == yhat))
        out["logloss"] = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
        out["auc_ovr"] = float(np.mean([auc((y == c).astype(int), P[:, c]) for c in range(3)]))
        out["move_auc"] = auc(moved.astype(int), 1 - P[:, 1])
        out["direction_auc"] = auc((y[moved] == 2).astype(int), (P[:, 2] - P[:, 0])[moved])
    return out


def ece(y, P, bins=10):
    e = []
    for c in range(3):
        b = np.minimum((P[:, c] * bins).astype(int), bins - 1)
        e.append(sum(abs(P[b == k, c].mean() - (y[b == k] == c).mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))
    return float(np.mean(e))


def month_groups(dates):
    m = pd.to_datetime(pd.Series(dates)).dt.to_period("M").astype(str).to_numpy()
    u, inv = np.unique(m, return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(u))]


def align(a, b):
    """Two prediction frames on identical events, same row order."""
    ids = np.intersect1d(a.event_id.to_numpy(), b.event_id.to_numpy())
    return a.set_index("event_id").loc[ids].reset_index(), b.set_index("event_id").loc[ids].reset_index()


def arr(d):
    return d.y.astype(int).to_numpy(), d.yhat.astype(int).to_numpy(), d[PCOLS].to_numpy(float)


def boot_single(d, b=B, seed=SEED):
    y, yh, P = arr(d); gr = month_groups(d.date); rng = np.random.default_rng(seed)
    pt = metrics(y, yh, P); pt["ece"] = ece(y, P)
    res = {k: [] for k in ("macro_f1", "accuracy", "logloss", "auc_ovr", "move_auc", "direction_auc")}
    for _ in range(b):
        idx = np.concatenate([gr[k] for k in rng.integers(0, len(gr), len(gr))])
        m = metrics(y[idx], yh[idx], P[idx])
        for k in res: res[k].append(m[k])
    out = dict(pt, n=len(d), n_months=len(gr))
    for k, v in res.items():
        out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(v, [2.5, 97.5])
    rng2 = np.random.default_rng(seed + 1)
    out["chance_shuffled"] = float(np.mean([macro_f1(y, rng2.permutation(yh)) for _ in range(200)]))
    return out


def boot_pair(a, b_, metric_names=("macro_f1", "logloss", "auc_ovr", "move_auc", "direction_auc"), b=B, seed=SEED, margin=0.01):
    """Paired month-block bootstrap of metric(a) - metric(b) on identical events."""
    a, b_ = align(a, b_)
    ya, yha, Pa = arr(a); _, yhb, Pb = arr(b_)
    gr = month_groups(a.date); rng = np.random.default_rng(seed)
    pa, pb = metrics(ya, yha, Pa), metrics(ya, yhb, Pb)
    draws = {k: [] for k in metric_names}
    full = metric_names != ("macro_f1",)
    for _ in range(b):
        idx = np.concatenate([gr[k] for k in rng.integers(0, len(gr), len(gr))])
        ma, mb = metrics(ya[idx], yha[idx], Pa[idx], full), metrics(ya[idx], yhb[idx], Pb[idx], full)
        for k in metric_names: draws[k].append(ma[k] - mb[k])
    rows = {}
    for k in metric_names:
        v = np.asarray(draws[k], float); v = v[np.isfinite(v)]
        d = pa[k] - pb[k]
        rows[k] = {"lift": d, "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5),
                   "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))), "share_ge_margin": float((v >= margin).mean()),
                   "n": len(a), "n_months": len(gr)}
    return rows


# ------------------------------------------------------------------------------------------------- load predictions
keep = ["event_id", "y", "date"] + PCOLS + ["yhat"]
p11 = pd.read_parquet(g("S11", "predictions.parquet"), columns=keep + ["run_id"])
p15 = pd.read_parquet(g("S15", "predictions.parquet"), columns=keep + ["run_id"])
dev = pd.read_parquet(g("S18", "dev_predictions.parquet"), columns=keep + ["name"])
lock = pd.read_parquet(g("S19", "locked_predictions.parquet"), columns=keep + ["name"])


def wf(run):
    src = p15 if run.startswith("common_emb") else p11
    return src[src.run_id == run].drop(columns="run_id").dropna(subset=["y"])


WF = {"L1": wf("common|L1|enet|-|T_trade|h1|tuned"), "L6": wf("common|L6|enet|-|T_trade|h1|tuned"),
      "L6_nochange": wf("common|L6_nochange|enet|-|T_trade|h1|tuned"), "L7": wf("common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned")}
PER = {"Walk-forward 2008-2019": WF,
       "Development 2020-2022": {n: dev[dev.name == n].drop(columns="name") for n in ("L1", "L6", "L6_nochange", "L7")},
       "Locked test 2023-2026": {n: lock[lock.name == n].drop(columns="name") for n in ("L1", "L6", "L6_nochange", "L7")}}

# ------------------------------------------------------------------------------------------------- performance and lifts
rows = []
for per, mods in PER.items():
    for n in ("L1", "L6", "L7"):
        rows.append(dict(period=per, model=n, **boot_single(mods[n])))
perf = pd.DataFrame(rows); perf.to_csv(OUT / "performance.csv", index=False)

rows = []
for per, mods in PER.items():
    for lab, a, b in (("Text over structured (L7 - L6)", "L7", "L6"), ("Change features (L6 - L6 no change)", "L6", "L6_nochange"),
                      ("Structured over price (L6 - L1)", "L6", "L1")):
        for met, r in boot_pair(mods[a], mods[b]).items():
            rows.append(dict(period=per, comparison=lab, metric=met, **r))
lifts = pd.DataFrame(rows); lifts.to_csv(OUT / "lifts.csv", index=False)

# same head, same encoder family, walk-forward only (the locked test scored only the frozen finalists)
rows = []
for lab, a, b in (("LightGBM + BGE vs LightGBM", "common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned", "common_emb|L6|lightgbm|-|T_trade|h1|tuned"),
                  ("LightGBM + ChronoBERT vs LightGBM", "common_emb|L7|lightgbm|chrono_bert|T_trade|h1|tuned", "common_emb|L6|lightgbm|-|T_trade|h1|tuned"),
                  ("Elastic net + BGE vs elastic net", "common_emb|L7|enet|bge_base|T_trade|h1|tuned", "common_emb|L6|enet|-|T_trade|h1|tuned"),
                  ("Elastic net + ChronoBERT vs elastic net", "common_emb|L7|enet|chrono_bert|T_trade|h1|tuned", "common_emb|L6|enet|-|T_trade|h1|tuned"),
                  ("ChronoBERT vs BGE (LightGBM, L7)", "common_emb|L7|lightgbm|chrono_bert|T_trade|h1|tuned", "common_emb|L7|lightgbm|bge_base|T_trade|h1|tuned")):
    rows.append(dict(comparison=lab, **boot_pair(wf(a), wf(b), ("macro_f1",))["macro_f1"]))
pd.DataFrame(rows).to_csv(OUT / "same_head_lifts.csv", index=False)

# disagreement between the text and the no-text model
d6, d7 = align(PER["Locked test 2023-2026"]["L6"], PER["Locked test 2023-2026"]["L7"])
facts["locked_L6L7_same_class"] = float((d6.yhat == d7.yhat).mean())
facts["locked_L6L7_signal_corr"] = float(np.corrcoef(d6.p_up - d6.p_down, d7.p_up - d7.p_down)[0, 1])

# ------------------------------------------------------------------------------------------------- event attributes
cars = pd.read_parquet(g("S20", "track_a_cars.parquet"))
feat = pd.read_parquet(g("S09", "features.parquet"))
ev = pd.read_parquet(g("S07", "events.parquet"))
uni = pd.read_parquet(g("S02", "universe.parquet"))
bk = event_buckets(cars[["event_id", "cik", "decision_date"]], uni)
attr = cars[["event_id", "event_type", "session_bucket", "earnings_type"]].merge(bk[["event_id", "sp500_member_at_event"]], on="event_id")
facts["member_at_event_share"] = float(attr.sp500_member_at_event.mean())

# text lift by subset (exploratory): pooled out-of-sample and locked
subsets = [("8-K", lambda a: a.event_type == "8K"), ("10-Q", lambda a: a.event_type == "periodic_10Q"), ("10-K", lambda a: a.event_type == "periodic_10K"),
           ("Earnings releases", lambda a: a.earnings_type == "earnings_announcement_verified"),
           ("Non-earnings events", lambda a: a.earnings_type != "earnings_announcement_verified"),
           ("Before the open", lambda a: a.session_bucket == "PRE"), ("During the session", lambda a: a.session_bucket == "INTRA"),
           ("After the close", lambda a: a.session_bucket == "POST")]
pool6 = pd.concat([PER[p]["L6"] for p in PER]); pool7 = pd.concat([PER[p]["L7"] for p in PER])
rows = []
for lab, fn in subsets:
    ids = attr[fn(attr)].event_id
    for scope, a, b in (("Pooled 2008-2026", pool7, pool6), ("Locked test", PER["Locked test 2023-2026"]["L7"], PER["Locked test 2023-2026"]["L6"])):
        A, Bb = a[a.event_id.isin(ids)], b[b.event_id.isin(ids)]
        rows.append(dict(subset=lab, scope=scope, **boot_pair(A, Bb, ("macro_f1",))["macro_f1"]))
pd.DataFrame(rows).to_csv(OUT / "text_lift_by_subset.csv", index=False)

# performance by calendar year (no bootstrap)
rows = []
for n in ("L1", "L6", "L7"):
    d = pd.concat([PER[p][n] for p in PER]); d["year"] = pd.to_datetime(d.date).dt.year
    for yr, x in d.groupby("year"):
        y, yh, P = arr(x); m = y != 1
        rows.append(dict(model=n, year=yr, n=len(x), macro_f1=macro_f1(y, yh), move_auc=auc(m.astype(int), 1 - P[:, 1]), direction_auc=auc((y[m] == 2).astype(int), (P[:, 2] - P[:, 0])[m])))
pd.DataFrame(rows).to_csv(OUT / "by_year.csv", index=False)

# ------------------------------------------------------------------------------------------------- T_info versus T_trade
rows = []
pti = pd.read_parquet(g("S11", "predictions.parquet"), columns=keep + ["run_id"], filters=[("run_id", "in", [f"common|{r}|enet|-|T_info|h1|tuned_transferred" for r in ("L1", "L4", "L6")])])
for rung in ("L1", "L4", "L6"):
    ti = pti[pti.run_id == f"common|{rung}|enet|-|T_info|h1|tuned_transferred"].drop(columns="run_id").dropna(subset=["y"])
    tt = wf(f"common|{rung}|enet|-|T_trade|h1|tuned")
    tt = tt[tt.event_id.isin(ti.event_id)]
    for tn, d in (("T_info", ti), ("T_trade", tt)):
        b = boot_single(d, b=1000)
        rows.append(dict(rung=rung, target=tn, n=b["n"], macro_f1=b["macro_f1"], move_auc=b["move_auc"], move_lo=b["move_auc_lo"], move_hi=b["move_auc_hi"],
                         direction_auc=b["direction_auc"], dir_lo=b["direction_auc_lo"], dir_hi=b["direction_auc_hi"]))
pd.DataFrame(rows).to_csv(OUT / "tinfo_vs_ttrade.csv", index=False)

# ------------------------------------------------------------------------------------------------- predicted move versus realised size
tg = pd.read_parquet(g("S08", "targets.parquet"), columns=["event_id", "target_def", "horizon", "ar_mm", "abs_std", "label3", "sigma", "eligible"],
                     filters=[("target_def", "=", "T_trade"), ("horizon", "=", 1)])
rows = []
for n in ("L1", "L6"):
    d = PER["Locked test 2023-2026"][n].merge(tg[["event_id", "ar_mm", "abs_std"]], on="event_id").dropna(subset=["ar_mm"])
    d["pmove"] = 1 - d.p_side
    d["dec"] = pd.qcut(d.pmove.rank(method="first"), 10, labels=False) + 1
    for k, x in d.groupby("dec"):
        rows.append(dict(model=n, decile=k, n=len(x), mean_pmove=x.pmove.mean(), share_moved=(x.y != 1).mean(), mean_abs_ar=x.ar_mm.abs().mean(), median_abs_ar=x.ar_mm.abs().median(), mean_abs_std=x.abs_std.mean()))
pd.DataFrame(rows).to_csv(OUT / "move_deciles.csv", index=False)

# reliability
rows = []
for n in ("L6", "L7"):
    y, _, P = arr(PER["Locked test 2023-2026"][n])
    for c, cn in enumerate(("Down", "Sideways", "Up")):
        b = pd.qcut(pd.Series(P[:, c]).rank(method="first"), 10, labels=False).to_numpy()
        for k in range(10):
            rows.append(dict(model=n, cls=cn, bin=k, mean_p=P[b == k, c].mean(), freq=(y[b == k] == c).mean()))
pd.DataFrame(rows).to_csv(OUT / "reliability.csv", index=False)

# ------------------------------------------------------------------------------------------------- sample description
tab = tg.merge(cars[["event_id", "event_type", "session_bucket", "decision_date"]], on="event_id").dropna(subset=["ar_mm"])
tab["year"] = pd.to_datetime(tab.decision_date).dt.year
rows = []
for lab, x in [("All events", tab)] + [(n, tab[tab.event_type == k]) for n, k in (("8-K", "8K"), ("10-Q", "periodic_10Q"), ("10-K", "periodic_10K"))]:
    rows.append(dict(sample=lab, n=len(x), mean_ar=x.ar_mm.mean(), sd_ar=x.ar_mm.std(), p05=x.ar_mm.quantile(.05), median=x.ar_mm.median(), p95=x.ar_mm.quantile(.95), mean_abs=x.ar_mm.abs().mean(),
                     down=(x.label3 == 0).mean(), side=(x.label3 == 1).mean(), up=(x.label3 == 2).mean()))
for lab, b_ in (("Before the open", "PRE"), ("During the session", "INTRA"), ("After the close", "POST")):
    x = tab[tab.session_bucket == b_]
    rows.append(dict(sample=lab, n=len(x), mean_ar=x.ar_mm.mean(), sd_ar=x.ar_mm.std(), p05=x.ar_mm.quantile(.05), median=x.ar_mm.median(), p95=x.ar_mm.quantile(.95), mean_abs=x.ar_mm.abs().mean(),
                     down=(x.label3 == 0).mean(), side=(x.label3 == 1).mean(), up=(x.label3 == 2).mean()))
pd.DataFrame(rows).to_csv(OUT / "sample_returns.csv", index=False)
facts["label_rows_t_trade_h1"] = int(len(tab))

# ------------------------------------------------------------------------------------------------- drift
r = cars[cars.earnings_type == "earnings_announcement_verified"].merge(
    feat[["event_id", "sue_event", "px_mom_12_2", "acc_log_mcap", "acc_book_to_market", "px_vol_60", "sue_n_diffs"]], on="event_id").merge(bk[["event_id", "sp500_member_at_event"]], on="event_id")
r["q"] = prior_quarter_quintiles(r, "sue_event")
r["day"] = pd.to_datetime(r.trackA_day0); r["yr"] = r.day.dt.year; r["qtr"] = r.day.dt.to_period("Q").astype(str)
qt = r.dropna(subset=["q"]).groupby("q").agg(
    reports=("event_id", "size"), sue=("sue_event", "mean"), ann_mm=("car_m1_1", "mean"), ann_c4=("car4_m1_1", "mean"), pre=("car_m20_m2", "mean"),
    d2_20_c4=("car4_2_20", "mean"), d2_60_mm=("car_2_60", "mean"), d2_60_c4=("car4_2_60", "mean"), median_d2_60_c4=("car4_2_60", "median"), n_d=("car4_2_60", "count"),
    pos_c4=("car4_2_60", lambda s: (s.dropna() > 0).mean()))
qt.to_csv(OUT / "drift_quintiles.csv")


def st(lab, df, col="car4_2_60", qcol="q"):
    res = spread_test(df, qcol, col, "cik", "trackA_day0", 1000, SEED)
    return dict(spec=lab, **{k: res[k] for k in ("estimate", "ci_low", "ci_high", "t", "n", "q5_mean", "q1_mean")}) if res else dict(spec=lab, n=0)


def within(df, col, by="qtr"):
    x = df.copy(); hi = x.groupby(by)[col].transform("median"); return x, x[col] >= hi


rel = r.dropna(subset=["q"])
rows = [st("Carhart-adjusted, baseline", rel), st("Market-model-adjusted", rel, "car_2_60"), st("One-session delay [+3,+61]", rel, "car4_3_61"), st("First leg only [+2,+20]", rel, "car4_2_20"),
        st("No other firm event within 60 sessions", rel[~rel.overlap_flag_h60])]
for lab, (a, b_) in (("2000-2009", (2000, 2009)), ("2010-2019", (2010, 2019)), ("2020-2026", (2020, 2026))):
    rows.append(st(lab, rel[rel.yr.between(a, b_)]))
rows.append(st("Excluding 2008-2009 and 2020", rel[~rel.yr.isin([2008, 2009, 2020])]))
rows.append(st("S&P 500 member at the event", rel[rel.sp500_member_at_event])); rows.append(st("Not yet / no longer a member", rel[~rel.sp500_member_at_event]))
x, big = within(rel.dropna(subset=["acc_log_mcap"]), "acc_log_mcap"); rows.append(st("Above-median market cap", x[big])); rows.append(st("Below-median market cap", x[~big]))
x, hv = within(rel.dropna(subset=["px_vol_60"]), "px_vol_60"); rows.append(st("Above-median volatility", x[hv])); rows.append(st("Below-median volatility", x[~hv]))
w = rel.copy(); lo_, hi_ = w.car4_2_60.quantile([.01, .99]); w["cw"] = w.car4_2_60.clip(lo_, hi_); rows.append(st("Returns winsorised at 1% / 99%", w, "cw"))
drift = pd.DataFrame(rows); drift.to_csv(OUT / "drift_robustness.csv", index=False)
print(drift.round(4).to_string())

# calendar-quarter spreads and Fama-MacBeth
sp = rel[rel.q.isin([1, 5])].dropna(subset=["car4_2_60"]).groupby(["qtr", "q"]).car4_2_60.mean().unstack().dropna()
s = (sp[5.0] - sp[1.0]).to_numpy()


def nw(v, lags=4):
    fit = sm.OLS(v, np.ones(len(v))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


facts["quarterly_spread_mean"], facts["quarterly_spread_t"] = nw(s); facts["quarterly_spread_n"] = int(len(s)); facts["quarterly_spread_negative_share"] = float((s < 0).mean())
facts["pct_quarters_min_obs"] = int(sp.shape[0])

fm = rel.dropna(subset=["car4_2_60", "car_m1_1", "car_m20_m2", "px_mom_12_2", "acc_log_mcap", "acc_book_to_market", "px_vol_60"]).copy()
fm["sue_rank"] = fm.groupby("qtr").sue_event.rank(pct=True)
specs = {"(1)": ["sue_rank"], "(2)": ["sue_rank", "car_m1_1"], "(3)": ["sue_rank", "car_m1_1", "car_m20_m2"], "(4)": ["sue_rank", "car_m1_1", "car_m20_m2", "px_mom_12_2", "acc_log_mcap", "acc_book_to_market", "px_vol_60"]}
for c in ["car4_2_60", "car_m1_1", "car_m20_m2", "px_mom_12_2", "acc_book_to_market", "px_vol_60"]:
    a, b_ = fm[c].quantile([.01, .99]); fm[c] = fm[c].clip(a, b_)
rows = []
for name, xs in specs.items():
    coefs, ns, r2s = [], [], []
    for q_, x in fm.groupby("qtr"):
        if len(x) < len(xs) + 15:
            continue
        z = x[xs].copy()
        for c in xs:
            if c != "sue_rank": z[c] = (z[c] - z[c].mean()) / (z[c].std() + 1e-12)
        o = sm.OLS(x.car4_2_60.to_numpy(), sm.add_constant(z.to_numpy(), has_constant="add")).fit()
        coefs.append(o.params[1:]); ns.append(len(x)); r2s.append(o.rsquared)
    coefs = np.asarray(coefs)
    for j, c in enumerate(xs):
        m_, t_ = nw(coefs[:, j]); rows.append(dict(spec=name, var=c, coef=m_, t=t_, quarters=len(coefs), obs=int(np.sum(ns)), mean_r2=float(np.mean(r2s))))
pd.DataFrame(rows).to_csv(OUT / "fama_macbeth.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())

# how much missing low-surprise loss would it take? (illustrative bound)
q1, q5 = qt.loc[1.0, "d2_60_c4"], qt.loc[5.0, "d2_60_c4"]
pd.DataFrame([dict(loss=L, share_missing=(q1 - q5) / (q1 - L)) for L in (-0.1, -0.2, -0.3, -0.4, -0.5, -0.75, -1.0)]).to_csv(OUT / "survivorship_bound.csv", index=False)
facts["bound_q1"], facts["bound_q5"] = float(q1), float(q5)
facts["sue_reports_all_quintiles"] = int(rel.car_2_60.notna().sum()); facts["sue_reports_with_q"] = int(len(rel)); facts["verified_releases"] = int(len(r)); facts["sue_nonmissing"] = int(r.sue_event.notna().sum())

# ------------------------------------------------------------------------------------------------- spearman of SUE and text with reactions
txt = cars.merge(feat[["event_id", "main_lm_net_tone", "novelty_share", "sue_event"] + [c for c in feat.columns if c == "chg_cosine"]], on="event_id")
rows = []
for var in ("main_lm_net_tone", "novelty_share", "chg_cosine", "sue_event"):
    if var not in txt: continue
    for lab, x in (("All events", txt), ("Earnings releases", txt[txt.earnings_type == "earnings_announcement_verified"])):
        for w_ in ("car_m1_1", "car_2_60"):
            z = x[[var, w_]].dropna()
            rows.append(dict(var=var, sample=lab, window=w_, spearman=z.corr(method="spearman").iloc[0, 1], n=len(z)))
pd.DataFrame(rows).to_csv(OUT / "text_reaction_corr.csv", index=False)

# earnings-only: does anything predict direction where news exists?
rows = []
for per, mods in PER.items():
    for n in ("L6", "L7"):
        d = mods[n][mods[n].event_id.isin(attr[attr.earnings_type == "earnings_announcement_verified"].event_id)]
        b = boot_single(d, b=500)
        rows.append(dict(period=per, model=n, n=b["n"], macro_f1=b["macro_f1"], move_auc=b["move_auc"], direction_auc=b["direction_auc"], dir_lo=b["direction_auc_lo"], dir_hi=b["direction_auc_hi"]))
pd.DataFrame(rows).to_csv(OUT / "earnings_only.csv", index=False)

facts["n_firms"] = int(cars.cik.nunique())
json.dump(facts, open(OUT / "facts.json", "w"), indent=1)
print(json.dumps(facts, indent=1))
print(pd.read_csv(OUT / "lifts.csv").query("metric=='macro_f1'").round(4).to_string())
