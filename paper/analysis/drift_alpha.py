"""Is the post-earnings 'reversal' real, or does it come from subtracting a pre-event alpha over the 59-day drift window?

The pipeline's CARs are r - alpha - beta'f with alpha estimated on sessions [-250,-30]. Over [+2,+60] that subtracts 59 x alpha.
Stocks that did well in the prior year have large alphas, and good earnings tend to follow good years, so the term can create
a negative surprise-drift relation by construction. This script measures drift four other ways, including calendar-time
portfolios, which need no pre-event alpha at all. Exploratory: written after the locked test, reads stored inputs only.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from secprice.evaluation.buckets import event_buckets
from secprice.track_a.drift import prior_quarter_quintiles, spread_test
from secprice.track_a.event_study import clustered_mean

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
OUT = Path(__file__).resolve().parents[1] / "tables"
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
pd.set_option("display.width", 220, "display.max_columns", 40)
SEED = 20261102
W0, W1, MINOBS = -250, -30, 120

cal = pd.to_datetime(pd.read_parquet(g("S03", "calendar.parquet")).date).dt.tz_localize(None).dt.normalize().to_numpy()
px = pd.read_parquet(g("S03", "prices.parquet"), columns=["ticker", "date", "total_return_index"])
px["date"] = pd.to_datetime(px.date).dt.tz_localize(None).dt.normalize()
wide = px.pivot_table(index="date", columns="ticker", values="total_return_index").reindex(cal)
ret = wide.pct_change(fill_method=None)
fac = pd.read_parquet(g("S03", "factors.parquet")); fac["date"] = pd.to_datetime(fac.date).dt.tz_localize(None).dt.normalize()
fac = fac.set_index("date").reindex(cal)
rm = ret["SPY"].to_numpy(); F4 = fac[["mkt_rf", "smb", "hml", "mom"]].to_numpy(float); rf = fac.rf.to_numpy(float)

ev = pd.read_parquet(g("S07", "events.parquet"), columns=["event_id", "ticker", "cik", "event_type", "earnings_type", "trackA_day0_idx", "trackA_day0", "overlap_flag_h60"])
cars = pd.read_parquet(g("S20", "track_a_cars.parquet"), columns=["event_id", "car_2_60", "car4_2_60", "car_m1_1", "car4_m1_1", "decision_date", "cik"])
feat = pd.read_parquet(g("S09", "features.parquet"), columns=["event_id", "sue_event", "px_mom_12_2", "acc_log_mcap", "acc_book_to_market", "px_vol_60"])
uni = pd.read_parquet(g("S02", "universe.parquet"))
bk = event_buckets(cars[["event_id", "cik", "decision_date"]], uni)
ev = ev.merge(cars.drop(columns="cik"), on="event_id").merge(feat, on="event_id").merge(bk[["event_id", "sp500_member_at_event"]], on="event_id")
n = len(cal)
A, B = 2, 60
rel_days = np.arange(-20, 61)
recs, path_rows = [], []
paths = {}
for tk, gp in ev.groupby("ticker", sort=False):
    if tk not in ret: continue
    r = ret[tk].to_numpy()
    for _, e in gp.iterrows():
        d = int(e.trackA_day0_idx)
        rec = {"event_id": e.event_id}
        lo, hi = d + W0, d + W1
        if lo < 1 or d + B >= n: recs.append(rec); continue
        y, x = r[lo:hi + 1], rm[lo:hi + 1]; m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < MINOBS: recs.append(rec); continue
        X = np.column_stack([np.ones(m.sum()), x[m]]); a_mm, b_mm = np.linalg.lstsq(X, y[m], rcond=None)[0]
        s = slice(d + A, d + B + 1); rs, ms = r[s], rm[s]
        if np.isfinite(rs).all() and np.isfinite(ms).all():
            rec["mm_alpha_replica"] = float((rs - a_mm - b_mm * ms).sum()); rec["mm_noalpha"] = float((rs - b_mm * ms).sum()); rec["mkt_adj"] = float((rs - ms).sum()); rec["raw"] = float(rs.sum())
            rec["alpha_term_mm"] = float(a_mm * (B - A + 1))
        ye, Xf = (r - rf)[lo:hi + 1], F4[lo:hi + 1]; m4 = np.isfinite(ye) & np.isfinite(Xf).all(1)
        if m4.sum() >= MINOBS:
            co = np.linalg.lstsq(np.column_stack([np.ones(m4.sum()), Xf[m4]]), ye[m4], rcond=None)[0]
            yy, XX = (r - rf)[s], F4[s]
            if np.isfinite(yy).all() and np.isfinite(XX).all():
                rec["c4_alpha_replica"] = float((yy - co[0] - XX @ co[1:]).sum()); rec["c4_noalpha"] = float((yy - XX @ co[1:]).sum()); rec["alpha_term_c4"] = float(co[0] * (B - A + 1))
                p0, p1 = d + 61, d + 280                                      # loadings and alpha from AFTER the window (measurement only)
                if p1 < n:
                    yp, Xp = (r - rf)[p0:p1 + 1], F4[p0:p1 + 1]; mp = np.isfinite(yp) & np.isfinite(Xp).all(1)
                    if mp.sum() >= MINOBS:
                        cp = np.linalg.lstsq(np.column_stack([np.ones(mp.sum()), Xp[mp]]), yp[mp], rcond=None)[0]
                        rec["c4_post"] = float((yy - cp[0] - XX @ cp[1:]).sum())
        # daily paths for the event-study figure
        if d - 20 >= 1 and d + 60 < n:
            seg = np.arange(d - 20, d + 61); rr = r[seg]; mm_ = rm[seg]
            if np.isfinite(rr).all() and np.isfinite(mm_).all():
                paths.setdefault(e.event_type, []).append((rr - a_mm - b_mm * mm_, rr - b_mm * mm_))
        recs.append(rec)
est = pd.DataFrame(recs)
ev = ev.merge(est, on="event_id", how="left")
print("replication of stored CARs: corr mm %.6f, c4 %.6f" % (ev[["car_2_60", "mm_alpha_replica"]].corr().iloc[0, 1], ev[["car4_2_60", "c4_alpha_replica"]].corr().iloc[0, 1]))
facts = {"replica_corr_mm": float(ev[["car_2_60", "mm_alpha_replica"]].corr().iloc[0, 1]), "replica_corr_c4": float(ev[["car4_2_60", "c4_alpha_replica"]].corr().iloc[0, 1])}

rows = []
for et, L in paths.items():
    a = np.array([x[0] for x in L]); b = np.array([x[1] for x in L])
    ca, cb = a.mean(0).cumsum(), b.mean(0).cumsum()
    for i, dd in enumerate(rel_days): rows.append(dict(event_type=et, rel_day=dd, caar_alpha=ca[i] - ca[20 - 1], caar_noalpha=cb[i] - cb[20 - 1], n=len(L)))
pd.DataFrame(rows).to_csv(OUT / "car_paths_alpha.csv", index=False)

# ------------------------------------------------------------------- reports sorted on surprise
r = ev[ev.earnings_type == "earnings_announcement_verified"].copy()
r["q"] = prior_quarter_quintiles(r, "sue_event"); r = r.dropna(subset=["q"]); r["yr"] = pd.to_datetime(r.trackA_day0).dt.year; r["qtr"] = pd.to_datetime(r.trackA_day0).dt.to_period("Q").astype(str)
meas = {"mm_alpha_replica": "Market model, pre-event alpha subtracted", "c4_alpha_replica": "Carhart, pre-event alpha subtracted", "mm_noalpha": "Market model, no alpha", "c4_noalpha": "Carhart, no alpha",
        "c4_post": "Carhart, alpha and loadings from days [+61,+280]", "mkt_adj": "Market-adjusted (stock minus S&P 500)", "raw": "Raw return"}
rows = []
for col, lab in meas.items():
    res = spread_test(r, "q", col, "cik", "trackA_day0", 1000, SEED)
    rows.append(dict(measure=lab, col=col, estimate=res["estimate"], ci_low=res["ci_low"], ci_high=res["ci_high"], t=res["t"], n=res["n"], q5=res["q5_mean"], q1=res["q1_mean"]))
main = pd.DataFrame(rows); main.to_csv(OUT / "drift_by_measure.csv", index=False); print(main.round(4).to_string())

qt = r.groupby("q").agg(reports=("event_id", "size"), alpha_term_mm=("alpha_term_mm", "mean"), alpha_term_c4=("alpha_term_c4", "mean"), mm_alpha=("mm_alpha_replica", "mean"), mm_noalpha=("mm_noalpha", "mean"), c4_alpha=("c4_alpha_replica", "mean"),
                        c4_noalpha=("c4_noalpha", "mean"), c4_post=("c4_post", "mean"), mkt_adj=("mkt_adj", "mean"), raw=("raw", "mean"), mom=("px_mom_12_2", "mean"), share_fell_mm_alpha=("mm_alpha_replica", lambda s: (s.dropna() < 0).mean()),
                        share_fell_mkt_adj=("mkt_adj", lambda s: (s.dropna() < 0).mean()), share_fell_c4_noalpha=("c4_noalpha", lambda s: (s.dropna() < 0).mean()))
qt.to_csv(OUT / "drift_quintiles_alpha.csv"); print(qt.round(4).T.to_string())
z = r[["sue_event", "px_mom_12_2", "alpha_term_c4", "alpha_term_mm"]].dropna()
facts["spearman_sue_alpha_c4"] = float(z[["sue_event", "alpha_term_c4"]].corr(method="spearman").iloc[0, 1]); facts["spearman_mom_alpha_c4"] = float(z[["px_mom_12_2", "alpha_term_c4"]].corr(method="spearman").iloc[0, 1]); facts["spearman_sue_mom"] = float(z[["sue_event", "px_mom_12_2"]].corr(method="spearman").iloc[0, 1])

# robustness of the no-alpha measures
rows = []
for lab, x in (("All reports", r), ("2000-2009", r[r.yr.between(2000, 2009)]), ("2010-2019", r[r.yr.between(2010, 2019)]), ("2020-2026", r[r.yr.between(2020, 2026)]), ("Member at the event", r[r.sp500_member_at_event]), ("Not a member at the event", r[~r.sp500_member_at_event]), ("No other firm event in 60 sessions", r[~r.overlap_flag_h60])):
    for col in ("c4_noalpha", "mkt_adj"):
        res = spread_test(x, "q", col, "cik", "trackA_day0", 1000, SEED)
        if res: rows.append(dict(sample=lab, measure=col, estimate=res["estimate"], ci_low=res["ci_low"], ci_high=res["ci_high"], t=res["t"], n=res["n"]))
rb = pd.DataFrame(rows); rb.to_csv(OUT / "drift_noalpha_robustness.csv", index=False); print(rb.round(4).to_string())


def nw(v, lags=4):
    fit = sm.OLS(v, np.ones(len(v))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}); return float(fit.params[0]), float(fit.tvalues[0])


# Fama-MacBeth by quarter: with and without the pre-event alpha term in the dependent variable
rows = []
for dep in ("c4_alpha_replica", "c4_noalpha", "mkt_adj"):
    fm = r.dropna(subset=[dep, "car_m1_1", "px_mom_12_2", "acc_log_mcap", "acc_book_to_market", "px_vol_60"]).copy(); fm["sue_rank"] = fm.groupby("qtr").sue_event.rank(pct=True)
    a, b_ = fm[dep].quantile([.01, .99]); fm[dep] = fm[dep].clip(a, b_)
    for sp, xs in (("(1)", ["sue_rank"]), ("(2)", ["sue_rank", "car_m1_1", "px_mom_12_2", "acc_log_mcap", "acc_book_to_market", "px_vol_60"])):
        co = []
        for _, x in fm.groupby("qtr"):
            if len(x) < len(xs) + 15: continue
            Z = x[xs].copy()
            for c in xs:
                if c != "sue_rank": Z[c] = (Z[c].clip(*Z[c].quantile([.01, .99])) - Z[c].mean()) / (Z[c].std() + 1e-12)
            co.append(sm.OLS(x[dep].to_numpy(), sm.add_constant(Z.to_numpy(), has_constant="add")).fit().params[1:])
        co = np.asarray(co)
        for j, c in enumerate(xs):
            m_, t_ = nw(co[:, j]); rows.append(dict(dep=dep, spec=sp, var=c, coef=m_, t=t_, quarters=len(co)))
fmt = pd.DataFrame(rows); fmt.to_csv(OUT / "fama_macbeth_alpha.csv", index=False); print(fmt.round(4).to_string())

# ------------------------------------------------------------------- calendar-time portfolios (no pre-event estimation at all)
tk = list(ret.columns); ti = {t: i for i, t in enumerate(tk)}
E = (ret.to_numpy() - rf[:, None])


def ct(x, label):
    H = {q: np.zeros((n, len(tk)), bool) for q in (1, 5)}
    for _, e in x[x.q.isin([1, 5])].iterrows():
        if e.ticker in ti:
            d = int(e.trackA_day0_idx); H[int(e.q)][d + A:min(d + B + 1, n), ti[e.ticker]] = True
    out = {}
    with np.errstate(invalid="ignore"):
        cnt5, cnt1 = (H[5] & np.isfinite(E)).sum(1), (H[1] & np.isfinite(E)).sum(1)
        p5 = np.where(cnt5 >= 5, np.nansum(np.where(H[5], E, 0), 1) / np.maximum(cnt5, 1), np.nan); p1 = np.where(cnt1 >= 5, np.nansum(np.where(H[1], E, 0), 1) / np.maximum(cnt1, 1), np.nan)
    ok = np.isfinite(p5) & np.isfinite(p1) & np.isfinite(F4).all(1)
    S = p5[ok] - p1[ok]; X = sm.add_constant(F4[ok])
    fit = sm.OLS(S, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5}); raw = sm.OLS(S, np.ones(len(S))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    f5 = sm.OLS(p5[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5}); f1 = sm.OLS(p1[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return dict(sample=label, days=int(ok.sum()), mean_names_q5=float(cnt5[ok].mean()), mean_names_q1=float(cnt1[ok].mean()), raw_59d=float(raw.params[0] * 59), raw_t=float(raw.tvalues[0]), alpha_59d=float(fit.params[0] * 59), alpha_lo=float(fit.conf_int()[0][0] * 59), alpha_hi=float(fit.conf_int()[0][1] * 59), alpha_t=float(fit.tvalues[0]),
                b_mkt=float(fit.params[1]), b_smb=float(fit.params[2]), b_hml=float(fit.params[3]), b_mom=float(fit.params[4]), b_mom_t=float(fit.tvalues[4]), q5_alpha_59d=float(f5.params[0] * 59), q5_t=float(f5.tvalues[0]), q1_alpha_59d=float(f1.params[0] * 59), q1_t=float(f1.tvalues[0]))


cts = [ct(r, "All reports"), ct(r[r.yr.between(2000, 2009)], "2000-2009"), ct(r[r.yr.between(2010, 2019)], "2010-2019"), ct(r[r.yr.between(2020, 2026)], "2020-2026"), ct(r[r.sp500_member_at_event], "Member at the event"), ct(r[~r.sp500_member_at_event], "Not a member at the event")]
pd.DataFrame(cts).to_csv(OUT / "calendar_time.csv", index=False); print(pd.DataFrame(cts).round(3).T.to_string())

# all events by form, without alpha, two-way clustered
rows = []
for et, lab in (("8K", "8-K"), ("periodic_10Q", "10-Q"), ("periodic_10K", "10-K")):
    x = ev[ev.event_type == et]
    for col in ("mm_alpha_replica", "mm_noalpha", "mkt_adj"):
        z = x.dropna(subset=[col]); o = clustered_mean(z[col].to_numpy(float), z.cik.to_numpy(), z.trackA_day0.astype(str).to_numpy())
        rows.append(dict(form=lab, measure=col, mean=o["mean"], t=o["t"], n=len(z)))
pd.DataFrame(rows).to_csv(OUT / "form_drift_alpha.csv", index=False); print(pd.DataFrame(rows).round(4).to_string())
rep = Path(__file__).resolve().parents[1] / "replication"; rep.mkdir(exist_ok=True)
r[["event_id", "ticker", "trackA_day0", "q", "sue_event", "car_m1_1", "car4_m1_1", "mm_alpha_replica", "c4_alpha_replica", "mm_noalpha", "c4_noalpha", "c4_post", "mkt_adj", "raw", "alpha_term_mm", "alpha_term_c4", "px_mom_12_2", "sp500_member_at_event", "overlap_flag_h60"]].to_parquet(rep / "drift_events.parquet", compression="zstd")
old = json.load(open(OUT / "facts.json")); old.update(facts); json.dump(old, open(OUT / "facts.json", "w"), indent=1); print(json.dumps(facts, indent=1))
