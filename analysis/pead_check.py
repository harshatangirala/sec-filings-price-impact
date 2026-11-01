import glob
import os

import numpy as np
import pandas as pd

from secprice.track_a.drift import prior_quarter_quintiles

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")


def g(s, f):
    return max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)


d = pd.read_parquet(g("S20", "track_a_cars.parquet"))
fe = pd.read_parquet(g("S09", "features.parquet"), columns=["event_id", "sue_event"])
r = d[d.earnings_type == "earnings_announcement_verified"].merge(fe, on="event_id")
r["q"] = prior_quarter_quintiles(r, "sue_event")
x = r.dropna(subset=["q"])
t = x.groupby("q").agg(n=("event_id", "size"), sue_mean=("sue_event", "mean"), announce_car_m1_1=("car_m1_1", "mean"),
                       pre_car_m20_m2=("car_m20_m2", "mean"), drift_car_2_60=("car_2_60", "mean")).round(4)
print(t.to_string())
print("\ncorr(SUE, announcement CAR[-1,+1]) =", round(x[["sue_event", "car_m1_1"]].corr(method="spearman").iloc[0, 1], 4))
print("corr(SUE, pre-event CAR[-20,-2])  =", round(x[["sue_event", "car_m20_m2"]].corr(method="spearman").iloc[0, 1], 4))
print("corr(SUE, drift CAR[+2,+60])      =", round(x[["sue_event", "car_2_60"]].corr(method="spearman").iloc[0, 1], 4))
x["period"] = pd.cut(pd.to_datetime(x.decision_date).dt.year, [1999, 2009, 2019, 2030], labels=["2000-09", "2010-19", "2020-26"])
x["top"] = x.q == 5
x["bot"] = x.q == 1
print("\nQ5 - Q1 drift by period:")
for p, gg in x.groupby("period", observed=True):
    print(p, round(gg[gg.q == 5].car_2_60.mean() - gg[gg.q == 1].car_2_60.mean(), 4), "n =", len(gg))
xc = x[~x.overlap_flag_h60]
print("\nclean events only (no nearby events): Q5-Q1 =", round(xc[xc.q == 5].car_2_60.mean() - xc[xc.q == 1].car_2_60.mean(), 4), "n =", len(xc))

