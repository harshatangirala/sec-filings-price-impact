import glob
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from secprice.track_a.drift import prior_quarter_quintiles

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)  # noqa: E731
d = pd.read_parquet(g("S20", "track_a_cars.parquet"))
fe = pd.read_parquet(g("S09", "features.parquet"), columns=["event_id", "sue_event"])
r = d[d.earnings_type == "earnings_announcement_verified"].merge(fe, on="event_id")
r["q"] = prior_quarter_quintiles(r, "sue_event")
x = r.dropna(subset=["q", "car_2_60", "car_m1_1"])
s = x[x.q == 5]
n = len(s)
a = s.car_m1_1 > 0
b = s.car_2_60 > 0
cells = {"rose then rose": (a & b).sum(), "rose then fell": (a & ~b).sum(), "fell then rose": (~a & b).sum(), "fell then fell": (~a & ~b).sum()}
print("n =", n)
for k, v in cells.items():
    sub = s[{"rose then rose": a & b, "rose then fell": a & ~b, "fell then rose": ~a & b, "fell then fell": ~a & ~b}[k]]
    print(f"{k:16} {v:5d} {v / n:6.1%}  avg announcement {sub.car_m1_1.mean():+.2%}  avg later {sub.car_2_60.mean():+.2%}")
print("later fell total:", (~b).sum(), f"{(~b).mean():.1%}")

