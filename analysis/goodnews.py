import glob
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from secprice.track_a.drift import prior_quarter_quintiles

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")


def g(s, f):
    return max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)


d = pd.read_parquet(g("S20", "track_a_cars.parquet"))
fe = pd.read_parquet(g("S09", "features.parquet"), columns=["event_id", "sue_event"])
r = d[d.earnings_type == "earnings_announcement_verified"].merge(fe, on="event_id")
r["q"] = prior_quarter_quintiles(r, "sue_event")
x = r.dropna(subset=["q", "car_2_60", "car_m1_1"])
print("earnings releases with a surprise score and quintile:", len(x))
for label, sel in (("Q5 best news", x.q == 5), ("Q4+Q5 good news", x.q >= 4), ("Q1 worst news", x.q == 1)):
    s = x[sel]
    print(f"\n{label}: n={len(s)}")
    print(f"  announcement reaction CAR[-1,+1] positive: {(s.car_m1_1 > 0).mean():.1%}  | negative: {(s.car_m1_1 < 0).mean():.1%}")
    print(f"  later drift CAR[+2,+60] positive: {(s.car_2_60 > 0).mean():.1%}  | negative: {(s.car_2_60 < 0).mean():.1%}")
s = x[x.q == 5]
up_then_down = s[(s.car_m1_1 > 0) & (s.car_2_60 < 0)]
print(f"\nQ5 events that rose on the announcement AND then fell over days +2..+60: {len(up_then_down)} of {len(s)} ({len(up_then_down) / len(s):.1%})")
print(f"  their average announcement gain {up_then_down.car_m1_1.mean():+.2%}, average later fall {up_then_down.car_2_60.mean():+.2%}")
down_on_good = s[s.car_m1_1 < 0]
print(f"Q5 events where the stock FELL on the announcement despite good news: {len(down_on_good)} of {len(s)} ({len(down_on_good) / len(s):.1%})")
print("median later drift: Q5", round(s.car_2_60.median() * 100, 2), "% | Q1", round(x[x.q == 1].car_2_60.median() * 100, 2), "%")

