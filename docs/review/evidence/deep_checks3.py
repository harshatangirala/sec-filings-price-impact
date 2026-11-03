"""Which features carried the pre-2020 direction signal, and did it decay? Univariate rank correlation of each feature with the
T_trade abnormal return, by era and by event type. Stored outputs only."""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = os.environ.get("RUN_ROOT", r"C:\Users\harsh\Downloads\sec_report_mini\full")
OUT = Path(__file__).resolve().parent
g = lambda s, f: max(glob.glob(f"{R}/{s}/*/{f}"), key=os.path.getmtime)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)
feat = pd.read_parquet(g("S09", "features.parquet"))
ev = pd.read_parquet(g("S07", "events.parquet"), columns=["event_id", "decision_time", "earnings_type", "event_type"])
ev["year"] = pd.to_datetime(ev.decision_time, utc=True).dt.tz_convert("America/New_York").dt.year
tg = pd.read_parquet(g("S08", "targets.parquet"), columns=["event_id", "target_def", "horizon", "ar_mm"], filters=[("target_def", "=", "T_trade"), ("horizon", "=", 1)]).dropna(subset=["ar_mm"])
m = feat.merge(ev, on="event_id").merge(tg[["event_id", "ar_mm"]], on="event_id")
num = [c for c in feat.columns if c != "event_id" and not c.startswith("_") and pd.api.types.is_numeric_dtype(feat[c]) and feat[c].nunique() > 2]
rows = []
for lab, a, b in (("2008-2019", 2008, 2019), ("2020-2026", 2020, 2026)):
    x = m[m.year.between(a, b)]
    for c in num:
        z = x[[c, "ar_mm"]].dropna()
        if len(z) > 3000:
            rho = spearmanr(z[c], z.ar_mm)[0]
            rows.append(dict(era=lab, feature=c, rho=rho, se=1 / np.sqrt(len(z)), n=len(z)))
r = pd.DataFrame(rows).pivot(index="feature", columns="era", values="rho"); r["diff"] = r["2020-2026"] - r["2008-2019"]
r.to_csv(OUT / "feature_direction_rho_by_era.csv")
print("TOP |rho| 2008-2019:"); print(r.reindex(r["2008-2019"].abs().sort_values(ascending=False).index).head(10).round(4).to_string())
print("TOP |rho| 2020-2026:"); print(r.reindex(r["2020-2026"].abs().sort_values(ascending=False).index).head(8).round(4).to_string())
# by event type
rows = []
for lab, x in (("earnings releases", m[m.earnings_type == "earnings_announcement_verified"]), ("other 8-K", m[(m.event_type == "8K") & (m.earnings_type != "earnings_announcement_verified")]), ("10-Q", m[m.event_type == "periodic_10Q"]), ("10-K", m[m.event_type == "periodic_10K"])):
    for era, a, b in (("2008-2019", 2008, 2019), ("2020-2026", 2020, 2026)):
        y = x[x.year.between(a, b)]
        for c in ("px_ret_5", "px_ret_20", "px_mom_12_2", "sue_event", "ear_last", "main_lm_net_tone"):
            z = y[[c, "ar_mm"]].dropna()
            if len(z) > 400:
                rows.append(dict(group=lab, era=era, feature=c, rho=spearmanr(z[c], z.ar_mm)[0], n=len(z)))
d = pd.DataFrame(rows).pivot_table(index=["group", "feature"], columns="era", values="rho"); d.to_csv(OUT / "direction_rho_by_type_era.csv"); print(d.round(3).to_string())
