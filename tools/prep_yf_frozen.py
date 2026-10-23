"""One-off input preparation: freeze yfinance warm-up bars, actions and SPY into DATA_ROOT/inputs.

Run locally (no Colab credits). The notebook reads the frozen file first and falls back to live yfinance.
Explicit parameters: auto_adjust=False, actions=True, back_adjust=False, repair=False, interval=1d.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

out = Path(sys.argv[1])
tickers = sorted(p.stem for p in Path(sys.argv[2]).glob("*.parquet")) + ["SPY"]
WARM_END = pd.Timestamp("2001-03-01")
parts, fails = [], []
t0 = time.time()
for i, tk in enumerate(tickers):
    h = None
    for attempt in range(4):
        try:
            h = yf.Ticker(tk.replace(".", "-")).history(start="1998-01-01", end="2026-10-24", interval="1d",
                                                         auto_adjust=False, actions=True, back_adjust=False, repair=False)
            break
        except Exception as e:
            time.sleep(10 * (attempt + 1))
    if h is None or h.empty:
        fails.append(tk)
        continue
    h = h.reset_index()
    d = pd.DataFrame({"ticker": tk, "date": pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize(),
                      "open": h.Open, "high": h.High, "low": h.Low, "close": h.Close, "volume": h.Volume,
                      "dividends": h.get("Dividends", 0.0), "splits": h.get("Stock Splits", 0.0)})
    keep = (d.date < WARM_END) | (d.dividends > 0) | (d.splits > 0) | (tk == "SPY")
    parts.append(d[keep])
    if i % 50 == 0:
        print(i, tk, round(time.time() - t0), flush=True)
df = pd.concat(parts, ignore_index=True)
df.to_parquet(out, index=False, compression="zstd")
meta = {"params": "auto_adjust=False, actions=True, back_adjust=False, repair=False, interval=1d",
        "yfinance": yf.__version__, "retrieved_utc": pd.Timestamp.utcnow().isoformat(), "fails": fails,
        "rows": len(df), "note": "bars kept for date < 2001-03-01 (warm-up + seam) and all SPY bars; all actions"}
import json
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print("done", len(df), "fails", fails)
