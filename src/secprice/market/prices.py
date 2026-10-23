"""S03 prices: pluggable providers and the firm x session panel (SPEC.md §13.5 'Prices', §16 prices.parquet).

Series kept separately (SPEC.md §5.5): close_raw, close_split_adj and a total-return index. Local TriFrame bars
are split-adjusted (Yahoo split history), not dividend-adjusted; dividends come from Yahoo actions (split-adjusted
amounts, the same basis). Before the local history starts, yfinance supplies the market warm-up (explicit
parameters recorded).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

BAR_COLS = ["date", "open", "high", "low", "close", "volume", "source"]


class LocalTriframeProvider:
    """DATA_ROOT/inputs/ohlcv_1day/<TICKER>.parquet with columns date, open, high, low, close, volume, source."""

    def __init__(self, folder: Path, version: str = "triframe_final_1day"):
        self.folder = Path(folder)
        self.version = version

    def tickers(self) -> list[str]:
        return sorted(p.stem for p in self.folder.glob("*.parquet"))

    def coverage(self) -> pd.DataFrame:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        rows = []
        for p in sorted(self.folder.glob("*.parquet")):
            t = pq.read_table(p, columns=["date"])
            rows.append({"ticker": p.stem, "price_start": pc.min(t["date"]).as_py(), "price_end": pc.max(t["date"]).as_py(),
                         "n_bars": t.num_rows})
        df = pd.DataFrame(rows)
        for c in ("price_start", "price_end"):
            df[c] = pd.to_datetime(df[c]).dt.normalize()
        return df

    def load(self, ticker: str) -> pd.DataFrame:
        p = self.folder / f"{ticker}.parquet"
        if not p.exists():
            return pd.DataFrame(columns=BAR_COLS)
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "source" not in df:
            df["source"] = "local"
        df["source"] = "local:" + df["source"].astype(str)
        return df[BAR_COLS].sort_values("date").reset_index(drop=True)


class YFinanceProvider:
    """yfinance with explicit parameters; reads a frozen cache file first when present (reproducible runs)."""

    def __init__(self, params: dict, cache_file: Path | None = None, log=None):
        self.params = dict(params)
        self.cache_file = Path(cache_file) if cache_file else None
        self.log = log
        self._cache = None
        try:
            import yfinance
            self.version = f"yfinance {yfinance.__version__}"
        except Exception:
            self.version = "yfinance unavailable"

    def _cached(self) -> pd.DataFrame | None:
        if self._cache is None and self.cache_file is not None and self.cache_file.exists():
            self._cache = pd.read_parquet(self.cache_file)
            self._cache["date"] = pd.to_datetime(self._cache["date"]).dt.normalize()
        return self._cache

    def history(self, ticker: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(bars, actions) for [start, end). Actions: date, type in {dividend, split}, value."""
        c = self._cached()
        if c is not None and ticker in set(c["ticker"].unique()):
            sub = c[(c.ticker == ticker) & (c.date >= pd.Timestamp(start)) & (c.date < pd.Timestamp(end))]
            bars = sub[["date", "open", "high", "low", "close", "volume"]].assign(source="yfinance_frozen")
            acts = sub.melt(id_vars="date", value_vars=["dividends", "splits"], var_name="type", value_name="value")
            acts = acts[acts.value > 0].assign(type=lambda d: d.type.map({"dividends": "dividend", "splits": "split"}))
            return bars.reset_index(drop=True), acts.reset_index(drop=True)
        import yfinance as yf
        p = self.params
        for attempt in range(4):
            try:
                h = yf.Ticker(ticker.replace(".", "-")).history(start=start, end=end, interval=p.get("interval", "1d"),
                                                               auto_adjust=p.get("auto_adjust", False),
                                                               actions=p.get("actions", True),
                                                               back_adjust=p.get("back_adjust", False),
                                                               repair=p.get("repair", False))
                break
            except Exception as e:  # rate limits: back off, never evade
                if self.log:
                    self.log.warning(f"yfinance {ticker} attempt {attempt + 1}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            return pd.DataFrame(columns=BAR_COLS), pd.DataFrame(columns=["date", "type", "value"])
        if h is None or h.empty:
            return pd.DataFrame(columns=BAR_COLS), pd.DataFrame(columns=["date", "type", "value"])
        h = h.reset_index()
        h["date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
        bars = pd.DataFrame({"date": h.date, "open": h.Open, "high": h.High, "low": h.Low, "close": h.Close,
                             "volume": h.Volume, "source": "yfinance_live"})
        acts = []
        if "Dividends" in h:
            acts.append(pd.DataFrame({"date": h.date, "type": "dividend", "value": h.Dividends}))
        if "Stock Splits" in h:
            acts.append(pd.DataFrame({"date": h.date, "type": "split", "value": h["Stock Splits"]}))
        acts = pd.concat(acts) if acts else pd.DataFrame(columns=["date", "type", "value"])
        return bars, acts[acts.value > 0].reset_index(drop=True)


def load_actions_file(path: Path) -> pd.DataFrame:
    """inputs/yahoo_actions.parquet: ticker, date, type (dividend|split), value."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["ticker", "date", "type", "value"])
    a = pd.read_parquet(path)
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    return a


def build_series(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """close_raw, close_split_adj, total_return_index from split-adjusted bars + actions (split-adjusted dividends).

    close_raw(d) = close_split_adj(d) x product of split ratios with split date > d.
    TRI(t) = TRI(t-1) x (C(t) + D(t)) / C(t-1) where D(t) is the dividend with ex-date t.
    """
    b = bars.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True).copy()
    b = b.rename(columns={"close": "close_split_adj"})
    splits = actions[actions.type == "split"][["date", "value"]] if len(actions) else pd.DataFrame(columns=["date", "value"])
    divs = actions[actions.type == "dividend"].groupby("date")["value"].sum() if len(actions) else pd.Series(dtype=float)
    factor = np.ones(len(b))
    for d, r in zip(pd.to_datetime(splits["date"]), splits["value"].astype(float)):
        if r > 0:
            factor[b["date"].values < np.datetime64(d)] *= r
    b["close_raw"] = b["close_split_adj"] * factor
    b["dividend"] = b["date"].map(divs).fillna(0.0).astype(float)
    c = b["close_split_adj"].to_numpy(float)
    dv = b["dividend"].to_numpy(float)
    gross = np.ones(len(b))
    if len(b) > 1:
        prev = c[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (c[1:] + dv[1:]) / prev
        g[~np.isfinite(g)] = np.nan
        gross[1:] = g
    tri = np.nancumprod(np.where(np.isnan(gross), 1.0, gross))
    tri[np.isnan(c)] = np.nan
    b["total_return_index"] = tri
    return b


def splice(local: pd.DataFrame, warm: pd.DataFrame, seam_days: int = 40) -> tuple[pd.DataFrame, dict]:
    """Use yfinance only before the first local date; report the seam agreement on overlapping days."""
    if local.empty:
        return warm.copy(), {"seam_overlap_days": 0, "seam_median_abs_pct": np.nan, "warmup_days": len(warm)}
    first = local["date"].min()
    ov = warm[(warm.date >= first)].merge(local[["date", "close"]], on="date", suffixes=("_yf", "_local")).head(seam_days)
    med = float((ov.close_local / ov.close_yf - 1).abs().median() * 100) if len(ov) else np.nan
    pre = warm[warm.date < first]
    out = pd.concat([pre, local], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out, {"seam_overlap_days": int(len(ov)), "seam_median_abs_pct": med, "warmup_days": int(len(pre))}


class PricePanel:
    """Per-firm arrays aligned to the global session calendar (NaN where no bar)."""

    def __init__(self, prices: pd.DataFrame, cal):
        self.cal = cal
        self.n = len(cal)
        self.firms: dict[str, dict[str, np.ndarray]] = {}
        sess = cal.sessions
        for key, g in prices.groupby("ticker", sort=False):
            idx = sess.searchsorted(pd.to_datetime(g["date"]).values)
            ok = (idx < self.n) & (sess[np.clip(idx, 0, self.n - 1)] == pd.to_datetime(g["date"]).values)
            arr = {}
            for col in ("open", "high", "low", "close_split_adj", "close_raw", "total_return_index", "volume", "dividend"):
                a = np.full(self.n, np.nan)
                if col in g:
                    a[idx[ok]] = g[col].to_numpy(float)[ok]
                arr[col] = a
            tri = arr["total_return_index"]
            r = np.full(self.n, np.nan)
            r[1:] = tri[1:] / tri[:-1] - 1
            arr["ret"] = r
            self.firms[str(key)] = arr

    def has(self, ticker: str) -> bool:
        return ticker in self.firms

    def __getitem__(self, ticker: str) -> dict[str, np.ndarray]:
        return self.firms[ticker]
