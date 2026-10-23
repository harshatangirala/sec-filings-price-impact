"""S02 Universe: firms common to our OHLCV files and the HF 10-K corpus, mapped ticker->CIK, S&P-500-history
aware, outcome-free selection (SPEC.md §3 universe.*, hard rule 9)."""
from __future__ import annotations

import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from ..sec.submissions import norm_ticker

SP500_URL = "https://raw.githubusercontent.com/fja05680/sp500/{commit}/sp500_ticker_start_end.csv"
FF12_NAMES = {1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems", 6: "BusEq", 7: "Telcm", 8: "Utils",
              9: "Shops", 10: "Hlth", 11: "Money", 12: "Other"}


def load_sp500(commit: str, cache_path=None) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        r = requests.get(SP500_URL.format(commit=commit), timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if cache_path is not None:
            cache_path.write_text(r.text, encoding="utf-8")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def load_ff12(url: str, cache_path=None) -> list[tuple[int, int, int]]:
    if cache_path is not None and cache_path.exists():
        raw = cache_path.read_bytes()
    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw = r.content
        if cache_path is not None:
            cache_path.write_bytes(raw)
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin1")
    ranges, cur = [], None
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(\w+)\s", line)
        if m and not re.match(r"\s*\d{4}-\d{4}", line):
            cur = int(m.group(1))
            continue
        m2 = re.match(r"\s*(\d{4})-(\d{4})", line)
        if m2 and cur:
            ranges.append((int(m2.group(1)), int(m2.group(2)), cur))
    return ranges


def sic_to_ff12(sic, ranges) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Other"
    for lo, hi, k in ranges:
        if lo <= s <= hi:
            return FF12_NAMES[k]
    return "Other"


def map_tickers(ohlcv: pd.DataFrame, sec_map: pd.DataFrame, hf_census: pd.DataFrame) -> pd.DataFrame:
    """ticker -> CIK via SEC's current ticker map, then the HF corpus tickers as a fallback."""
    m = dict(zip(sec_map["ticker"], sec_map["cik"].astype(int)))
    hf = {}
    for c, ts in zip(hf_census["cik"].astype(int), hf_census["tickers"].fillna("")):
        for t in str(ts).split("|"):
            if t:
                hf.setdefault(t.upper(), c)
    out = ohlcv.copy()
    nt = out["ticker"].map(norm_ticker)
    out["cik_sec"] = nt.map(m).astype("Int64")
    out["cik_hf"] = nt.map(hf).astype("Int64")
    out["cik"] = out["cik_sec"].fillna(out["cik_hf"]).astype("Int64")
    out["mapping_source"] = np.where(out["cik_sec"].notna(), "sec_company_tickers_exchange",
                                     np.where(out["cik_hf"].notna(), "hf_corpus_tickers", None))
    # a reorganisation (new holding-company CIK) leaves the history under the predecessor CIK: keep both
    out["ciks_all"] = [json.dumps(sorted({int(x) for x in (a, b) if pd.notna(x)})) for a, b in zip(out.cik_sec, out.cik_hf)]
    return out


def sp500_intervals(sp: pd.DataFrame, ticker: str) -> list[tuple]:
    rows = sp[sp.ticker.isin({ticker.upper(), norm_ticker(ticker), ticker.upper().replace("-", ".")})]
    return [(a, b) for a, b in zip(rows.start_date, rows.end_date)]


def select_universe(cand: pd.DataFrame, n: int, seed: int, fixed: list[str] | None = None,
                    balanced: bool = False) -> pd.DataFrame:
    """Deterministic, outcome-free selection: S&P-500 members first, then seeded sector-stratified fill.
    Never reads price returns (hard rule 9)."""
    cand = cand.copy()
    cand["selected"] = False
    cand["selection_rank"] = np.nan
    if fixed:
        sel = cand["ticker"].isin([t.upper() for t in fixed]) & cand["eligible"]
        cand.loc[sel, "selected"] = True
        cand.loc[sel, "selection_rank"] = np.arange(sel.sum())
        return cand
    elig = cand[cand["eligible"]].copy()
    rng = np.random.default_rng(seed)
    elig["_u"] = rng.random(len(elig))
    picked: list[int] = []

    def take(pool: pd.DataFrame, k: int) -> list[int]:
        if k <= 0 or pool.empty:
            return []
        if k >= len(pool):
            return pool.sort_values("_u").index.tolist()
        counts = pool.groupby("ff12_industry").size()
        if balanced:
            quota = pd.Series(k // len(counts), index=counts.index).clip(upper=counts)
        else:
            raw = counts / counts.sum() * k
            quota = np.floor(raw).astype(int)
            rem = k - quota.sum()
            for ind in (raw - quota).sort_values(ascending=False).index[:rem]:
                quota[ind] += 1
            quota = quota.clip(upper=counts)
        out = []
        for ind, q in quota.items():
            out += pool[pool.ff12_industry == ind].sort_values("_u").index[:q].tolist()
        short = k - len(out)
        if short > 0:
            rest = pool.drop(index=out).sort_values("_u")
            out += rest.index[:short].tolist()
        return out

    a = elig[elig["ever_sp500"]]
    b = elig[~elig["ever_sp500"]]
    picked += take(a, min(n, len(a)))
    picked += take(b, n - len(picked))
    cand.loc[picked, "selected"] = True
    order = elig.loc[picked].sort_values(["ever_sp500", "_u"], ascending=[False, True]).index
    cand.loc[order, "selection_rank"] = np.arange(len(order))
    return cand


def firms_per_year(univ: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sel = univ[univ.selected]
    rows = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        y0, y1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        has_px = ((pd.to_datetime(sel.price_start) <= y1) & (pd.to_datetime(sel.price_end) >= y0)).sum()
        in_idx = 0
        for iv in sel["sp500_intervals"]:
            ivs = json.loads(iv) if isinstance(iv, str) else (iv or [])
            if any(pd.Timestamp(a) <= y1 and (pd.isna(pd.Timestamp(b)) or pd.Timestamp(b) >= y0) for a, b in ivs):
                in_idx += 1
        rows.append({"year": y, "firms_with_prices": int(has_px), "firms_in_sp500": int(in_idx), "unique_firms": len(sel)})
    return pd.DataFrame(rows)
