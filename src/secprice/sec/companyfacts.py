"""XBRL company facts (data.sec.gov/api/xbrl/companyfacts) -> long fact table for selected tags (S06)."""
from __future__ import annotations

import pandas as pd

CF_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FACT_COLS = ["cik", "accession", "taxonomy", "tag", "unit", "value", "period_start", "period_end", "fy", "fp",
             "form", "filed", "frame"]


def fetch_companyfacts(client, cik: int):
    return client.get_json(CF_URL.format(cik=int(cik)))


def facts_frame(js: dict, tags: list[str], dei_tags: list[str] | None = None) -> pd.DataFrame:
    if not js:
        return pd.DataFrame(columns=FACT_COLS)
    rows = []
    cik = int(js.get("cik", 0))
    want = {("us-gaap", t) for t in tags} | {("dei", t) for t in (dei_tags or [])}
    for tax, block in (js.get("facts") or {}).items():
        for tag, spec in block.items():
            if (tax, tag) not in want:
                continue
            for unit, vals in (spec.get("units") or {}).items():
                for v in vals:
                    rows.append((cik, v.get("accn"), tax, tag, unit, v.get("val"), v.get("start"), v.get("end"),
                                 v.get("fy"), v.get("fp"), v.get("form"), v.get("filed"), v.get("frame")))
    df = pd.DataFrame(rows, columns=FACT_COLS)
    if len(df):
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df
