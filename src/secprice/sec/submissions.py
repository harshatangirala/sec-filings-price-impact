"""Submissions JSON (data.sec.gov/submissions): firm metadata and the full filing list, including the older
paged files (SPEC.md §13.5, S04)."""
from __future__ import annotations

import pandas as pd

SUB_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUB_PAGE_URL = "https://data.sec.gov/submissions/{name}"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

_COLS = {
    "accessionNumber": "accession", "filingDate": "filing_date", "reportDate": "report_date",
    "acceptanceDateTime": "acceptance_raw", "act": "act", "form": "form", "fileNumber": "file_number",
    "items": "items_raw", "size": "size", "isXBRL": "is_xbrl", "isInlineXBRL": "is_inline_xbrl",
    "primaryDocument": "primary_doc", "primaryDocDescription": "primary_doc_desc",
}


def _frame(block: dict) -> pd.DataFrame:
    if not block:
        return pd.DataFrame(columns=list(_COLS.values()))
    df = pd.DataFrame({k: block.get(k, [None] * len(block.get("accessionNumber", []))) for k in _COLS})
    return df.rename(columns=_COLS)


def fetch_submissions(client, cik: int) -> tuple[dict, pd.DataFrame]:
    js = client.get_json(SUB_URL.format(cik=int(cik)))
    if js is None:
        return {}, pd.DataFrame(columns=list(_COLS.values()) + ["cik"])
    frames = [_frame(js.get("filings", {}).get("recent", {}))]
    for f in js.get("filings", {}).get("files", []) or []:
        page = client.get_json(SUB_PAGE_URL.format(name=f["name"]))
        if page:
            frames.append(_frame(page))
    df = pd.concat(frames, ignore_index=True)
    df["cik"] = int(cik)
    df = df.drop_duplicates("accession").reset_index(drop=True)
    meta = {k: js.get(k) for k in ("cik", "name", "tickers", "exchanges", "sic", "sicDescription", "formerNames",
                                   "fiscalYearEnd", "stateOfIncorporation", "entityType", "category", "ein")}
    meta["cik"] = int(cik)
    return meta, df


def fetch_ticker_map(client) -> pd.DataFrame:
    js = client.get_json(TICKERS_URL)
    if not js:
        return pd.DataFrame(columns=["cik", "name", "ticker", "exchange"])
    df = pd.DataFrame(js["data"], columns=js["fields"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def norm_ticker(t: str) -> str:
    """Our tickers use '.' for share classes (BRK.B); SEC maps use '-' (BRK-B)."""
    return str(t).upper().replace(".", "-").strip()
