"""Deterministic synthetic world for the offline integration smoke (SPEC.md §18.2; addendum P1.5).

Three firms (AAA, BBB, CCC), 2019-06 to 2024-06: prices (with a split and dividends), SPY, factors, and SEC
responses in the real formats (submissions JSON with the double-shifted acceptance times observed on EDGAR, 10-K /
10-Q HTML primary documents, 8-K full submissions with SGML headers and EX-99 releases, XBRL company facts).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FIRMS = [("AAA", 1001, "AAA TECHNOLOGIES INC", "3571", 0.0004, 0.018),
         ("BBB", 1002, "BBB PHARMACEUTICAL CORP", "2834", 0.0002, 0.014),
         ("CCC", 1003, "CCC BANCORP", "6021", 0.0001, 0.016)]
POS = ["strong", "growth", "improved", "record", "gains", "achieved", "success", "favorable"]
NEG = ["loss", "decline", "adverse", "impairment", "litigation", "weak", "deterioration", "uncertain"]
UNC = ["uncertain", "may", "approximately", "possible", "risk", "unpredictable"]
ET = "America/New_York"


def _json_time(t_et: pd.Timestamp) -> str:
    """Encode an ET time the way the submissions JSON does (ET shifted by the ET->UTC offset twice, 'Z')."""
    off = -t_et.utcoffset().total_seconds()
    return (t_et.tz_localize(None) + pd.Timedelta(seconds=2 * off)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sentences(rng, tone: float, n: int, firm: str) -> str:
    out = []
    for _ in range(n):
        words = list(rng.choice(POS if rng.random() < 0.5 + tone / 2 else NEG, 2)) + list(rng.choice(UNC, 1))
        out.append(f"{firm} reported {words[0]} results with {words[1]} trends and {words[2]} conditions in the period "
                   f"across segment {int(rng.integers(1, 9))} operations.")
    return " ".join(out)


def make_world(root: Path, seed: int = 7) -> dict:
    import pandas_market_calendars as mcal
    rng = np.random.default_rng(seed)
    root = Path(root)
    inputs = root / "inputs"
    sec = root / "sec_fixtures"
    (inputs / "ohlcv_1day").mkdir(parents=True, exist_ok=True)
    (inputs / "lexicons").mkdir(parents=True, exist_ok=True)
    sch = mcal.get_calendar("NYSE").schedule(start_date="2019-06-01", end_date="2024-06-30")
    days = pd.DatetimeIndex(sch.index).tz_localize(None)
    n = len(days)
    mkt = rng.normal(0.0003, 0.01, n)
    # SPY + factors
    spy_close = 300 * np.exp(np.cumsum(mkt))
    spy_open = spy_close * np.exp(rng.normal(0, 0.002, n))
    yf_rows = []
    for d, o, c in zip(days, spy_open, spy_close):
        yf_rows.append(("SPY", d, o, max(o, c) * 1.003, min(o, c) * 0.997, c, 5e7, 0.0, 0.0))
    for k, d in enumerate(days):
        if d.month in (3, 6, 9, 12) and d.day >= 15 and (k == 0 or days[k - 1].month != d.month or days[k - 1].day < 15):
            yf_rows[k] = yf_rows[k][:7] + (1.5, 0.0)
    fac = pd.DataFrame({"date": days, "mkt_rf": mkt - 0.0001, "smb": rng.normal(0, 0.004, n), "hml": rng.normal(0, 0.004, n),
                        "rmw": rng.normal(0, 0.003, n), "cma": rng.normal(0, 0.003, n), "mom": rng.normal(0, 0.005, n),
                        "rf": np.full(n, 0.0001)})
    tickers_ex, hf_rows, sp_rows, facts_all = [], [], [], {}
    for tk, cik, name, sic, mu, vol in FIRMS:
        beta = 0.8 + 0.4 * rng.random()
        r = mu + beta * mkt + rng.normal(0, vol, n)
        close = 50 * np.exp(np.cumsum(r))
        opn = close * np.exp(rng.normal(0, 0.004, n))
        vol_sh = rng.integers(1_000_000, 5_000_000, n).astype(float)
        local = pd.DataFrame({"date": days, "open": opn, "high": np.maximum(opn, close) * 1.01, "low": np.minimum(opn, close) * 0.99,
                              "close": close, "volume": vol_sh, "minute_bars": 390, "source": "1min"})
        local[local.date >= "2020-01-02"].to_parquet(inputs / "ohlcv_1day" / f"{tk}.parquet", index=False)
        for d, o, c, v in zip(days, opn, close, vol_sh):
            if d < pd.Timestamp("2020-03-01"):
                yf_rows.append((tk, d, o, max(o, c) * 1.01, min(o, c) * 0.99, c, v, 0.0, 0.0))
        if tk == "AAA":
            yf_rows.append((tk, pd.Timestamp("2022-06-06"), np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 2.0))
        if tk == "BBB":
            for d in days[(days.month.isin([2, 5, 8, 11])) & (days.day <= 7)][::5]:
                yf_rows.append((tk, d, np.nan, np.nan, np.nan, np.nan, np.nan, 0.45, 0.0))
        tickers_ex.append([cik, name, tk, "NYSE"])
        if tk != "CCC":                               # CCC is never an S&P 500 member: both index buckets are exercised (D-056)
            sp_rows.append((tk, "2010-01-01", "" if tk != "BBB" else "2022-12-31"))   # BBB leaves the index (point-in-time flag)
        # ---------------- filings
        filings = []
        sub_docs = {}
        eps_hist = []
        eps_q = 1.0 + rng.normal(0, 0.1)
        for y in range(2018, 2024):
            for q, (m10q, mrel) in enumerate([(5, 4), (8, 7), (11, 10), (2, 1)]):
                fy = y if q < 3 else y + 1
                pe = pd.Timestamp(year=y, month=[3, 6, 9, 12][q], day=1) + pd.offsets.MonthEnd(0)
                eps_q = eps_q * (1 + rng.normal(0.01, 0.08))
                eps_hist.append((pe, round(eps_q, 2), q))
                rel_day = pd.Timestamp(year=fy, month=mrel, day=1) + pd.offsets.BDay(int(rng.integers(15, 20)))
                hour = [7, 16, 12][int(rng.integers(0, 3))]
                rel_t = (rel_day + pd.Timedelta(hours=hour, minutes=int(rng.integers(1, 55)), seconds=int(rng.integers(0, 59)))).tz_localize(ET)
                filings.append(("8-K", rel_t, "2.02,9.01", pe, {"eps": round(eps_q, 2), "pe": pe}))
                per_day = pd.Timestamp(year=fy, month=m10q, day=1) + pd.offsets.BDay(int(rng.integers(2, 8)))
                per_t = (per_day + pd.Timedelta(hours=int(rng.choice([6, 16, 17])), minutes=int(rng.integers(0, 59)))).tz_localize(ET)
                filings.append(("10-K" if q == 3 else "10-Q", per_t, "", pe, {"eps": round(eps_q, 2), "pe": pe, "q": q}))
            mg = (pd.Timestamp(year=y, month=9, day=10) + pd.offsets.BDay(int(rng.integers(0, 10)))) + pd.Timedelta(hours=8, minutes=30)
            filings.append(("8-K", mg.tz_localize(ET), "5.02", None, {}))
        filings.sort(key=lambda x: x[1])
        recent = {k: [] for k in ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "act", "form", "fileNumber",
                                  "filmNumber", "items", "core_type", "size", "isXBRL", "isInlineXBRL", "isXBRLNumeric",
                                  "primaryDocument", "primaryDocDescription")}
        cf_eps = []
        for j, (form, t, items, pe, info) in enumerate(filings):
            acc = f"{cik:010d}-{t.year % 100:02d}-{j:06d}"
            nd = acc.replace("-", "")
            fdate = (t.tz_localize(None) + (pd.Timedelta(days=1) if t.hour >= 22 else pd.Timedelta(0))).strftime("%Y-%m-%d")
            doc = f"{tk.lower()}-{nd}.htm"
            recent["accessionNumber"].append(acc)
            recent["filingDate"].append(fdate)
            recent["reportDate"].append(pe.strftime("%Y-%m-%d") if pe is not None else fdate)
            recent["acceptanceDateTime"].append(_json_time(t))
            recent["act"].append("34")
            recent["form"].append(form)
            recent["fileNumber"].append("001-0000")
            recent["filmNumber"].append("0")
            recent["items"].append(items)
            recent["core_type"].append(form)
            recent["size"].append(10000)
            recent["isXBRL"].append(1)
            recent["isInlineXBRL"].append(1)
            recent["isXBRLNumeric"].append(1)
            recent["primaryDocument"].append(doc)
            recent["primaryDocDescription"].append(form)
            base = sec / "www.sec.gov" / "Archives" / "edgar" / "data" / str(cik) / nd
            base.mkdir(parents=True, exist_ok=True)
            tone = float(rng.normal(0, 0.4))
            hdr = (f"<SEC-HEADER>\nACCESSION NUMBER: {acc}\nCONFORMED SUBMISSION TYPE: {form}\n<ACCEPTANCE-DATETIME>{t.strftime('%Y%m%d%H%M%S')}\n"
                   f"FILED AS OF DATE: {fdate.replace('-', '')}\n")
            if form == "8-K":
                titles = ["Results of Operations and Financial Condition", "Financial Statements and Exhibits"] if "2.02" in items else \
                    ["Departure of Directors or Certain Officers; Election of Directors"]
                hdr += "".join(f"ITEM INFORMATION: {x}\n" for x in titles) + "</SEC-HEADER>\n"
                body8k = (f"<html><body><p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 8-K</p>"
                          f"<p>{name}</p>" + "".join(f"<p>Item {it}. {tt}</p><p>{_sentences(rng, tone, 3, name)}</p>"
                                                     for it, tt in zip(items.split(","), titles)) +
                          "<p>SIGNATURES</p><p>Pursuant to the requirements of the Securities Exchange Act of 1934, the registrant has duly caused</p></body></html>")
                txt = hdr + f"<DOCUMENT>\n<TYPE>8-K\n<SEQUENCE>1\n<FILENAME>{doc}\n<TEXT>\n{body8k}\n</TEXT>\n</DOCUMENT>\n"
                if "2.02" in items:
                    e = info["eps"]
                    pe_s = info["pe"].strftime("%B %d, %Y").replace(" 0", " ")
                    ex = (f"<html><body><p>{name} Reports Results for the Quarter Ended {pe_s}</p><p>{_sentences(rng, tone, 4, name)}</p>"
                          f"<p>Net income for the three months ended {pe_s} increased. Revenue grew.</p>"
                          f"<table><tr><td>Earnings per share:</td></tr><tr><td>Diluted earnings per share</td><td>$</td><td>{e:.2f}</td>"
                          f"<td>$</td><td>{e * 0.9:.2f}</td></tr></table></body></html>")
                    txt += f"<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>ex991.htm\n<TEXT>\n{ex}\n</TEXT>\n</DOCUMENT>\n"
                (base / f"{acc}.txt").write_text(txt, encoding="utf-8")
            else:
                hdr += "</SEC-HEADER>\n"
                if form == "10-K":
                    html = ("<html><body><p>FORM 10-K</p><p>TABLE OF CONTENTS</p><p>Item 1. Business</p><p>Item 1A. Risk Factors</p>"
                            "<p>Item 7. Management's Discussion and Analysis</p>"
                            f"<p>PART I</p><p>Item 1. Business</p><p>{_sentences(rng, tone, 25, name)}</p>"
                            f"<p>Item 1A. Risk Factors</p><p>{_sentences(rng, tone - 0.3, 30, name)}</p>"
                            "<p>Item 2. Properties</p><p>We own offices.</p>"
                            f"<p>PART II</p><p>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>"
                            f"<p>{_sentences(rng, tone, 40, name)}</p>"
                            f"<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p><p>{_sentences(rng, tone, 8, name)}</p>"
                            "<p>Item 8. Financial Statements and Supplementary Data</p><table><tr><td>Total assets</td><td>1,000</td></tr></table>"
                            "</body></html>")
                else:
                    html = ("<html><body><p>FORM 10-Q</p><p>PART I - FINANCIAL INFORMATION</p><p>Item 1. Financial Statements</p>"
                            "<table><tr><td>Revenue</td><td>100</td></tr></table>"
                            f"<p>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>"
                            f"<p>{_sentences(rng, tone, 25, name)}</p>"
                            f"<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p><p>{_sentences(rng, tone, 5, name)}</p>"
                            "<p>Item 4. Controls and Procedures</p><p>Effective.</p><p>PART II - OTHER INFORMATION</p>"
                            f"<p>Item 1A. Risk Factors</p><p>{_sentences(rng, tone - 0.3, 6, name)}</p><p>Item 6. Exhibits</p></body></html>")
                (base / doc).write_text(html, encoding="utf-8")
                (base / f"{acc}.txt").write_text(hdr + f"<DOCUMENT>\n<TYPE>{form}\n<TEXT>\n{html}\n</TEXT>\n</DOCUMENT>\n", encoding="utf-8")
                (base / f"{acc}-index-headers.html").write_text(hdr, encoding="utf-8")
                # XBRL EPS facts reported in this filing (quarterly for 10-Q; FY + 9M for 10-K)
                qn = info["q"]
                pe = info["pe"]
                st = (pe - pd.offsets.MonthEnd(3) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if form == "10-Q":
                    cf_eps.append({"start": st, "end": pe.strftime("%Y-%m-%d"), "val": info["eps"], "accn": acc, "fy": pe.year,
                                   "fp": f"Q{qn + 1}", "form": form, "filed": fdate})
                else:
                    fy_start = (pe - pd.offsets.MonthEnd(12) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    prev3 = [e for (p_, e, q_) in eps_hist if p_ < pe][-3:]
                    cf_eps.append({"start": fy_start, "end": pe.strftime("%Y-%m-%d"), "val": round(sum(prev3) + info["eps"], 2), "accn": acc,
                                   "fy": pe.year, "fp": "FY", "form": form, "filed": fdate})
                    cf_eps.append({"start": fy_start, "end": (pe - pd.offsets.MonthEnd(3)).strftime("%Y-%m-%d"), "val": round(sum(prev3), 2),
                                   "accn": acc, "fy": pe.year, "fp": "FY", "form": form, "filed": fdate})
        subj = {"cik": str(cik), "entityType": "operating", "sic": sic, "sicDescription": "synthetic", "name": name, "tickers": [tk],
                "exchanges": ["NYSE"], "fiscalYearEnd": "1231", "stateOfIncorporation": "DE", "category": "Large accelerated filer",
                "formerNames": [], "filings": {"recent": recent, "files": []}}
        p = sec / "data.sec.gov" / "submissions"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"CIK{cik:010d}.json").write_text(json.dumps(subj), encoding="utf-8")
        assets = [{"end": e["end"], "val": 1e9 * (1 + 0.1 * k), "accn": e["accn"], "fy": e["fy"], "fp": e["fp"], "form": e["form"],
                   "filed": e["filed"]} for k, e in enumerate(cf_eps)]
        cf = {"cik": cik, "entityName": name, "facts": {
            "us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": cf_eps}},
                        "Assets": {"units": {"USD": assets}},
                        "Liabilities": {"units": {"USD": [{**a, "val": a["val"] * 0.6} for a in assets]}},
                        "StockholdersEquity": {"units": {"USD": [{**a, "val": a["val"] * 0.4} for a in assets]}}},
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [{**a, "val": 1e8} for a in assets]}}}}}
        p = sec / "data.sec.gov" / "api" / "xbrl" / "companyfacts"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"CIK{cik:010d}.json").write_text(json.dumps(cf), encoding="utf-8")
        for (form, t, *_r) in filings:
            if form == "10-K" and t.year <= 2021:
                hf_rows.append({"cik": f"{cik:010d}", "name": name, "tickers": tk, "exchanges": "NYSE", "sic": sic,
                                "entity_type": "operating", "form": "10-K", "filing_date": t.strftime("%Y-%m-%d"),
                                "acceptance_raw": _json_time(t), "report_date": None, "doc_id": f"{cik:010d}_10-K_{t.year}",
                                "n_sections_nonempty": 4, "n_sentences": 100, "split": "train", "shard": "synthetic"})
    p = sec / "www.sec.gov" / "files"
    p.mkdir(parents=True, exist_ok=True)
    (p / "company_tickers_exchange.json").write_text(json.dumps({"fields": ["cik", "name", "ticker", "exchange"], "data": tickers_ex}))
    yf = pd.DataFrame(yf_rows, columns=["ticker", "date", "open", "high", "low", "close", "volume", "dividends", "splits"])
    yf.to_parquet(inputs / "yf_frozen.parquet", index=False)
    pd.DataFrame(hf_rows).to_parquet(inputs / "hf_census_5da7e3c8b920.parquet", index=False)
    sp = pd.DataFrame(sp_rows, columns=["ticker", "start_date", "end_date"])
    sp.to_csv(inputs / "sp500_ticker_start_end_a2430f2a.csv", index=False)
    txt = (" 1 NoDur  Consumer Nondurables\n          0100-0999\n\n 5 Chems  Chemicals\n          2800-2829\n\n"
           " 6 BusEq  Business Equipment\n          3570-3579\n\n10 Hlth   Healthcare\n          2830-2839\n\n11 Money  Finance\n          6000-6999\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Siccodes12.txt", txt)
    (inputs / "Siccodes12.zip").write_bytes(buf.getvalue())
    lm = pd.DataFrame({"Word": [w.upper() for w in POS + NEG + UNC]})
    for c in ["Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal", "Constraining"]:
        lm[c] = 0
    lm.loc[lm.Word.isin([w.upper() for w in NEG]), "Negative"] = 2009
    lm.loc[lm.Word.isin([w.upper() for w in POS]), "Positive"] = 2009
    lm.loc[lm.Word.isin([w.upper() for w in UNC]), "Uncertainty"] = 2009
    lm.loc[lm.Word == "LITIGATION", "Litigious"] = 2009
    lm.to_csv(inputs / "lexicons" / "Loughran-McDonald_MasterDictionary_1993-2025.csv", index=False)
    return {"inputs": str(inputs), "sec": str(sec), "factors": fac, "tickers": [f[0] for f in FIRMS]}


def offline_overrides() -> dict:
    return {"universe": {"n_stocks": 3, "fixed_tickers": ["AAA", "BBB", "CCC"]},
            "study": {"start": "2021-01-01", "end": "2023-12-31"},
            "market": {"warmup_start": "2019-06-03"},
            "smoke": {"max_filings_per_firm_form": 20},
            "tuning": {"trials": {"lightgbm": 2, "catboost": 1, "enet": 2}, "permutations": 2},
            "bootstrap": {"n": 50}}
