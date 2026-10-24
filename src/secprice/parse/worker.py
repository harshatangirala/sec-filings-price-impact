"""S05 per-firm download + parse worker. One process handles one firm at a time with its own SEC client whose
rate is max_rps / n_processes, so the global rate never exceeds sec.max_rps. Outputs are written per firm
atomically (resume never duplicates rows; SPEC.md §17.3)."""
from __future__ import annotations

import hashlib
import os
import time
import traceback
from pathlib import Path

import pandas as pd

from ..sec.archives import doc_url, full_txt_url, parse_sgml
from ..sec.client import FixtureSECClient, SECBlockedError, SECClient
from .anonymize import build_masker
from .html import to_text
from .sections import eightk_items_text, extract_sections, form_family, mdna_from_ex13

SECTION_COLS = ["accession", "cik", "form", "section_id", "text_raw", "text_masked", "n_chars", "n_tokens_est",
                "parse_method", "by_reference", "truncated", "mask_firm", "mask_ticker"]
EXHIBIT_COLS = ["accession", "cik", "exhibit_type", "sequence", "filename", "description", "text", "n_chars"]
LEDGER_COLS = ["accession", "cik", "form", "url", "status", "error", "bytes", "checksum", "n_tables", "text_method",
               "header_acceptance_raw", "header_items", "header_item_titles", "header_period", "truncated",
               "fallback", "n_sections", "elapsed_s"]
MDNA_FALLBACK_CHARS = 15000


def _client(job: dict):
    if job.get("fixture_root"):
        return FixtureSECClient(job["fixture_root"])
    return SECClient(job["user_agent"], max_rps=job["rps_per_proc"], cache_dir=job["cache_dir"],
                     cache_days=job["cache_days"], timeout_s=job["timeout_s"], max_retries=job["max_retries"],
                     backoff_base_s=job["backoff_base_s"], blocked_403_limit=job["blocked_403_limit"],
                     cache_max_bytes=job.get("cache_max_bytes", 0))


def _sec_rows(acc, cik, form, secs: dict, mask) -> list[dict]:
    rows = []
    for sid, s in secs.items():
        txt = s["text"]
        if not txt or len(txt) < 40:
            continue
        masked, mc = mask(txt)
        rows.append({"accession": acc, "cik": cik, "form": form, "section_id": sid, "text_raw": txt,
                     "text_masked": masked, "n_chars": len(txt), "n_tokens_est": int(len(txt.split()) * 1.3),
                     "parse_method": s["method"], "by_reference": bool(s.get("by_reference")),
                     "truncated": bool(s.get("truncated")), "mask_firm": mc["firm"], "mask_ticker": mc["ticker"]})
    return rows


def process_filing(client, f: dict, mask, max_chars: int, max_8k: int, max_doc: int) -> tuple[list, list, dict]:
    cik, acc, form = int(f["cik"]), f["accession"], f["form"]
    filer = int(f.get("filer_cik") or cik)               # URLs use the filer's CIK; outputs use the firm id
    fam = form_family(form)
    t0 = time.time()
    led = {"accession": acc, "cik": cik, "form": form, "status": "ok", "error": None, "bytes": 0, "checksum": None,
           "n_tables": 0, "text_method": None, "header_acceptance_raw": None, "header_items": None,
           "header_item_titles": None, "header_period": None, "truncated": False, "fallback": None, "n_sections": 0}
    sec_rows, ex_rows = [], []
    try:
        if fam == "8-K":
            url = full_txt_url(filer, acc)
            raw = client.get(url, max_bytes=max_8k)
            led["url"] = url
            if raw is None:
                led["status"] = "http_404"
                return sec_rows, ex_rows, led
            led["bytes"], led["checksum"] = len(raw), hashlib.sha256(raw).hexdigest()
            sub = parse_sgml(raw, truncated=len(raw) >= max_8k)
            led.update(header_acceptance_raw=sub.acceptance_raw, header_items=",".join(sub.items),
                       header_item_titles=" || ".join(sub.item_titles), header_period=sub.period,
                       truncated=sub.truncated)
            main = [d for d in sub.documents if d.type.upper().startswith("8-K")] or sub.documents[:1]
            secs = {}
            if main:
                txt, nt, meth = to_text(main[0].text)
                led["n_tables"], led["text_method"] = nt, meth
                secs["8K_ITEMS"] = eightk_items_text(txt, max_chars)
            ex_txt = []
            for d in sub.documents:
                if d.type.upper().startswith("EX-99"):
                    t, nt, _ = to_text(d.text)
                    led["n_tables"] += nt
                    ex_rows.append({"accession": acc, "cik": cik, "exhibit_type": d.type.upper(), "sequence": d.sequence,
                                    "filename": d.filename, "description": d.description, "text": t[:max_chars],
                                    "n_chars": len(t)})
                    ex_txt.append(t)
            if ex_txt:
                joined = "\n\n".join(ex_txt)
                secs["8K_EX99"] = {"text": joined[:max_chars], "method": "ex99", "by_reference": False,
                                   "truncated": len(joined) > max_chars}
            sec_rows = _sec_rows(acc, cik, form, secs, mask)
        else:
            sub = None
            if f.get("primary_doc"):
                url = doc_url(filer, acc, f["primary_doc"])
                raw = client.get(url, max_bytes=max_doc)
                body = raw
            else:
                url, raw, body = full_txt_url(filer, acc), None, None
            if raw is None:
                url = full_txt_url(filer, acc)
                raw = client.get(url, max_bytes=max_doc)
                led["fallback"] = "full_txt"
                if raw is None:
                    led["url"], led["status"] = url, "http_404"
                    return sec_rows, ex_rows, led
                sub = parse_sgml(raw)
                led["header_acceptance_raw"] = sub.acceptance_raw
                led["header_period"] = sub.period
                docs = [d for d in sub.documents if d.type.upper().startswith(fam[:4])] or sub.documents[:1]
                body = docs[0].text if docs else ""
            led["url"], led["bytes"], led["checksum"] = url, len(raw), hashlib.sha256(raw).hexdigest()
            txt, nt, meth = to_text(body)
            led["n_tables"], led["text_method"] = nt, meth
            secs = extract_sections(txt, form, max_chars)
            if fam == "10-K":
                m7 = secs.get("10K_7")
                if m7 is None or m7.get("by_reference") or len(m7["text"]) < MDNA_FALLBACK_CHARS:
                    if sub is None:
                        full = client.get(full_txt_url(filer, acc), max_bytes=max_doc)
                        sub = parse_sgml(full) if full else None
                        if sub is not None and not led["header_acceptance_raw"]:
                            led["header_acceptance_raw"] = sub.acceptance_raw
                    if sub is not None:
                        for d in sub.documents:
                            if d.type.upper().startswith("EX-13"):
                                t13, _, _ = to_text(d.text)
                                alt = mdna_from_ex13(t13, max_chars)
                                if alt and (m7 is None or len(alt["text"]) > 2 * len(m7["text"])):
                                    secs["10K_7"] = alt
                                    led["fallback"] = (led["fallback"] or "") + "+ex13_mdna"
                                    break
            sec_rows = _sec_rows(acc, cik, form, secs, mask)
            if not led["header_acceptance_raw"]:
                # the submissions-JSON time is unreliable (two conventions observed); read the SGML header from the first
                # 20 KB of the full submission (one small request) - decision log D-047
                pre = client.get(full_txt_url(filer, acc), max_bytes=20000)
                if pre:
                    hs = parse_sgml(pre, truncated=True)
                    led["header_acceptance_raw"] = hs.acceptance_raw
                    led["header_period"] = led["header_period"] or hs.period
        led["n_sections"] = len(sec_rows)
        if not sec_rows:
            led["status"] = "no_sections"
    except SECBlockedError:
        raise
    except Exception as e:
        led["status"], led["error"] = "error", f"{type(e).__name__}: {e}"[:300]
    led["elapsed_s"] = round(time.time() - t0, 2)
    return sec_rows, ex_rows, led


def run_firm(job: dict) -> dict:
    """Download + parse every task filing of one firm; write sections/exhibits/ledger parquet atomically."""
    out = Path(job["out_dir"])
    cik = int(job["cik"])
    done_flag = out / "ledger" / f"{cik:010d}.parquet"
    if done_flag.exists():
        if job.get("retry_errors"):
            return _retry_failed(job, out, cik, done_flag)
        return {"cik": cik, "skipped": True}
    client = _client(job)
    mask = build_masker(job["names"], job["tickers"])
    secs, exs, leds = [], [], []
    for f in job["filings"]:
        s, e, l = process_filing(client, f, mask, job["max_chars"], job["max_8k_bytes"], job["max_doc_bytes"])
        secs += s
        exs += e
        leds.append(l)
    for sub, rows, cols in (("sections", secs, SECTION_COLS), ("exhibits", exs, EXHIBIT_COLS)):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=cols)
        tmp = d / f".tmp_{cik:010d}.parquet"
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, d / f"{cik:010d}.parquet")
    (out / "ledger").mkdir(parents=True, exist_ok=True)
    led = pd.DataFrame(leds, columns=LEDGER_COLS)
    tmp = out / "ledger" / f".tmp_{cik:010d}.parquet"
    led.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, done_flag)
    return {"cik": cik, "skipped": False, "n": len(leds), "ok": int((led.status == "ok").sum()),
            "stats": getattr(client, "stats", {})}


def _retry_failed(job: dict, out: Path, cik: int, done_flag: Path) -> dict:
    """Re-process only the filings whose earlier attempt ended in a transient error (e.g. repeated HTTP 503) and replace
    their rows; every other filing of the firm is untouched (sec.retry_failed_filings)."""
    led = pd.read_parquet(done_flag)
    bad = set(led.loc[led.status == "error", "accession"])
    if not bad:
        return {"cik": cik, "skipped": True}
    client = _client(job)
    mask = build_masker(job["names"], job["tickers"])
    secs, exs, leds = [], [], []
    for f in job["filings"]:
        if f["accession"] in bad:
            s_, e_, l_ = process_filing(client, f, mask, job["max_chars"], job["max_8k_bytes"], job["max_doc_bytes"])
            secs += s_
            exs += e_
            leds.append(l_)
    for sub, rows, cols in (("sections", secs, SECTION_COLS), ("exhibits", exs, EXHIBIT_COLS)):
        p = out / sub / f"{cik:010d}.parquet"
        old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=cols)
        new = pd.concat([old[~old.accession.isin(bad)], pd.DataFrame(rows, columns=cols)], ignore_index=True)
        tmp = out / sub / f".tmp_{cik:010d}.parquet"
        new.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, p)
    led2 = pd.concat([led[~led.accession.isin(bad)], pd.DataFrame(leds, columns=LEDGER_COLS)], ignore_index=True)
    tmp = out / "ledger" / f".tmp_{cik:010d}.parquet"
    led2.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, done_flag)
    return {"cik": cik, "skipped": False, "retried": len(bad), "ok": int((pd.DataFrame(leds).status == "ok").sum()) if leds else 0}


def run_firm_safe(job: dict) -> dict:
    try:
        return run_firm(job)
    except SECBlockedError as e:
        return {"cik": job["cik"], "blocked": str(e)}
    except Exception as e:
        return {"cik": job["cik"], "crash": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
