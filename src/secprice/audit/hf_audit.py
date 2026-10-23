"""S01 Hugging Face SEC dataset audit (SPEC.md §13.5, Appendix C). Audit/cross-check only: the dataset's return
labels and returns are never read into any artifact (assertion)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

CENSUS_COLS = ["cik", "name", "tickers", "exchanges", "sic", "entity_type", "form", "filing_date", "acceptance_raw",
               "report_date", "doc_id", "n_sections_nonempty", "n_sentences", "split", "shard"]
FORBIDDEN_KEYS = {"labels", "returns"}
QA_SECTIONS = {"section_1": "10K_1", "section_1A": "10K_1A", "section_7": "10K_7", "section_7A": "10K_7A"}


def shard_files() -> list[str]:
    return [f"data/large/{s}/shard_{i}.jsonl" for s in ("train", "validate", "test") for i in range(10)]


def _url(repo: str, rev: str, fn: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{fn}"


def _iter_lines(url: str, retries: int = 5):
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                for line in r.iter_lines(chunk_size=1 << 22):
                    if line:
                        yield line
            return
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def census_shard(repo: str, rev: str, fn: str) -> pd.DataFrame:
    rows = []
    for line in _iter_lines(_url(repo, rev, fn)):
        d = json.loads(line)
        for f in d.get("filings", []):
            rep = f.get("report") or {}
            rows.append((d.get("cik"), d.get("name"), "|".join(d.get("tickers") or []), "|".join(d.get("exchanges") or []),
                         d.get("sic"), d.get("entityType"), f.get("form"), f.get("filingDate"), f.get("acceptanceDateTime"),
                         f.get("reportDate"), f"{d.get('cik')}_{f.get('form')}_{str(f.get('reportDate') or '')[:4]}",
                         sum(1 for v in rep.values() if v), sum(len(v) for v in rep.values() if v),
                         fn.split("/")[2], fn))
    return pd.DataFrame(rows, columns=CENSUS_COLS)


def build_census(repo: str, rev: str, work_dir: Path, log=None) -> pd.DataFrame:
    """Stream every raw JSONL shard (~13.4 GB) and keep filing metadata only. Checkpointed per shard."""
    work_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for fn in shard_files():
        part = work_dir / (fn.replace("/", "__") + ".parquet")
        if not part.exists():
            t0 = time.time()
            df = census_shard(repo, rev, fn)
            df.to_parquet(part, index=False)
            if log:
                log.info(f"HF census {fn}: {len(df)} filings in {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(part))
    out = pd.concat(parts, ignore_index=True)
    assert not (set(out.columns) & FORBIDDEN_KEYS), "HF labels/returns must never be read (SPEC.md §13.5)"
    return out


def qa_sections_for(repo: str, rev: str, census: pd.DataFrame, wanted: pd.DataFrame) -> pd.DataFrame:
    """Section text (joined sentences) for a QA sample of (cik, filing_date) pairs; streams only needed shards."""
    want = set(zip(wanted["cik"].astype(str).str.zfill(10), wanted["filing_date"].astype(str)))
    shards = census[census.apply(lambda r: (str(r.cik).zfill(10), str(r.filing_date)) in want, axis=1)].shard.unique()
    rows = []
    for fn in shards:
        for line in _iter_lines(_url(repo, rev, fn)):
            d = json.loads(line)
            cik = str(d.get("cik")).zfill(10)
            for f in d.get("filings", []):
                if (cik, str(f.get("filingDate"))) not in want:
                    continue
                rep = f.get("report") or {}
                for k, sid in QA_SECTIONS.items():
                    sents = rep.get(k) or []
                    rows.append((int(cik), f.get("filingDate"), sid, " ".join(sents), len(sents)))
    return pd.DataFrame(rows, columns=["cik", "filing_date", "section_id", "hf_text", "hf_n_sentences"])


def audit_markdown(census: pd.DataFrame, repo: str, rev: str, card_note: str = "") -> str:
    c = census.copy()
    c["year"] = c.filing_date.astype(str).str[:4]
    by_year = c.groupby("year").size()
    forms = c.form.value_counts().to_dict()
    dup = int(c.duplicated(["cik", "filing_date", "form"]).sum())
    acc_null = int(c.acceptance_raw.isna().sum())
    empty = int((c.n_sentences == 0).sum())
    lines = [
        "# Hugging Face SEC dataset audit (S01)", "",
        f"- Repository: `{repo}` at revision `{rev}` (raw JSONL shards, `large` config)",
        "- Use: **audit, firm-overlap definition and 10-K section-parser QA only**. Its return labels and `returns` "
        "fields are never read (assertion in code). Filing text for the study comes from our own point-in-time "
        "EDGAR pipeline (SPEC.md §13.5).",
        f"- Firms (CIK): **{c.cik.nunique():,}**; filings: **{len(c):,}**; forms present: {forms}",
        f"- Filing dates: {c.filing_date.min()} to {c.filing_date.max()}",
        "- Forms **not** present: 10-Q and 8-K (they come from EDGAR).",
        f"- Duplicate (cik, filing date, form) rows: {dup}; missing acceptance timestamps: {acc_null}; "
        f"filings with zero sentences: {empty}",
        f"- Median sentences per filing: {int(c.n_sentences.median())}; median non-empty sections: "
        f"{int(c.n_sections_nonempty.median())} of 20", "",
        "## Filings by year", "", "| Year | Filings |", "|---|---|",
        *[f"| {y} | {n:,} |" for y, n in by_year.items()], "", card_note,
    ]
    return "\n".join(lines)
