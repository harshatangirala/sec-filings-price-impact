"""Firm anonymization: mask the firm's own names and tickers (firm-memorization control, SPEC.md §6.3 #4).
Regex masking runs on CPU for every section; NER-based PER/ORG masking (dslim/bert-base-NER, never money) is
applied on GPU before embedding when enabled."""
from __future__ import annotations

import re

_SUFFIX = re.compile(r"\b(incorporated|inc|corporation|corp|company|co|ltd|limited|plc|llc|lp|l\.p|holdings?|group|"
                     r"the|n\.?v|s\.?a|ag|trust|bancorp(oration)?|/[a-z]{2}/)\b\.?", re.I)
_TICKER_STOP = {"A", "I", "IT", "ON", "ALL", "ARE", "AN", "AM", "BE", "BY", "DO", "GO", "HAS", "HE", "IN", "IS", "ME",
                "NO", "OF", "OR", "SO", "TO", "UP", "US", "WE", "AT", "AS", "CEO", "CFO", "GAAP", "SEC", "USA", "EPS",
                "OUT", "NEW", "NOW", "ONE", "KEY", "WELL", "LOW", "HIGH", "BIG", "CAN", "FAST", "SEE", "TV", "PM", "AI",
                "RE", "EX", "IR", "PR", "Q", "K", "L", "R", "V", "X", "Y", "C", "D", "F", "T", "M", "O", "J", "S"}


def name_variants(names: list[str]) -> list[str]:
    out = set()
    for n in names:
        if not n:
            continue
        n = re.sub(r"\s+", " ", str(n)).strip()
        core = re.sub(r"\s+", " ", _SUFFIX.sub(" ", n.replace(",", " "))).strip(" .,&-")
        for v in (n, core):
            v = v.strip(" .,")
            if len(v) >= 4:
                out.add(v)
    return sorted(out, key=len, reverse=True)


def build_masker(names: list[str], tickers: list[str]):
    variants = name_variants(names)
    pats = []
    for v in variants:
        toks = v.split()
        esc = r"\s+".join(re.escape(t) for t in toks)
        if len(toks) >= 2:
            pats.append(re.compile(rf"(?<![\w]){esc}(?![\w])", re.I))
        else:
            # single-word names: exact Title/UPPER case only (avoid masking common words like 'target')
            pats.append(re.compile(rf"(?<![\w])(?:{re.escape(v.title())}|{re.escape(v.upper())})(?![\w])"))
    tick = [t.upper() for t in tickers if t and t.upper() not in _TICKER_STOP and len(t) >= 2]
    tpat = re.compile(r"(?<![\w$])(?:" + "|".join(re.escape(t) for t in sorted(tick, key=len, reverse=True)) + r")(?![\w])") if tick else None

    def mask(text: str) -> tuple[str, dict]:
        counts = {"firm": 0, "ticker": 0}
        for p in pats:
            text, n = p.subn("[FIRM]", text)
            counts["firm"] += n
        if tpat is not None:
            text, n = tpat.subn("[TICKER]", text)
            counts["ticker"] += n
        return text, counts

    return mask
