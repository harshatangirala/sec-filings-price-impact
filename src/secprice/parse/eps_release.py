"""Earnings-release exhibit (EX-99.x) checks and release-EPS parsing (SPEC.md §5.2 verified releases, §5.6).

Diluted GAAP EPS is preferred; basic is the flagged fallback. Rows mentioning non-GAAP / adjusted measures are
skipped. The parsed value is compared with the later XBRL value only as a QA metric (never as an input).
"""
from __future__ import annotations

import re

import pandas as pd

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", re.I)
_PERIOD = re.compile(rf"(three|3|thirteen|twelve|twelve-week|sixteen|quarter|fourth quarter|first quarter|second quarter|"
                     rf"third quarter|fiscal quarter)[\w\s\-]{{0,30}}(ended)\s+({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", re.I)
_NUM = re.compile(r"(\(\s*\$?\s*(?:\d{1,3}(?:,\d{3})*)?\.\d{1,3}\s*\)?|-?\$?\s*(?:\d{1,3}(?:,\d{3})*)?\.\d{1,3})")
_SHARES = re.compile(r"shares|weighted|average\s+number|outstanding", re.I)
_ANNUAL = re.compile(r"full[- ]year|for\s+the\s+(fiscal\s+)?year|twelve\s+months|year\s+ended|annual|outlook|guidance|"
                     r"expects?|forecast|range\s+of|fiscal\s+20\d\d\s+(earnings|eps)", re.I)
_CONT = re.compile(r"continuing\s+operations", re.I)
_DISC = re.compile(r"discontinued", re.I)
_RANGE = re.compile(r"\d\.\d{2}\s*(?:-|\u2013|\u2014|to)\s*\$?\s*\d*\.\d{2}|in\s+the\s+range|range\s+of", re.I)
_STATEMENT = re.compile(r"statements?\s+of\s+(consolidated\s+)?(income|operations|earnings)|income\s+statements?|"
                        r"consolidated\s+(results|statements?)|condensed\s+consolidated", re.I)
_IMPACT = re.compile(r"\b(impact\w*|charges?|benefit\w*|includ\w*|effects?|reduc\w*|related\s+to|costs?|gain\s+on|"
                     r"loss\s+on|items?|after[- ]tax|pre-?tax|dilution\s+from|accretive|dilutive)\b", re.I)
_NONGAAP = re.compile(r"non-?gaap|adjusted|excluding|excluded|pro\s*forma|core\b|operating\s+(eps|earnings per)|before\s+special|"
                      r"comparable|underlying|cash\s+eps|normalized|street|ongoing|specified\s+items", re.I)
_EPS_CTX = re.compile(r"per\s+(common\s+)?share|earnings\s+per|\beps\b|per\s+diluted", re.I)
_DIL = re.compile(r"dilut", re.I)
_V = r"(\(?-?\$?\s*\d*\.\d{2}\)?)"
# Ordered prose patterns: (regex, confidence)
_PROSE = [
    (re.compile(r"diluted\s+(?:earnings|eps)[^.$]{0,40}?(?:was|were|of)\s+\$\s*(\(?\d*\.\d{2}\)?)\s+GAAP", re.I), "medium_gaap_marked"),
    (re.compile(r"\$\s*(\(?\d*\.\d{2}\)?)\s+(?:per|a)\s+diluted\s+(?:common\s+)?share", re.I), "medium"),
    (re.compile(r"(?:diluted\s+(?:earnings|net\s+income|income|eps)[^.$]{0,50}?|\(assuming\s+dilution\)\s*|"
                r"assuming\s+dilution[^.$]{0,30}?)(?:of|was|were|totaled|to|:)\s*" + _V, re.I), "medium"),
    (re.compile(r"diluted\s+earnings\s+per\s+share[^.]{0,80}?were\s+\$[\d.,]+\s+(?:million|billion)\s+and\s+" + _V, re.I), "medium"),
    (re.compile(r"(?:net\s+income|net\s+earnings|earnings|net\s+profit)[^.]{0,80}?,?\s+or\s+\$\s*(\(?\d*\.\d{2}\)?)\s+per\s+(?:common\s+)?share", re.I),
     "low"),
]
USABLE_CONFIDENCE = {"high", "medium", "medium_table", "medium_gaap_marked"}   # SUE_release uses only these; 'low' stays for QA


def results_release_score(text: str) -> int:
    t = (text or "")[:80000].lower()
    s = 0
    s += bool(re.search(r"net income|net earnings|net loss|earnings per (diluted )?share|per diluted share|diluted eps", t))
    s += bool(re.search(r"(first|second|third|fourth) quarter|three months ended|quarter ended|fiscal (year|quarter)|full[- ]year", t))
    s += bool(re.search(r"(report|reports|reported|announce|announces|announced)\b.{0,80}(results|earnings)|financial results|results of operations", t))
    s += bool(re.search(r"\brevenues?\b|net sales|total sales", t))
    return s


def is_results_release(text: str) -> bool:
    return results_release_score(text) >= 3


def _to_float(s: str) -> float | None:
    s = s.strip()
    neg = s.startswith("(") or s.startswith("-")
    v = re.sub(r"[^\d.]", "", s)
    try:
        x = float(v)
    except ValueError:
        return None
    return -x if neg else x


def _row_value(line: str) -> float | None:
    """First EPS-like value after the label cell (keeps '(0.15)' negatives intact)."""
    cut = line.find("|")
    if cut < 0:
        d = re.search(r"\d", line)
        cut = (d.start() - 1 if d and d.start() > 0 and line[d.start() - 1] in "($-" else (d.start() if d else 0))
    nums = [_to_float(m.group(0)) for m in _NUM.finditer(line[max(0, cut):])]
    nums = [x for x in nums if x is not None and abs(x) < 200]
    return nums[0] if nums else None


def parse_release_eps(text: str) -> dict:
    """{'eps_diluted', 'eps_basic', 'parse_confidence', 'source_line', 'period_end', 'period_source'}"""
    out = {"eps_diluted": None, "eps_basic": None, "parse_confidence": "none", "source_line": None,
           "period_end": None, "period_source": None}
    if not text:
        return out
    raw_lines = text.split("\n")
    lines = []
    i = 0
    while i < len(raw_lines):
        cur = raw_lines[i]
        nxt_i = i + 1
        while nxt_i < len(raw_lines) and not raw_lines[nxt_i].strip():
            nxt_i += 1
        nxt = raw_lines[nxt_i] if nxt_i < len(raw_lines) else ""
        if (cur.strip() and not re.search(r"\d", cur) and len(cur) < 80 and re.search(r"dilut|basic", cur, re.I)
                and re.match(r"^\s*[\$\(\-\d.|\s,)]+$", nxt) and re.search(r"\d", nxt)):
            lines.append(cur.strip() + " | " + nxt.strip())
            i = nxt_i + 1
            continue
        lines.append(cur)
        i += 1
    best = None
    # 1) table rows: every candidate is scored; the best GAAP total-EPS row under an income-statement heading wins
    cands = []
    for i, line in enumerate(lines[:6000]):
        if "|" not in line:
            continue
        low = line.lower()
        is_dil, is_bas = bool(_DIL.search(low)), "basic" in low
        label = line.split("|")[0]
        if not (is_dil or is_bas) or _NONGAAP.search(line) or _SHARES.search(label) or _DISC.search(label):
            continue
        if _RANGE.search(line):
            continue                                   # guidance ranges ("$6.50 - $6.90", "in the range of ...")
        ctx = " ".join(lines[max(0, i - 4):i + 1])
        if not _EPS_CTX.search(ctx) or _NONGAAP.search(" ".join(lines[max(0, i - 2):i])):
            continue
        above = " ".join(lines[max(0, i - 8):i])
        if _ANNUAL.search(above) and not re.search(r"quarter|three\s+months|13\s+weeks", above, re.I):
            continue                                   # an annual / outlook table, not the quarter
        v = _row_value(line)
        if v is None:
            continue
        kind = "diluted" if is_dil else "basic"
        conf = "high" if _EPS_CTX.search(line) else "medium_table"
        score = 0
        if _CONT.search(label):
            conf, score = "continuing_ops", -2          # differs from XBRL EarningsPerShareDiluted (total): QA only
        if _STATEMENT.search(" ".join(lines[max(0, i - 60):i])):
            score += 2                                  # inside the income statement
        if re.search(r"quarter|three\s+months|13\s+weeks", " ".join(lines[max(0, i - 30):i]), re.I):
            score += 1
        cands.append((score, -i, kind, v, conf, line[:200]))
    dil = sorted([c for c in cands if c[2] == "diluted"], reverse=True)
    bas = sorted([c for c in cands if c[2] == "basic"], reverse=True)
    if dil:
        out["eps_diluted"] = dil[0][3]
        best = ("diluted", dil[0][3], dil[0][4], dil[0][5])
    if bas:
        out["eps_basic"] = bas[0][3]
        best = best or ("basic", bas[0][3], bas[0][4], bas[0][5])
    # 2) prose, in order of preference, skipping sentences about non-GAAP measures
    if out["eps_diluted"] is None:
        head = text[:60000]
        for rx, conf in _PROSE:
            hit = None
            for m in rx.finditer(head):
                s0 = head.rfind(".", 0, m.start()) + 1
                sentence = head[s0:m.end()]
                if _ANNUAL.search(sentence) and not re.search(r"quarter", sentence, re.I):
                    continue
                if _RANGE.search(sentence) or (_IMPACT.search(sentence) and conf != "medium_gaap_marked"):
                    continue                           # guidance ranges / impact-of-an-item sentences
                if conf == "medium_gaap_marked" or not _NONGAAP.search(sentence):
                    hit = m
                    break
            if hit:
                out["eps_diluted"] = _to_float(hit.group(hit.lastindex))
                best = ("diluted", out["eps_diluted"], conf, hit.group(0)[:200])
                break
    if best:
        out["parse_confidence"] = best[2] if best[0] == "diluted" else "basic_fallback"
        out["source_line"] = best[3]
    # current period = the latest 'quarter ended <date>' phrase near the top of the release
    dates = []
    for pm in _PERIOD.finditer(text[:25000]):
        try:
            dates.append(pd.Timestamp(f"{pm.group(3)} {pm.group(4)}, {pm.group(5)}"))
        except ValueError:
            pass
    if dates:
        out["period_end"] = max(dates).date().isoformat()
        out["period_source"] = "text_period_phrase_latest"
    return out


def infer_quarter_end(filing_date, fiscal_year_end: str | None) -> tuple[str | None, str]:
    """Most recent fiscal quarter end at least 7 days before the filing date (flagged as inferred)."""
    fd = pd.Timestamp(filing_date)
    try:
        mm = int(str(fiscal_year_end)[:2]) if fiscal_year_end else 12
        dd = int(str(fiscal_year_end)[2:4]) if fiscal_year_end and len(str(fiscal_year_end)) >= 4 else 28
        if not 1 <= mm <= 12:
            mm = 12
        if dd <= 7:                       # 52/53-week year ending early in a month (e.g. '0103') = previous month
            mm = 12 if mm == 1 else mm - 1
    except ValueError:
        mm = 12
    q_months = sorted({(mm - 1 - 3 * q) % 12 + 1 for q in range(4)})
    ends = [pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
            for y in (fd.year - 1, fd.year) for m in q_months]
    ends = sorted(e for e in ends if e <= fd - pd.Timedelta(days=7))
    return (ends[-1].date().isoformat(), "inferred_fiscal_calendar") if ends else (None, "none")
