"""Section extraction for 10-K / 10-Q and 8-K item text (SPEC.md §5.2, §16 sections.parquet).

Headings are found by item number + title keywords; among candidate starts (table-of-contents lines repeat the
headings) the one with the longest span to the next different heading wins. Sections incorporated by reference
are flagged so the caller can fall back to Exhibit 13 (annual report) for MD&A.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ITEM = re.compile(
    r"^\s*(?:part\s+(?P<part>i{1,3}|iv|1|2)\s*[,.\-–—:]*\s*)?item[\s\xa0]*(?P<num>\d{1,2})\s*(?P<let>\(?[a-c]\)?)?"
    r"(?![\d.]\d)\s*[.:\-–—]?\s*(?P<title>.{0,160})$", re.I)
_PART = re.compile(r"^\s*part\s+(i{1,3}|iv|1|2)\b\s*[.\-–—:]?\s*(.{0,80})$", re.I)
_BYREF = re.compile(r"incorporated\s+(herein\s+)?by\s+reference|is\s+included\s+in\s+(the\s+)?(registrant'?s\s+)?annual\s+report|"
                    r"set\s+forth\s+(under|in)\s+.{0,80}annual\s+report|appears?\s+(under|in)\s+.{0,80}annual\s+report|"
                    r"reference\s+is\s+made\s+to|see\s+.{0,80}(section|caption)\s+(entitled|captioned)|"
                    r"contained\s+in\s+.{0,80}(annual\s+report|financial\s+section)", re.I)

# (form family, section id) -> (item key, title regex, part requirement)
TARGETS = {
    "10-K": {
        "10K_1": ("1", r"business", None),
        "10K_1A": ("1A", r"risk\s+factors", None),
        "10K_7": ("7", r"management.{0,4}s?\s+discussion|discussion\s+and\s+analysis", None),
        "10K_7A": ("7A", r"quantitative\s+and\s+qualitative|market\s+risk", None),
    },
    "10-Q": {
        "10Q_MDA": ("2", r"management.{0,4}s?\s+discussion|discussion\s+and\s+analysis", "I"),
        "10Q_MR": ("3", r"quantitative\s+and\s+qualitative|market\s+risk", "I"),
        "10Q_RF": ("1A", r"risk\s+factors", "II"),
    },
}
TITLE_ONLY = {
    "10K_7": r"^\s*management.{0,4}s?\s+discussion\s+and\s+analysis\s+of\s+(financial\s+condition|results)",
    "10Q_MDA": r"^\s*management.{0,4}s?\s+discussion\s+and\s+analysis\s+of\s+(financial\s+condition|results)",
    "10K_1A": r"^\s*risk\s+factors\s*$",
    "10K_7A": r"^\s*quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk",
}
# Annual-report exhibits (EX-13) name the MD&A in many ways ("FINANCIAL REVIEW", "Management Discussion", ...). Used for EX-13 only.
EX13_MDNA_TITLE = (r"^\s*(management.{0,4}s?\s+discussion(\s+and\s+analysis)?(\s+of\s+(financial\s+condition|results)[\w\s,&]{0,60})?"
                   r"|financial\s+review|financial\s+review\s+and\s+analysis|review\s+of\s+operations|"
                   r"financial\s+(condition\s+and\s+)?results\s+of\s+operations)\s*:?\s*$")
_ROMAN = {"1": "I", "2": "II", "i": "I", "ii": "II", "iii": "III", "iv": "IV"}


@dataclass
class Heading:
    line_no: int
    char_pos: int
    key: str           # e.g. "7A"
    title: str
    part: str | None


def form_family(form: str) -> str:
    f = (form or "").upper()
    if f.startswith("10-K"):
        return "10-K"
    if f.startswith("10-Q"):
        return "10-Q"
    if f.startswith("8-K"):
        return "8-K"
    return f


def find_headings(text: str) -> list[Heading]:
    heads: list[Heading] = []
    pos = 0
    lines = text.split("\n")
    cur_part = None
    for i, line in enumerate(lines):
        L = len(line)
        if L <= 220:
            pm = _PART.match(line)
            if pm and not re.search(r"item", line, re.I):
                cur_part = _ROMAN.get(pm.group(1).lower(), pm.group(1).upper())
            m = _ITEM.match(line)
            if m:
                let = (m.group("let") or "").strip("() ").upper()
                key = f"{int(m.group('num'))}{let}"
                title = (m.group("title") or "").strip(" .:-|")
                if len(title) < 4 and i + 1 < len(lines):
                    title = (title + " " + lines[i + 1][:160]).strip()
                part = m.group("part")
                part = _ROMAN.get(part.lower(), part.upper()) if part else cur_part
                heads.append(Heading(i, pos, key, title, part))
        pos += L + 1
    return heads


def _span(text: str, heads: list[Heading], j: int) -> tuple[int, int]:
    start = heads[j].char_pos
    end = len(text)
    for k in range(j + 1, len(heads)):
        if heads[k].key != heads[j].key:
            end = heads[k].char_pos
            break
    return start, end


def extract_sections(text: str, form: str, max_chars: int = 300000) -> dict[str, dict]:
    """{section_id: {text, start, end, method, by_reference}} for the form's target sections."""
    fam = form_family(form)
    targets = TARGETS.get(fam, {})
    heads = find_headings(text)
    out: dict[str, dict] = {}
    for sid, (key, title_rx, part_req) in targets.items():
        best = None
        for j, h in enumerate(heads):
            if h.key != key:
                continue
            if part_req and h.part and h.part != part_req:
                continue
            if title_rx and not re.search(title_rx, h.title, re.I):
                if not (fam == "10-K" and key in ("1", "7") and not h.title):
                    continue
            s, e = _span(text, heads, j)
            if best is None or (e - s) > (best[1] - best[0]):
                best = (s, e)
        method = "item_heading"
        if best is None or (best[1] - best[0]) < 200:
            alt = _title_only(text, sid)
            if alt:
                best, method = alt, "title_only"
        elif (best[1] - best[0]) < 2500 and sid in TITLE_ONLY:
            # Item points elsewhere in the same document (e.g. 'Reference is made to ... Financial Section')
            alt = _title_only(text, sid)
            if alt and (alt[1] - alt[0]) > 3 * (best[1] - best[0]) and alt[0] != best[0]:
                best, method = alt, "title_in_doc"
        if best is None:
            continue
        s, e = best
        body = text[s:e].strip()
        first_nl = body.find("\n")
        if 0 < first_nl < 240:
            body_wo_head = body[first_nl + 1:].strip()
        else:
            body_wo_head = body
        by_ref = len(body_wo_head) < 3000 and bool(_BYREF.search(body_wo_head[:3000]))
        out[sid] = {"text": body[:max_chars], "start": s, "end": e, "method": method, "by_reference": by_ref,
                    "truncated": len(body) > max_chars}
    return out


def _title_only(text: str, sid: str, rx_override: str | None = None):
    rx = rx_override or TITLE_ONLY.get(sid)
    if not rx:
        return None
    best = None
    # line-anchored override patterns need re.M (the slice below spans several lines); built-in patterns keep their behaviour
    pat = re.compile(rx, re.I | re.M) if rx_override else re.compile(rx, re.I)
    starts = [m.start() for m in re.finditer(r"(?m)^.{0,200}$", text) if pat.match(text[m.start():m.start() + 200])]
    stops = [m.start() for m in re.finditer(
        r"(?mi)^\s*(item\s*\d|quantitative\s+and\s+qualitative|financial\s+statements\s+and\s+supplementary|"
        r"report\s+of\s+independent|consolidated\s+statements?\s+of|controls\s+and\s+procedures)", text)]
    for s in starts:
        nxt = [p for p in stops if p > s + 50]
        e = nxt[0] if nxt else len(text)
        if best is None or e - s > best[1] - best[0]:
            best = (s, e)
    return best


def mdna_from_ex13(text: str, max_chars: int = 300000) -> dict | None:
    """MD&A located by title inside an annual-report exhibit (EX-13)."""
    span = _title_only(text, "10K_7")
    if not span or span[1] - span[0] < 1500:
        span = _title_only(text, "10K_7", EX13_MDNA_TITLE)
    if not span or span[1] - span[0] < 1500:
        return None
    s, e = span
    return {"text": text[s:e][:max_chars].strip(), "start": s, "end": e, "method": "ex13_title", "by_reference": False,
            "truncated": e - s > max_chars}


_8K_ITEM = re.compile(r"(?mi)^\s*item[\s\xa0]*(\d{1,2}(?:\.\d{2})?)\b[.:\s\-–—]*(.{0,120})$")
_SIG = re.compile(r"(?mi)^\s*signatures?\s*$|pursuant to the requirements of the securities exchange act of 1934, the registrant has duly caused")


def eightk_items_text(text: str, max_chars: int = 300000) -> dict:
    """Narrative of the 8-K body from the first item heading to the signature block; item codes seen in text."""
    m = list(_8K_ITEM.finditer(text))
    start = m[0].start() if m else 0
    sig = _SIG.search(text, start)
    end = sig.start() if sig else len(text)
    body = text[start:end].strip()
    codes = sorted({x.group(1) for x in m})
    return {"text": body[:max_chars], "start": start, "end": end, "method": "8k_body", "item_codes_text": codes,
            "by_reference": False, "truncated": len(body) > max_chars}
