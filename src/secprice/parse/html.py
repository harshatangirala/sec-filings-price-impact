"""HTML / SGML-text -> normalized plain text. Tables are flattened to 'cell | cell' rows so numeric parsers
(release EPS) can read them; narrative consumers drop numeric-heavy lines later (text.drop_numeric_lines)."""
from __future__ import annotations

import html as _html
import re

_BLOCK = {"p", "div", "br", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section",
          "article", "center", "blockquote", "pre", "hr", "dt", "dd", "title", "page"}
_CELL = {"td", "th"}
_HTML_HINT = re.compile(r"<(html|body|div|p|table|font|br|span)\b", re.I)
_SGML_JUNK = re.compile(r"</?(PAGE|S|C|FN|F\d|R\d|CAPTION|TABLE)>", re.I)
_WS = re.compile(r"[ \t\f\v    ​]+")
_MULTI_NL = re.compile(r"\n{3,}")
_PAGE_LINE = re.compile(r"^\s{0,3}(page\s{0,3})?[-–]?\s{0,3}\d{1,3}\s{0,3}[-–]?\s{0,3}$|^\s{0,3}table of contents\s{0,3}$", re.I)


def decode(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def is_html(text: str) -> bool:
    return bool(_HTML_HINT.search(text[:20000]))


def _in_cell(el) -> bool:
    for a in el.iterancestors():
        t = a.tag.split("}")[-1].lower() if isinstance(a.tag, str) else ""
        if t in ("td", "th"):
            return True
        if t in ("tr", "table", "body"):
            return False
    return False


def html_to_text(text: str) -> tuple[str, int]:
    """Returns (text, n_tables)."""
    import lxml.html
    from lxml import etree
    try:
        doc = lxml.html.document_fromstring(text.encode("utf-8", errors="replace"))
    except (etree.ParserError, ValueError):
        return plain_to_text(re.sub(r"<[^>]+>", " ", text)), 0
    for bad in doc.xpath("//script|//style|//head|//*[local-name()='header' and contains(name(), 'ix:')]"):
        bad.drop_tree()
    for hid in doc.xpath("//*[@style]"):
        st = (hid.get("style") or "").replace(" ", "").lower()
        if "display:none" in st:
            hid.drop_tree()
    n_tables = 0
    for el in doc.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        tag = tag.split("}")[-1].lower()
        if tag == "table":
            n_tables += 1
        if tag in _BLOCK and tag not in ("tr", "table") and _in_cell(el):
            el.tail = " " + (el.tail or "")          # a <p>/<div>/<br> inside a cell must not split the table row
            continue
        if tag in _BLOCK:
            el.tail = "\n" + (el.tail or "")
            if tag != "br":
                el.text = "\n" + (el.text or "")
        elif tag in _CELL:
            el.tail = " | " + (el.tail or "")
    out = doc.text_content()
    return normalize(out), n_tables


def plain_to_text(text: str) -> str:
    return normalize(_SGML_JUNK.sub(" ", text))


def _c1_map() -> dict:
    m = {}
    for c in range(0x80, 0xA0):
        try:
            m[c] = bytes([c]).decode("cp1252")
        except UnicodeDecodeError:
            m[c] = " "
    return m


_C1 = _c1_map()


_MULTI_PIPES = re.compile(r"\s*\|(?:\s*\|)+\s*")


def normalize(text: str) -> str:
    text = _html.unescape(text).translate(_C1)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line).strip()
        # unambiguous forms (each run of whitespace belongs to exactly one group): the older (\s*\|\s*)+ backtracked
        # exponentially on wide table rows with hundreds of empty cells (found on 2000s insurance statistical supplements)
        line = line.strip(" |")                 # leading / trailing empty cells (the line is already whitespace-collapsed)
        line = _MULTI_PIPES.sub(" | ", line)
        if not line or _PAGE_LINE.match(line):
            lines.append("")
            continue
        lines.append(line)
    out = "\n".join(lines)
    return _MULTI_NL.sub("\n\n", out).strip()


def to_text(raw: bytes | str) -> tuple[str, int, str]:
    """(text, n_tables, method)."""
    t = decode(raw)
    if is_html(t):
        txt, n = html_to_text(t)
        return txt, n, "html"
    return plain_to_text(t), t.upper().count("<TABLE>"), "plain"


_NUMTOK = re.compile(r"^[\(\$\-–—]*[\d,.]+%?\)?$")


def drop_numeric_lines(text: str, max_numeric_share: float = 0.5) -> str:
    """Narrative view: drop table rows / lines dominated by numbers."""
    keep = []
    for line in text.split("\n"):
        toks = line.replace("|", " ").split()
        if not toks:
            keep.append("")
            continue
        num = sum(1 for t in toks if _NUMTOK.match(t))
        if "|" in line and num / len(toks) > 0.3:
            continue
        if num / len(toks) > max_numeric_share:
            continue
        keep.append(line)
    return _MULTI_NL.sub("\n\n", "\n".join(keep)).strip()
