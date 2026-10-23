"""EDGAR archive URLs and the SGML full-submission parser (headers, documents, exhibits)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ARCH = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/"


def nodash(acc: str) -> str:
    return acc.replace("-", "")


def doc_url(cik: int, acc: str, doc: str) -> str:
    return ARCH.format(cik=int(cik), nodash=nodash(acc)) + doc


def full_txt_url(cik: int, acc: str) -> str:
    return ARCH.format(cik=int(cik), nodash=nodash(acc)) + f"{acc}.txt"


def index_json_url(cik: int, acc: str) -> str:
    return ARCH.format(cik=int(cik), nodash=nodash(acc)) + "index.json"


def index_headers_url(cik: int, acc: str) -> str:
    return ARCH.format(cik=int(cik), nodash=nodash(acc)) + f"{acc}-index-headers.html"


@dataclass
class SgmlDocument:
    type: str
    sequence: str
    filename: str
    description: str
    text: str


@dataclass
class SgmlSubmission:
    header: str
    acceptance_raw: str | None
    form_type: str | None
    period: str | None
    filed: str | None
    items: list[str] = field(default_factory=list)
    item_titles: list[str] = field(default_factory=list)
    documents: list[SgmlDocument] = field(default_factory=list)
    truncated: bool = False


_DOC_RE = re.compile(r"<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)", re.S | re.I)
_TAG = lambda t: re.compile(rf"<{t}>\s*([^\n<]*)", re.I)  # noqa: E731
_TYPE, _SEQ, _FN, _DESC = _TAG("TYPE"), _TAG("SEQUENCE"), _TAG("FILENAME"), _TAG("DESCRIPTION")
_TEXT_RE = re.compile(r"<TEXT>(.*?)(?:</TEXT>|\Z)", re.S | re.I)
_ITEMS_RE = re.compile(r"<ITEMS>\s*([0-9.]+)", re.I)
_ITEM_INFO_RE = re.compile(r"ITEM INFORMATION:\s*(.+)", re.I)
_SKIP_TYPES = re.compile(r"^(GRAPHIC|ZIP|EXCEL|PDF|XML|JSON|EX-101|EX-27|EX-100|EX-104|EX-102|COVER)", re.I)


def _first(rx: re.Pattern, block: str) -> str:
    m = rx.search(block[:2000])
    return m.group(1).strip() if m else ""


def parse_sgml(raw: bytes | str, truncated: bool = False) -> SgmlSubmission:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
    hend = text.find("<DOCUMENT>")
    header = text[: hend if hend > 0 else 20000]
    acc = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})|ACCEPTANCE-DATETIME:\s*(\d{14})", header)
    form = re.search(r"CONFORMED SUBMISSION TYPE:\s*(\S+)|<TYPE>\s*(\S+)", header)
    period = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})|<PERIOD>\s*(\d{8})", header)
    filed = re.search(r"FILED AS OF DATE:\s*(\d{8})|<FILING-DATE>\s*(\d{8})", header)
    items = [m.group(1) for m in _ITEMS_RE.finditer(header)]
    titles = [m.group(1).strip() for m in _ITEM_INFO_RE.finditer(header)]
    docs = []
    for m in _DOC_RE.finditer(text):
        block = m.group(1)
        t = _TYPE.search(block)
        dtype = t.group(1).strip() if t else ""
        if _SKIP_TYPES.match(dtype):
            continue
        tm = _TEXT_RE.search(block)
        body = tm.group(1) if tm else ""
        docs.append(SgmlDocument(dtype, _first(_SEQ, block), _first(_FN, block), _first(_DESC, block), body))
    g = lambda m: (m.group(1) or m.group(2)) if m else None  # noqa: E731
    return SgmlSubmission(header=header, acceptance_raw=g(acc), form_type=g(form), period=g(period), filed=g(filed),
                          items=items, item_titles=titles, documents=docs, truncated=truncated)
