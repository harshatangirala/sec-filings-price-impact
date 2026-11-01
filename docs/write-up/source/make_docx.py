"""Converts report.html (the same text and figures as the PDF) into a Word document."""
import base64, io, re
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = r"C:\Users\harsh\Downloads\sec_report_build\report.html"
OUT = r"C:\Users\harsh\Downloads\sec_report_build\report.docx"
TEXT_W_CM = 17.0
NAVY, GREY, BLUE = RGBColor(0x14, 0x30, 0x4D), RGBColor(0x6B, 0x72, 0x80), RGBColor(0x1D, 0x4E, 0xD8)
BODY_FONT, HEAD_FONT = "Georgia", "Segoe UI"

soup = BeautifulSoup(open(SRC, encoding="utf-8").read(), "html.parser")
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = Cm(2.0); sec.bottom_margin = Cm(2.2)


def set_font(style, name, size, color=None, bold=None):
    style.font.name = name; style.font.size = Pt(size)
    rpr = style.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), name)
    if color is not None: style.font.color.rgb = color
    if bold is not None: style.font.bold = bold


st = doc.styles
set_font(st["Normal"], BODY_FONT, 10.5)
st["Normal"].paragraph_format.space_after = Pt(7); st["Normal"].paragraph_format.line_spacing = 1.25
set_font(st["Heading 1"], HEAD_FONT, 22, NAVY, True); set_font(st["Heading 2"], HEAD_FONT, 14.5, NAVY, True)
st["Heading 1"].paragraph_format.space_after = Pt(8)
st["Heading 2"].paragraph_format.space_before = Pt(18); st["Heading 2"].paragraph_format.space_after = Pt(6)
st["Heading 2"].paragraph_format.keep_with_next = True


def add_field(run, instr):
    for t, txt in (("begin", None), (None, instr), ("end", None)):
        if t:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


# footer: title + page x of y
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Can a company's own filing tell you what its stock will do next?  ·  page "); r.font.size = Pt(8); r.font.color.rgb = GREY; r.font.name = HEAD_FONT
r2 = fp.add_run(); r2.font.size = Pt(8); r2.font.color.rgb = GREY; add_field(r2, "PAGE")
r3 = fp.add_run(" of "); r3.font.size = Pt(8); r3.font.color.rgb = GREY
r4 = fp.add_run(); r4.font.size = Pt(8); r4.font.color.rgb = GREY; add_field(r4, "NUMPAGES")


def hyperlink(par, url, text, size=None):
    rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
    rr = OxmlElement("w:r"); pr = OxmlElement("w:rPr")
    c = OxmlElement("w:color"); c.set(qn("w:val"), "1D4ED8"); pr.append(c)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); pr.append(u)
    if size:
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); pr.append(sz)
    rr.append(pr); t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); rr.append(t); h.append(rr)
    par._p.append(h)


def inline(par, node, bold=False, ital=False, size=None, color=None):
    for ch in node.children:
        if isinstance(ch, NavigableString):
            txt = re.sub(r"\s+", " ", str(ch))
            if not txt: continue
            run = par.add_run(txt); run.bold = bold or None; run.italic = ital or None
            if size: run.font.size = Pt(size)
            if color is not None: run.font.color.rgb = color
        elif isinstance(ch, Tag):
            if ch.name == "b": inline(par, ch, True, ital, size, color)
            elif ch.name == "i": inline(par, ch, bold, True, size, color)
            elif ch.name == "br": par.add_run().add_break(WD_BREAK.LINE)
            elif ch.name == "a": hyperlink(par, ch.get("href"), ch.get_text(), size)
            elif ch.name == "span": inline(par, ch, bold, ital, 8.5 if "small" in (ch.get("class") or []) else size, GREY if "small" in (ch.get("class") or []) else color)
            else: inline(par, ch, bold, ital, size, color)


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_); tcPr.append(sh)


def borders(tbl, color="D5DBE3"):
    tblPr = tbl._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:space"), "0"); x.set(qn("w:color"), color); b.append(x)
    anchor = None
    for tag in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption"):
        anchor = tblPr.find(qn(tag))
        if anchor is not None: break
    if anchor is not None: anchor.addprevious(b)
    else: tblPr.append(b)


def add_table(t, links=False):
    rows = t.find_all("tr")
    ncol = max(len(r.find_all(["td", "th"])) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=ncol); tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    if not links: borders(tbl)
    for i, r in enumerate(rows):
        for j, c in enumerate(r.find_all(["td", "th"])):
            cell = tbl.cell(i, j); cell.text = ""
            par = cell.paragraphs[0]; par.paragraph_format.space_after = Pt(1); par.paragraph_format.line_spacing = 1.05
            sz = 9.5 if links else 8.5
            inline(par, c, bold=(c.name == "th"), size=sz, color=(GREY if links and j == 0 else None))
            for run in par.runs: run.font.name = HEAD_FONT if (links and j == 0) or c.name == "th" else run.font.name
            if c.name == "th": shade(cell, "EEF2F7")
    if links:
        for row in tbl.rows:
            row.cells[0].width = Cm(3.6); row.cells[1].width = Cm(13.4)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(fig):
    img = fig.find("img"); data = base64.b64decode(img["src"].split(",", 1)[1])
    m = re.search(r"width:\s*(\d+)%", fig.get("style", "")); frac = int(m.group(1)) / 100 if m else 1.0
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(io.BytesIO(data), width=Cm(TEXT_W_CM * frac))
    cap = doc.add_paragraph(); cap.paragraph_format.space_after = Pt(10)
    inline(cap, fig.find("figcaption"), size=8.5, color=GREY)


def add_par(node, style=None):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph(); inline(p, node); return p


body = soup.body
first_h2_seen = False
for el in body.find_all(recursive=False):
    cls = el.get("class") or []
    if el.name == "div" and "cover" in cls:
        k = el.find(class_="kicker"); p = doc.add_paragraph(); inline(p, k, size=9, color=RGBColor(0x8A, 0x5A, 0x2B))
        for run in p.runs: run.font.name = HEAD_FONT; run.font.all_caps = True
        doc.add_heading(el.find("h1").get_text(), level=1)
        p = doc.add_paragraph(); inline(p, el.find(class_="sub"), size=12)
        p = doc.add_paragraph(); inline(p, el.find(class_="by"), size=10.5, bold=True)
        add_table(el.find("table"), links=True)
    elif el.name == "h2":
        h = doc.add_heading(el.get_text().strip(), level=2)
        if "page-break-before" in el.get("style", ""): h.paragraph_format.page_break_before = True
    elif el.name == "p": add_par(el)
    elif el.name == "figure": add_figure(el)
    elif el.name == "table": add_table(el)
    elif el.name == "ol":
        refs = "refs" in cls
        for n, li in enumerate(el.find_all("li", recursive=False), 1):
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.6)
            p.paragraph_format.space_after = Pt(5 if refs else 3)
            run = p.add_run(f"{n}.  ");
            inline(p, li, size=9.5 if refs else None)
    elif el.name == "div" and "refnote" in cls:
        for q in el.find_all("p"):
            p = add_par(q);
            if "small" in (q.get("class") or []):
                for run in p.runs: run.font.size = Pt(8.5); run.font.color.rgb = GREY

core = doc.core_properties
core.title = "Can a company's own filing tell you what its stock will do next?"
core.author = "Krishna Harsha Tangirala"; core.subject = "SEC filings and stock reactions: a study of 150 S&P 500 stocks, 2000-2026"
for z in doc.settings.element.findall(qn("w:zoom")):
    z.set(qn("w:percent"), "100")
doc.save(OUT); print("saved", OUT)
