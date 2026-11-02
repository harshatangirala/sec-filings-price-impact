"""Prints paper/paper.html to a Letter-size PDF with page numbers (Playwright + Chromium)."""
from pathlib import Path

import pymupdf
from playwright.sync_api import sync_playwright

P = Path(__file__).resolve().parent
OUT = P / "Tangirala_SEC_Filings_Point_in_Time_Test.pdf"
with sync_playwright() as s:
    b = s.chromium.launch(); g = b.new_page()
    g.goto((P / "paper.html").as_uri()); g.wait_for_timeout(2500)
    foot = "<div style='font-size:9px;width:100%;text-align:center;font-family:Times New Roman,serif'><span class='pageNumber'></span></div>"
    g.pdf(path=str(OUT), format="Letter", print_background=True, display_header_footer=True, header_template="<div></div>", footer_template=foot, margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"})
    b.close()
d = pymupdf.open(OUT); print("pages:", len(d), "words:", sum(len(p.get_text().split()) for p in d))
d.set_metadata({"title": "Do the Words in SEC Filings Predict the Next Trading Session? A Point-in-Time Test", "author": "Krishna Harsha Tangirala", "subject": "SEC filings, textual analysis, out-of-sample prediction, post-earnings drift"}); d.saveIncr()
