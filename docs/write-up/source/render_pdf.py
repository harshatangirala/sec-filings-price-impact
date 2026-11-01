from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(r"file:///C:/Users/harsh/Downloads/sec_report_build/report.html")
    pg.wait_for_timeout(1500)
    foot = ("<div style='font-size:8px;color:#6b7280;width:100%;text-align:center;font-family:Segoe UI,Arial'>"
            "Can a company's own filing tell you what its stock will do next? &nbsp;·&nbsp; page <span class='pageNumber'></span> of <span class='totalPages'></span></div>")
    pg.pdf(path=r"C:\Users\harsh\Downloads\sec_report_build\report.pdf", format="A4", print_background=True, display_header_footer=True,
           header_template="<div></div>", footer_template=foot, margin={"top": "18mm", "bottom": "20mm", "left": "19mm", "right": "19mm"})
    b.close()
import pymupdf

d = pymupdf.open(r"C:\Users\harsh\Downloads\sec_report_build\report.pdf")
print("pages:", len(d))
words = sum(len(pg.get_text().split()) for pg in d)
print("approx words:", words)
d.set_metadata({"title": "Can a company's own filing tell you what its stock will do next?", "author": "Krishna Harsha Tangirala",
                "subject": "SEC filings and stock reactions: a study of 150 S&P 500 stocks, 2000-2026"})
d.saveIncr()
