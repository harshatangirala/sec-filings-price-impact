# Write-up

- `SEC_Filings_Price_Impact_Report.pdf`: the written report (21 pages).
- `SEC_Filings_Study_13_Slides.pptx`: the slide deck.
- `source/`: scripts that made them. `make_charts.py` draws the charts from `dashboard/data`, `build_report.py` writes the HTML,
  `render_pdf.py` prints it to PDF (Playwright + Chromium), `build_deck.js` builds the deck (pptxgenjs).
  They contain Windows paths from the machine they were written on; edit the constants at the top before running elsewhere.
