# SEC filings → price impact: interactive dashboard

A Streamlit app that explains the whole project, from data collection to the findings, in two windows:

| Window | What it shows |
|---|---|
| **🗺️ Project Flow** | A clickable, colour-coded flowchart of the 16 project steps (research design → data collection → EDA → parsing → NLP pre-processing → events & labels → features → splits & leakage control → baseline models → GPU embeddings → text models & fine-tuning gate → RAG extraction → model selection & locked test → event study & PEAD → backtest → findings). Click a step, use **Prev / Next** for a guided tour, or highlight phases. Each step opens four tabs: *What we did & why*, *Interactive evidence* (Plotly charts and searchable tables), *Report figures* (the report's PNGs) and *Guardrails & decisions* (with the pipeline stages, their outputs and run times). |
| **🎞️ Slide Deck** | 18 presentation slides built from the same numbers: Prev / Next, a slide slider, an overview grid to jump to any slide, speaker notes, and a downloadable self-contained HTML handout. |

Every number comes from `data/`, a snapshot extracted from the finished run. Nothing is typed in by hand, so the
dashboard always matches the report.

## Run it

```bash
pip install -r dashboard/requirements.txt
cd dashboard
streamlit run app.py          # opens http://localhost:8501
```

On Windows you can double-click `run_dashboard.bat`.

## Rebuild the data snapshot (after a new run)

```bash
python dashboard/build_snapshot.py --root <DATA_ROOT>/full     # e.g. C:\Users\harsh\Downloads\sec_report_mini\full
```

The builder imports the `secprice` package (run it from the repo's virtual environment). It writes the CSV, JSON and
parquet files, copies the report figures and the PDF, and computes the results in context (chance level, move-vs-direction
AUCs, paired lifts by period) with the same functions as the report.

## Layout

```
dashboard/
  app.py                 entry point: page config, theme, sidebar, top navigation between the two windows
  views/flow_page.py     window 1: flowchart + step detail
  views/deck_page.py     window 2: slide deck
  lib/data.py            cached loaders for data/ (missing files return None, never a guess)
  lib/theme.py           phase colours, CSS, cards, metric tiles, Plotly template
  lib/charts.py          Plotly figure builders
  lib/flow.py            the 16 steps and the flowchart figure
  lib/steps.py           content and interactive evidence for each step
  lib/slides.py          the 18 slides and the HTML handout
  build_snapshot.py      run artifacts -> data/
  data/                  the snapshot (about 7.5 MB including figures and the PDF report)
```

Tests: `pytest tests/dashboard` renders both windows, every step and every slide headlessly (Streamlit `AppTest`).
