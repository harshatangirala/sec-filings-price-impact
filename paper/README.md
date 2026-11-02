# Working paper

**Do the Words in SEC Filings Predict the Next Trading Session? A Point-in-Time Test**, Krishna Harsha Tangirala.

| File | What it is |
|---|---|
| `Tangirala_SEC_Filings_Point_in_Time_Test.pdf` | The paper |
| `paper.html`, `figures/` | Its source and figures (`build_paper.py` writes the HTML, `render_pdf.py` prints it) |
| `tables/` | Every table behind the paper as CSV, plus `facts.json` |
| `replication/predictions.parquet` | Out-of-sample class probabilities and labels for the models compared in the paper |
| `replication/drift_events.parquet` | Event-level earnings surprises and the drift measures in Section VII |
| `references.json` | Bibliography, looked up by DOI (`analysis/verify_references.py`) |

## Rebuilding

Scripts read a finished run folder from the pipeline, set with `RUN_ROOT`. In order:

```
python paper/analysis/make_tables.py          # performance, paired lifts, subsets, targets (slow: 2,000 bootstrap draws per comparison)
python paper/analysis/drift_alpha.py          # drift with and without the pre-event alpha term, calendar-time portfolios
python paper/analysis/subset_same_head.py     # same-classifier text lift by session
python paper/analysis/no_text_rung.py         # numbers-only rung (a new fit, exploratory)
python paper/analysis/export_replication.py   # the files in replication/
python paper/analysis/make_figures.py
python paper/build_paper.py
python paper/render_pdf.py
```

Nothing here changes the frozen models or the stored stage outputs. The locked test was scored once by the pipeline; anything in `analysis/` that touches that period is exploratory, and the paper says which.

Raw prices and the Loughran-McDonald dictionary are not included. Filings come from EDGAR, and the pinned sources are listed in Table I of the paper.

`references_check.md` records the comparison of every reference with Crossref and OpenAlex.
