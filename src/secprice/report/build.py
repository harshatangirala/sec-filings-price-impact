"""S24 report: reports/results_summary.md (SPEC.md §21, computed values only) + a self-contained HTML report with
figures embedded at their sections + a PDF. Every number in the text is read from an artifact; missing results are stated as such.

Figures are placed with slots ('[[FIG:F18]]') that each renderer resolves: Markdown -> image link, HTML -> embedded image,
PDF -> inline image. A figure that was not produced is replaced by a one-line note with the reason."""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SLOT = re.compile(r"^\[\[FIG:(\w+)\]\]\s*$", re.M)


def fmt(v, nd=3, col: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v)) if any(k in str(col).lower() for k in ("year", "fold", "item", "horizon", "lag")) else f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        c = str(col).lower()
        if float(v).is_integer() and (c == "n" or c.startswith("n_") or c.endswith("_n") or "count" in c or "days" in c
                                      or c in ("observed", "parsed", "firm_years", "lag", "cost_bps", "fold", "horizon")):
            return f"{int(v):,}"
        return f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame | None, max_rows: int = 40, nd: int = 3) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_"
    d = df.head(max_rows)
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd, c).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    if len(df) > max_rows:
        out.append(f"\n_{len(df) - max_rows} more rows in the CSV._")
    return "\n".join(out)


def slot(fid: str) -> str:
    return f"\n[[FIG:{fid}]]\n"


def smallest_defensible_claim(f: dict) -> str:
    h1 = f.get("h1") or {}
    lift, lo, hi = h1.get("value"), h1.get("ci_low"), h1.get("ci_high")
    n = h1.get("n_events")
    block = h1.get("block", "walk-forward validation")
    if lift is None:
        return ("The core text-versus-baseline comparison (H1) could not be evaluated in this run; the study supports "
                "only the data-integrity and event-study statements above.")
    if lo is not None and lo > 0:
        txt = (f"On the common cohort ({fmt(n)} events, {block}), adding SEC-filing text to price, metadata, accounting "
               f"and lexical baselines raised macro-F1 by {fmt(lift)} (95% month-block CI {fmt(lo)} to {fmt(hi)}) under "
               f"the assumed availability/latency policy. This is an out-of-time association with modest predictive "
               f"value, not evidence of causation or of tradable profit.")
    elif hi is not None and hi < 0:
        txt = (f"On the common cohort ({fmt(n)} events, {block}), adding SEC-filing text lowered macro-F1 by "
               f"{fmt(-lift)} (CI {fmt(lo)} to {fmt(hi)}): text did not add out-of-time value beyond the baselines.")
    else:
        txt = (f"On the common cohort ({fmt(n)} events, {block}), the lift from adding SEC-filing text was {fmt(lift)} "
               f"macro-F1 with a 95% CI of {fmt(lo)} to {fmt(hi)} that includes zero: we cannot claim that filing text adds "
               f"reliable out-of-time information beyond the structured baselines. A credible null is a valid result.")
    return txt + (f.get("claim_tail") or "")


def build_markdown(f: dict) -> str:
    L = []
    a = L.append
    N = f.get("narr") or {}

    def sec(key, fallback="_(not built)_"):
        a(N.get(key, fallback))

    a(f"# SEC filings -> price impact: results report ({f.get('profile', '?')} profile)")
    a("")
    a(f"Generated {f.get('generated')} | config hash `{f.get('config_hash')}` | code `{f.get('code')}` | "
      f"pre-registration hash `{f.get('prereg_hash', 'not frozen')}`")
    a(f"\n**Timing policy:** {f.get('timing_policy')}")
    a("\n**Reading rules:** association is not causation; prediction is not profitability. Results are "
      "validation-selected, never a global optimum. Power labels: n<30 warning, 30-99 exploratory, wide CI inconclusive.")
    a("\n## Executive summary")
    for b in f.get("exec_bullets", []):
        a(f"- {b}")
    a(f"\n**Smallest defensible claim.** {smallest_defensible_claim(f)} All results are conditioned on a survivor-only "
      "universe (firms listed in 2026), which biases drift and backtest results towards survivors.")
    a("\n## Reader's guide")
    sec("guide")
    a("\n## A. Study design and system")
    a("\n### A1. Research questions and hypotheses")
    sec("a1")
    a("\n### A2. System architecture")
    sec("a2")
    a(slot("A01"))
    a(slot("R04"))
    a("\n### A3. Data sources and universe")
    sec("a3")
    a(slot("A02"))
    a(slot("R05"))
    a("\n### A4. Pre-processing and NLP pipeline")
    sec("a4")
    a("\n### A5. Events, clocks, targets and labels")
    sec("a5")
    a(slot("F07"))
    a(slot("F10"))
    a("\n### A6. Features and the ablation ladder")
    sec("a6")
    a(slot("A03"))
    a("\n### A7. Models, encoders and training")
    sec("a7")
    a("\n### A8. RAG and structured-extraction system")
    sec("a8")
    a(slot("A04"))
    a(slot("R06"))
    a("\n### A9. Evaluation design: splits, metrics and statistics")
    sec("a9")
    a(slot("F17"))
    a("\n## 1. Data integrity")
    a(f.get("universe_text", ""))
    a("\n### Universe by year")
    a(md_table(f.get("firms_per_year")))
    a(slot("F04"))
    a("\n### Filing coverage (observed vs parsed) and audit reason codes")
    a(md_table(f.get("coverage_by_form")))
    a(md_table(f.get("reason_codes")))
    a(slot("F01"))
    a(slot("F02"))
    a("\n### 8-K category availability (taxonomy v1)")
    a(md_table(f.get("category_by_year"), 30))
    a(slot("F03"))
    a("\n### Timestamp quality")
    a(md_table(f.get("ts_quality")))
    a(f"\n{f.get('timestamp_text', '')}")
    a(slot("F05"))
    a(slot("F06"))
    a("\n### Hugging Face dataset audit")
    a(re.sub(r"^#+ ", "#### ", f.get("hf_text", ""), flags=re.M))
    a("\n### Limitations of the data")
    for x in f.get("data_limitations", []):
        a(f"- {x}")
    a("\n## 2. Design")
    a(f.get("design_text", ""))
    a(md_table(f.get("event_counts")))
    a(slot("F08"))
    a(slot("F09"))
    a("\n## 3. Leakage and provenance audit")
    a(md_table(f.get("leakage_audit"), 30))
    a("\n### Model provenance ledger")
    a(md_table(f.get("ledger"), 30))
    a("\n### Sanity experiments and review triggers")
    a(md_table(f.get("sanity")))
    a(slot("F37"))
    a(slot("F38"))
    a(slot("F39"))
    a(N.get("mcs", ""))
    a(slot("F40"))
    a("\n## 4. Track A: event study (association only)")
    a(md_table(f.get("track_a_main"), 40))
    for fid in ("F11", "F12", "F13", "F14", "F15"):
        a(slot(fid))
    a("\n### Drift tests and verdicts")
    a(md_table(f.get("hypothesis_tests"), 30))
    a(f.get("drift_text", ""))
    a(slot("F16"))
    a("\n## 5. Track B: prediction")
    a(f.get("trackb_text", ""))
    a("\n### Ladder (walk-forward, primary target)")
    a(md_table(f.get("ladder"), 60))
    a(slot("F18"))
    a("\n### Pre-registered guardrails (log-loss no worse than the class prior; ECE <= 0.10; 1 = met)")
    a(md_table(f.get("guardrails"), 40))
    a("\n### Hypotheses H1-H4 (with CIs)")
    a(md_table(f.get("hypotheses_core")))
    a(slot("F19"))
    a("\n### 5a. How good is the accuracy, really? Chance level, move versus direction, locked-test lifts")
    a(N.get("context", "_(not built)_"))
    a(slot("R01"))
    a(slot("R02"))
    a(slot("R03"))
    a(N.get("selection", ""))
    a("\n### Tuning: default vs tuned")
    a(md_table(f.get("tuning_before_after"), 40))
    for fid in ("F20", "F21", "F22", "F23"):
        a(slot(fid))
    a("\n### Development holdout")
    a(md_table(f.get("dev_results"), 40))
    a("\n### Locked test")
    a(f.get("locked_text", ""))
    a(md_table(f.get("locked_results"), 40))
    for fid in ("F24", "F25", "F26"):
        a(slot(fid))
    a("\n### Selective prediction (coverage)")
    a(md_table(f.get("selective"), 20))
    a(slot("F27"))
    a(slot("F28"))
    a(slot("F31"))
    a("\n### T_info vs T_trade")
    a(md_table(f.get("tinfo_vs_ttrade"), 20))
    a(slot("F29"))
    a("\n### Backtest (economic-relevance check; not an investable strategy)")
    a(f.get("backtest_text", ""))
    a(md_table(f.get("backtest"), 20))
    a(slot("F30"))
    a("\n### Index membership: S&P 500 vs non-S&P 500 vs combined")
    comp = f.get("sp_composition")
    if comp is not None and len(comp) and int((comp.sp_bucket != "Combined").sum()) <= 1:
        a("Only one index bucket exists in this universe (every firm is an S&P 500 member), so the S&P-vs-non-S&P comparison does not apply "
          "to this study; it is part of the 500-firm design. Composition:\n")
        a(md_table(comp, 10))
    else:
        a(f.get("sp_text", ""))
        a("\n**Composition**")
        a(md_table(comp, 10))
        a("\n**Prediction (walk-forward, primary target): the combined model scored per bucket**")
        a(md_table(f.get("sp_scores_wf"), 30))
        a(slot("FX2"))
        a("\n**Pooled vs bucket-only models (paired, identical rows)**")
        a(md_table(f.get("sp_pooled_vs_own"), 10))
        a("\n**Development holdout / locked test by bucket**")
        a(md_table(f.get("sp_scores_later"), 30))
        a("\n**Track A by bucket (association only)**")
        a(md_table(f.get("sp_track_a"), 30))
        a(slot("FX3"))
        a("\n**SUE-conditioned drift by bucket**")
        a(md_table(f.get("sp_drift"), 10))
        a("\n**Backtest by bucket (lowest cost, no lag)**")
        a(md_table(f.get("sp_backtest"), 20))
    a("\n## 6. Track C: content validity")
    a(f.get("trackc_text", ""))
    a(md_table(f.get("trackc"), 30))
    for fid in ("F32", "F33", "F34", "F35", "F36"):
        a(slot(fid))
    a("\n## 7. Fit diagnostics")
    a(md_table(f.get("fit_diag"), 40))
    a(f.get("optim_text", ""))
    a(slot("FX1"))
    a("\n## 8. Extensions run")
    a(md_table(f.get("extensions"), 20))
    a("\n## 9. Limitations, resource use, decision log")
    for x in f.get("limitations", []):
        a(f"- {x}")
    a(md_table(f.get("resources"), 40))
    a(f"\nDecision log: `{f.get('decision_log', 'docs/decision_log.md')}`")
    a("\n## 10. What the results mean: answers to the research questions")
    a(N.get("answers", "_(not built)_"))
    a("\n## 11. Known issues, reproducibility and next steps")
    a(N.get("issues", "_(not built)_"))
    a("\n## Appendix: decision log, file map and how to re-run")
    a(N.get("appendix", "_(not built)_"))
    a("\n## Figure index")
    for fig in f.get("figures", []):
        st = "" if fig["status"] == "ok" else f" - **{fig['status']}**"
        a(f"- **{fig['id']}** {fig['title']}{st}. {fig['caption']}")
    return "\n".join(L)


def _fig_map(figures: list[dict]) -> dict:
    return {fg["id"]: fg for fg in figures}


def resolve_slots_md(md_text: str, figures: list[dict]) -> str:
    fm = _fig_map(figures)

    def rep(m):
        fg = fm.get(m.group(1))
        if fg is None:
            return ""
        if fg.get("path") and fg["status"] == "ok":
            return f"\n![{fg['id']} {fg['title']}](figures/{Path(fg['path']).name})\n\n*{fg['id']} {fg['title']}.* {fg['caption']}\n"
        return f"\n*{fg['id']} {fg['title']}: {fg['status']}*\n"
    return _SLOT.sub(rep, md_text)


def build_html(md_text: str, figures: list[dict]) -> str:
    fm = _fig_map(figures)
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(md_text) + "</pre>"
    used = set()

    def rep(m):
        fg = fm.get(m.group(1))
        if fg is None:
            return ""
        used.add(fg["id"])
        if fg.get("path") and Path(fg["path"]).exists() and fg["status"] == "ok":
            b64 = base64.b64encode(Path(fg["path"]).read_bytes()).decode()
            return (f"<figure><img src='data:image/png;base64,{b64}' style='max-width:100%'/>"
                    f"<figcaption><b>{fg['id']}</b> {html.escape(fg['title'])}. {html.escape(fg['caption'])}</figcaption></figure>")
        return f"<p class='skip'><i>{fg['id']} {html.escape(fg['title'])}: {html.escape(fg['status'])}</i></p>"
    body = re.sub(r"<p>\[\[FIG:(\w+)\]\]</p>", rep, body)
    extra = []
    for fg in figures:
        if fg["id"] not in used and fg.get("path") and Path(fg["path"]).exists() and fg["status"] == "ok":
            b64 = base64.b64encode(Path(fg["path"]).read_bytes()).decode()
            extra.append(f"<figure><img src='data:image/png;base64,{b64}' style='max-width:100%'/>"
                         f"<figcaption><b>{fg['id']}</b> {html.escape(fg['title'])}. {html.escape(fg['caption'])}</figcaption></figure>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1150px;margin:auto;padding:16px;line-height:1.5;color:#222}"
           "h1{border-bottom:2px solid #2563eb;padding-bottom:6px}h2{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:3px}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto;max-width:100%}td,th{border:1px solid #ccc;padding:3px 6px;vertical-align:top}"
           "th{background:#f3f3f3}figure{margin:18px 0;border-top:1px solid #eee;padding-top:8px}"
           "figcaption{font-size:12px;color:#444}code{background:#f5f5f5;padding:1px 3px}.skip{color:#777;font-size:12px}")
    more = f"<h2>Figures not placed in a section</h2>{''.join(extra)}" if extra else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>SEC filings price impact report</title>"
            f"<style>{css}</style></head><body>{body}{more}</body></html>")


def _pdf_table(rows: list[str], ss, width):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    cells = []
    for r in rows:
        if re.match(r"^\|[-| :]+\|\s*$", r):
            continue
        cells.append([c.strip() for c in r.strip().strip("|").split("|")])
    if not cells:
        return None
    ncol = max(len(c) for c in cells)
    cells = [c + [""] * (ncol - len(c)) for c in cells]
    fs = 7 if ncol <= 5 else 6 if ncol <= 9 else 5
    st = ParagraphStyle("t", parent=ss["BodyText"], fontSize=fs, leading=fs + 1.6)
    data = [[Paragraph(html.escape(c)[:400], st) for c in row] for row in cells]
    lens = [max(min(len(row[j]), 60) for row in cells) + 4 for j in range(ncol)]
    widths = [width * x / sum(lens) for x in lens]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


def build_pdf(md_text: str, figures: list[dict], path: Path) -> str | None:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except Exception as e:
        return f"reportlab unavailable: {e}"
    ss = getSampleStyleSheet()
    fm = _fig_map(figures)
    page = landscape(A4)
    width = page[0] - 3 * cm
    story = []
    table_buf: list[str] = []
    used: set = set()

    def flush():
        if table_buf:
            t = _pdf_table(table_buf, ss, width)
            if t is not None:
                story.append(t)
                story.append(Spacer(1, 5))
            table_buf.clear()

    def add_fig(fg):
        used.add(fg["id"])
        if fg.get("path") and Path(fg["path"]).exists() and fg["status"] == "ok":
            story.append(Image(fg["path"], width=width, height=14.5 * cm, kind="proportional"))
            story.append(Paragraph(f"<b>{fg['id']}</b> {html.escape(fg['title'])}. {html.escape(fg['caption'])}", ss["BodyText"]))
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(f"<i>{fg['id']} {html.escape(fg['title'])}: {html.escape(fg['status'])}</i>", ss["BodyText"]))

    for line in md_text.splitlines():
        if line.startswith("|"):
            table_buf.append(line)
            continue
        flush()
        m = _SLOT.match(line)
        if m:
            if m.group(1) in fm:
                add_fig(fm[m.group(1)])
            continue
        esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(line))
        esc = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", esc)
        if line.startswith("# "):
            story.append(Paragraph(esc[2:], ss["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(esc[3:], ss["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(esc[4:], ss["Heading3"]))
        elif line.strip():
            story.append(Paragraph(esc, ss["BodyText"]))
        else:
            story.append(Spacer(1, 4))
    flush()
    rest = [fg for fg in figures if fg["id"] not in used and fg.get("path") and fg["status"] == "ok"]
    if rest:
        story.append(PageBreak())
        story.append(Paragraph("Figures not placed in a section", ss["Heading2"]))
        for fg in rest:
            add_fig(fg)
    SimpleDocTemplate(str(path), pagesize=page, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm).build(story)
    return None
