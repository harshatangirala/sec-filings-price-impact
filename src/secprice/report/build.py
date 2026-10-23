"""S24 report: reports/results_summary.md (SPEC.md §21, computed values only) + a self-contained HTML report with
figures embedded + a PDF. Every number in the text is read from an artifact; missing results are stated as such."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
        out.append("| " + " | ".join(fmt(r[c], nd, c) for c in cols) + " |")
    if len(df) > max_rows:
        out.append(f"\n_{len(df) - max_rows} more rows in the CSV._")
    return "\n".join(out)


def smallest_defensible_claim(f: dict) -> str:
    h1 = f.get("h1") or {}
    lift, lo, hi = h1.get("value"), h1.get("ci_low"), h1.get("ci_high")
    n = h1.get("n_events")
    block = h1.get("block", "walk-forward validation")
    if lift is None:
        return ("The core text-versus-baseline comparison (H1) could not be evaluated in this run; the study supports "
                "only the data-integrity and event-study statements above.")
    if lo is not None and lo > 0:
        return (f"On the common cohort ({fmt(n)} events, {block}), adding SEC-filing text to price, metadata, accounting "
                f"and lexical baselines raised macro-F1 by {fmt(lift)} (95% month-block CI {fmt(lo)} to {fmt(hi)}) under "
                f"the assumed availability/latency policy. This is an out-of-time association with modest predictive "
                f"value, not evidence of causation or of tradable profit.")
    if hi is not None and hi < 0:
        return (f"On the common cohort ({fmt(n)} events, {block}), adding SEC-filing text lowered macro-F1 by "
                f"{fmt(-lift)} (CI {fmt(lo)} to {fmt(hi)}): text did not add out-of-time value beyond the baselines.")
    return (f"On the common cohort ({fmt(n)} events, {block}), the lift from adding SEC-filing text was {fmt(lift)} "
            f"macro-F1 with a 95% CI of {fmt(lo)} to {fmt(hi)} that includes zero: we cannot claim that filing text adds "
            f"reliable out-of-time information beyond the structured baselines. A credible null is a valid result.")


def build_markdown(f: dict) -> str:
    L = []
    a = L.append
    a(f"# SEC filings -> price impact: results summary ({f.get('profile', '?')} profile)")
    a("")
    a(f"Generated {f.get('generated')} | config hash `{f.get('config_hash')}` | code `{f.get('code')}` | "
      f"pre-registration hash `{f.get('prereg_hash', 'not frozen')}`")
    a(f"\n**Timing policy:** {f.get('timing_policy')}")
    a("\n**Reading rules:** association is not causation; prediction is not profitability. Results are "
      "validation-selected, never a global optimum. Power labels: n<30 warning, 30-99 exploratory, wide CI inconclusive.")
    a("\n## 10. Executive summary")
    for b in f.get("exec_bullets", []):
        a(f"- {b}")
    a(f"\n**Smallest defensible claim.** {smallest_defensible_claim(f)}")
    a("\n## 1. Data integrity")
    a(f.get("universe_text", ""))
    a("\n### Universe by year")
    a(md_table(f.get("firms_per_year")))
    a("\n### Filing coverage (observed vs parsed) and audit reason codes")
    a(md_table(f.get("coverage_by_form")))
    a(md_table(f.get("reason_codes")))
    a("\n### 8-K category availability (taxonomy v1)")
    a(md_table(f.get("category_by_year"), 30))
    a("\n### Timestamp quality")
    a(md_table(f.get("ts_quality")))
    a(f"\n{f.get('timestamp_text', '')}")
    a("\n### Hugging Face dataset audit")
    a(f.get("hf_text", ""))
    a("\n### Limitations of the data")
    for x in f.get("data_limitations", []):
        a(f"- {x}")
    a("\n## 2. Design")
    a(f.get("design_text", ""))
    a(md_table(f.get("event_counts")))
    a("\n## 3. Leakage and provenance audit")
    a(md_table(f.get("leakage_audit"), 30))
    a("\n### Model provenance ledger")
    a(md_table(f.get("ledger"), 30))
    a("\n### Sanity experiments and review triggers")
    a(md_table(f.get("sanity")))
    a("\n## 4. Track A: event study (association only)")
    a(md_table(f.get("track_a_main"), 40))
    a("\n### Drift tests and verdicts")
    a(md_table(f.get("hypothesis_tests"), 30))
    a("\n## 5. Track B: prediction")
    a(f.get("trackb_text", ""))
    a("\n### Ladder (walk-forward, primary target)")
    a(md_table(f.get("ladder"), 60))
    a("\n### Hypotheses H1-H4 (with CIs)")
    a(md_table(f.get("hypotheses_core")))
    a("\n### Tuning: default vs tuned")
    a(md_table(f.get("tuning_before_after"), 40))
    a("\n### Development holdout")
    a(md_table(f.get("dev_results"), 40))
    a("\n### Locked test")
    a(f.get("locked_text", ""))
    a(md_table(f.get("locked_results"), 40))
    a("\n### Selective prediction (coverage)")
    a(md_table(f.get("selective"), 20))
    a("\n### T_info vs T_trade")
    a(md_table(f.get("tinfo_vs_ttrade"), 20))
    a("\n### Backtest (economic-relevance check; not an investable strategy)")
    a(md_table(f.get("backtest"), 20))
    a("\n## 6. Track C: content validity")
    a(f.get("trackc_text", ""))
    a(md_table(f.get("trackc"), 30))
    a("\n## 7. Fit diagnostics")
    a(md_table(f.get("fit_diag"), 40))
    a(f.get("optim_text", ""))
    a("\n## 8. Extensions run")
    a(md_table(f.get("extensions"), 20))
    a("\n## 9. Limitations, resource use, decision log")
    for x in f.get("limitations", []):
        a(f"- {x}")
    a(md_table(f.get("resources"), 40))
    a(f"\nDecision log: `{f.get('decision_log', 'docs/decision_log.md')}`")
    a("\n## Figures")
    for fig in f.get("figures", []):
        st = "" if fig["status"] == "ok" else f" - **{fig['status']}**"
        a(f"- **{fig['id']}** {fig['title']}{st}. {fig['caption']}")
    return "\n".join(L)


def build_html(md_text: str, figures: list[dict]) -> str:
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(md_text) + "</pre>"
    imgs = []
    for fg in figures:
        if fg.get("path") and Path(fg["path"]).exists():
            b64 = base64.b64encode(Path(fg["path"]).read_bytes()).decode()
            imgs.append(f"<figure><img src='data:image/png;base64,{b64}' style='max-width:100%'/>"
                        f"<figcaption><b>{fg['id']}</b> {html.escape(fg['title'])}. {html.escape(fg['caption'])}</figcaption></figure>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.45;color:#222}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px}"
           "th{background:#f3f3f3}figure{margin:18px 0;border-top:1px solid #eee;padding-top:8px}"
           "figcaption{font-size:12px;color:#444}code{background:#f5f5f5;padding:1px 3px}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>SEC filings price impact report</title>"
            f"<style>{css}</style></head><body>{body}<h2>Figures</h2>{''.join(imgs)}</body></html>")


def build_pdf(md_text: str, figures: list[dict], path: Path) -> str | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except Exception as e:
        return f"reportlab unavailable: {e}"
    ss = getSampleStyleSheet()
    story = []
    table_buf: list[str] = []

    def flush():
        if table_buf:
            story.append(Preformatted("\n".join(x[:180] for x in table_buf[:60]), ss["Code"], maxLineLength=180))
            table_buf.clear()

    for line in md_text.splitlines():
        if line.startswith("|"):
            table_buf.append(line)
            continue
        flush()
        esc = html.escape(line).replace("**", "")
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
    story.append(PageBreak())
    for fg in figures:
        if fg.get("path") and Path(fg["path"]).exists():
            story.append(Paragraph(f"{fg['id']} {html.escape(fg['title'])}", ss["Heading3"]))
            story.append(Image(fg["path"], width=17 * cm, height=9 * cm, kind="proportional"))
            story.append(Paragraph(html.escape(fg["caption"]), ss["BodyText"]))
            story.append(Spacer(1, 10))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm).build(story)
    return None
