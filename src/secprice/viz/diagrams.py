"""Architecture diagrams for the report (A01-A04). Boxes carry numbers read from the run (passed in), never typed here."""
from __future__ import annotations

import textwrap

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {"CPU": "#dbeafe", "GPU": "#fde68a", "CPU-GPU": "#fbcfe8", "data": "#e5e7eb", "ok": "#d1fae5", "warn": "#fed7aa",
          "off": "#f3f4f6", "edge": "#374151", "strict": "#bbf7d0", "retro": "#fecaca"}


def _box(ax, x, y, w, h, text, fc, ec=None, fs=7.5, ls="-", bold=False, wrap=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec or COLORS["edge"], lw=1.0, ls=ls))
    if wrap:
        text = "\n".join(textwrap.fill(t, wrap) for t in text.split("\n"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1, color="#6b7280", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def _canvas(F, w=11.5, h=6.2):
    fig, ax = F.plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def pipeline(F, fid, title, stages: list[dict]):
    """stages: dicts with id, name, part, hours (float|None), note (str|None: 'ran empty' flag)."""
    fig, ax = _canvas(F, 12.5, 7.6)
    bands = [("PART 1 - CPU: data, events, features, baselines, event study", ["CPU"], lambda s: s["id"] in
              {"S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S20", "S21"}),
             ("PART 2 - GPU: embeddings, text models, RAG", ["GPU"], lambda s: s["id"] in {"S13", "S14", "S15", "S16", "S17"}),
             ("PART 3 - CPU: holdout, locked test, backtest, report", ["CPU"], lambda s: s["id"] in {"S18", "S19", "S22", "S23", "S24"})]
    y_top, per_row = 96, 8
    for title_b, _, pick in bands:
        sel = [s for s in stages if pick(s)]
        rows = max(1, -(-len(sel) // per_row))
        ax.text(1, y_top, title_b, fontsize=9, fontweight="bold", va="top")
        y0 = y_top - 4.5
        for k, s in enumerate(sel):
            r, c = divmod(k, per_row)
            x, y = 1.5 + c * 12.2, y0 - r * 14.5 - 11
            warn = bool(s.get("note"))
            fc = COLORS["warn"] if warn else COLORS.get(s["part"], COLORS["CPU"])
            hrs = s.get("hours")
            lab = f"{s['id']}  {('%.1f h' % hrs) if hrs is not None and hrs >= 0.1 else ('<0.1 h' if hrs is not None else '')}\n" \
                  + textwrap.fill(s["name"], 17) + (f"\n[{s['note']}]" if warn else "")
            _box(ax, x, y, 11, 11.5, lab, fc, fs=6.4, ls="--" if warn else "-")
            if c < per_row - 1 and k + 1 < len(sel) and divmod(k + 1, per_row)[0] == r:
                _arrow(ax, x + 11.1, y + 5.7, x + 12.3, y + 5.7)
        y_top = y0 - rows * 14.5 - 6
    ax.text(1, 2.5, "blue = CPU stage, yellow = GPU stage, pink = CPU/GPU, orange dashed = ran but produced no result (needs gold labels). "
                    "Hours = wall time of the run that produced the result.", fontsize=7, color="#4b5563")
    F.save(fig, fid, title)


def data_flow(F, fid, title, c: dict):
    """c: strings/numbers for the boxes (all computed)."""
    fig, ax = _canvas(F, 13, 7.4)
    cols = [("1 SOURCES", 1), ("2 ACQUIRE (S02-S06)", 20.5), ("3 CLEAN + SECTION (S05)", 40), ("4 EVENTS + CLOCKS (S07-S08)", 59.5),
            ("5 FEATURES (S09)", 79)]
    for t, x in cols:
        ax.text(x, 97, t, fontsize=8.5, fontweight="bold")
    W, H = 18, 14
    src = [("SEC EDGAR\n" + c["src_edgar"], 82), ("Hugging Face SEC corpus\n" + c["src_hf"], 66), ("Daily prices (OHLCV)\n" + c["src_px"], 50),
           ("Factors + industry\n" + c["src_fac"], 34), ("S&P 500 history\n" + c["src_sp"], 18), ("Tone dictionary\n" + c["src_lm"], 3)]
    for t, y in src:
        _box(ax, 1, y, W, H - 1, t, COLORS["data"], fs=6.6)
    acq = [("Universe + eligibility\n" + c["universe"], 82), ("Filing manifest + 4 timestamps\n" + c["manifest"], 66),
           ("Prices, splits, factors, SPY\n" + c["prices"], 50), ("XBRL facts + release EPS -> SUE\n" + c["xbrl"], 34)]
    for t, y in acq:
        _box(ax, 20.5, y, W, H - 1, t, COLORS["CPU"], fs=6.6)
    prep = [("Download + HTML -> text\n" + c["download"], 82), ("Item sections (10-K 1/1A/7/7A,\n10-Q MD&A/RF/MR, 8-K items+EX-99)\n" + c["sections"], 64),
            ("Mask firm names + tickers\n(firm-memorisation control)\n" + c["mask"], 46), ("Drop numeric lines, cap length,\nchunk 350 words / 50 overlap\n" + c["chunks"], 28)]
    for t, y in prep:
        _box(ax, 40, y, W, H + 1, t, COLORS["CPU"], fs=6.2)
    ev = [("Events: 10-K, 10-Q, same-day 8-K groups\n" + c["events"], 82), ("Clocks: acceptance -> public (+5 min)\n-> decision (+30 min)", 66),
          ("Targets: T_trade (primary), T_info\n3 classes at +/- k*sigma*sqrt(h)\n" + c["targets"], 46), ("Overlap flags + clean cohort\n" + c["overlap"], 30)]
    for t, y in ev:
        _box(ax, 59.5, y, W, H + 1, t, COLORS["CPU"], fs=6.4)
    ft = [(c["ft_a"], 82), (c["ft_b"], 66),
          ("Chunk embeddings (GPU):\nChronoBERT + BGE", 50), ("Fold-fitted TF-IDF + SVD,\nPCA, imputers, scalers", 34),
          ("Availability rule: every feature's\nlatest input <= decision time", 18)]
    for t, y in ft:
        _box(ax, 79, y, 20, H - 1, t, COLORS["GPU"] if "GPU" in t else COLORS["CPU"], fs=6.4)
    for x0, x1 in ((19.1, 20.4), (38.6, 39.9), (58.1, 59.4), (77.6, 78.9)):
        _arrow(ax, x0, 50, x1, 50)
    ax.text(1, 0.5, c["footer"], fontsize=7, color="#4b5563")
    F.save(fig, fid, title)


def ladder(F, fid, title, rungs: list[dict]):
    """rungs: dicts rung, adds, wf (walk-forward macro-F1 or None), lb ('strict'|'retrospective'), status."""
    fig, ax = _canvas(F, 12, 6.2)
    n = len(rungs)
    h = 86 / n
    ax.text(1, 98, "Ablation ladder (SPEC.md section 9.1): each rung ADDS information to the one before; every comparison is on identical events",
            fontsize=8.5, fontweight="bold")
    for k, r in enumerate(rungs):
        y = 90 - (k + 1) * h + 0.8
        off = r.get("status") == "not run"
        fc = COLORS["off"] if off else (COLORS["strict"] if r["lb"] == "strict" else COLORS["retro"])
        _box(ax, 1, y, 9, h - 2, r["rung"], fc, fs=9, bold=True, ls="--" if off else "-")
        _box(ax, 11.5, y, 58, h - 2, r["adds"], "white", fs=7, ls="--" if off else "-", wrap=88)
        sc = "not run (needs gold labels)" if off else (f"walk-forward macro-F1 {r['wf']:.3f}" if r.get("wf") is not None else "")
        _box(ax, 71, y, 28, h - 2, sc + (f"\n{r['lb']} leaderboard" if not off else ""), fc, fs=7)
        if k:
            _arrow(ax, 5.5, y + h - 1.6, 5.5, y + h - 2.2 + 0.3)
    ax.text(1, 1, "green = strict point-in-time leaderboard (documented cutoff before every evaluated event); red = retrospective only (cutoff unknown); grey = not run.",
            fontsize=7, color="#4b5563")
    F.save(fig, fid, title)


def rag(F, fid, title, c: dict):
    fig, ax = _canvas(F, 13, 7.2)
    ax.text(1, 97, "Retrieval-augmented, evidence-grounded extraction (S16-S17). Solid = ran in this study, dashed grey = built but not run (needs gold labels).",
            fontsize=8.5, fontweight="bold")
    # row 1: retrieval
    _box(ax, 1, 70, 14, 18, "Filing sections\n(masked text)\n" + c["sections"], COLORS["data"], fs=6.6)
    _box(ax, 19, 70, 14, 18, "Chunks\n350 words,\n50 overlap\n" + c["chunks"], COLORS["CPU"], fs=6.6)
    _box(ax, 38, 79, 15, 10, "BM25\n(k1 1.5, b 0.75)", COLORS["CPU"], fs=6.6)
    _box(ax, 38, 66, 15, 10, "Dense BGE-base\n(cosine)", COLORS["GPU"], fs=6.6)
    _box(ax, 58, 70, 12, 18, "Reciprocal rank fusion\n(k=60)\n-> top 30", COLORS["CPU"], fs=6.4)
    _box(ax, 74, 70, 12, 18, "BGE reranker\n(cross-encoder)\n-> top 8", COLORS["GPU"], fs=6.4)
    _box(ax, 89, 70, 10, 18, "8 passages\nwith chunk ids", COLORS["data"], fs=6.4)
    for x0, x1, y in ((15.2, 18.8, 79), (33.2, 37.8, 84), (33.2, 37.8, 71), (53.2, 57.8, 79), (70.2, 73.8, 79), (86.2, 88.8, 79)):
        _arrow(ax, x0, y, x1, y)
    # row 2: generation + guardrails
    _box(ax, 1, 40, 17, 20, "Question bank v1\n12 fields: guidance, results tone,\nimpairment, restructuring, liquidity,\nfinancing, M&A, dividend, buyback,\nlitigation, management, risk factors", COLORS["data"], fs=6.0)
    _box(ax, 22, 40, 18, 20, "Qwen2.5-7B-Instruct\n(local, no hosted API)\nJSON-schema constrained\ngeneration\n" + c["gen"], COLORS["GPU"], fs=6.4)
    _box(ax, 44, 36, 28, 28, "GUARDRAILS\n1 schema validation + one retry\n2 numeric cross-check: a magnitude must\n   appear in its cited evidence\n3 faithfulness: every quote must be a\n   substring of its cited chunk, else null\n4 citation offsets verified",
         COLORS["ok"], fs=6.3)
    _box(ax, 76, 40, 11, 20, "Response cache\n(model revision +\nprompt + passage\nhash)\n" + c["cache"], COLORS["data"], fs=6.0)
    _box(ax, 89, 40, 10, 20, "Pre-filled\nannotation\ntemplates\n" + c["templates"], COLORS["ok"], fs=6.2)
    for x0, x1, y in ((18.2, 21.8, 50), (40.2, 43.8, 50), (72.2, 75.8, 50), (87.2, 88.8, 50)):
        _arrow(ax, x0, y, x1, y)
    _arrow(ax, 94, 69.5, 31, 60.5, rad=-0.25)
    # row 3: downstream, not run
    off = dict(fc=COLORS["off"], ec="#9ca3af", ls="--")
    _box(ax, 1, 8, 20, 18, "HUMAN GOLD LABELS\n(gold-dev / gold-val / gold-test)\nannotators fix the pre-filled\ntemplates", off["fc"], off["ec"], fs=6.2, ls=off["ls"])
    _box(ax, 26, 8, 20, 18, "Gate G3b: extraction F1 >= 0.70 and\nfaithfulness >= 0.90 on gold-val\n-> allows teacher run at scale", off["fc"], off["ec"], fs=6.2, ls=off["ls"])
    _box(ax, 51, 8, 20, 18, "Teacher run on a stratified sample of\ntraining-period filings (S17)\n-> student ModernBERT-base\n(multi-task)", off["fc"], off["ec"], fs=6.2, ls=off["ls"])
    _box(ax, 76, 8, 23, 18, "L8 = L7 + extracted-event features;\nTrack C: content validity on gold-test\n(never validated against prices)", off["fc"], off["ec"], fs=6.2, ls=off["ls"])
    for x0, x1 in ((21.2, 25.8), (46.2, 50.8), (71.2, 75.8)):
        _arrow(ax, x0, 17, x1, 17, color="#9ca3af", ls="--")
    _arrow(ax, 94, 39.5, 12, 26.5, color="#9ca3af", ls="--", rad=0.2)
    ax.text(1, 31, "Historical-evidence path (SPEC.md section 8): the firm's PRIOR filings with public time < decision time only; never the current or a later filing "
                   "(self-test + planted-leak test).", fontsize=6.6, color="#374151")
    F.save(fig, fid, title)
