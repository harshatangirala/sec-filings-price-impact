"""Figures for the paper, drawn from paper/tables and the dashboard's data snapshot. Writes SVG to paper/figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
T, D, F = ROOT / "paper" / "tables", ROOT / "dashboard" / "data", ROOT / "paper" / "figures"
F.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, INK, MUTE, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#111111", "#52514e", "#e4e3df"
COL = {"L6": BLUE, "L7": ORANGE, "L1": AQUA}
MARK = {"L6": "o", "L7": "s", "L1": "^"}
NAME = {"L1": "Price only (L1)", "L6": "All non-neural (L6)", "L7": "L6 + embeddings (L7)"}
PERS = ["Walk-forward 2008-2019", "Development 2020-2022", "Locked test 2023-2026"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTE, "axes.linewidth": .7, "xtick.color": MUTE, "ytick.color": MUTE, "xtick.major.size": 2.5, "ytick.major.size": 0,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5, "svg.fonttype": "path", "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)


def fwd(a):
    return a


# ---------------------------------------------------------------- design
def fig_timeline():
    b = pd.read_csv(D / "blocks.csv")
    fig, ax = plt.subplots(figsize=(6.7, 2.25))
    for i, r in b[b.kind == "walk_forward_val"].reset_index(drop=True).iterrows():
        s, e = pd.Timestamp(r.start).year, pd.Timestamp(r.end_exclusive).year
        ax.add_patch(Rectangle((2000, i), s - 2000 - .25, .68, color="#c9d9ee", lw=0))
        ax.add_patch(Rectangle((s, i), e - s, .68, color=BLUE, lw=0))
        ax.text(s + (e - s) / 2, i + .34, f"{int(r.events):,}", ha="center", va="center", color="white", fontsize=7.3)
    ax.add_patch(Rectangle((2000, 6), 19.75, .68, color="#c9d9ee", lw=0)); ax.add_patch(Rectangle((2020, 6), 3, .68, color=YELLOW, lw=0)); ax.text(2021.5, 6.34, "7,455", ha="center", va="center", fontsize=7.3)
    ax.add_patch(Rectangle((2000, 7), 22.75, .68, color="#c9d9ee", lw=0)); ax.add_patch(Rectangle((2023, 7), 3.58, .68, color=ORANGE, lw=0)); ax.text(2024.8, 7.34, "8,213", ha="center", va="center", color="white", fontsize=7.3)
    ax.set_yticks(np.arange(8) + .34, [f"Fold {i + 1}" for i in range(6)] + ["Development", "Locked test"]); ax.invert_yaxis()
    ax.set_xlim(2000, 2026.7); ax.set_ylim(7.9, -.2); ax.grid(False); ax.spines["left"].set_visible(False); ax.set_xticks(range(2000, 2027, 2)); ax.tick_params(axis="x", labelsize=8)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ("#c9d9ee", BLUE, YELLOW, ORANGE)]
    ax.legend(handles, ["Training window", "Validation block", "Development holdout", "Locked test (scored once)"], ncol=4, loc="upper center", bbox_to_anchor=(.5, -.13), fontsize=8)
    save(fig, "timeline")


def fig_clock():
    fig, ax = plt.subplots(figsize=(6.7, 1.95)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-1.4, 3.3)
    ax.plot([2, 98], [1, 1], color=MUTE, lw=1)
    pts = [(14, "EDGAR\nacceptance"), (28, "Assumed public\n(+5 min)"), (46, "Decision time\n(+30 min)")]
    for x, t in pts:
        ax.plot([x, x], [.8, 1.2], color=INK, lw=1.2); ax.text(x, 1.45, t, ha="center", va="bottom", fontsize=8.2)
    ax.plot([58, 58], [.8, 1.2], color=BLUE, lw=1.6); ax.text(58, 1.45, "First open\nat or after", ha="center", va="bottom", fontsize=8.2)
    ax.plot([82, 82], [.8, 1.2], color=BLUE, lw=1.6); ax.text(82, 1.45, "Close of\nthat session", ha="center", va="bottom", fontsize=8.2)
    ax.plot([3, 3], [.8, 1.2], color=ORANGE, lw=1.6); ax.text(3, 1.45, "Last\nclose", ha="center", va="bottom", fontsize=8.2)
    ax.annotate("", xy=(82, .15), xytext=(58, .15), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2)); ax.text(70, -.05, "$T_{trade}$: open-to-close, primary target", ha="center", va="top", fontsize=8.2)
    ax.annotate("", xy=(82, -.75), xytext=(3, -.75), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2)); ax.text(42, -.93, "$T_{info}$: close-to-close, contains the first reaction", ha="center", va="top", fontsize=8.2)
    ax.axvspan(2, 46, ymin=.46, ymax=.53, color="#c9d9ee", lw=0, zorder=0)
    fig.savefig(F / "clock.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_sample():
    e = pd.read_csv(D / "events_by_year.csv").pivot(index="year", columns="event_type", values="events").fillna(0); e = e[e.index <= 2026]
    s = pd.read_csv(D / "session_buckets.csv").pivot(index="event_type", columns="session_bucket", values="events").fillna(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    bot = np.zeros(len(e))
    for c, col, lab in (("8K", BLUE, "8-K"), ("periodic_10Q", ORANGE, "10-Q"), ("periodic_10K", AQUA, "10-K")):
        a.bar(e.index, e[c], bottom=bot, color=col, width=.82, label=lab, edgecolor="white", linewidth=.4); bot += e[c].to_numpy()
    a.set_title("A. Events per year"); a.legend(ncol=3, loc="upper left"); a.grid(axis="x", visible=False); a.set_ylabel("Events"); a.set_ylim(0, bot.max() * 1.18)
    sh = s.div(s.sum(axis=1), axis=0); labs = {"8K": "8-K", "periodic_10Q": "10-Q", "periodic_10K": "10-K"}; order = ["periodic_10K", "periodic_10Q", "8K"]; left = np.zeros(3)
    for c, col, lab in (("PRE", YELLOW, "Pre-open"), ("INTRA", AQUA, "Session"), ("POST", BLUE, "After close")):
        v = sh.loc[order, c].to_numpy(); b.barh([labs[o] for o in order], v, left=left, color=col, label=lab, edgecolor="white", linewidth=.6, height=.62)
        for i, x in enumerate(v):
            if x > .08: b.text(left[i] + x / 2, i, f"{x:.0%}", ha="center", va="center", fontsize=7.8, color="white" if c == "POST" else INK)
        left += v
    b.set_title("B. When events reach EDGAR"); b.set_xlim(0, 1); b.set_xticks([0, .5, 1], ["0", "50%", "100%"]); b.grid(False); b.legend(ncol=3, loc="upper center", bbox_to_anchor=(.45, -.1), fontsize=7.6, columnspacing=.8, handlelength=1)
    save(fig, "sample")


def fig_target():
    h = pd.read_csv(D / "ar_hist.csv"); l = pd.read_csv(D / "labels_by_year.csv"); l = l[l.year <= 2026]
    pv = l.pivot(index="year", columns="label", values="events").fillna(0); pv = pv.div(pv.sum(axis=1), axis=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.45))
    h = h[h.abnormal_return_pct.abs() < 9.8]
    a.bar(h.abnormal_return_pct, h.events, width=.23, color=BLUE, lw=0); a.set_xlabel("Abnormal open-to-close return (%)"); a.set_ylabel("Events"); a.set_title("A. Distribution of the target"); a.grid(axis="x", visible=False)
    for c, col, m in (("Down", ORANGE, "v"), ("Sideways", MUTE, "o"), ("Up", BLUE, "^")):
        b.plot(pv.index, pv[c], color=col, marker=m, ms=3.5, lw=1.5, label=c)
    b.axvline(2007.5, color=GRID); b.set_title("B. Class shares by year"); b.set_yticks([.2, .3, .4, .5], ["20%", "30%", "40%", "50%"]); b.set_xlim(1999.5, 2026.5); b.set_xticks(range(2000, 2027, 5)); b.legend(ncol=3, loc="upper center", fontsize=7.8); b.set_ylim(.15, .62)
    save(fig, "target")


# ---------------------------------------------------------------- prediction
def fig_ladder():
    d = pd.read_csv(D / "ladder.csv"); d = d[(d.tuned == "tuned") | (d.rung == "L0")]
    rows = []
    for rung, lab in (("L0", "L0 Class prior"), ("L1", "L1 Price"), ("L2", "L2 Metadata"), ("L3", "L3 Accounting"), ("L4", "L4 Classical text"), ("L5", "L5 Embeddings"), ("L6", "L6 All non-neural"), ("L7", "L7 L6 + embeddings")):
        x = d[d.rung == rung]
        if rung in ("L5", "L7"):
            x = x[x.cohort == "common_emb"]
        elif rung == "L6":
            x = x[x.cohort == "common"]
        rows.append((lab, x))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    heads = {"lightgbm": (BLUE, "o", "LightGBM"), "catboost": (YELLOW, "D", "CatBoost"), "enet": (AQUA, "^", "Elastic net")}
    seen = set()
    for i, (lab, x) in enumerate(rows):
        off = {"lightgbm": -.22, "catboost": 0, "enet": .22}
        for _, r in x.iterrows():
            if r["head"] not in heads:
                ax.plot(r.macro_f1, i, "x", color=MUTE); continue
            c, m, nm = heads[r["head"]]; y = i + off[r["head"]] + (.08 if str(r.encoder) == "chrono_bert" else 0)
            ax.errorbar(r.macro_f1, y, xerr=[[r.macro_f1 - r.ci_low], [r.ci_high - r.macro_f1]], fmt=m, color=c, ms=4, elinewidth=.9, label=nm if nm not in seen else None, mfc=c if str(r.encoder) != "chrono_bert" else "white"); seen.add(nm)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis(); ax.axvline(.33, color=INK, ls="--", lw=1); ax.text(.3315, -.62, "shuffled-prediction chance, about 0.33", fontsize=7.8, va="center")
    ax.set_xlabel("Pooled macro-F1, walk-forward 2008-2019, with 95% interval"); ax.grid(axis="y", visible=False); ax.legend(loc="lower left")
    save(fig, "ladder")


def fig_perf():
    p = pd.read_csv(T / "performance.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.6)); w = .25
    for j, m in enumerate(("L1", "L6", "L7")):
        x = p[p.model == m].set_index("period").loc[PERS]
        ax.bar(np.arange(3) + (j - 1) * w, x.macro_f1, w * .94, color=COL[m], label=NAME[m], yerr=[x.macro_f1 - x.macro_f1_lo, x.macro_f1_hi - x.macro_f1], error_kw=dict(lw=.8, ecolor=INK))
        for i, v in enumerate(x.macro_f1): ax.text(i + (j - 1) * w, .262, f"{v:.3f}", ha="center", fontsize=7.6, color="white")
    ax.axhline(p.chance_shuffled[p.model != "L1"].mean(), color=INK, ls="--", lw=1, label="Chance (shuffled predictions)")
    ax.set_xticks(range(3), [s.replace(" 20", "\n20", 1) for s in PERS]); ax.set_ylim(.25, .46); ax.set_ylabel("Macro-F1"); ax.grid(axis="x", visible=False); ax.legend(ncol=4, loc="upper center", fontsize=7.6, columnspacing=.9, handlelength=1.4)
    save(fig, "performance")


def fig_auc():
    p = pd.read_csv(T / "performance.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    for ax, k, t in ((axs[0], "move_auc", "A. Will the stock move? (move AUC)"), (axs[1], "direction_auc", "B. Which way, given a move? (direction AUC)")):
        for j, m in enumerate(("L1", "L6", "L7")):
            x = p[p.model == m].set_index("period").loc[PERS]
            ax.errorbar(x[k], np.arange(3) + (j - 1) * .22, xerr=[x[k] - x[k + "_lo"], x[k + "_hi"] - x[k]], fmt=MARK[m], color=COL[m], ms=4.5, elinewidth=1, label=NAME[m])
        ax.axvline(.5, color=INK, ls="--", lw=1); ax.set_title(t); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(3), [s.split(" 20")[0] + "\n20" + s.split(" 20", 1)[1] for s in PERS]); axs[0].invert_yaxis(); axs[0].set_xlim(.49, .68); axs[1].set_xlim(.46, .54); h_, l_ = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, .08, 1, 1)); fig.legend(h_, l_, ncol=3, loc="lower center", fontsize=8.2)
    fig.savefig(F / "auc.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_year():
    d = pd.read_csv(T / "by_year.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5))
    for m in ("L1", "L6", "L7"):
        x = d[(d.model == m) & (d.n >= 200)]
        axs[0].plot(x.year, x.macro_f1, color=COL[m], marker=MARK[m], ms=3.2, lw=1.3, label=NAME[m])
        axs[1].plot(x.year, x.direction_auc, color=COL[m], marker=MARK[m], ms=3.2, lw=1.3)
    axs[0].axhline(.33, color=INK, ls="--", lw=.9); axs[1].axhline(.5, color=INK, ls="--", lw=.9)
    for a in axs:
        a.axvspan(2019.5, 2022.5, color="#fbecc8", lw=0, zorder=0); a.axvspan(2022.5, 2026.6, color="#fbdccc", lw=0, zorder=0); a.set_xlim(2007.5, 2026.6); a.set_xticks(range(2008, 2027, 3))
    axs[0].set_title("A. Macro-F1 by calendar year"); axs[1].set_title("B. Direction AUC by calendar year"); h_, l_ = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, .08, 1, 1)); fig.legend(h_, l_, ncol=3, loc="lower center", fontsize=8.2)
    fig.savefig(F / "by_year.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_deciles():
    d = pd.read_csv(T / "move_deciles.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45))
    for m in ("L6", "L1"):
        x = d[d.model == m]
        axs[0].plot(x.decile, x.mean_abs_ar * 100, color=COL[m], marker=MARK[m], ms=4, lw=1.5, label=NAME[m]); axs[1].plot(x.decile, x.share_moved, color=COL[m], marker=MARK[m], ms=4, lw=1.5)
    axs[0].set_title("A. Mean absolute abnormal return"); axs[0].set_ylabel("%"); axs[1].set_title("B. Share of events leaving the Sideways band"); axs[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for a in axs: a.set_xlabel("Decile of predicted P(move), locked test"); a.set_xticks(range(1, 11))
    axs[0].legend(loc="upper left")
    save(fig, "move_deciles")


def fig_reliability():
    d = pd.read_csv(T / "reliability.csv"); fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True, sharex=True)
    for ax, c in zip(axs, ("Down", "Sideways", "Up")):
        ax.plot([.15, .6], [.15, .6], color=MUTE, lw=.8, ls="--")
        for m in ("L6", "L7"):
            x = d[(d.model == m) & (d.cls == c)]; ax.plot(x.mean_p, x.freq, marker=MARK[m], color=COL[m], ms=3.5, lw=1.2, label=NAME[m])
        ax.set_title(c); ax.set_xlabel("Mean predicted probability"); ax.set_xlim(.15, .6); ax.set_ylim(.15, .6)
    axs[0].set_ylabel("Observed frequency"); axs[0].legend(loc="upper left", fontsize=7.2)
    save(fig, "reliability")


def fig_confusion():
    c = pd.read_csv(D / "locked_confusion.csv"); c = c[c.model == "L6"].pivot(index="true", columns="pred", values="events").loc[["Down", "Sideways", "Up"], ["Down", "Sideways", "Up"]]
    rn = c.div(c.sum(axis=1), axis=0); fig, ax = plt.subplots(figsize=(3.1, 2.5)); ax.imshow(rn, cmap="Blues", vmin=.1, vmax=.65); ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rn.iloc[i, j]:.0%}\n({int(c.iloc[i, j]):,})", ha="center", va="center", fontsize=8, color="white" if rn.iloc[i, j] > .45 else INK)
    ax.set_xticks(range(3), c.columns); ax.set_yticks(range(3), c.index); ax.set_xlabel("Predicted"); ax.set_ylabel("Realised")
    save(fig, "confusion")


def fig_lifts():
    d = pd.read_csv(T / "lifts.csv"); d = d[d.metric == "macro_f1"]; cmps = ["Text over structured (L7 - L6)", "Change features (L6 - L6 no change)", "Structured over price (L6 - L1)"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.15), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    for ax, c in zip(axs, cmps):
        x = d[d.comparison == c].set_index("period").loc[PERS]
        ax.errorbar(x.lift, range(3), xerr=[x.lift - x.lo, x.hi - x.lift], fmt="o", color=BLUE, ms=4.5, elinewidth=1.1); ax.axvline(0, color=INK, lw=.9); ax.set_title({"Text": "Embeddings (L7 - L6)", "Chan": "Change features", "Stru": "L6 over price only"}[c[:4]], fontsize=9); ax.grid(axis="y", visible=False)
        if c.startswith("Text"): ax.axvline(.01, color=ORANGE, ls="--", lw=1); ax.text(.0112, -.32, "0.01 gate", color=INK, fontsize=7.6)
        if not c.startswith("Struct"): ax.set_xlim(-.02, .025)
    axs[0].set_yticks(range(3), ["Walk-forward", "Development", "Locked test"]); axs[0].invert_yaxis(); fig.supxlabel("Paired change in macro-F1, with 95% month-block interval", fontsize=9, y=.02)
    save(fig, "lifts")


def fig_subsets():
    d = pd.read_csv(T / "text_lift_by_subset.csv"); subs = d.subset.drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(6.2, 3))
    for j, (sc, col, m) in enumerate((("Pooled 2008-2026", BLUE, "o"), ("Locked test", ORANGE, "s"))):
        x = d[d.scope == sc].set_index("subset").loc[subs]; ax.errorbar(x.lift, np.arange(len(subs)) + (j - .5) * .32, xerr=[x.lift - x.lo, x.hi - x.lift], fmt=m, color=col, ms=4, elinewidth=1, label=sc)
    ax.axvline(0, color=INK, lw=.9); ax.axhline(2.5, color=GRID); ax.axhline(4.5, color=GRID); ax.set_yticks(range(len(subs)), subs); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("Change in macro-F1 from adding text vectors (L7 - L6), with 95% interval"); ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.1))
    save(fig, "subsets")


def fig_tinfo():
    d = pd.read_csv(T / "tinfo_vs_ttrade.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.1), sharey=True); rungs = ["L1", "L4", "L6"]; lab = ["L1 Price", "L4 Classical text", "L6 All non-neural"]
    for ax, k, lo, hi, tt in ((axs[0], "move_auc", "move_lo", "move_hi", "A. Move AUC"), (axs[1], "direction_auc", "dir_lo", "dir_hi", "B. Direction AUC")):
        for j, (tg, col, m, nm) in enumerate((("T_info", ORANGE, "s", "$T_{info}$ (close-to-close)"), ("T_trade", BLUE, "o", "$T_{trade}$ (open-to-close)"))):
            x = d[d.target == tg].set_index("rung").loc[rungs]; ax.errorbar(x[k], np.arange(3) + (j - .5) * .3, xerr=[x[k] - x[lo], x[hi] - x[k]], fmt=m, color=col, ms=4.5, elinewidth=1, label=nm)
        ax.axvline(.5, color=INK, ls="--", lw=.9); ax.set_title(tt); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(3), lab); axs[0].invert_yaxis(); h_, l_ = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, .1, 1, 1)); fig.legend(h_, l_, ncol=2, loc="lower center", fontsize=8.2)
    fig.savefig(F / "tinfo.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------- event study and drift
def fig_car():
    c = pd.read_csv(D / "car_paths.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True)
    for ax, s, t in ((axs[0], "all", "A. All events"), (axs[1], "clean", "B. No other firm event within 60 sessions")):
        for g_, col, m, lab in (("8K", BLUE, None, "8-K"), ("periodic_10Q", ORANGE, None, "10-Q"), ("periodic_10K", AQUA, None, "10-K")):
            x = c[(c.group == g_) & (c.event_set == s)]
            if s == "clean" and g_ == "periodic_10K": continue
            ax.plot(x.rel_day, x.caar * 100, color=col, lw=1.5, label=lab)
        ax.axvline(0, color=INK, lw=.8); ax.axhline(0, color=MUTE, lw=.6); ax.set_title(t); ax.set_xlabel("Trading days relative to the event")
    axs[0].set_ylabel("Cumulative average abnormal return (%)"); axs[0].legend(loc="upper left")
    save(fig, "car_paths")


def fig_sue():
    q = pd.read_csv(T / "drift_quintiles.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5)); x = q.q.to_numpy()
    axs[0].bar(x, q.ann_c4 * 100, color=BLUE, width=.65); axs[0].set_title("A. Reaction, days [-1,+1]"); axs[0].set_ylabel("Carhart abnormal return (%)")
    w = .36; axs[1].bar(x - w / 2, q.d2_60_c4 * 100, w, color=BLUE, label="Carhart"); axs[1].bar(x + w / 2, q.d2_60_mm * 100, w, color=ORANGE, label="Market model"); axs[1].set_title("B. Drift, days [+2,+60]"); axs[1].legend()
    for a in axs:
        a.axhline(0, color=INK, lw=.8); a.set_xticks(x, ["Q1\nworst", "Q2", "Q3", "Q4", "Q5\nbest"]); a.grid(axis="x", visible=False); a.set_xlabel("Earnings-surprise quintile")
    save(fig, "sue_quintiles")


def fig_drift():
    d = pd.read_csv(T / "drift_robustness.csv").dropna(subset=["estimate"]); fig, ax = plt.subplots(figsize=(6.3, 4)); y = np.arange(len(d))
    ax.errorbar(d.estimate * 100, y, xerr=[(d.estimate - d.ci_low) * 100, (d.ci_high - d.estimate) * 100], fmt="o", color=BLUE, ms=4.3, elinewidth=1)
    ax.axvline(0, color=INK, lw=.9); ax.axvline(d.estimate.iloc[0] * 100, color=ORANGE, ls="--", lw=.9); ax.set_yticks(y, [f"{a}  (n = {int(n):,})" for a, n in zip(d.spec, d.n)], fontsize=8.3); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Q5 minus Q1 abnormal return, percent, with 95% month-block interval")
    save(fig, "drift_robustness")


def fig_bound():
    b = pd.read_csv(T / "survivorship_bound.csv"); L = np.linspace(-.1, -1, 100); q = pd.read_csv(T / "drift_quintiles.csv").set_index("q"); q1, q5 = q.loc[1, "d2_60_c4"], q.loc[5, "d2_60_c4"]
    fig, ax = plt.subplots(figsize=(4.8, 2.5)); ax.plot(-L * 100, (q1 - q5) / (q1 - L) * 100, color=BLUE, lw=1.8)
    v = (q1 - q5) / (q1 + .3) * 100; ax.plot([30], [v], "o", color=ORANGE, ms=6); ax.annotate(f"{v:.1f}% at a 30% loss", (30, v), xytext=(40, v + 3.5), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTE, lw=.7))
    ax.set_xlabel("Assumed mean loss of a missing low-surprise report (%)"); ax.set_ylabel("Share of Q1 reports that\nmust be missing (%)"); ax.set_ylim(0, None)
    save(fig, "survivorship_bound")


def fig_backtest():
    d = pd.read_parquet(D / "backtest_daily.parquet"); d["date"] = pd.to_datetime(d.date); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5))
    for ax, b, t in ((axs[0], "dev", "A. Development 2020-2022"), (axs[1], "locked", "B. Locked test 2023-2026")):
        x = d[(d.block == b) & (d.lag == 0) & (d.cost_bps == 10)].sort_values("date")
        for c, col, lab in (("long_ret", BLUE, "Long book"), ("spy_matched", MUTE, "Matched S&P 500"), ("ls_ret", ORANGE, "Long-short (hypothetical)")):
            ax.plot(x.date, (1 + x[c]).cumprod() * 100 - 100, color=col, lw=1.4, label=lab, ls="--" if c == "ls_ret" else "-")
        ax.axhline(0, color=INK, lw=.7); ax.set_title(t); ax.tick_params(axis="x", labelsize=7.8); ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axs[0].set_ylabel("Cumulative return (%)"); h_, l_ = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, .07, 1, 1)); fig.legend(h_, l_, ncol=3, loc="lower center", fontsize=8.2)
    fig.savefig(F / "backtest.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_notext():
    d = pd.read_csv(T / "no_text_lifts.csv"); d = d[d.metric == "macro_f1"]; fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.2), sharey=True, sharex=True)
    rows = (("All text, LightGBM on both sides", "Classical text + embeddings (L7 - numbers only, both LightGBM)", ORANGE, "s"), ("Classical text, elastic net", "All classical text (L6 - numbers only, elastic net)", BLUE, "o"))
    for ax, per, t in zip(axs, PERS, ("Walk-forward", "Development", "Locked test")):
        for i, (lab, c, col, mk) in enumerate(rows):
            x = d[(d.comparison == c) & (d.period == per)].iloc[0]; ax.errorbar(x.lift, i, xerr=[[x.lift - x.lo], [x.hi - x.lift]], fmt=mk, color=col, ms=5, elinewidth=1.1)
        ax.axvline(0, color=INK, lw=.9); ax.axvline(.01, color=ORANGE, ls="--", lw=.8); ax.set_title(t); ax.grid(axis="y", visible=False); ax.set_ylim(1.6, -.6)
    axs[0].set_yticks([0, 1], [r[0].replace(", ", ",\n") for r in rows]); fig.supxlabel("Change in macro-F1 relative to numbers only, with 95% interval", fontsize=9, y=.02)
    save(fig, "no_text")


def fig_car():  # noqa: F811
    c = pd.read_csv(T / "car_paths_alpha.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6), sharex=True, sharey=True)
    for ax, k, t in ((axs[0], "caar_alpha", "A. Pre-event alpha subtracted"), (axs[1], "caar_noalpha", "B. No alpha subtracted")):
        for g_, col, lab in (("8K", BLUE, "8-K"), ("periodic_10Q", ORANGE, "10-Q"), ("periodic_10K", AQUA, "10-K")):
            x = c[c.event_type == g_]; ax.plot(x.rel_day, x[k] * 100, color=col, lw=1.5, label=lab)
        ax.axvline(0, color=INK, lw=.8); ax.axhline(0, color=MUTE, lw=.6); ax.set_title(t); ax.set_xlabel("Trading days relative to the event")
    axs[0].set_ylabel("Cumulative abnormal return\nfrom day -1 (%)"); axs[1].legend(loc="upper left")
    save(fig, "car_paths")


def fig_sue():  # noqa: F811
    q0 = pd.read_csv(T / "drift_quintiles.csv"); q = pd.read_csv(T / "drift_quintiles_alpha.csv"); x = q.q.to_numpy(); w = .37
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.55))
    axs[0].bar(x, q0.ann_c4 * 100, color=BLUE, width=.65); axs[0].set_title("A. Reaction, days [-1,+1]"); axs[0].set_ylabel("Carhart abnormal return (%)")
    axs[1].bar(x - w / 2, q.c4_alpha * 100, w, color=ORANGE, label="Pre-event alpha subtracted"); axs[1].bar(x + w / 2, q.c4_noalpha * 100, w, color=BLUE, label="No alpha subtracted"); axs[1].set_title("B. Drift, days [+2,+60]"); axs[1].legend(loc="lower left", fontsize=7.8)
    for a in axs:
        a.axhline(0, color=INK, lw=.8); a.set_xticks(x, ["Q1\nworst", "Q2", "Q3", "Q4", "Q5\nbest"]); a.grid(axis="x", visible=False); a.set_xlabel("Earnings-surprise quintile")
    save(fig, "sue_quintiles")


def fig_alpha():
    q = pd.read_csv(T / "drift_quintiles_alpha.csv"); x = q.q.to_numpy(); fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.35))
    axs[0].bar(x, q.mom * 100, color=AQUA, width=.65); axs[0].set_title("A. Mean past return, months -12 to -2"); axs[0].set_ylabel("%")
    axs[1].bar(x, q.alpha_term_c4 * 100, color=ORANGE, width=.65); axs[1].set_title("B. Amount subtracted: 59 x pre-event alpha"); axs[1].set_ylabel("%")
    axs[1].set_ylim(-.95, 2.95)
    for i, v in zip(x, q.alpha_term_c4 * 100): axs[1].text(i, v + (.09 if v >= 0 else -.09), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for a in axs:
        a.axhline(0, color=INK, lw=.8); a.set_xticks(x, ["Q1\nworst", "Q2", "Q3", "Q4", "Q5\nbest"]); a.grid(axis="x", visible=False); a.set_xlabel("Earnings-surprise quintile")
    save(fig, "alpha_term")


def fig_drift():  # noqa: F811
    m = pd.read_csv(T / "drift_by_measure.csv"); c = pd.read_csv(T / "calendar_time.csv"); r = pd.read_csv(T / "drift_noalpha_robustness.csv"); r = r[(r.measure == "c4_noalpha") & (r["sample"] != "All reports")]
    rows = [(a, e * 100, l * 100, h * 100, ORANGE if "pre-event" in a else BLUE, "s" if "pre-event" in a else "o") for a, e, l, h in zip(m.measure, m.estimate, m.ci_low, m.ci_high)]
    for _, x in c.iterrows():
        if x["sample"] == "All reports": rows.append(("Calendar-time Carhart alpha, 59 days", x.alpha_59d * 100, x.alpha_lo * 100, x.alpha_hi * 100, AQUA, "D"))
    rows += [(f"Carhart, no alpha: {a}", e * 100, l * 100, h * 100, BLUE, "o") for a, e, l, h in zip(r["sample"], r.estimate, r.ci_low, r.ci_high)]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for i, (a, e, l, h, col, mk) in enumerate(rows): ax.errorbar(e, i, xerr=[[e - l], [h - e]], fmt=mk, color=col, ms=4.3, elinewidth=1)
    ax.axvline(0, color=INK, lw=.9); ax.set_yticks(range(len(rows)), [x[0] for x in rows], fontsize=8.1); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.axhline(6.5, color=GRID); ax.axhline(7.5, color=GRID)
    ax.set_xlabel("Q5 minus Q1 over days [+2,+60], percent, with 95% interval")
    save(fig, "drift_robustness")


def fig_bound():  # noqa: F811
    q = pd.read_csv(T / "drift_quintiles_alpha.csv").set_index("q"); q1, q5 = q.loc[1, "c4_noalpha"], q.loc[5, "c4_noalpha"]; d = np.linspace(0, .10, 60)
    fig, ax = plt.subplots(figsize=(4.9, 2.6))
    for L, col, ls in ((-.2, AQUA, "-"), (-.3, BLUE, "-"), (-.5, ORANGE, "-")):
        ax.plot(d * 100, (q5 - ((1 - d) * q1 + d * L)) * 100, color=col, lw=1.7, ls=ls); ax.text(10.1, (q5 - (.9 * q1 + .1 * L)) * 100, f"{L:.0%}", va="center", fontsize=8.5)
    ax.text(10.1, (q5 - (.9 * q1 + .1 * -.5)) * 100 + .75, "Mean loss", fontsize=8, va="bottom"); ax.set_xlim(0, 11.6); ax.set_xlabel("Share of low-surprise reports missing from the sample (%)"); ax.set_ylabel("Implied Q5 minus Q1 (%)")
    save(fig, "survivorship_bound")


for fn in (fig_timeline, fig_clock, fig_sample, fig_target, fig_ladder, fig_perf, fig_auc, fig_year, fig_deciles, fig_reliability, fig_confusion, fig_lifts, fig_subsets, fig_tinfo, fig_notext, fig_car, fig_sue, fig_alpha, fig_drift, fig_bound, fig_backtest):
    fn()
print("figures:", len(list(F.glob("*.svg"))))
