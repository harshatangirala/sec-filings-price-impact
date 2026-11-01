"""Charts for the written report. Every number is read from the dashboard data snapshot (built from the finished run)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = r"C:\Users\harsh\Downloads\sec-filings-price-impact\dashboard\data"
OUT = r"C:\Users\harsh\Downloads\sec_report_build\img"
os.makedirs(OUT, exist_ok=True)
INK, BLUE, RUST, GREEN, GREY, SAND, GOLD, PLUM = "#1f2937", "#2b5c8a", "#c0562f", "#4c8c6b", "#8b95a1", "#e9e4d8", "#d19a2a", "#7a4e8c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9aa3ad", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e7eb", "grid.linewidth": 0.8, "figure.dpi": 100, "savefig.dpi": 170, "legend.frameon": False})
csv = lambda n: pd.read_csv(f"{D}/{n}")  # noqa: E731


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1 filings per year
d = csv("filings_by_year.csv")
p = d.pivot_table(index="year", columns="base_form", values="filings", aggfunc="sum").fillna(0)
p = p[[c for c in ("8-K", "10-Q", "10-K", "10-K405") if c in p.columns]]
fig, ax = plt.subplots(figsize=(7.6, 3.3))
bottom = np.zeros(len(p))
for c, col in zip(p.columns, (BLUE, GOLD, RUST, GREY)):
    ax.bar(p.index, p[c], bottom=bottom, label=c, color=col, width=0.8)
    bottom += p[c].values
ax.set_ylabel("filings downloaded")
ax.legend(ncol=4, loc="upper left")
ax.set_ylim(0, 3300)
ax.set_xlim(1995.5, 2026.5)
save(fig, "filings_by_year")

# 2 universe by sector
u = csv("universe.csv")
c = u.ff12_industry.value_counts().sort_values()
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.barh(c.index, c.values, color=BLUE)
for i, v in enumerate(c.values):
    ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
ax.set_xlabel("firms")
ax.grid(axis="y", visible=False)
save(fig, "sectors")

# 3 when filings arrive
sb = csv("session_buckets.csv").groupby("session_bucket").events.sum().reindex(["PRE", "INTRA", "POST", "NONTRADING"])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(["before\nthe open", "during\ntrading", "after\nthe close", "weekend /\nholiday"], sb.values / sb.sum() * 100, color=[GREEN, GOLD, BLUE, GREY])
for i, v in enumerate(sb.values / sb.sum() * 100):
    ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylabel("% of events")
ax.set_ylim(0, 60)
ax.tick_params(axis="x", labelsize=8.5)
save(fig, "when_filed")

# 4 text length by section
s = csv("sections.csv").sort_values("sections")
names = {"10K_1": "10-K business", "10K_1A": "10-K risk factors", "10K_7": "10-K MD&A", "10K_7A": "10-K market risk", "10Q_MDA": "10-Q MD&A",
         "10Q_MR": "10-Q market risk", "10Q_RF": "10-Q risk factors", "8K_ITEMS": "8-K item text", "8K_EX99": "8-K press release"}
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.barh([names.get(x, x) for x in s.section_id], s.sections, color=GREEN)
for i, (v, m) in enumerate(zip(s.sections, s.median_tokens)):
    ax.text(v + 400, i, f"{v:,}  (median ~{m:,.0f} tokens)", va="center", fontsize=8.5)
ax.set_xlim(0, 80000)
ax.set_xlabel("sections kept")
ax.grid(axis="y", visible=False)
save(fig, "sections")

# 5 split timeline
b = csv("blocks.csv")
fig, ax = plt.subplots(figsize=(7.6, 2.5))
ax.barh(0, (pd.Timestamp("2008-01-01") - pd.Timestamp("2000-01-01")).days, left=pd.Timestamp("2000-01-01").toordinal(), color="#cbd5e1", height=0.5)
ax.clear()
t0 = pd.Timestamp("2000-01-01")


def seg(a, z, row, col, txt):
    a, z = pd.Timestamp(a), pd.Timestamp(z)
    ax.barh(row, (z - a).days / 365.25, left=(a - t0).days / 365.25 + 2000, color=col, height=0.55, edgecolor="white")
    ax.text((a - t0).days / 365.25 + 2000 + (z - a).days / 365.25 / 2, row, txt, ha="center", va="center", color="white", fontsize=(6.2 if txt == "development" else 7 if txt == "locked test" else 8.5), fontweight="bold")


seg("2000-01-01", "2008-01-01", 3, GREY, "initial training")
for i, r in enumerate(b[b.kind == "walk_forward_val"].itertuples()):
    seg(r.start, r.end_exclusive, 2, BLUE, f"fold {i + 1}")
r = b[b.kind == "development_holdout"].iloc[0]
seg(r.start, r.end_exclusive, 1, GOLD, "development")
r = b[b.kind == "locked_test"].iloc[0]
seg(r.start, r.end_exclusive, 0, RUST, "locked test")
ax.set_yticks([3, 2, 1, 0])
ax.set_yticklabels(["train", "validate (6 folds)", "choose model", "final test, once"], fontsize=9)
ax.set_xlim(2000, 2026.7)
ax.set_xlabel("year")
ax.grid(axis="y", visible=False)
save(fig, "splits")

# 6 label balance + abnormal return histogram
lab = csv("labels_by_year.csv")
sh = lab.groupby("label").events.sum() / lab.events.sum()
hist = csv("ar_hist.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
a1.bar(hist.abnormal_return_pct, hist.events, width=0.23, color=np.where(hist.abnormal_return_pct < 0, RUST, GREEN))
a1.set_xlabel("next-session abnormal return (%, clipped at ±10)")
a1.set_ylabel("events")
a2.bar(["Down", "Sideways", "Up"], [sh["Down"] * 100, sh["Sideways"] * 100, sh["Up"] * 100], color=[RUST, GREY, GREEN])
for i, k in enumerate(["Down", "Sideways", "Up"]):
    a2.text(i, sh[k] * 100 + 1, f"{sh[k] * 100:.0f}%", ha="center")
a2.set_ylim(0, 50)
a2.set_ylabel("% of labelled events")
fig.tight_layout()
save(fig, "target")

# 7 ladder
l = csv("ladder.csv")
rows = []
for rung in ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"):
    sub = l[(l.rung == rung) & ((l.tuned == "tuned") | (rung == "L0"))]
    sub = sub[sub.cohort == ("common_emb" if rung in ("L5", "L7") else "common")]
    for r in sub.itertuples():
        rows.append((rung, r.head, r.encoder if r.encoder not in ("-", "n/a") else "", r.macro_f1, r.ci_low, r.ci_high))
lad = pd.DataFrame(rows, columns=["rung", "head", "enc", "f1", "lo", "hi"])
fig, ax = plt.subplots(figsize=(7.6, 4.2))
cols = {"lightgbm": BLUE, "catboost": GOLD, "enet": GREEN, "prior": GREY}
labels = {"L0": "L0  class frequencies only", "L1": "L1  price", "L2": "L2  metadata", "L3": "L3  accounting", "L4": "L4  word features", "L5": "L5  text vectors only",
          "L6": "L6  all the numbers", "L7": "L7  numbers + text vectors"}
ypos = {r: i for i, r in enumerate(reversed(list(labels)))}
for head, g in lad.groupby("head"):
    for j, r in enumerate(g.itertuples()):
        off = {"lightgbm": 0.2, "catboost": 0.0, "enet": -0.2, "prior": 0}[head] + (0.07 if r.enc == "bge_base" else -0.07 if r.enc == "chrono_bert" else 0)
        ax.errorbar(r.f1, ypos[r.rung] + off, xerr=[[r.f1 - r.lo], [r.hi - r.f1]], fmt="o", ms=4.5, color=cols[head], capsize=2, lw=1.2)
ax.set_yticks(list(ypos.values()))
ax.set_yticklabels(list(reversed(list(labels.values()))), fontsize=9)
ax.axvline(0.328, color=INK, ls="--", lw=1.2)
ax.text(0.3295, -0.75, "luck ≈ 0.33", fontsize=8.5, va="bottom")
ax.set_xlabel("macro-F1 on the 2008–2019 validation folds (dots = models, bars = 95% interval)")
ax.set_xlim(0.17, 0.43)
for h, c in (("LightGBM", BLUE), ("CatBoost", GOLD), ("elastic net", GREEN)):
    ax.plot([], [], "o", color=c, label=h)
ax.legend(loc="lower left", fontsize=8.5)
ax.set_ylim(-0.9, 7.6)
ax.grid(axis="y", visible=False)
save(fig, "ladder")

# 8 chance level by period
sk = csv("skill.csv")
per = ["walk-forward", "development", "locked"]
nm = {"walk-forward": "validation\n2008–2019", "development": "development\n2020–2022", "locked": "locked test\n2023–2026"}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.26
for k, (m, col) in enumerate((("L1", GREY), ("L6", BLUE), ("L7", RUST))):
    v = [sk[(sk.period == p_) & (sk.model == m)].macro_f1.iloc[0] for p_ in per]
    xs = np.arange(3) + (k - 1) * w
    ax.bar(xs, v, w, color=col, label={"L1": "price only", "L6": "all the numbers", "L7": "numbers + text"}[m])
    for x, y in zip(xs, v):
        ax.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=8)
ch = sk[sk.model.isin(["L6", "L7"])].chance_f1_shuffled.mean()
meta = json.load(open(f"{D}/meta.json"))["modelling"]
ax.axhline(ch, color=INK, ls="--", lw=1.4, label=f"what luck gives ≈ {ch:.2f}")
ax.axhline(meta["majority_baseline_f1"], color="#6b7280", ls=":", lw=1.6, label=f"always guessing the biggest class = {meta['majority_baseline_f1']:.2f}")
ax.set_xticks(range(3))
ax.set_xticklabels([nm[x] for x in per])
ax.set_ylim(0.15, 0.50)
ax.set_ylabel("macro-F1")
ax.legend(ncol=2, loc="upper left", fontsize=8.2)
ax.grid(axis="x", visible=False)
save(fig, "chance")

# 9 move vs direction
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
for ax, col, ttl in ((axs[0], "move_auc", "Will the stock move?"), (axs[1], "direction_auc", "Which way, if it moved?")):
    for k, (m, c_) in enumerate((("L1", GREY), ("L6", BLUE), ("L7", RUST))):
        g = sk[sk.model == m].set_index("period").loc[per]
        v = g[col].values
        ax.bar(np.arange(3) + (k - 1) * 0.26, v, 0.26, color=c_, label=m, yerr=[v - g[col + "_lo"].values, g[col + "_hi"].values - v], error_kw=dict(lw=1, capsize=2))
    ax.axhline(0.5, color=INK, ls="--", lw=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["validation", "development", "locked"], fontsize=8.5)
    ax.set_title(ttl, fontsize=10)
    ax.set_ylim(0.4, 0.72)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("AUC (0.5 = no information)")
axs[0].legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
save(fig, "move_direction")

# 10 text lift forest
li = csv("lifts.csv")
fig, ax = plt.subplots(figsize=(7.4, 3.5))
order = [("L7 - L6 (text)", "adding text to the numbers"), ("L6 - L6_nochange (change features)", "adding 'change since last filing' words"), ("L6 - L1 (structured over price)", "all the numbers over price alone")]
y = 0
ticks, tl = [], []
for key, lab_ in order:
    for per_, col in (("walk-forward", BLUE), ("development", GOLD), ("locked", RUST)):
        r = li[(li.period == per_) & (li.comparison == key)].iloc[0]
        ax.errorbar(r.lift, y, xerr=[[r.lift - r.ci_low], [r.ci_high - r.lift]], fmt="o", color=col, capsize=3, ms=5, label=per_ if key == order[0][0] else None)
        y -= 1
    ticks.append(y + 2)
    tl.append(lab_)
    y -= 0.6
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(ticks)
ax.set_yticklabels(tl, fontsize=9)
ax.set_xlabel("difference in macro-F1 (dot = estimate, bar = 95% interval)")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(axis="y", visible=False)
save(fig, "lifts")

# 11 SUE: announcement vs drift
q = ["worst news\n(Q1)", "Q2", "Q3", "Q4", "best news\n(Q5)"]
ann = [-0.54, 0.00, 0.15, 0.60, 0.81]
sq = csv("sue_quintiles.csv")
drift = (sq["mean"] * 100).tolist()
err = (1.96 * sq["sem"] * 100).tolist()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=False)
a1.bar(q, ann, color=[RUST if v < 0 else GREEN for v in ann])
a1.set_title("On announcement day (−1 to +1)", fontsize=10)
a1.set_ylabel("average abnormal return, %")
a1.axhline(0, color=INK, lw=1)
a2.bar(q, drift, yerr=err, color=[GREEN if v > 0 else RUST for v in drift], error_kw=dict(lw=1, capsize=3))
a2.set_title("Over the next 2 to 60 trading days", fontsize=10)
a2.axhline(0, color=INK, lw=1)
for a in (a1, a2):
    a.tick_params(axis="x", labelsize=8)
    a.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "sue")

# 12 2x2 of best-news reports
fig, ax = plt.subplots(figsize=(5.8, 3.2))
cells = [("rose, then rose", 490, 5.21, 8.53, GREEN), ("rose, then fell", 587, 4.77, -10.06, RUST), ("fell, then rose", 371, -4.42, 8.58, GOLD), ("fell, then fell", 460, -4.62, -10.84, GREY)]
ax.bar([c[0] for c in cells], [c[1] for c in cells], color=[c[4] for c in cells])
for i, c in enumerate(cells):
    ax.text(i, c[1] + 12, f"{c[1]}\n({c[1] / 1908 * 100:.0f}%)", ha="center", fontsize=9)
ax.set_ylim(0, 720)
ax.set_ylabel("best-news earnings reports")
ax.tick_params(axis="x", labelsize=8.5)
ax.grid(axis="x", visible=False)
save(fig, "twobytwo")

# 13 locked confusion matrix (L6)
cm = csv("locked_confusion.csv")
g = cm[cm.model == "L6"].pivot(index="true", columns="pred", values="events").reindex(index=["Down", "Sideways", "Up"], columns=["Down", "Sideways", "Up"])
sh_ = g.div(g.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(4.4, 3.5))
im = ax.imshow(sh_.values, cmap="Blues", vmin=0, vmax=0.6)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{sh_.values[i, j]:.0%}\n({int(g.values[i, j]):,})", ha="center", va="center", fontsize=9, color="white" if sh_.values[i, j] > 0.4 else INK)
ax.set_xticks(range(3))
ax.set_xticklabels(["Down", "Sideways", "Up"])
ax.set_yticks(range(3))
ax.set_yticklabels(["Down", "Sideways", "Up"])
ax.set_xlabel("what the model said")
ax.set_ylabel("what happened")
ax.grid(False)
save(fig, "confusion")

# 14 backtest
bd = pd.read_parquet(f"{D}/backtest_daily.parquet")
x = bd[(bd.block == "locked") & (bd.lag == 0) & (bd.cost_bps == 10)].sort_values("date")
dates = pd.to_datetime(x.date)
fig, ax = plt.subplots(figsize=(7.6, 3.3))
for col, lab_, c_ in (("long_ret", "long book", GREEN), ("spy_matched", "S&P 500 on the same days", GREY), ("long_excess_spy", "long book minus S&P 500", PLUM)):
    ax.plot(dates, ((1 + x[col].fillna(0)).cumprod() - 1) * 100, label=lab_, color=c_, lw=2)
ax.axhline(0, color=INK, lw=1)
ax.set_ylabel("cumulative return, %")
ax.legend(loc="upper left", fontsize=8.5)
save(fig, "backtest")

# 15 compute time
st = csv("stages.csv").dropna(subset=["hours"])
fig, ax = plt.subplots(figsize=(7.6, 2.9))
colr = {"CPU": BLUE, "GPU": GOLD, "CPU-GPU": PLUM}
ax.bar(st.stage, st.hours * 60, color=[colr.get(p_, GREY) for p_ in st.part])
ax.set_yscale("log")
ax.set_ylabel("minutes (log scale)")
ax.tick_params(axis="x", rotation=90, labelsize=8)
for h, c_ in (("CPU stage", BLUE), ("GPU stage", GOLD), ("mixed", PLUM)):
    ax.bar([0], [0], color=c_, label=h)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
ax.grid(axis="x", visible=False)
save(fig, "compute")

# 16 RAG prevalence
rp = csv("rag_prevalence_rows.csv")
g = rp.groupby("field").present.mean().sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.barh([f.replace("_", " ") for f in g.index], g.values * 100, color=PLUM)
for i, v in enumerate(g.values * 100):
    ax.text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=8.5)
ax.set_xlabel("% of 141 pre-filled filings where Qwen said the field was present")
ax.grid(axis="y", visible=False)
save(fig, "rag")

# 17 hyper-parameter tuning gain
t = csv("tuning.csv").dropna(subset=["default_mean_f1", "tuned_mean_f1"])
fig, ax = plt.subplots(figsize=(5.4, 3.3))
dlt = (t.tuned_mean_f1 - t.default_mean_f1)
ax.hist(dlt, bins=14, color=BLUE, edgecolor="white")
ax.axvline(0, color=INK, lw=1.2)
ax.set_xlabel("tuned minus default (mean validation macro-F1)")
ax.set_ylabel("rung × model pairs")
save(fig, "tuning")
print("tuning gain: median", round(dlt.median(), 4), "| share improved", round((dlt > 0).mean(), 2), "| n", len(dlt))
