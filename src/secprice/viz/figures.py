"""Figures F01-F40 (SPEC.md §20.1), one function per ID. Each figure is saved with its 'how to read' caption;
a figure whose inputs do not exist is recorded with the reason instead of being drawn empty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CAPTIONS = {
    "F01": "Coverage heatmap: share of observed filings that parsed into at least one section, by year x form. Pale cells = parser or download gaps.",
    "F02": "Audit reason codes by year (periodic-filing expectations only; 8-Ks have no expectation).",
    "F03": "8-K category availability by year under taxonomy v1 (legacy items before 2004-08-23).",
    "F04": "Universe: unique firms (constant) vs firms with prices and firms in the S&P 500 each year. Survivorship-aware, not survivorship-free.",
    "F05": "Timestamp quality (A exact / B date-only / C uncertain) and session buckets by year.",
    "F06": "Pre-publication window (last close -> acceptance) and the share of events whose clock changes under public-lag sensitivity.",
    "F07": "Event-clock diagram: acceptance, assumed public time, decision time; T_info base/exit; T_trade entry/exit.",
    "F08": "Earnings event types over time (verified releases, periodic-only, ambiguous).",
    "F09": "Same-firm overlap counts by horizon and the clean-event share.",
    "F10": "Target distributions (abnormal returns) and 3-class balance by target definition.",
    "F11": "Average cumulative abnormal return path (market model) around day 0 by event type, all events vs clean events.",
    "F12": "CAR[0,+1] by session bucket (INTRA day 0 is a mixed pre/post interval).",
    "F13": "Pre-trend diagnostics: CAR[-20,-2] and CAR[-5,-1] by event type (interpreted, not forced to zero).",
    "F14": "Abnormal log volume and volatility response on days [0,+1] by event type.",
    "F15": "Spearman correlations between text features and event-window CARs (association only).",
    "F16": "SUE-conditioned post-earnings drift: CAR[+2,+60] by SUE_release quintile (prior-quarter breakpoints).",
    "F17": "Split timeline: walk-forward validation blocks, purge/embargo gap, development holdout, locked test; per-fold strict/retrospective labels.",
    "F18": "Ablation ladder: walk-forward macro-F1 by rung with month-block bootstrap CIs (common and maximum-available cohorts).",
    "F19": "H4: point-in-time ChronoBERT vintages vs static encoders on identical rows and heads.",
    "F20": "Loss curves: train vs inner-validation loss per boosting round (GBDT) and per step/epoch (fine-tuning).",
    "F21": "Validation curves: walk-forward objective vs each tuned hyperparameter (Optuna trials).",
    "F22": "Learning curves: validation macro-F1 vs training-set fraction (most recent rows kept).",
    "F23": "Before vs after tuning: default vs tuned walk-forward macro-F1 per rung and head.",
    "F24": "Confusion matrices (row-normalised) for the selected models.",
    "F25": "One-vs-rest ROC and precision-recall curves for the selected primary model.",
    "F26": "Calibration: reliability curves and expected calibration error.",
    "F27": "Coverage-accuracy curve: accuracy and macro-F1 when only the most confident predictions are kept.",
    "F28": "Rolling monthly rank IC of the abnormal-return regression head.",
    "F29": "T_info vs T_trade: the same rung scored on the information-response and the tradable target.",
    "F30": "Backtest: cumulative long-only (vs interval-matched SPY) and hypothetical long-short returns, by cost level and with a one-session lag.",
    "F31": "Primary-model macro-F1 by form, category, bucket, timestamp quality, size, sector and year.",
    "F32": "Retrieval metrics by setup on gold-val qrels (BM25, hybrid, hybrid + rerank).",
    "F33": "Extraction field F1 and faithfulness on gold-test.",
    "F34": "Numeric fidelity: exact and tolerance matches against source/XBRL.",
    "F35": "Faithfulness judge vs human ratings (kappa).",
    "F36": "Student vs teacher vs human agreement.",
    "F37": "Permutation null: distribution of macro-F1 under within-block class-stratified label permutation vs the observed value.",
    "F38": "Random-split demo (pedagogical only): random split vs walk-forward macro-F1.",
    "F39": "Adversarial validation ROC per fold (an AUC near 0.5 does not prove a split is unbiased).",
    "F40": "Selection diagnostics: Model Confidence Set membership and PBO (diagnostics only).",
    "FX1": "Supplementary - optimisation diagnostics: Optuna best-so-far objective per trial and the elastic-net convexity certificate (equal objective from several random starts = global optimum of that convex problem; tree and transformer optima are validation-selected, not global).",
}


class Figs:
    def __init__(self, out_dir: Path, dpi: int = 130):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.index: list[dict] = []
        plt.rcParams.update({"figure.figsize": (9, 4.8), "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})

    def save(self, fig, fid: str, title: str):
        p = self.out / f"{fid}.png"
        fig.suptitle(f"{fid} - {title}", fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(p, dpi=self.dpi)
        self.plt.close(fig)
        self.index.append({"id": fid, "title": title, "path": str(p), "caption": CAPTIONS.get(fid, ""), "status": "ok"})

    def skip(self, fid: str, title: str, reason: str):
        self.index.append({"id": fid, "title": title, "path": None, "caption": CAPTIONS.get(fid, ""),
                           "status": f"not produced: {reason}"})

    def safe(self, fid, title, fn, *a, **k):
        try:
            fn(fid, title, *a, **k)
        except Exception as e:
            self.plt.close("all")
            self.skip(fid, title, f"error {type(e).__name__}: {str(e)[:150]}")

    # ------------------------------------------------------------------ generic helpers
    def heat(self, fid, title, mat: pd.DataFrame, fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None):
        fig, ax = self.plt.subplots(figsize=(min(16, 2 + 0.45 * mat.shape[1]), min(10, 1.5 + 0.35 * mat.shape[0])))
        im = ax.imshow(mat.values.astype(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(mat.shape[1]), [str(c) for c in mat.columns], rotation=90)
        ax.set_yticks(range(mat.shape[0]), [str(i) for i in mat.index])
        if mat.size <= 400:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6, color="w")
        fig.colorbar(im, ax=ax)
        ax.grid(False)
        self.save(fig, fid, title)

    def stacked(self, fid, title, df: pd.DataFrame, xlabel="year", ylabel="count", share=False):
        d = df.div(df.sum(axis=1), axis=0) if share else df
        fig, ax = self.plt.subplots()
        bottom = np.zeros(len(d))
        for c in d.columns:
            ax.bar(d.index.astype(str), d[c].values, bottom=bottom, label=str(c))
            bottom += d[c].fillna(0).values
        ax.set_xlabel(xlabel)
        ax.set_ylabel("share" if share else ylabel)
        ax.tick_params(axis="x", rotation=90)
        ax.legend(fontsize=7, ncol=3)
        self.save(fig, fid, title)

    def bars_ci(self, fid, title, df: pd.DataFrame, x="label", y="value", lo="ci_low", hi="ci_high", hue=None, ref=None):
        fig, ax = self.plt.subplots(figsize=(10, 4.8))
        groups = [None] if hue is None else list(df[hue].unique())
        width = 0.8 / len(groups)
        labels = list(dict.fromkeys(df[x]))
        for k, gname in enumerate(groups):
            g = df if gname is None else df[df[hue] == gname]
            g = g.set_index(x).reindex(labels)
            xs = np.arange(len(labels)) + k * width - 0.4 + width / 2
            err = np.vstack([(g[y] - g[lo]).clip(lower=0).fillna(0), (g[hi] - g[y]).clip(lower=0).fillna(0)]) if lo in g else None
            ax.bar(xs, g[y].values, width, yerr=err, capsize=3, label=str(gname) if gname is not None else None)
        if ref is not None:
            ax.axhline(ref, color="k", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        if hue is not None:
            ax.legend(fontsize=7)
        self.save(fig, fid, title)

    def lines(self, fid, title, series: dict, xlabel="", ylabel="", logy=False):
        fig, ax = self.plt.subplots()
        for name, (x, y) in series.items():
            ax.plot(x, y, label=name, lw=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        self.save(fig, fid, title)

    # ------------------------------------------------------------------ specific figures
    def f07_clock(self, fid, title):
        plt = self.plt
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3.2)
        ax.axis("off")
        for x0, x1, lab in ((0.5, 3.0, "session D-1"), (4.5, 7.0, "session D (next)")):
            ax.add_patch(plt.Rectangle((x0, 2.2), x1 - x0, 0.4, color="#ddd"))
            ax.text((x0 + x1) / 2, 2.72, lab, ha="center", fontsize=8)
        marks = [(3.0, "T_info base\n(last close <= acceptance)"), (3.4, "acceptance"), (3.55, "public\n(+lag, assumed)"),
                 (3.85, "decision\n(+latency)"), (4.5, "T_trade entry\n(first open >= decision)"),
                 (7.0, "T_info exit h=1 /\nT_trade exit h=1")]
        for i, (x, lab) in enumerate(marks):
            ax.axvline(x, ymin=0.25, ymax=0.85, color=["k", "C3", "C1", "C2", "C0", "C4"][i], lw=1.2)
            ax.text(x, 0.25 + 0.42 * (i % 2), lab, ha="center", fontsize=7)
        ax.annotate("", xy=(7.0, 1.75), xytext=(3.0, 1.75), arrowprops=dict(arrowstyle="<->", color="k"))
        ax.text(5.0, 1.82, "T_info: close -> close (INTRA excluded)", ha="center", fontsize=7)
        ax.annotate("", xy=(7.0, 1.45), xytext=(4.5, 1.45), arrowprops=dict(arrowstyle="<->", color="C0"))
        ax.text(5.75, 1.52, "T_trade: open -> close, SPY on the same interval", ha="center", fontsize=7, color="C0")
        self.save(fig, fid, title)

    def f17_timeline(self, fid, title, blocks: pd.DataFrame, prov: pd.DataFrame | None, start, embargo_days):
        fig, ax = self.plt.subplots(figsize=(11, 4.5))
        y = 0
        for _, b in blocks.iterrows():
            colr = {"walk_forward_val": "C0", "development_holdout": "C1", "locked_test": "C3"}[b.kind]
            ax.barh(y, (b.end_exclusive - b.start).days, left=b.start, color=colr, alpha=0.8)
            if b.kind == "walk_forward_val":
                ax.barh(y, (b.start - pd.Timestamp(start)).days, left=pd.Timestamp(start), color="C2", alpha=0.35)
                ax.barh(y, embargo_days, left=b.start - pd.Timedelta(days=embargo_days), color="k", alpha=0.5)
            lab = b.block
            if prov is not None and len(prov):
                p = prov[prov.block == b.block]
                st = ", ".join(f"{m}:{'S' if l == 'strict' else 'R'}" for m, l in zip(p.model, p.leaderboard)
                               if m in ("sec_bert", "finbert_tone", "chrono_bert", "bge_base", "qwen"))
                lab += f"  [{st}]"
            ax.text(b.start, y + 0.38, lab, fontsize=6)
            y += 1
        ax.set_yticks([])
        ax.set_xlabel("decision date")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="C2", alpha=0.35, label="train (expanding)"), Patch(color="k", alpha=0.5, label="purge/embargo"),
                           Patch(color="C0", label="validation block"), Patch(color="C1", label="development holdout"),
                           Patch(color="C3", label="locked test")], fontsize=7, loc="lower right")
        self.save(fig, fid, title)

    def f20_loss(self, fid, title, curves: list[dict], ft_curves: list[dict]):
        n = (1 if curves else 0) + (1 if ft_curves else 0)
        if n == 0:
            return self.skip(fid, title, "no recorded curves")
        fig, axes = self.plt.subplots(1, n, figsize=(6 * n, 4.2), squeeze=False)
        k = 0
        if curves:
            ax = axes[0, k]
            for c in curves[:6]:
                lab = f"{c['rung']}/{c['head']}{'/' + c['enc'] if c.get('enc') else ''} f{c['fold']}"
                ax.plot(c["train"], lw=1, label=f"{lab} train")
                if c.get("val"):
                    ax.plot(c["val"], lw=1, ls="--", label=f"{lab} val")
            ax.set_xlabel(curves[0].get("unit", "iteration"))
            ax.set_ylabel(curves[0].get("metric", "loss"))
            ax.legend(fontsize=6)
            k += 1
        if ft_curves:
            ax = axes[0, k]
            for c in ft_curves[:4]:
                cu = c["curve"]
                st = np.array(cu["step"])
                tl = pd.Series(cu["train_loss"]).rolling(20, min_periods=1).mean()
                ax.plot(st, tl, lw=1, label=f"{c['label']} train (smoothed)")
                ax.plot(cu["eval_step"], cu["inner_val_loss"], "o--", ms=3, lw=1, label=f"{c['label']} inner-val")
            ax.set_xlabel("step")
            ax.set_ylabel("cross-entropy")
            ax.legend(fontsize=6)
        self.save(fig, fid, title)

    def f24_cm(self, fid, title, cms: dict):
        if not cms:
            return self.skip(fid, title, "no predictions")
        k = len(cms)
        fig, axes = self.plt.subplots(1, k, figsize=(3.4 * k, 3.4), squeeze=False)
        for ax, (name, cm) in zip(axes[0], cms.items()):
            cm = np.asarray(cm, float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(3), ["down", "side", "up"])
            ax.set_yticks(range(3), ["down", "side", "up"])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(name, fontsize=8)
            ax.grid(False)
        self.save(fig, fid, title)

    def f25_roc(self, fid, title, y, proba):
        from sklearn.metrics import precision_recall_curve, roc_curve
        fig, (a1, a2) = self.plt.subplots(1, 2, figsize=(10, 4.2))
        for c, nm in enumerate(["down", "side", "up"]):
            pos = (y == c)
            if pos.any() and (~pos).any():
                fpr, tpr, _ = roc_curve(pos, proba[:, c])
                a1.plot(fpr, tpr, label=nm)
                pr, rc, _ = precision_recall_curve(pos, proba[:, c])
                a2.plot(rc, pr, label=f"{nm} (base {pos.mean():.2f})")
        a1.plot([0, 1], [0, 1], "k--", lw=0.7)
        a1.set_xlabel("FPR")
        a1.set_ylabel("TPR")
        a2.set_xlabel("recall")
        a2.set_ylabel("precision")
        a1.legend(fontsize=7)
        a2.legend(fontsize=7)
        self.save(fig, fid, title)

    def f26_calib(self, fid, title, y, proba, ece):
        from ..evaluation.metrics import reliability
        fig, ax = self.plt.subplots(figsize=(5.5, 4.5))
        for c, nm in enumerate(["down", "side", "up"]):
            r = reliability(proba, y, c)
            if len(r):
                ax.plot(r.mean_pred, r.frac_pos, "o-", ms=3, label=nm)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"top-label ECE = {ece:.3f}" if ece is not None else "", fontsize=8)
        ax.legend(fontsize=7)
        self.save(fig, fid, title)

    def f27_coverage(self, fid, title, y, proba, yhat):
        from ..evaluation.metrics import confusion, f1_from_cm
        conf = proba.max(1)
        order = np.argsort(-conf)
        covs = np.linspace(0.05, 1, 20)
        acc, f1 = [], []
        for c in covs:
            m = order[: max(1, int(c * len(y)))]
            f = f1_from_cm(confusion(y[m], yhat[m]))
            acc.append(f["accuracy"])
            f1.append(f["macro_f1"])
        self.lines(fid, title, {"accuracy": (covs, acc), "macro-F1": (covs, f1)}, "coverage", "metric")

    def write_index(self):
        (self.out / "figure_index.json").write_text(json.dumps(self.index, indent=2), encoding="utf-8")
        return self.index
