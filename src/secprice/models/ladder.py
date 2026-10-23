"""Track B ablation ladder engine (SPEC.md §9.1-9.4).

Every comparison declares a cohort: common (events eligible for every compared rung) or maximum-available.
Fold-level preprocessing (TF-IDF + SVD, embedding PCA, imputers, scalers, calibration, clip bounds) is fitted on
the training rows of that fold only. Selection uses mean walk-forward macro-F1 ("validation-selected", never a
global optimum).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.bootstrap import macro_f1_ci, paired_macro_f1_diff
from ..evaluation.metrics import classification_metrics, decide, regression_metrics
from .heads import Head, enet_certificate, space

FAMILY_PREFIX = {
    "price": ("px_", "mkt_"), "metadata": ("meta_",), "accounting": ("acc_", "sue_"),
    "lexical": ("lm_", "main_lm_", "txt_doc"), "change": ("chg_", "rf_new", "rf_removed", "boilerplate_"),
    "novelty": ("novelty_",), "extraction": ("ext_",), "ft": ("ft_",),
}
RUNGS = {
    "L0": [], "L1": ["price"], "L2": ["metadata"], "L3": ["accounting"],
    "L4": ["lexical", "change", "novelty", "tfidf"], "L4_levels": ["lexical", "tfidf"], "L4_change": ["change", "novelty"],
    "L5": ["emb"], "L6": ["price", "metadata", "accounting", "lexical", "change", "novelty", "tfidf"],
    "L6_nochange": ["price", "metadata", "accounting", "lexical", "tfidf"],
    "L7": ["price", "metadata", "accounting", "lexical", "change", "novelty", "tfidf", "emb"],
    "L8": ["price", "metadata", "accounting", "lexical", "change", "novelty", "tfidf", "emb", "extraction", "ft"],
}
REQUIRES = {"price": "has_price", "accounting": "has_accounting", "lexical": "has_text", "change": "has_text",
            "novelty": "has_text", "tfidf": "has_text", "extraction": "has_extraction"}


def _hash_ids(ids) -> str:
    return hashlib.sha1(",".join(sorted(map(str, ids))).encode()).hexdigest()[:12]


class Ladder:
    def __init__(self, cfg: dict, events: pd.DataFrame, feats: pd.DataFrame, targets: pd.DataFrame,
                 splits: pd.DataFrame, texts: pd.Series | None = None, embs: dict | None = None,
                 out_dir: Path | None = None, log=None, provenance: pd.DataFrame | None = None):
        self.cfg, self.log = cfg, log
        self.ev = events.set_index("event_id")
        self.F = feats.set_index("event_id").reindex(self.ev.index)
        self.T = targets
        self.S = splits
        self.texts = texts
        self.embs = embs or {}
        self.out = Path(out_dir) if out_dir else None
        self.cache: dict = {}
        self.curves: list[dict] = []
        self.config_counter = 0
        self.prov = provenance
        self.inner_frac = float(cfg["tuning"]["inner_val_frac"])
        self.min_train = int(cfg["tuning"].get("min_train_rows", 50))
        self.min_test = int(cfg["tuning"].get("min_test_rows", 10))
        self.seed = int(cfg["seed"])
        self.clean_col = f"overlap_flag_h{int(cfg['contam_window_sessions'])}"
        self.avail = self._availability()

    # ------------------------------------------------------------------ data
    def _availability(self) -> pd.DataFrame:
        F = self.F
        a = pd.DataFrame(index=F.index)
        px = [c for c in F.columns if c.startswith("px_ret_20") or c.startswith("px_vol_20")]
        a["has_price"] = F[px].notna().all(axis=1) if px else False
        acc = [c for c in F.columns if c.startswith(FAMILY_PREFIX["accounting"])]
        a["has_accounting"] = F[acc].notna().any(axis=1) if acc else False
        a["has_text"] = F["txt_has_text"].fillna(False).astype(bool) if "txt_has_text" in F else False
        a["has_extraction"] = F[[c for c in F.columns if c.startswith("ext_")]].notna().any(axis=1) if any(
            c.startswith("ext_") for c in F.columns) else False
        for enc, E in self.embs.items():
            a[f"has_emb_{enc}"] = a.index.isin(E.index)
        return a

    def labelled(self, tdef: str, h: int, col: str = "label3") -> pd.DataFrame:
        t = self.T[(self.T.target_def == tdef) & (self.T.horizon == h)][["event_id", col, "weight_uniq"]]
        t = t[t[col].notna()].set_index("event_id")
        t = t[t.index.isin(self.ev.index)]
        out = pd.DataFrame({"y": t[col], "w": t["weight_uniq"].fillna(1.0)})
        out["date"] = pd.to_datetime(self.ev.loc[out.index, "decision_date"])
        return out.sort_values("date")

    def requirement_mask(self, rung: str, enc: str | None, ids) -> np.ndarray:
        m = np.ones(len(ids), bool)
        av = self.avail.loc[ids]
        for fam in RUNGS[rung]:
            if fam == "emb":
                m &= av[f"has_emb_{enc}"].to_numpy() if enc and f"has_emb_{enc}" in av else False
            elif REQUIRES.get(fam):
                m &= av[REQUIRES[fam]].to_numpy()
        return m

    def cohort(self, rungs: list[str], encs: list[str | None], ids) -> np.ndarray:
        m = np.ones(len(ids), bool)
        for r in rungs:
            for e in (encs if "emb" in RUNGS[r] else [None]):
                m &= self.requirement_mask(r, e, ids)
        return m

    def fold_ids(self, scheme: str, fold: int, ids: pd.Index) -> tuple[pd.Index, pd.Index]:
        s = self.S[(self.S.scheme == scheme) & (self.S.fold == fold)].set_index("event_id")["role"]
        s = s.reindex(ids)
        test_role = {"walk_forward": "val", "dev_holdout": "dev", "locked_test": "locked"}.get(scheme, "test")
        return ids[(s == "train").to_numpy()], ids[(s == test_role).to_numpy()]

    def folds(self, scheme: str) -> list[int]:
        return sorted(self.S[self.S.scheme == scheme].fold.unique().tolist())

    # ------------------------------------------------------------------ design matrices
    def _family_cols(self, fam: str) -> list[str]:
        pre = FAMILY_PREFIX.get(fam, ())
        return [c for c in self.F.columns if c.startswith(pre) and not c.startswith("_")]

    def _tfidf(self, tr_ids, key: str):
        ck = ("tfidf", key)
        if ck not in self.cache:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            docs = self.texts.reindex(tr_ids).fillna("").tolist()
            nd = len(docs)
            vec = TfidfVectorizer(max_features=20000, min_df=min(5, max(1, nd // 50)), sublinear_tf=True,
                                  dtype=np.float32, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", stop_words="english")
            M = vec.fit_transform(docs)
            k = int(min(64, max(2, M.shape[1] - 1), max(2, nd - 1)))
            svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(M)
            self.cache[ck] = (vec, svd)
        return self.cache[ck]

    def _pca(self, enc: str, tr_ids, key: str):
        ck = ("pca", enc, key)
        if ck not in self.cache:
            from sklearn.decomposition import PCA
            E = self.embs[enc]
            Xtr = E.reindex(tr_ids).dropna().to_numpy(np.float32)
            k = int(min(int(self.cfg["models"]["pca_dims"]), max(1, Xtr.shape[0] - 1), Xtr.shape[1]))
            self.cache[ck] = PCA(n_components=k, random_state=self.seed).fit(Xtr)
        return self.cache[ck]

    def _cached(self, key, fn):
        """Transformed blocks are reused across trials/heads; bounded so memory stays flat."""
        if key not in self.cache:
            if len([k for k in self.cache if k[0] in ("tfidf_x", "pca_x")]) > 400:
                for k in [k for k in self.cache if k[0] in ("tfidf_x", "pca_x")][:200]:
                    del self.cache[k]
            self.cache[key] = fn()
        return self.cache[key]

    def matrices(self, rung: str, tr_ids, eval_ids: list, enc: str | None = None):
        key = _hash_ids(tr_ids)
        blocks_tr, blocks_ev, names = [], [[] for _ in eval_ids], []
        for fam in RUNGS[rung]:
            if fam == "tfidf":
                if self.texts is None:
                    continue
                vec, svd = self._tfidf(tr_ids, key)
                f = lambda ids, k=key: self._cached(("tfidf_x", k, _hash_ids(ids)), lambda: svd.transform(  # noqa: E731
                    vec.transform(self.texts.reindex(ids).fillna("").tolist())).astype(np.float32))
                blocks_tr.append(f(tr_ids))
                for j, ids in enumerate(eval_ids):
                    blocks_ev[j].append(f(ids))
                names += [f"tfidf_svd{i}" for i in range(blocks_tr[-1].shape[1])]
            elif fam == "emb":
                pca = self._pca(enc, tr_ids, key)
                E = self.embs[enc]
                f = lambda ids, k=key: self._cached(("pca_x", enc, k, _hash_ids(ids)), lambda: pca.transform(  # noqa: E731
                    E.reindex(ids).fillna(0).to_numpy(np.float32)).astype(np.float32))
                blocks_tr.append(f(tr_ids))
                for j, ids in enumerate(eval_ids):
                    blocks_ev[j].append(f(ids))
                names += [f"emb_{enc}_pc{i}" for i in range(blocks_tr[-1].shape[1])]
            else:
                cols = self._family_cols(fam)
                if not cols:
                    continue
                blocks_tr.append(self.F.loc[tr_ids, cols].to_numpy(np.float32))
                for j, ids in enumerate(eval_ids):
                    blocks_ev[j].append(self.F.loc[ids, cols].to_numpy(np.float32))
                names += cols
        if not blocks_tr:
            return np.zeros((len(tr_ids), 0), np.float32), [np.zeros((len(i), 0), np.float32) for i in eval_ids], names
        return np.hstack(blocks_tr), [np.hstack(b) for b in blocks_ev], names

    # ------------------------------------------------------------------ one configuration over folds
    def run(self, rung: str, head: str, params: dict | None, lab: pd.DataFrame, cohort_ids: pd.Index, scheme: str,
            task: str = "cls", enc: str | None = None, keep: bool = True, folds: list[int] | None = None,
            trial=None, curves: bool = False, max_train: int | None = None) -> dict:
        self.config_counter += 1
        out = {"fold_metrics": [], "preds": [], "fold_scores": [], "train_scores": []}
        ids_all = cohort_ids
        flist = folds if folds is not None else self.folds(scheme)
        for step, f in enumerate(flist):
            tr, te = self.fold_ids(scheme, f, ids_all)
            if len(tr) < self.min_train or len(te) < self.min_test or (task == "cls" and lab.loc[tr, "y"].nunique() < 2):
                out["fold_metrics"].append({"fold": f, "skipped": f"insufficient rows (train {len(tr)}, test {len(te)})"})
                continue
            trs = lab.loc[tr].sort_values("date")
            if max_train and len(trs) > max_train:
                # tuning screen only: time-stratified subsample (every k-th row in date order keeps all years)
                trs = trs.iloc[np.linspace(0, len(trs) - 1, int(max_train)).round().astype(int)]
            cut = int(len(trs) * (1 - self.inner_frac))
            itr, iva = trs.index[:cut], trs.index[cut:]
            ytr, yva, yte = lab.loc[itr, "y"].to_numpy(), lab.loc[iva, "y"].to_numpy(), lab.loc[te, "y"].to_numpy()
            wtr, wva = lab.loc[itr, "w"].to_numpy(), lab.loc[iva, "w"].to_numpy()
            if task == "reg":
                lo, hi = np.nanquantile(ytr, [0.005, 0.995])          # clip bounds from training rows only
                ytr_c, yva_c = np.clip(ytr, lo, hi), np.clip(yva, lo, hi)
            if rung == "L0":
                if task == "cls":
                    prior = np.bincount(ytr.astype(int), minlength=3) / len(ytr)
                    proba = np.tile(prior, (len(te), 1))
                    yhat = np.full(len(te), int(np.argmax(prior)))
                    m = classification_metrics(yte, proba, prior, yhat)
                    tr_m = m
                else:
                    yhat = np.full(len(te), float(np.mean(ytr)))
                    m = regression_metrics(yte, yhat, kind="return")
                    tr_m = m
            else:
                Xtr, (Xva, Xte), names = self.matrices(rung, itr, [iva, te], enc)
                if Xtr.shape[1] == 0:
                    out["fold_metrics"].append({"fold": f, "skipped": "no features"})
                    continue
                hd = Head(head, params, task=task, seed=self.seed)
                if task == "cls":
                    hd.fit(Xtr, ytr, wtr, Xva, yva, wva)
                    proba = hd.predict_proba(Xte)
                    yhat = decide(proba, hd.prior)
                    m = classification_metrics(yte, proba, hd.prior, yhat)
                    ptr = hd.predict_proba(Xtr)
                    tr_m = classification_metrics(ytr, ptr, hd.prior)
                else:
                    hd.fit(Xtr, ytr_c, wtr, Xva, yva_c, wva)
                    yhat = hd.predict(Xte)
                    m = regression_metrics(yte, yhat, kind="return")
                    tr_m = regression_metrics(ytr, hd.predict(Xtr), kind="return")
                if curves and hd.curve:
                    self.curves.append({"rung": rung, "head": head, "enc": enc, "fold": f, "scheme": scheme, **hd.curve})
                m["head_info"] = hd.info
            m["fold"] = f
            m["n_train"] = int(len(itr))
            out["fold_metrics"].append(m)
            key = "macro_f1" if task == "cls" else "r2"
            out["fold_scores"].append(m.get(key))
            out["train_scores"].append(tr_m.get(key))
            if keep:
                p = pd.DataFrame({"event_id": te, "fold": f, "y": yte, "date": lab.loc[te, "date"].values})
                if task == "cls":
                    p[["p_down", "p_side", "p_up"]] = proba
                    p["yhat"] = yhat
                else:
                    p["yhat_reg"] = yhat
                out["preds"].append(p)
            if trial is not None:
                sc = [s for s in out["fold_scores"] if s is not None]
                trial.report(float(np.mean(sc)) if sc else 0.0, step)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        sc = [s for s in out["fold_scores"] if s is not None]
        out["mean_score"] = float(np.mean(sc)) if sc else None
        out["std_score"] = float(np.std(sc)) if len(sc) > 1 else None
        trs = [s for s in out["train_scores"] if s is not None]
        out["mean_train_score"] = float(np.mean(trs)) if trs else None
        out["preds"] = pd.concat(out["preds"], ignore_index=True) if out["preds"] else pd.DataFrame()
        return out

    # ------------------------------------------------------------------ tuning
    def tune(self, rung, head, lab, cohort_ids, n_trials, timeout, storage_path: Path, study_name: str,
             task="cls", enc=None) -> tuple[dict, pd.DataFrame]:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{storage_path.as_posix()}"
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
        done = len([t for t in study.trials if t.state.is_finished()])
        if done == 0:
            from .heads import DEFAULTS
            study.enqueue_trial(DEFAULTS[head])

        tc = self.cfg["tuning"]
        last = int(tc.get("tune_last_folds") or 0)
        fl = self.folds("walk_forward")
        tune_folds = fl[-last:] if last else fl
        cap = int(tc.get("tune_max_train_rows") or 0) or None

        def obj(trial):
            params = space(trial, head)
            r = self.run(rung, head, params, lab, cohort_ids, "walk_forward", task, enc, keep=False, trial=trial,
                         folds=tune_folds, max_train=cap)
            return r["mean_score"] if r["mean_score"] is not None else 0.0

        remaining = max(0, int(n_trials) - done)
        if remaining:
            study.optimize(obj, n_trials=remaining, timeout=timeout, gc_after_trial=True)
        trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        best = study.best_trial.params if study.best_trial is not None else {}
        return best, trials


def summarize(res: dict, lab: pd.DataFrame, n_boot: int, seed: int, task: str = "cls") -> dict:
    """Pooled-OOF metrics with month-block CI + mean fold score."""
    p = res["preds"]
    if p.empty:
        return {"mean_fold": res.get("mean_score"), "std_fold": res.get("std_score"), "mean_train": res.get("mean_train_score"),
                "pooled": {}, "ci": {}, "n": 0}
    if task == "cls":
        prior = np.bincount(p.y.astype(int), minlength=3) / len(p)
        pooled = classification_metrics(p.y.values, p[["p_down", "p_side", "p_up"]].values, prior, p.yhat.values)
        ci = macro_f1_ci(p.y.values, p.yhat.values, p.date.values, n_boot, seed)
    else:
        pooled = regression_metrics(p.y.values, p.yhat_reg.values, dates=p.date, kind="return")
        ci = {}
    return {"mean_fold": res.get("mean_score"), "std_fold": res.get("std_score"),
            "mean_train": res.get("mean_train_score"), "pooled": pooled, "ci": ci, "n": int(len(p))}


def fit_verdict(mean_val, mean_train, std, majority, audit: dict) -> str:
    if mean_val is None:
        return "not_evaluable"
    v = []
    if mean_train is not None and mean_train - mean_val > float(audit["overfit_gap"]):
        v.append("overfit")
    if majority is not None and mean_val - majority <= float(audit["underfit_margin"]):
        v.append("underfit (may mean weak signal)")
    if std is not None and std > float(audit["unstable_std"]):
        v.append("unstable")
    return ", ".join(v) or "good fit"
