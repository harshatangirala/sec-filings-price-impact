"""Tabular heads: LightGBM, CatBoost, ElasticNet-logistic (classification) and LightGBM/ElasticNet (regression).

Losses (SPEC.md §9.4): class-weighted cross-entropy (balanced weights x label-uniqueness weights); Huber for
regression. Early stopping and probability calibration use an inner, time-ordered slice of the training fold
only. Every fitted head keeps its train/validation loss curve.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULTS = {
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
                 "colsample_bytree": 0.8, "reg_alpha": 1e-3, "reg_lambda": 1.0, "max_depth": -1},
    "catboost": {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0, "random_strength": 1.0},
    "enet": {"C": 0.1, "l1_ratio": 0.5},
}


def space(trial, head: str) -> dict:
    """Search spaces fixed before the holdout is unlocked (SPEC.md §9.4; v3 §13.1)."""
    if head == "lightgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8])}
    if head == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True)}
    if head == "enet":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    raise ValueError(head)


def prior_correct(p_balanced: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """A model trained with class-balanced weights estimates posteriors under uniform class priors. Multiplying by
    the training prior and renormalising returns true-frequency probabilities (then temperature-calibrated); the
    cost-balanced decision argmax p / prior is applied once, in evaluation.metrics.decide."""
    q = np.asarray(p_balanced, float) * np.maximum(np.asarray(prior, float), 1e-6)
    return q / q.sum(1, keepdims=True)


def _balanced_w(y: np.ndarray, k: int = 3) -> np.ndarray:
    cnt = np.bincount(y.astype(int), minlength=k).astype(float)
    cw = np.where(cnt > 0, len(y) / (k * np.maximum(cnt, 1)), 0.0)
    return cw[y.astype(int)]


def _gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar
        y = y.astype(int)

        def nll(t):
            z = logp / t
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(y)), y].mean()

        if len(y) >= 30 and len(np.unique(y)) > 1:
            self.T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(proba, 1e-12, 1)) / self.T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class Head:
    def __init__(self, kind: str, params: dict | None, task: str = "cls", seed: int = 0, n_jobs: int | None = None):
        self.kind, self.task, self.seed = kind, task, seed
        self.params = {**DEFAULTS.get(kind, {}), **(params or {})}
        self.n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
        self.curve: dict = {}
        self.model = None
        self.cal = TemperatureScaler()
        self.prior = np.ones(3) / 3
        self.info: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, w, Xv, yv, wv):
        y = np.asarray(y)
        yv = np.asarray(yv)
        if self.task == "cls":
            self.prior = np.bincount(y.astype(int), minlength=3) / max(1, len(y))
            w = np.asarray(w) * _balanced_w(y)
            wv = np.asarray(wv) * _balanced_w(yv) if len(yv) else wv
        getattr(self, f"_fit_{self.kind}")(X, y, w, Xv, yv, wv)
        if self.task == "cls" and len(yv):
            self.cal.fit(np.log(np.clip(self._raw_proba(Xv), 1e-12, 1)), yv)
        return self

    def _fit_lightgbm(self, X, y, w, Xv, yv, wv):
        import lightgbm as lgb
        p = dict(self.params)
        common = dict(n_estimators=3000, subsample_freq=1, random_state=self.seed, n_jobs=self.n_jobs, verbose=-1, max_bin=127, **p)
        if self.task == "cls":
            self.model = lgb.LGBMClassifier(objective="multiclass", num_class=3, **common)
            metric = "multi_logloss"
        else:
            self.model = lgb.LGBMRegressor(objective="huber", alpha=1.0, **common)
            metric = "huber"
        ev = {}
        cbs = [lgb.record_evaluation(ev)]
        es = len(yv) > 0
        if es:
            cbs.append(lgb.early_stopping(100, verbose=False))
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
            self.model.fit(X, y, sample_weight=w, eval_set=[(X, y), (Xv, yv)] if es else [(X, y)],
                           eval_sample_weight=[w, wv] if es else [w], eval_metric=metric, callbacks=cbs)
        keys = list(ev.keys())
        self.curve = {"train": list(ev[keys[0]][metric]) if keys else [],
                      "val": list(ev[keys[1]][metric]) if len(keys) > 1 else [], "metric": metric, "unit": "boosting round"}
        self.info["best_iteration"] = int(getattr(self.model, "best_iteration_", 0) or 0)

    def _fit_catboost(self, X, y, w, Xv, yv, wv):
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        p = dict(self.params)
        gpu = _gpu()
        kw = dict(iterations=3000, random_seed=self.seed, verbose=False, allow_writing_files=False,
                  od_type="Iter", od_wait=100, task_type="GPU" if gpu else "CPU", **p)
        if not gpu:
            kw["thread_count"] = self.n_jobs
        if gpu:
            kw.pop("random_strength", None)
        if self.task == "cls":
            self.model = CatBoostClassifier(loss_function="MultiClass", **kw)
        else:
            self.model = CatBoostRegressor(loss_function="Huber:delta=1.0", **kw)
        tr = Pool(X, y, weight=w)
        if self.task == "cls" and len(yv):
            keep = np.isin(yv, np.unique(y))           # CatBoost rejects eval classes unseen in training
            Xv, yv, wv = Xv[keep], yv[keep], np.asarray(wv)[keep]
        va = Pool(Xv, yv, weight=wv) if len(yv) else None
        self.model.fit(tr, eval_set=va, use_best_model=va is not None)
        er = self.model.get_evals_result()
        lk = "MultiClass" if self.task == "cls" else list(er.get("learn", {}).keys())[0] if er.get("learn") else None
        self.curve = {"train": list(er.get("learn", {}).get(lk, [])), "val": list(er.get("validation", {}).get(lk, [])),
                      "metric": lk, "unit": "iteration"}
        self.info["best_iteration"] = int(self.model.get_best_iteration() or 0)

    def _fit_enet(self, X, y, w, Xv, yv, wv):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        p = self.params
        if self.task == "cls":
            est = LogisticRegression(solver="saga", penalty="elasticnet", C=p["C"], l1_ratio=p["l1_ratio"], max_iter=400,
                                     tol=1e-3, random_state=self.seed)
        else:
            est = ElasticNet(alpha=1.0 / max(p["C"], 1e-6) / 100, l1_ratio=p["l1_ratio"], max_iter=2000, random_state=self.seed)
        self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                   StandardScaler(), est)
        self.model.fit(X, y, **{f"{type(est).__name__.lower()}__sample_weight": w})
        last = self.model[-1]
        self.info["n_iter"] = int(np.max(getattr(last, "n_iter_", [0])))
        self.info["converged"] = bool(self.info["n_iter"] < (400 if self.task == "cls" else 2000))
        self.curve = {}

    # ------------------------------------------------------------------ predict
    def _raw_proba(self, X) -> np.ndarray:
        """Always 3 columns (down, sideways, up) even when a training fold lacked a class."""
        P = np.asarray(self.model.predict_proba(X), float)
        cls = getattr(self.model, "classes_", None)
        if cls is None and hasattr(self.model, "steps"):
            cls = getattr(self.model[-1], "classes_", None)
        if not (P.shape[1] == 3 and (cls is None or list(np.asarray(cls).astype(int)) == [0, 1, 2])):
            full = np.full((P.shape[0], 3), 1e-6)
            for j, c in enumerate(np.asarray(cls).astype(int).ravel()):
                full[:, int(c)] = P[:, j]
            P = full / full.sum(1, keepdims=True)
        return prior_correct(P, self.prior) if self.task == "cls" else P

    def predict_proba(self, X) -> np.ndarray:
        return self.cal.transform(self._raw_proba(X))

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X), float).ravel()


def enet_certificate(X, y, w, params: dict, seeds=(0, 1, 2)) -> dict:
    """Convex problem -> a unique global optimum. Refit from several random starts and compare the penalized
    objective; agreement within tolerance certifies the optimizer reached the global minimum."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xi = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X))
    y = np.asarray(y).astype(int)
    w = np.asarray(w) * _balanced_w(y)
    objs = []
    for s in seeds:
        m = LogisticRegression(solver="saga", penalty="elasticnet", C=params["C"], l1_ratio=params["l1_ratio"],
                               max_iter=3000, tol=1e-6, random_state=s, warm_start=False)
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(y))
        m.fit(Xi[perm], y[perm], sample_weight=w[perm])
        p = np.clip(m.predict_proba(Xi), 1e-12, 1)
        nll = -np.sum(w * np.log(p[np.arange(len(y)), y])) / np.sum(w)
        coef = m.coef_.ravel()
        pen = (params["l1_ratio"] * np.abs(coef).sum() + 0.5 * (1 - params["l1_ratio"]) * (coef ** 2).sum()) / (params["C"] * np.sum(w))
        objs.append(float(nll + pen))
    spread = float(np.max(objs) - np.min(objs))
    return {"objective_by_start": objs, "spread": spread, "certified_global_min": bool(spread < 1e-4),
            "note": "elastic-net logistic loss is convex; equal objectives from different starts = global optimum"}
