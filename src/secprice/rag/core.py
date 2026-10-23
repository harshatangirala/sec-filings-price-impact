"""Core RAG (SPEC.md §8): BM25, hybrid BM25 + dense (RRF), hybrid + cross-encoder rerank; two evidence paths.

Current-document path: chunks of the filing being processed (allowed at decision_time).
Historical path: the firm's prior filings with public_available_time < decision_time, never the current filing
and never a later one (SPEC.md §6.3 #16). The filter is the only way to build a historical candidate set.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(t: str) -> list[str]:
    return _TOK.findall(t.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = np.mean([len(t) for t in self.toks]) if self.toks else 0.0
        df: dict = {}
        for t in self.toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [pd.Series(t).value_counts().to_dict() if t else {} for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(self.N)
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            dl = len(toks)
            for w in q:
                f = tf.get(w)
                if f:
                    s[i] += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
        return s


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion of score arrays (higher = better)."""
    n = len(rankings[0])
    out = np.zeros(n)
    for sc in rankings:
        order = np.argsort(-sc, kind="stable")
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
        out += 1.0 / (k + r)
    return out


def historical_candidates(chunks: pd.DataFrame, cik: int, decision_time, current_accessions) -> pd.DataFrame:
    """The only constructor of the historical evidence set: same firm, public strictly before decision_time,
    current filing excluded."""
    dt = pd.Timestamp(decision_time)
    pub = pd.to_datetime(chunks["public_time"], utc=True)
    m = (chunks["cik"] == cik) & (pub < dt.tz_convert("UTC") if dt.tzinfo else pub < dt.tz_localize("UTC"))
    m &= ~chunks["accession"].isin(list(current_accessions))
    return chunks[m]


def retrieval_guard() -> bool:
    """Planted-leak self-test: a future document and the current filing must never enter the historical set."""
    now = pd.Timestamp("2015-06-01 12:00", tz="America/New_York")
    ch = pd.DataFrame({"chunk_id": ["past", "current", "future", "other_firm"], "cik": [1, 1, 1, 2],
                       "accession": ["a-past", "a-cur", "a-fut", "b-past"],
                       "public_time": [now - pd.Timedelta(days=90), now - pd.Timedelta(minutes=35),
                                       now + pd.Timedelta(days=1), now - pd.Timedelta(days=5)],
                       "text": ["x", "y", "z", "w"]})
    got = set(historical_candidates(ch, 1, now, ["a-cur"]).chunk_id)
    return got == {"past"}


class Retriever:
    """setup in {'bm25', 'hybrid', 'hybrid_rerank'}; dense / rerank models are injected (GPU) or None."""

    def __init__(self, setup: str, dense=None, reranker=None, k1=1.5, b=0.75, rrf_k=60, rerank_top_n=30):
        self.setup, self.dense, self.reranker = setup, dense, reranker
        self.k1, self.b, self.rrf_k, self.top_n = k1, b, rrf_k, rerank_top_n

    def rank(self, query: str, cands: pd.DataFrame, k: int) -> pd.DataFrame:
        if cands.empty:
            return cands.assign(score=[])
        texts = cands["text"].tolist()
        s_bm = BM25(texts, self.k1, self.b).scores(query)
        if self.setup == "bm25" or self.dense is None:
            score = s_bm
        else:
            s_d = self.dense(query, texts)
            score = rrf([s_bm, s_d], self.rrf_k)
        order = np.argsort(-score, kind="stable")
        out = cands.iloc[order].assign(score=score[order])
        if self.setup == "hybrid_rerank" and self.reranker is not None:
            top = out.head(self.top_n)
            rs = self.reranker(query, top["text"].tolist())
            top = top.assign(score=rs).sort_values("score", ascending=False)
            out = top
        return out.head(k)


def ir_metrics(ranked_ids: list[str], relevant: set, k: int) -> dict:
    """Recall@k, nDCG@k, MRR, AP for one query (binary relevance)."""
    r = ranked_ids[:k]
    hits = [1 if x in relevant else 0 for x in r]
    rec = sum(hits) / len(relevant) if relevant else None
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    ap, n = 0.0, 0
    for i, h in enumerate(hits):
        if h:
            n += 1
            ap += n / (i + 1)
    return {"recall": rec, "ndcg": dcg / idcg if idcg else None, "mrr": mrr, "ap": ap / len(relevant) if relevant else None}
