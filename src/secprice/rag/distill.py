"""S17 teacher sample -> student distillation (SPEC.md §8 Distillation). The teacher (frozen RAG + generator) labels a
stratified sample of TRAINING-PERIOD filings; a multi-task student (ModernBERT-base) is trained on those outputs and
applied to every event. Reported: student vs teacher agreement (and student vs human on gold-test in S23)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import common as C
from ..storage import write_json


def teacher_sample(ev: pd.DataFrame, dev_start: pd.Timestamp, n: int, seed: int) -> pd.DataFrame:
    tr = ev[pd.to_datetime(ev.decision_date) < dev_start].copy()
    tr["stratum"] = tr.event_type + "|" + (tr.year // 5 * 5).astype(str)
    k = min(n, len(tr))
    w = tr.groupby("stratum").event_id.transform("size")
    p = (1 / np.sqrt(w)) / (1 / np.sqrt(w)).sum()
    idx = np.random.default_rng(seed).choice(len(tr), size=k, replace=False, p=p.to_numpy())
    s = tr.iloc[idx]
    assert (pd.to_datetime(s.decision_date) < dev_start).all(), "teacher sample must be training-period only"
    return s


def run_distillation(ctx, exploratory: bool) -> dict:
    import torch
    from ..models.embeddings import event_chunks
    from ..rag.core import Retriever
    from ..rag.extract import ResponseCache, build_prompt, cache_key, field_schema, guard
    from ..stages_model import _dense_and_reranker, _generator
    cfg = ctx.cfg
    qb = C.question_bank(cfg)
    fields = list(qb["fields"])
    ev = C.events(ctx)
    smp = teacher_sample(ev, pd.Timestamp(cfg["splits"]["dev_start"]), int(cfg["budget"]["llm_teacher_filings"]), int(cfg["seed"]))
    gen, gname = _generator(ctx, qb)
    dense, rerank = _dense_and_reranker(ctx) if gname != "mock" else (None, None)
    frozen = json.loads((ctx.dep("S16") / "rag_frozen.json").read_text())
    ret = Retriever(frozen["setup"] if dense else "bm25", dense, rerank, cfg["rag"]["bm25_k1"], cfg["rag"]["bm25_b"],
                    cfg["rag"]["rrf_k"], cfg["rag"]["rerank_top_n"])
    cache = ResponseCache(ctx.out / "teacher_cache.jsonl")
    secs = []
    for cik in smp.cik.unique():
        p = ctx.dep("S05") / "sections" / f"{int(cik):010d}.parquet"
        if p.exists():
            secs.append(pd.read_parquet(p, columns=["accession", "section_id", "text_masked"]))
    ch = event_chunks(smp, pd.concat(secs, ignore_index=True), cfg) if secs else pd.DataFrame()
    ch["chunk_id"] = ch.event_id + ":" + ch.section_id + ":" + ch.chunk_no.astype(str)
    q = " ".join(f["question"] for f in qb["fields"].values())
    rows = []
    pmap, todo = {}, []
    for eid in smp.event_id:
        c = ch[ch.event_id == eid]
        if c.empty:
            continue
        top = ret.rank(q, c, int(cfg["rag"]["top_k"]))
        passages = [{"chunk_id": r.chunk_id, "text": r.text} for r in top.itertuples(index=False)]
        k = cache_key(getattr(gen, "revision", gname), build_prompt(qb, passages), passages)
        pmap[eid] = (k, passages)
        if cache.get(k) is None:
            todo.append(eid)
    for b0 in range(0, len(todo), 64):                      # checkpointed through the response cache
        ids_b = todo[b0:b0 + 64]
        for e, raw in zip(ids_b, gen.batch(qb, [pmap[e][1] for e in ids_b])):
            cache.put(pmap[e][0], raw)
    for eid, (k, passages) in pmap.items():
        clean, _ = guard(cache.get(k) or {}, qb, passages, field_schema(qb))
        for f in fields:
            v = clean.get(f) or {}
            rows.append({"event_id": eid, "field": f, "present": v.get("present"), "direction": v.get("direction"),
                         "magnitude": v.get("magnitude"), "unit": v.get("unit"), "certainty": v.get("certainty"),
                         "evidence": json.dumps(v.get("evidence") or []), "source": "teacher", "confidence": None})
    teach = pd.DataFrame(rows)
    texts = C.read(ctx, "S09", "event_text.parquet").set_index("event_id")["text"]
    student = train_student(cfg, teach, texts, fields, qb, ctx.log)
    allp = student["predict"](texts.index.tolist(), texts.tolist())
    feats = pd.DataFrame({"event_id": texts.index})
    for j, f in enumerate(fields):
        feats[f"ext_{f}_present"] = allp[:, j]
    ctx.write(feats, "ext_features.parquet")
    ext = teach.copy()
    stud = feats.melt(id_vars="event_id", var_name="field", value_name="confidence")
    stud["field"] = stud.field.str.replace("ext_", "").str.replace("_present", "")
    stud["present"] = stud.confidence > 0.5
    stud["source"] = "student"
    ctx.write(pd.concat([ext, stud], ignore_index=True), "extractions.parquet")
    rep = {"ran": True, "exploratory": exploratory, "generator": gname, "teacher_filings": int(smp.shape[0]),
           "student_vs_teacher": student["agreement"], "label": "EXPLORATORY: G3b not passed" if exploratory else "G3b passed"}
    write_json(ctx.out / "student_report.json", rep)
    return rep


def train_student(cfg, teach: pd.DataFrame, texts: pd.Series, fields: list[str], qb: dict, log=None) -> dict:
    """Multi-label student: present probability per field (BCE). ModernBERT-base on GPU; logistic TF-IDF on CPU."""
    import torch
    piv = teach.pivot_table(index="event_id", columns="field", values="present", aggfunc="first").reindex(columns=fields)
    piv = piv.dropna(how="all").fillna(False).astype(float)
    X = texts.reindex(piv.index).fillna("").tolist()
    Y = piv.to_numpy(np.float32)
    n = len(X)
    cut = max(1, int(n * 0.85))
    if not torch.cuda.is_available() or n < 50:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vec = TfidfVectorizer(max_features=20000, min_df=1, sublinear_tf=True).fit(X[:cut])
        Xt = vec.transform(X)
        models = []
        for j in range(len(fields)):
            y = Y[:cut, j]
            models.append(LogisticRegression(max_iter=500).fit(Xt[:cut], y) if len(set(y)) > 1 else float(y.mean() if len(y) else 0))
        def predict(ids, txts):
            Z = vec.transform(txts)
            return np.column_stack([m.predict_proba(Z)[:, 1] if hasattr(m, "predict_proba") else np.full(len(txts), m) for m in models])
        kind = "tfidf_logistic_cpu_fallback"
    else:
        from transformers import AutoModel, AutoTokenizer
        st = C.roster(cfg)["student"][cfg["distill"]["student_model"]]
        tok = AutoTokenizer.from_pretrained(st["model_id"], revision=st["revision"])
        base = AutoModel.from_pretrained(st["model_id"], revision=st["revision"]).cuda()
        head = torch.nn.Linear(base.config.hidden_size, len(fields)).cuda()
        opt = torch.optim.AdamW(list(base.parameters()) + list(head.parameters()), lr=float(cfg["distill"]["lr"]))
        enc = tok(X, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        yt = torch.tensor(Y)
        bs = int(cfg["distill"]["batch"])
        base.train()
        for ep in range(int(cfg["distill"]["epochs"])):
            perm = torch.randperm(cut)
            for k in range(0, cut, bs):
                b = perm[k:k + bs]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = base(input_ids=enc["input_ids"][b].cuda(), attention_mask=enc["attention_mask"][b].cuda()).last_hidden_state[:, 0]
                    lo = head(h.float())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), yt[b].cuda())
                loss.backward()
                opt.step()
                opt.zero_grad()
            if log:
                log.info(f"student epoch {ep}: loss {loss.item():.4f}")
        base.eval()

        def predict(ids, txts):
            out = []
            with torch.no_grad():
                for k in range(0, len(txts), 64):
                    e = tok(txts[k:k + 64], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = base(**e).last_hidden_state[:, 0]
                    out.append(torch.sigmoid(head(h.float())).float().cpu().numpy())
            return np.vstack(out) if out else np.zeros((0, len(fields)))
        kind = "modernbert_multitask"
    hold = predict(None, X[cut:]) if n > cut else np.zeros((0, len(fields)))
    from sklearn.metrics import f1_score
    agr = {}
    for j, f in enumerate(fields):
        yt_ = Y[cut:, j] > 0.5
        if len(yt_) and yt_.any():
            agr[f] = float(f1_score(yt_, hold[:, j] > 0.5))
    return {"predict": predict, "agreement": {"kind": kind, "heldout_n": int(n - cut), "f1_vs_teacher": agr}}
