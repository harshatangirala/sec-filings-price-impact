"""S14 frozen encoder embeddings (SPEC.md §7.2): one model in memory at a time -> embed -> persist float16 + index
-> unload. Section-aware chunks; mean/max pooling to section vectors; event vector = chunk-weighted mean of
section means; change = cosine to the firm's prior same-type event vector (prior decision time)."""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from ..parse.html import drop_numeric_lines


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    w = text.split()
    if not w:
        return []
    step = max(1, size - overlap)
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - overlap), step)]


def select_chunks(chunks: list[str], max_n: int) -> list[str]:
    if len(chunks) <= max_n:
        return chunks
    idx = np.unique(np.linspace(0, len(chunks) - 1, max_n).round().astype(int))
    return [chunks[i] for i in idx]


def device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return "cpu", torch.float32


def load_tokenizer(model_id: str, revision: str | None):
    """AutoTokenizer, falling back to BertTokenizerFast for BERT checkpoints whose config lacks `model_type`
    (e.g. yiyanghkust/finbert-tone under transformers 5)."""
    from transformers import AutoTokenizer, BertTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(model_id, revision=revision)
    except Exception:
        return BertTokenizerFast.from_pretrained(model_id, revision=revision)


def load_model(model_id: str, revision: str | None, kind: str, dtype=None, **kw):
    """kind: 'base' | 'classifier'. Falls back to the BERT classes when the config has no model_type."""
    from transformers import AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, BertModel
    auto = AutoModelForSequenceClassification if kind == "classifier" else AutoModel
    bert = BertForSequenceClassification if kind == "classifier" else BertModel
    last = None
    for cls in (auto, bert):
        for dkw in (({"dtype": dtype} if dtype is not None else {}), ({"torch_dtype": dtype} if dtype is not None else {})):
            try:
                return cls.from_pretrained(model_id, revision=revision, **dkw, **kw)
            except TypeError as e:
                last = e
                continue
            except (ValueError, KeyError, OSError) as e:
                last = e
                break
    raise RuntimeError(f"could not load {model_id}@{revision}: {last}")


def load_encoder(model_id: str, revision: str | None, kind: str):
    dev, dt = device_and_dtype()
    tok = load_tokenizer(model_id, revision)
    m = load_model(model_id, revision, kind, dt)
    m.to(dev).eval()
    return tok, m, dev


def unload(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode(tok, model, dev, texts: list[str], max_len: int, batch: int, pooling: str, classifier: bool = False,
           log=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Length-bucketed batched encoding with OOM retry that halves the batch."""
    import torch
    order = np.argsort([len(t) for t in texts])
    H = None
    probs = None
    out_vec, out_p = [None] * len(texts), [None] * len(texts)
    i, b = 0, batch
    while i < len(order):
        idx = order[i:i + b]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(dev == "cuda")):
                o = model(**enc, output_hidden_states=classifier)
            hs = o.hidden_states[-1] if classifier else o.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            if pooling == "cls":
                v = hs[:, 0]
                v = torch.nn.functional.normalize(v.float(), dim=-1)
            else:
                v = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            v = v.float().cpu().numpy().astype(np.float16)
            p = torch.softmax(o.logits.float(), -1).cpu().numpy() if classifier else None
            for k, j in enumerate(idx):
                out_vec[j] = v[k]
                if p is not None:
                    out_p[j] = p[k]
            i += b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if b == 1:
                raise
            b = max(1, b // 2)
            if log:
                log.warning(f"OOM: batch halved to {b}")
    V = np.vstack(out_vec) if out_vec else np.zeros((0, 768), np.float16)
    P = np.vstack(out_p) if classifier and out_p and out_p[0] is not None else None
    return V, P


def event_chunks(events: pd.DataFrame, sec_index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows: event_id, section_id, chunk_no, text. sec_index: accession, section_id, text_masked."""
    t = cfg["text"]
    by_acc = {a: g for a, g in sec_index.groupby("accession")}
    rows = []
    for r in events.itertuples(index=False):
        for a in r.accessions:
            g = by_acc.get(a)
            if g is None:
                continue
            for s in g.itertuples(index=False):
                txt = drop_numeric_lines(s.text_masked) if t.get("drop_numeric_lines", True) else s.text_masked
                ch = select_chunks(chunk_words(txt, int(t["chunk_words"]), int(t["chunk_overlap_words"])),
                                   int(t["max_chunks_per_section"]))
                rows += [(r.event_id, s.section_id, i, c) for i, c in enumerate(ch)]
    return pd.DataFrame(rows, columns=["event_id", "section_id", "chunk_no", "text"])


def pool_events(chunks: pd.DataFrame, V: np.ndarray, P: np.ndarray | None = None) -> tuple[pd.Index, np.ndarray, pd.DataFrame]:
    """Event vector = chunk-count-weighted mean of section mean vectors (= mean over chunks); plus max-pooled
    section vectors averaged; native classifier probabilities averaged."""
    df = chunks[["event_id", "section_id"]].copy()
    df["row"] = np.arange(len(df))
    ev_ids, vecs, extra = [], [], []
    for eid, g in df.groupby("event_id", sort=False):
        rows = g.row.to_numpy()
        Vm = V[rows].astype(np.float32)
        mean_v = Vm.mean(0)
        max_v = np.mean([V[s.row.to_numpy()].astype(np.float32).max(0) for _, s in g.groupby("section_id")], axis=0)
        ev_ids.append(eid)
        vecs.append(np.concatenate([mean_v, max_v]).astype(np.float16))
        e = {"event_id": eid, "n_chunks": len(rows)}
        if P is not None:
            pm = P[rows].mean(0)
            e.update({"tone_neutral": pm[0], "tone_positive": pm[1], "tone_negative": pm[2]})
        extra.append(e)
    return pd.Index(ev_ids, name="event_id"), np.vstack(vecs) if vecs else np.zeros((0, 2 * V.shape[1]), np.float16), pd.DataFrame(extra)


def change_cosine(events: pd.DataFrame, ids: pd.Index, M: np.ndarray, dim: int) -> pd.Series:
    """Cosine between an event's mean vector and the firm's previous same-type event vector."""
    pos = {e: i for i, e in enumerate(ids)}
    out = {}
    ev = events[events.event_id.isin(pos)].sort_values("decision_time")
    for (cik, et), g in ev.groupby(["cik", "event_type"]):
        prev = None
        for eid in g.event_id:
            v = M[pos[eid], :dim].astype(np.float32)
            if prev is not None:
                den = np.linalg.norm(v) * np.linalg.norm(prev)
                out[eid] = float(v @ prev / den) if den > 0 else np.nan
            prev = v
    return pd.Series(out, name="chg_emb_cosine")


def chrono_vintage(year: int, vintages: list[int]) -> int:
    """Event year Y uses vintage Y-1; clipped to the available range (always precedes the event)."""
    v = min(year - 1, max(vintages))
    return max(v, min(vintages))


def benchmark(tok, model, dev, n: int = 256, max_len: int = 512) -> float:
    """Chunks per second on synthetic 350-word texts (preflight micro-benchmark)."""
    texts = [("financial results revenue margin outlook guidance risk " * 50)[:2400]] * n
    t0 = time.time()
    encode(tok, model, dev, texts, max_len, 64, "mean")
    return n / max(1e-6, time.time() - t0)
