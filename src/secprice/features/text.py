"""Lexical (Loughran-McDonald), filing-change and novelty features (SPEC.md §9.1 L4; H2).

No statistic is fitted here (no IDF, no vocabulary): similarity uses a fixed hashing space, so nothing is learned
from future data. Prior documents are those of the same firm whose decision time precedes the current event
(their public time is therefore strictly earlier); ties are processed as a batch so simultaneous events never see
each other.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

from ..parse.html import drop_numeric_lines

LM_CATS = ["Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal", "Constraining"]
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_DIG = re.compile(r"\d")
MAIN_SECTION = {"periodic_10K": ["10K_7", "10K_1A", "10K_1", "10K_7A"], "periodic_10Q": ["10Q_MDA", "10Q_RF", "10Q_MR"],
                "8K": ["8K_EX99", "8K_ITEMS"]}


def load_lm(path) -> dict:
    df = pd.read_csv(path)
    lm = {}
    for c in LM_CATS:
        if c in df.columns:
            for w in df.loc[df[c] > 0, "Word"].astype(str).str.upper():
                lm.setdefault(w, set()).add(c)
    return lm


def lm_props(text: str, lm: dict) -> dict:
    words = [w.upper() for w in _WORD.findall(text)]
    n = len(words)
    counts = dict.fromkeys(LM_CATS, 0)
    for w in words:
        cats = lm.get(w)
        if cats:
            for c in cats:
                counts[c] += 1
    out = {f"lm_{c.lower()}": (counts[c] / n if n else np.nan) for c in LM_CATS}
    out["lm_net_tone"] = ((counts["Positive"] - counts["Negative"]) / (counts["Positive"] + counts["Negative"])
                          if (counts["Positive"] + counts["Negative"]) else np.nan)
    out["lm_n_words"] = n
    return out


def sentence_hashes(text: str) -> set:
    out = set()
    for s in _SENT.split(text):
        s = " ".join(s.lower().split())
        if len(s.split()) < 6:
            continue
        s = _DIG.sub("0", s)
        out.add(zlib.crc32(s.encode()) ^ (len(s) << 32))
    return out


class _Vec:
    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.hv = HashingVectorizer(n_features=2 ** 18, alternate_sign=False, norm="l2", stop_words="english",
                                    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")

    def __call__(self, text: str):
        return self.hv.transform([text])


def event_doc(sec_texts: dict, event_type: str, narrative: bool = True) -> tuple[str, str]:
    """(full doc text, main-section text) from {section_id: text}."""
    order = MAIN_SECTION.get(event_type, [])
    parts = [sec_texts[s] for s in order if s in sec_texts] + [v for k, v in sec_texts.items() if k not in order]
    doc = "\n\n".join(parts)
    main = next((sec_texts[s] for s in order if s in sec_texts), "")
    if narrative:
        doc, main = drop_numeric_lines(doc), drop_numeric_lines(main)
    return doc, main


def firm_text_features(events: pd.DataFrame, texts: dict, lm: dict) -> pd.DataFrame:
    """events (one firm): event_id, event_type, decision_time, accessions, public_available_time.
    texts: accession -> {section_id: text_raw}."""
    vec = _Vec()
    ev = events.sort_values("decision_time").reset_index(drop=True)
    memory: set = set()
    last: dict = {}           # event_type -> (vector, lm props, sentence set, n_words, rf_sentences)
    rows = []
    for _, tie in ev.groupby("decision_time", sort=True):
        staged = []
        for r in tie.itertuples(index=False):
            secs = {}
            for a in r.accessions:
                for sid, t in (texts.get(a) or {}).items():
                    secs[sid] = (secs[sid] + "\n\n" + t) if sid in secs else t
            doc, main = event_doc(secs, r.event_type)
            f = {"event_id": r.event_id, "txt_has_text": bool(doc.strip()), "txt_doc_chars": len(doc)}
            if not doc.strip():
                rows.append(f)
                continue
            p_doc = lm_props(doc, lm)
            p_main = lm_props(main, lm) if main else {}
            f.update({k: v for k, v in p_doc.items()})
            f.update({f"main_{k}": v for k, v in p_main.items()})
            v = vec(doc)
            sh = sentence_hashes(doc)
            rf = sentence_hashes(drop_numeric_lines(secs.get("10K_1A", "") or secs.get("10Q_RF", "")))
            prev = last.get(r.event_type)
            if prev is not None:
                pv, pp, psh, pn, prf = prev
                f["chg_cosine"] = float(v.multiply(pv).sum())
                f["chg_jaccard_sent"] = len(sh & psh) / max(1, len(sh | psh))
                f["chg_log_len"] = np.log((p_doc["lm_n_words"] + 1) / (pn + 1))
                for c in ("lm_negative", "lm_positive", "lm_uncertainty", "lm_litigious", "lm_net_tone"):
                    a, b = p_doc.get(c), pp.get(c)
                    f[f"chg_{c}"] = (a - b) if (a is not None and b is not None and np.isfinite(a) and np.isfinite(b)) else np.nan
                f["boilerplate_share"] = len(sh & psh) / max(1, len(sh))
                if rf and prf:
                    f["rf_new_share"] = len(rf - prf) / max(1, len(rf))
                    f["rf_removed_share"] = len(prf - rf) / max(1, len(prf))
            f["novelty_share"] = len(sh - memory) / max(1, len(sh)) if memory else np.nan
            staged.append((r.event_type, v, p_doc, sh, p_doc["lm_n_words"], rf))
            rows.append(f)
        for et, v, p, sh, n, rf in staged:      # tie batch: update memory only after the whole batch
            memory |= sh
            prev = last.get(et)
            last[et] = (v, p, sh, n, rf if rf else (prev[4] if prev else set()))
    return pd.DataFrame(rows)


def model_text(sec_texts_masked: dict, event_type: str, max_chars: int) -> str:
    """Masked narrative text fed to TF-IDF and fine-tuning (main section first)."""
    doc, _ = event_doc(sec_texts_masked, event_type)
    return doc[:max_chars]
