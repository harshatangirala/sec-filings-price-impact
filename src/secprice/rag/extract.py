"""Evidence-grounded extraction (SPEC.md §8): prompt -> JSON-constrained generation -> guardrails -> cache.

Guardrails: schema validation + one retry; numeric cross-check (a magnitude must appear in the cited evidence);
faithfulness gate (every evidence quote must be a substring of its cited chunk, otherwise the field becomes null);
citation offsets verified. The response cache is keyed by model revision, prompt hash and passage hash.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

SYSTEM = ("You extract facts from U.S. SEC filings. Use ONLY the numbered passages. If the passages do not support a "
          "field, set present=false. Quote evidence verbatim (<= 30 words) and cite the passage id. Output JSON only.")


def field_schema(qb: dict) -> dict:
    props = {}
    for name, f in qb["fields"].items():
        props[name] = {"type": "object", "properties": {
            "present": {"type": "boolean"},
            "direction": {"anyOf": [{"type": "string", "enum": f["directions"]}, {"type": "null"}]},
            "magnitude": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "certainty": {"type": "string", "enum": qb.get("certainty_levels", ["low", "medium", "high"])},
            "evidence": {"type": "array", "maxItems": 2, "items": {"type": "object", "properties": {
                "chunk_id": {"type": "string"}, "quote": {"type": "string"}}, "required": ["chunk_id", "quote"]}}},
            "required": ["present", "direction", "magnitude", "unit", "certainty", "evidence"]}
    return {"type": "object", "properties": props, "required": list(props)}


def build_prompt(qb: dict, passages: list[dict]) -> str:
    qs = "\n".join(f"- {k}: {f['question']} directions={f['directions']}" for k, f in qb["fields"].items())
    ps = "\n\n".join(f"[{p['chunk_id']}] {p['text'][:1800]}" for p in passages)
    return f"Fields:\n{qs}\n\nPassages:\n{ps}\n\nReturn one JSON object with every field."


def cache_key(model_rev: str, prompt: str, passages: list[dict]) -> str:
    ph = hashlib.sha256(prompt.encode()).hexdigest()
    pa = hashlib.sha256("".join(p["chunk_id"] + p["text"] for p in passages).encode()).hexdigest()
    return hashlib.sha256(f"{model_rev}|{ph}|{pa}".encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.d[r["k"]] = r["v"]
                except Exception:
                    continue

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}) + "\n")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w$%.,-]+", " ", s.lower()).split())


def guard(raw: dict, qb: dict, passages: list[dict], schema: dict) -> tuple[dict, dict]:
    """Apply guardrails. Returns (clean fields, flags)."""
    import jsonschema
    flags = {"schema_ok": True, "unsupported_nulled": 0, "numeric_nulled": 0, "citations_verified": 0, "citations_total": 0}
    try:
        jsonschema.validate(raw, schema)
    except Exception:
        flags["schema_ok"] = False
    by_id = {p["chunk_id"]: p["text"] for p in passages}
    out = {}
    for name, fdef in qb["fields"].items():
        f = raw.get(name) if isinstance(raw, dict) else None
        if not isinstance(f, dict):
            out[name] = {"present": None, "direction": None, "magnitude": None, "unit": None, "certainty": None,
                         "evidence": [], "status": "missing"}
            continue
        ev_ok = []
        for e in (f.get("evidence") or [])[:2]:
            flags["citations_total"] += 1
            txt = by_id.get(str(e.get("chunk_id")))
            q = str(e.get("quote") or "")
            if txt and q and _norm(q) and _norm(q) in _norm(txt):
                st = txt.lower().find(q.lower()[:40])
                ev_ok.append({"chunk_id": e["chunk_id"], "quote": q, "start": int(st) if st >= 0 else None,
                              "end": int(st + len(q)) if st >= 0 else None})
                flags["citations_verified"] += 1
        rec = {"present": bool(f.get("present")) if f.get("present") is not None else None,
               "direction": f.get("direction") if f.get("direction") in fdef["directions"] else None,
               "magnitude": f.get("magnitude") if isinstance(f.get("magnitude"), (int, float)) else None,
               "unit": f.get("unit"), "certainty": f.get("certainty"), "evidence": ev_ok, "status": "ok"}
        if rec["present"] and not ev_ok:          # faithfulness gate
            rec.update(present=None, direction=None, magnitude=None, status="unsupported_nulled")
            flags["unsupported_nulled"] += 1
        if rec["magnitude"] is not None:
            nums = " ".join(e["quote"] for e in ev_ok)
            mag = rec["magnitude"]
            cand = {f"{mag:g}", f"{mag:,.0f}", f"{mag:.1f}", f"{mag:.2f}"}
            if not any(c in nums.replace("$", "") for c in cand):
                rec["magnitude"] = None
                rec["status"] = "numeric_nulled"
                flags["numeric_nulled"] += 1
        out[name] = rec
    return out, flags


class MockGenerator:
    """Deterministic stub for the CPU smoke path (addendum P1.5): keyword rules, quotes copied from passages."""
    revision = "mock-v1"

    RULES = {"guidance": r"guidance|outlook", "results_tone": r"record|increase|decrease|declin", "impairment": r"impairment|write-?down",
             "restructuring": r"restructur|layoff|workforce reduction", "liquidity_stress": r"going concern|liquidity",
             "financing_covenant": r"credit agreement|notes due|covenant|indenture", "mna": r"acqui|merger|divest",
             "dividend": r"dividend", "buyback": r"repurchase|buyback", "litigation_regulatory": r"lawsuit|litigation|investigation",
             "management_change": r"chief executive|chief financial|resign|appoint", "risk_factor_change": r"risk factor"}

    def batch(self, qb: dict, passages_list: list, batch_size: int = 16) -> list:
        return [self(qb, p) for p in passages_list]

    def __call__(self, qb: dict, passages: list[dict]) -> dict:
        out = {}
        for name, fdef in qb["fields"].items():
            rx = re.compile(self.RULES.get(name, "$^"), re.I)
            hit = next(((p["chunk_id"], m) for p in passages for m in [rx.search(p["text"])] if m), None)
            if hit:
                cid, m = hit
                txt = next(p["text"] for p in passages if p["chunk_id"] == cid)
                s = max(0, m.start() - 60)
                quote = txt[s:m.end() + 60]
                out[name] = {"present": True, "direction": fdef["directions"][0], "magnitude": None, "unit": None,
                             "certainty": "low", "evidence": [{"chunk_id": cid, "quote": quote}]}
            else:
                out[name] = {"present": False, "direction": None, "magnitude": None, "unit": None, "certainty": "low",
                             "evidence": []}
        return out


def _shim_tokenization_utils() -> None:
    """transformers 5 removed `transformers.tokenization_utils`; lm-format-enforcer (<= 0.11) still imports
    PreTrainedTokenizerBase from it. Alias the module only when it is missing."""
    import importlib
    import sys
    base = importlib.import_module("transformers.tokenization_utils_base")
    try:
        mod = importlib.import_module("transformers.tokenization_utils")
    except ImportError:
        sys.modules["transformers.tokenization_utils"] = base
        return
    if not hasattr(mod, "PreTrainedTokenizerBase"):
        try:
            setattr(mod, "PreTrainedTokenizerBase", base.PreTrainedTokenizerBase)
        except Exception:
            sys.modules["transformers.tokenization_utils"] = base


class QwenGenerator:
    """Qwen2.5-7B-Instruct via transformers; JSON-schema-constrained decoding with lm-format-enforcer when
    available, otherwise schema validation + retry (logged)."""

    def __init__(self, model_id: str, revision: str, schema: dict, max_new_tokens: int = 900, log=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.revision = revision
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        kw = dict(revision=revision, device_map="auto")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
        self.model.eval()
        self.max_new = max_new_tokens
        self.schema = schema
        self.log = log
        self.constrained = False
        try:
            _shim_tokenization_utils()
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            self._parser = JsonSchemaParser(schema)
            self._fn = build_transformers_prefix_allowed_tokens_fn(self.tok, self._parser)
            self.constrained = True
        except Exception as e:
            if log:
                log.warning(f"lm-format-enforcer unavailable ({e}); using validation + retry")

    def generate(self, prompts: list[str]) -> list[str]:
        import torch
        msgs = [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(msgs, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(self.model.device)
        kw = dict(max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        if self.constrained:
            kw["prefix_allowed_tokens_fn"] = self._fn
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        return [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    @staticmethod
    def _parse(txt: str):
        try:
            return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception:
            return None

    def __call__(self, qb: dict, passages: list[dict]) -> dict:
        return self.batch(qb, [passages])[0]

    def batch(self, qb: dict, passages_list: list, batch_size: int = 16) -> list:
        """Batched generation (left padding); failures are retried one by one with an explicit JSON instruction."""
        import torch
        out: list = [None] * len(passages_list)
        prompts = [build_prompt(qb, p) for p in passages_list]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        bs, k = batch_size, 0
        while k < len(order):
            idx = order[k:k + bs]
            try:
                txts = self.generate([prompts[i] for i in idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                continue
            for i, t in zip(idx, txts):
                out[i] = self._parse(t)
            k += len(idx)
        for i, o in enumerate(out):
            if o is None:
                out[i] = self._parse(self.generate([prompts[i] + "\nReturn valid JSON only."])[0]) or {}
        return out
