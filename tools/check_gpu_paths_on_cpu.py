"""CPU check of the GPU-stage code paths (S14 encoders, S15 fine-tuning loop, S16 constrained generation).

Uses the real core encoders at pinned revisions, a short fine-tuning run, and Qwen2.5-0.5B-Instruct (same family/chat
template as the 7B roster model) for the JSON-constrained generator. Run locally before spending Colab credits.
"""
import json
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, "src")
from secprice.models.embeddings import encode, load_encoder, unload  # noqa: E402
from secprice.models.finetune import train_eval  # noqa: E402
from secprice.rag.extract import QwenGenerator, field_schema, guard  # noqa: E402

rost = yaml.safe_load(open("configs/models.yaml"))
qb = yaml.safe_load(open("configs/question_bank_v1.yaml"))
texts = ["Net sales increased 8% driven by strong demand; we raised our full-year guidance.",
         "The company recorded a goodwill impairment of $250 million and announced a restructuring plan.",
         "Risk factors: litigation and regulatory investigations may adversely affect our results."] * 3

out = {}
for key in ("sec_bert", "finbert_tone", "chrono_bert", "bge_base"):
    sp = rost["encoders"][key]
    mid = sp.get("model_id") or sp["model_id_template"].format(year=2010)
    rev = sp.get("revision") or (sp.get("revisions") or {}).get("2010")
    t0 = time.time()
    tok, m, dev = load_encoder(mid, rev, "classifier" if sp.get("native_outputs") else "base")
    V, P = encode(tok, m, dev, texts, 512, 4, sp.get("pooling", "mean"), classifier=bool(sp.get("native_outputs")))
    out[key] = {"shape": list(V.shape), "finite": bool(np.isfinite(V.astype(np.float32)).all()),
                "native": None if P is None else P.round(3)[:3].tolist(), "s": round(time.time() - t0, 1)}
    unload(m)
print(json.dumps(out, indent=1))

y = np.array([0, 1, 2] * 3)
r = train_eval("nlpaueb/sec-bert-base", rost["encoders"]["sec_bert"]["revision"], texts, y, np.ones(9), texts[:3], y[:3],
               [texts[:4]], {"lr": 2e-5, "batch": 4, "weight_decay": 0.0, "epochs": 1}, 64, 0)
print("finetune ok:", r["preds"][0].shape, "steps", len(r["curve"]["step"]), "evals", len(r["curve"]["inner_val_loss"]))

gen = QwenGenerator("Qwen/Qwen2.5-0.5B-Instruct", None, field_schema(qb), max_new_tokens=700)
passages = [{"chunk_id": "c1", "text": texts[0]}, {"chunk_id": "c2", "text": texts[1]}]
t0 = time.time()
raw = gen(qb, passages)
clean, flags = guard(raw, qb, passages, field_schema(qb))
print("generator constrained:", gen.constrained, "| seconds", round(time.time() - t0, 1), "| fields", len(raw), "| flags", flags)
print(json.dumps({k: clean[k] for k in list(clean)[:3]}, indent=1)[:1500])
