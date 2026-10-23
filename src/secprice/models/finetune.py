"""S15 encoder fine-tuning (SPEC.md §7.2, §9.4): top-2 encoders by validation, only if G3 passes (or the owner
override, labelled exploratory). Fixed grid: lr {1e-5,2e-5,3e-5} x batch {8,16} x epochs <= 3 with early stopping
x weight decay {0, 0.01}. Class-weighted cross-entropy. Early stopping on an inner, time-ordered slice of the
training fold (never on the validation block); grid selection on the walk-forward validation block."""
from __future__ import annotations

import gc
import itertools
import math
import time

import numpy as np


def grid(cfg: dict) -> list[dict]:
    g = cfg["tuning"]["finetune_grid"]
    combos = [dict(lr=lr, batch=b, weight_decay=wd, epochs=int(g["epochs"]))
              for lr, b, wd in itertools.product(g["lr"], g["batch"], g["weight_decay"])]
    return combos[: int(g.get("max_trials", 12))]


def _class_weights(y: np.ndarray) -> np.ndarray:
    cnt = np.bincount(y.astype(int), minlength=3).astype(float)
    return np.where(cnt > 0, len(y) / (3 * np.maximum(cnt, 1)), 0.0)


def train_eval(model_id: str, revision: str | None, texts_tr, y_tr, w_tr, texts_iv, y_iv, texts_ev: list, hp: dict,
               max_len: int, seed: int, log=None) -> dict:
    """Fine-tune once; return per-step/epoch curves, inner-val best checkpoint predictions for each eval set."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_linear_schedule_with_warmup
    from .embeddings import load_model, load_tokenizer
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(model_id, revision)
    model = load_model(model_id, revision, "classifier", None, num_labels=3, ignore_mismatched_sizes=True).to(dev)
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()

    def enc(texts):
        e = tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    ids, am = enc(texts_tr)
    ds = TensorDataset(ids, am, torch.tensor(y_tr, dtype=torch.long), torch.tensor(w_tr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=int(hp["batch"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    iv_ids, iv_am = enc(texts_iv)
    cw = torch.tensor(_class_weights(np.asarray(y_tr)), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    total = len(dl) * int(hp["epochs"])
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), max(1, total))
    evals_per_epoch = 4
    eval_every = max(1, len(dl) // evals_per_epoch)
    curve = {"step": [], "train_loss": [], "eval_step": [], "eval_epoch": [], "inner_val_loss": [], "lr": []}
    best = {"loss": math.inf, "state": None, "step": 0}
    bad, patience, step = 0, 2, 0

    def predict(i, a, bs=64):
        model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(i), bs):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    lo = model(input_ids=i[k:k + bs].to(dev), attention_mask=a[k:k + bs].to(dev)).logits
                out.append(torch.softmax(lo.float(), -1).cpu().numpy())
        model.train()
        return np.vstack(out) if out else np.zeros((0, 3))

    model.train()
    stop = False
    t0 = time.time()
    for ep in range(int(hp["epochs"])):
        for bi, am_, yb, wb in dl:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=bi.to(dev), attention_mask=am_.to(dev)).logits
            loss_vec = torch.nn.functional.cross_entropy(logits.float(), yb.to(dev), weight=cw, reduction="none")
            loss = (loss_vec * wb.to(dev)).sum() / wb.to(dev).sum().clamp(min=1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            curve["step"].append(step)
            curve["train_loss"].append(float(loss.item()))
            curve["lr"].append(float(sch.get_last_lr()[0]))
            if step % eval_every == 0 or step == total:
                p = predict(iv_ids, iv_am)
                yv = np.asarray(y_iv).astype(int)
                vl = float(-np.mean(np.log(np.clip(p[np.arange(len(yv)), yv], 1e-9, 1)))) if len(yv) else math.nan
                curve["eval_step"].append(step)
                curve["eval_epoch"].append(step / max(1, len(dl)))
                curve["inner_val_loss"].append(vl)
                if vl < best["loss"] - 1e-4:
                    best = {"loss": vl, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "step": step}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience * evals_per_epoch // 2 + 1:
                        stop = True
                        break
        if stop:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    preds = [predict(*enc(t)) if len(t) else np.zeros((0, 3)) for t in texts_ev]
    p_tr = predict(ids[: min(len(ids), 4000)], am[: min(len(ids), 4000)])
    out = {"curve": curve, "best_step": best["step"], "best_inner_val_loss": best["loss"], "preds": preds,
           "train_proba_sample": p_tr, "train_y_sample": np.asarray(y_tr)[: len(p_tr)], "seconds": time.time() - t0,
           "early_stopped": stop}
    del model, opt, best
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out
