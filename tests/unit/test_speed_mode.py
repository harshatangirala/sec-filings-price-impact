"""D-074: the speed switch changes only the GPU part and must never invalidate finished CPU-part results."""
from secprice import pipeline as P
from secprice.config import gpu_finetune_allowed, gpu_heads, gpu_trials, load_config
from secprice.runner import REGISTRY, _stage_key

OV = {"full": {"owner_opt_in_full": True, "require_pilot_success": False}}
CPU_PART1 = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S20", "S21", "S24"]


def test_fast_mode_swaps_gpu_settings_only():
    full = load_config("full", OV)
    fast = load_config("full", {**OV, "speed": {"mode": "fast"}})
    assert full["models"]["encoders"] == ["sec_bert", "finbert_tone", "chrono_bert", "bge_base"]
    assert fast["models"]["encoders"] == ["chrono_bert", "bge_base"]
    assert fast["tuning"] == full["tuning"]                                   # the finished S11 inputs are untouched
    assert gpu_heads(full) == ["lightgbm", "catboost", "enet"] and gpu_heads(fast) == ["lightgbm", "enet"]
    assert gpu_trials(fast) == {"lightgbm": 8, "enet": 4} and gpu_trials(full)["catboost"] == 6
    assert gpu_finetune_allowed(full) and not gpu_finetune_allowed(fast)


def test_fast_mode_does_not_change_finished_stage_hashes(tmp_path):
    a = P.make_runner("full", OV, tmp_path, tmp_path / "h")
    b = P.make_runner("full", {**OV, "speed": {"mode": "fast"}}, tmp_path, tmp_path / "h")
    for sid in CPU_PART1:
        assert a.stage_hash(sid) == b.stage_hash(sid), f"{sid} would be re-run when the speed mode changes"
    for sid in ("S13", "S14", "S15"):
        assert a.stage_hash(sid) != b.stage_hash(sid), f"{sid} must depend on the speed mode"
    assert [s for s in sorted(REGISTRY, key=_stage_key) if s in CPU_PART1]


def test_invalid_speed_mode_rejected():
    import pytest
    from secprice.config import ConfigError
    with pytest.raises(ConfigError):
        load_config("full", {"speed": {"mode": "turbo"}})


def test_benchmark_excludes_warmup_and_keeps_best_pass(monkeypatch):
    """D-075 (found on a real A100 run): the first call carries one-off GPU start-up cost; the old benchmark timed it and
    projected 19.3 GPU-hours from ~190 chunks/s."""
    import time as _t
    from secprice.models import embeddings as E
    calls = []

    def fake_encode(tok, model, dev, texts, max_len, batch, pooling, **kw):
        calls.append(len(texts))
        _t.sleep(0.6 if len(calls) == 1 else 0.02 * (len(texts) / 128))      # slow first call, steady afterwards

    monkeypatch.setattr(E, "encode", fake_encode)
    rate = E.benchmark(None, None, "cuda", n=512, batch=128)
    assert calls == [256, 512, 512]                                             # warm-up (2 batches), then two timed passes
    assert rate > 5000, rate                                                    # steady state: 512 / 0.08 s, not 512 / 0.68 s
