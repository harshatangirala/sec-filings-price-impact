"""Build notebooks/SEC_Filings_Price_Impact_v5_EndToEnd.ipynb from the repository (single source of truth).

The notebook embeds the package, configs, tests and docs as %%writefile cells, so it runs on Colab without cloning.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "SEC_Filings_Price_Impact_v5_EndToEnd.ipynb"

EMBED = sorted([p for p in (ROOT / "src" / "secprice").rglob("*.py")] +
               [p for p in (ROOT / "configs").rglob("*.yaml")] +
               [p for p in (ROOT / "tests").rglob("*.py")] +
               [p for p in (ROOT / "tests" / "fixtures").glob("*") if p.is_file()] +
               [ROOT / "pyproject.toml", ROOT / "SPEC.md", ROOT / "SPEC_ADDENDUM.md", ROOT / "CLAUDE.md",
                ROOT / "README.md", ROOT / "docs" / "decision_log.md"])


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def build():
    nb = nbf.v4.new_notebook()
    C = []
    C.append(md(f"""
# SEC filings → price impact — end-to-end study (SPEC v5 + pre-coding addendum)

**What this notebook does.** It builds a point-in-time event panel of **10-K, 10-Q and 8-K filings** for up to **500 US firms**
that are common to our OHLCV files and the Hugging Face 10-K corpus (`JanosAudran/financial-reports-sec`), from **Jan 2000 to
Jul 2026**. On that panel it runs three separate tracks:

| Track | Question | Main outputs |
|---|---|---|
| **A — event study** | What happens to prices, volume and volatility around disclosures? (association only) | CAR paths, pre-trends, SUE-conditioned drift, verdicts |
| **B — prediction** | What could have been predicted at decision time? Primary target **T_trade h=1** (Up / Sideways / Down) | ablation ladder L0–L8, tuned heads, fine-tuned encoders, stacking, dev holdout, locked test, backtest |
| **C — content validity** | Are sections, item mappings and extracted events correct? (human gold labels only) | gold templates, extraction F1 / faithfulness |

**How to run.** Set the owner settings in the first code cell, then *Runtime → Run all*. Every stage is resumable: re-running skips
stages whose `_SUCCESS` marker matches; the EDGAR download (S05) checkpoints per firm.

1. **smoke** (default, hard rule 4): 5 firms, 2021–2023, ~30–60 min on CPU. Verifies the whole chain on real data.
2. **pilot**: ~40 sector-balanced firms from 2012 — a few hours (download dominated).
3. **full**: 500 firms, 2000–2026. Gate **G2** requires a completed pilot and `owner_opt_in_full: True`. Expect **~10–14 h**
   for the EDGAR download/parse on a CPU runtime (fair-access limit 6 requests/s), then **~3–6 h of A100** for embeddings,
   tuning and fine-tuning. Recommended: run **Part 1 on a CPU (High-RAM) runtime**, switch to **A100** for Part 2, back to CPU for
   Part 3 (`PART` setting). Restart the runtime when switching.

**Rules this notebook enforces** (SPEC.md §0.2): SEC filings only as text; no news; four timestamps per event; features available no
later than decision time; the report contains computed values only; association ≠ causation; prediction ≠ profitability;
results are *validation-selected*, never a global optimum; the locked test is scored **once**, and only when you unseal it.

Repository build: `{commit()}` · Artifacts: `MyDrive/sec-price-study/<profile>/…` · Report: `…/<profile>/reports/results_report.html`
"""))
    C.append(md("## 0. Owner settings"))
    C.append(code("""
RUN_PROFILE = "smoke"      # "smoke" -> "pilot" -> "full"   (hard rule 4: smoke is the default)
PART = "all"               # "cpu1" (S00-S12, S20-S21) | "gpu" (S13-S17) | "cpu2" (S18-S24) | "all"
DRIVE_FOLDER = "sec-price-study"

# Study window: configs/default.yaml sets Jan 2000 - Jul 2026 for pilot/full (owner); smoke uses 2021-2023.
OWNER_OVERRIDES = {
    "extensions": {"ext_stacking": True},                 # owner: ensemble of the best models (H5), gated by G3c
    "gates": {"owner_override_g3_finetune": False},       # True = fine-tune even if G3 fails (results labelled EXPLORATORY)
    "rag": {"run_extraction_without_gold": False},        # True = teacher/student extraction without gold-val (EXPLORATORY)
    "lockbox": {"unlock_locked_test": False,              # True only after reviewing the development-holdout results
                "acknowledge_reuse": False},
    "full": {"owner_opt_in_full": False, "require_pilot_success": True},
}
"""))
    C.append(md("""
## 1. Environment
Colab: mounts Google Drive and reads the SEC User-Agent from **Colab secrets** (🔑 icon → add `SEC_USER_AGENT` with value
`Your Name your.email@example.com`; optional `HF_TOKEN`). The User-Agent is never hard-coded (hard rule 13).
"""))
    C.append(code(f"""
import os, sys, json, time, subprocess, importlib, shutil
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    DATA_ROOT = Path("/content/drive/MyDrive") / DRIVE_FOLDER
    HOT_ROOT = Path("/content/hot")
    REPO = Path("/content/sec-filings-price-impact")
    for k in ("SEC_USER_AGENT", "HF_TOKEN"):
        try:
            v = userdata.get(k)
            if v: os.environ[k] = v
        except Exception:
            pass
else:
    DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.cwd() / "sec-price-study"))
    HOT_ROOT = Path(os.environ.get("HOT_ROOT", DATA_ROOT / "_hot"))
    REPO = Path(os.environ.get("SECPRICE_NB_REPO", Path.cwd() / "sec-filings-price-impact"))
for p in (DATA_ROOT, HOT_ROOT, REPO):
    p.mkdir(parents=True, exist_ok=True)
os.environ.update({{"DATA_ROOT": str(DATA_ROOT), "HOT_ROOT": str(HOT_ROOT), "SECPRICE_REPO": str(REPO),
                    "SECPRICE_BUILD_COMMIT": "{commit()}", "TOKENIZERS_PARALLELISM": "false"}})
print("DATA_ROOT", DATA_ROOT, "| HOT_ROOT", HOT_ROOT, "| REPO", REPO)
print("SEC_USER_AGENT set:", bool(os.environ.get("SEC_USER_AGENT")))
try:
    import torch; print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch not available:", e)
"""))
    C.append(md("Install only what is missing, pinned against the runtime's own core libraries (torch / pandas / numpy are never reinstalled)."))
    C.append(code("""
need = {"catboost": "catboost", "optuna": "optuna", "arch": "arch", "pandas_market_calendars": "pandas_market_calendars",
        "reportlab": "reportlab", "lmformatenforcer": "lm-format-enforcer", "yfinance": "yfinance", "lightgbm": "lightgbm",
        "statsmodels": "statsmodels", "markdown": "markdown", "jsonschema": "jsonschema", "openpyxl": "openpyxl",
        "transformers": "transformers", "pytest": "pytest", "nbformat": "nbformat", "lxml": "lxml"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
pins = []
for mod in ("torch", "pandas", "numpy", "scipy", "sklearn", "pyarrow"):
    try:
        m = importlib.import_module(mod)
        pins.append(f"{'scikit-learn' if mod == 'sklearn' else mod}=={m.__version__.split('+')[0]}")
    except Exception:
        pass
Path("/tmp").mkdir(exist_ok=True)
Path("/tmp/secprice_constraints.txt").write_text("\\n".join(pins))
if missing:
    print("installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/secprice_constraints.txt", *missing], check=True)
import transformers
print("ready; transformers", transformers.__version__, "| missing installed:", missing)
"""))
    C.append(md("""
## 2. Package source (embedded from the repository — do not edit here)
The cells below write the `secprice` package, its configs, tests and documents to `REPO`. They are generated from the
repository by `tools/build_notebook.py`, so the notebook and the tested code are identical.
"""))
    dirs = sorted({str(p.relative_to(ROOT).parent).replace("\\", "/") for p in EMBED})
    C.append(code("for d in " + repr(dirs) + ":\n    (REPO / d).mkdir(parents=True, exist_ok=True)\nprint(len(" + repr(dirs) + "), 'folders ready')"))
    for p in EMBED:
        rel = p.relative_to(ROOT).as_posix()
        txt = p.read_text(encoding="utf-8", errors="replace")
        C.append(nbf.v4.new_code_cell(f"%%writefile $REPO/{rel}\n{txt}"))      # byte-exact (no stripping)
    C.append(code("""
sys.path.insert(0, str(REPO / "src"))
for m in [m for m in list(sys.modules) if m.startswith("secprice")]:
    del sys.modules[m]
import secprice, pandas as pd
from secprice import pipeline as P
from secprice.storage import code_hash
P.quiet()
pd.set_option("display.max_columns", 40, "display.width", 220, "display.max_colwidth", 80)
print("secprice", secprice.__version__, "| code hash", code_hash())

def run(stages, force=False):
    t0 = time.time()
    r = P.run(stages, RUN_PROFILE, OWNER_OVERRIDES, force=force, data_root=DATA_ROOT, hot_root=HOT_ROOT)
    print(f"{r}  ({time.time() - t0:.0f}s)")
    return r

def art(stage, name):
    rr = P.make_runner(RUN_PROFILE, OWNER_OVERRIDES, DATA_ROOT, HOT_ROOT)
    return rr.storage.profile_root / stage / rr.stage_hash(stage) / name

def show(stage, name, n=20):
    p = art(stage, name)
    if not p.exists():
        print("(missing)", p); return None
    if p.suffix == ".parquet":
        d = pd.read_parquet(p)
    elif p.suffix == ".csv":
        try:
            d = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print("(empty table)", p.name); return None
    elif p.suffix == ".json":
        print(json.dumps(json.loads(p.read_text()), indent=1, default=str)[:4000]); return None
    else:
        print(p.read_text()[:4000]); return None
    display(d.head(n)); return d

def card(stage):
    p = DATA_ROOT / RUN_PROFILE / "stage_cards" / f"{stage}.md"
    print(p.read_text()[:3000] if p.exists() else "(no stage card yet)")
"""))
    C.append(md("""
## 3. Gate G1 — unit tests + offline integration smoke (must pass before any GPU stage)
43 unit tests cover the spec's §18.1 list and every addendum acceptance test (timing, prospective target, interval-matched
benchmark, SUE history counts, gold-test lock). The offline smoke runs S00→S12 and S20→S24 on a synthetic world with no network.
A pass is recorded per code hash, so it is not repeated after a runtime restart.
"""))
    C.append(code("""
g1 = DATA_ROOT / "_g1" / f"g1_pass_{code_hash()}.json"
if g1.exists():
    print("G1 already passed for this code:", g1.read_text())
else:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(REPO / "tests")], cwd=str(REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, "Gate G1 failed: fix before running GPU stages"
    g1.parent.mkdir(parents=True, exist_ok=True)
    g1.write_text(json.dumps({"passed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": r.stdout.strip().splitlines()[-1]}))
    print("G1 passed")
"""))
    C.append(md("""
## 4. Inputs (`DATA_ROOT/inputs`)
| File | Content | Source |
|---|---|---|
| `ohlcv_1day/<TICKER>.parquet` | our 979 verified daily OHLCV series (split-adjusted) | TriFrame `final/1day` |
| `yf_frozen.parquet` | yfinance warm-up bars 1998–2001, all dividends/splits, SPY | frozen 2026-10-22 (explicit parameters) |
| `lexicons/Loughran-McDonald_MasterDictionary_1993-2025.csv` | LM dictionary (academic use) | SRAF Notre Dame |
| `hf_census_5da7e3c8b920.parquet` | metadata census of the HF 10-K corpus (no labels) | built by `secprice.audit.hf_audit` |

All four were prepared and copied to `MyDrive/sec-price-study/inputs`. Missing optional files are rebuilt automatically
(the census by streaming 13 GB of HF shards; the LM dictionary from SRAF).
"""))
    C.append(code("""
inp = DATA_ROOT / "inputs"
chk = {"ohlcv_1day files": len(list((inp / "ohlcv_1day").glob("*.parquet"))) if (inp / "ohlcv_1day").exists() else 0,
       "yf_frozen.parquet": (inp / "yf_frozen.parquet").exists(),
       "LM dictionary": (inp / "lexicons" / "Loughran-McDonald_MasterDictionary_1993-2025.csv").exists(),
       "HF census": (inp / "hf_census_5da7e3c8b920.parquet").exists()}
print(json.dumps(chk, indent=1))
assert chk["ohlcv_1day files"] > 0, "copy the OHLCV parquet files to DATA_ROOT/inputs/ohlcv_1day first"
display(P.status(RUN_PROFILE, OWNER_OVERRIDES, DATA_ROOT))
"""))
    # ---------------- Part 1
    C.append(md("""
# Part 1 — data, events, targets, features, baselines, Track A (CPU)
"""))
    p1 = [
        ("S00 environment · S01 Hugging Face audit · S02 universe",
         "S02 maps our tickers to SEC CIKs, keeps firms that are also in the HF 10-K corpus and file 10-K/10-Q, then selects "
         "S&P-500 members first and fills the rest with a seeded, sector-stratified draw (no price returns are read).",
         "run(['S00', 'S01', 'S02'])\nu = show('S02', 'universe.parquet', 5)\nshow('S02', 'firms_per_year.csv', 30)\n"
         "print(u.selected.sum() if u is not None else None, 'firms selected')"),
        ("S03 calendar, prices, corporate actions, factors",
         "NYSE sessions (holidays, early closes), our split-adjusted bars + yfinance warm-up, raw / split-adjusted / total-return "
         "series, SPY, Kenneth French FF5 + momentum.", "run(['S03'])\nshow('S03', 'price_coverage.csv', 10)"),
        ("S04 EDGAR index + coverage audit",
         "Every 10-K / 10-Q / 8-K in the window with corrected acceptance times (verified rule; SGML header sample), periodic "
         "expectations and reason codes. 8-Ks carry no expectation.",
         "run(['S04'])\nshow('S04', 'data_coverage_summary.md')\nshow('S04', 'ts_quality_by_year.csv', 30)"),
        ("S05 download + parse + taxonomy + anonymize (long, resumable)",
         "Downloads primary documents / full 8-K submissions at ≤ 6 requests/s with your User-Agent, extracts the target sections "
         "(Exhibit-13 fallback for MD&A by reference), masks firm names and tickers. If the runtime disconnects, re-run this cell: "
         "finished firms are skipped.", "run(['S05'])\nshow('S05', 'parse_qa.csv', 40)\ncard('S05')"),
        ("S06 XBRL + release EPS + SUE",
         "Company facts with public-availability times, as-first-reported quarterly EPS, EPS parsed from earnings-release "
         "exhibits, cluster-aware SUE (history strictly before decision time).",
         "run(['S06'])\nshow('S06', 'sue_coverage.csv', 40)\ncard('S06')"),
        ("S07 events + four timestamps · S08 targets and labels",
         "Events (periodic filings, same-day 8-K groups), session buckets, earnings types, clusters, overlaps; T_trade (primary, "
         "entry at the first open at/after decision time) and T_info targets.",
         "run(['S07', 'S08'])\nshow('S07', 'event_counts.csv')\nshow('S07', 'usable_cutoff.json')\nshow('S08', 'target_summary.csv', 40)"),
        ("S09 features · S10 splits, cohorts and the leakage audit",
         "Price, metadata, accounting, Loughran-McDonald, filing-change and novelty features (availability ≤ decision time). "
         "Walk-forward folds → development holdout → locked test with purge and embargo; 24-item leakage audit and four planted leaks.",
         "run(['S09', 'S10'])\nshow('S10', 'blocks.csv')\nshow('S10', 'leakage_audit.csv', 30)\nshow('S10', 'planted_leaks.json')"),
        ("S11 baselines L0–L4, L6 (default and tuned)",
         "Majority, price, metadata, accounting and lexical baselines and the best structured set, with Optuna tuning on walk-forward "
         "macro-F1, loss curves, learning curves, adversarial validation, permutation null and the elastic-net convexity certificate.",
         "run(['S11'])\nshow('S11', 'ladder_summary.csv', 40)\nshow('S11', 'fit_diagnosis.csv', 40)\nshow('S11', 'enet_certificate.json')"),
        ("S12 gold-set export / import (Track C)",
         "Stratified gold sample from training-period filings and annotation templates in Drive. Annotate them (gold-test blind), put "
         "the completed CSVs in `inputs/gold/`, then re-run S12 with force=True.", "run(['S12'])\nshow('S12', 'gold_status.json')"),
        ("S20 Track A event study · S21 drift tests",
         "CARs, abnormal volume and volatility by form / 8-K category / bucket, all vs clean events, pre-trends, correlations, topics; "
         "SUE-conditioned drift and other tests with the pre-registered verdict rubric.",
         "run(['S20', 'S21'])\nshow('S21', 'hypothesis_tests.csv', 30)"),
    ]
    for title, expl, body in p1:
        C.append(md(f"### {title}\n{expl}"))
        C.append(code("if PART in ('all', 'cpu1'):\n" + "\n".join("    " + l for l in body.splitlines())))
    # ---------------- Part 2
    C.append(md("""
# Part 2 — embeddings, tuned heads, G3, fine-tuning, RAG (GPU, A100)
Switch the runtime to **A100** (Runtime → Change runtime type), restart, set `PART = "gpu"` and *Run all* (stages already done
are skipped). On CPU these stages run only for the smoke profile.
"""))
    p2 = [
        ("S13 preflight", "Workload, storage and GPU-hour projection from a timed micro-benchmark; refuses `full` over budget.",
         "run(['S13'])\nshow('S13', 'preflight.json')"),
        ("S14 frozen encoder embeddings", "SEC-BERT, FinBERT-tone (with native tone), ChronoBERT (vintage Y−1 for year-Y events) and "
         "BGE-base; one model in memory at a time; mean/max pooled section vectors; change cosine.", "run(['S14'])\ncard('S14')"),
        ("S15 L5 / L7 heads, G3 and fine-tuning", "Embedding-only and structured+embedding heads per encoder (same rows, heads, budget), "
         "H4 vintage-vs-static, G3 text-lift gate, and — only if G3 passes or the owner override is set — fine-tuning of the top-2 "
         "encoders with the fixed grid (loss curves saved).",
         "run(['S15'])\nshow('S15', 'g3.json')\nshow('S15', 'emb_ladder_summary.csv', 40)\nshow('S15', 'h4.csv', 20)\nshow('S15', 'finetune_status.json')"),
        ("S16 core RAG · S17 teacher/student distillation", "BM25 / hybrid / hybrid+rerank retrieval and Qwen2.5-7B extraction scored on "
         "gold-val; G3b; LLM pre-fill for gold-dev/val templates. S17 runs only after G3b passes (or the exploratory owner flag).",
         "run(['S16', 'S17'])\nshow('S16', 'rag_tournament.csv')\nshow('S16', 'g3b.json')\nshow('S17', 'student_report.json')"),
    ]
    for title, expl, body in p2:
        C.append(md(f"### {title}\n{expl}"))
        C.append(code("if PART in ('all', 'gpu'):\n" + "\n".join("    " + l for l in body.splitlines())))
    # ---------------- Part 3
    C.append(md("""
# Part 3 — development holdout, G4 freeze, locked test, backtest, Track C, report (CPU)
"""))
    C.append(md("### Freeze the pre-registration (required before the development holdout is scored)"))
    C.append(code("if PART in ('all', 'cpu2'):\n    print('prereg hash:', P.prereg_freeze(RUN_PROFILE, OWNER_OVERRIDES, DATA_ROOT))"))
    p3 = [
        ("S18 development holdout, MCS / PBO, stacking (H5) and the G4 freeze",
         "Finalists retrained on all pre-holdout data and scored on 2020–2022; Model Confidence Set and PBO as diagnostics only; "
         "the stacking extension; the frozen architecture is hashed.",
         "run(['S18'])\nshow('S18', 'dev_lifts.csv')\nshow('S18', 'frozen_architecture.json')\nshow('S18', 'review_triggers.json')"),
        ("S19 locked test (single use)",
         "Runs only when `lockbox.unlock_locked_test` is True. A second scoring is refused unless `acknowledge_reuse` is True, which "
         "relabels the block as development.",
         "if OWNER_OVERRIDES['lockbox'].get('unlock_locked_test'):\n    run(['S19'])\n    show('S19', 'locked_results.parquet', 40)\n"
         "else:\n    print('Locked test sealed (set lockbox.unlock_locked_test = True after reviewing the development results).')"),
        ("S22 backtest (economic-relevance check)",
         "Signal P(Up) − P(Down) of the frozen T_trade model; training-period 80/20 thresholds; entry at the first open after decision "
         "time; 10/20 bps costs; long-only vs interval-matched SPY and hypothetical long-short.",
         "run(['S22'])\nshow('S22', 'backtest.csv', 30)"),
        ("S23 Track C final (gold-test, once) · S24 figures and the full report",
         "Content validity on human gold-test labels (when present), then all figures F01–F40 and the full extracted report "
         "(Markdown, HTML with embedded figures, PDF), built from computed artifacts only.",
         "run(['S23'])\nrun(['S24'], force=True)"),
    ]
    for title, expl, body in p3:
        C.append(md(f"### {title}\n{expl}"))
        C.append(code("if PART in ('all', 'cpu2'):\n" + "\n".join("    " + l for l in body.splitlines())))
    C.append(md("## Results — executive summary, smallest defensible claim, and every figure"))
    C.append(code("""
from IPython.display import Markdown, Image
rep = DATA_ROOT / RUN_PROFILE / "reports"
md_ = (rep / "results_summary.md").read_text(encoding="utf-8") if (rep / "results_summary.md").exists() else "(report not built yet)"
display(Markdown(md_.split("## 1. Data integrity")[0]))
idx = json.loads((rep / "figures" / "figure_index.json").read_text()) if (rep / "figures" / "figure_index.json").exists() else []
for f in idx:
    if f["status"] == "ok":
        display(Markdown(f"**{f['id']} — {f['title']}.** {f['caption']}"))
        display(Image(filename=f["path"]))
    else:
        print(f"{f['id']} {f['title']}: {f['status']}")
print("Full report:", rep / "results_report.html", "|", rep / "results_report.pdf", "|", rep / "results_summary.md")
"""))
    C.append(code("display(Markdown(md_))"))
    nb["cells"] = C
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                      "accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT, len(C), "cells", round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    build()
