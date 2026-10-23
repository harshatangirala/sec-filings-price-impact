"""Gold set (Track C, SPEC.md §12; addendum P1.6): stratified sampling from training-period filings, disjoint
gold-dev / gold-val / gold-test partitions, CSV/Excel templates + annotation guide, validated import, Cohen's
kappa, and a code-enforced single use of gold-test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_COLS = ["gold_split", "item_id", "event_id", "accession", "form", "filing_date", "section_id", "field",
                 "question", "present", "direction", "magnitude", "unit", "certainty", "evidence_quote",
                 "evidence_chunk_id", "annotator", "notes", "llm_prefill_json"]
PASSAGE_COLS = ["gold_split", "item_id", "event_id", "chunk_id", "assigned_label", "passage_text", "supports_label",
                "annotator", "notes"]


def era(year: int) -> str:
    return "2000-08" if year <= 2008 else ("2009-19" if year <= 2019 else "2020+")


def sample_gold(events: pd.DataFrame, train_end: pd.Timestamp, sizes: dict, seed: int) -> pd.DataFrame:
    """Stratify by form x era x category x sector over training-period events; oversample rare categories."""
    ev = events[pd.to_datetime(events["decision_date"]) < train_end].copy()
    ev["era"] = ev["year"].map(era)
    ev["cat0"] = ev["categories"].map(lambda c: sorted(c)[0] if len(c) else "periodic")
    ev["stratum"] = ev["event_type"] + "|" + ev["era"] + "|" + ev["cat0"].astype(str) + "|" + ev["ff12_industry"].fillna("Other")
    rng = np.random.default_rng(seed)
    total = sum(sizes.values())
    counts = ev["stratum"].value_counts()
    # inverse-sqrt frequency weights oversample rare strata; sampling weights kept for prevalence estimates
    w = ev["stratum"].map(lambda s: 1 / np.sqrt(counts[s]))
    p = (w / w.sum()).to_numpy()
    k = min(total, len(ev))
    idx = rng.choice(len(ev), size=k, replace=False, p=p)
    s = ev.iloc[idx].copy()
    s["sampling_weight"] = 1.0 / (p[idx] * len(ev))
    s = s.sample(frac=1.0, random_state=seed)
    split = []
    for name, n in sizes.items():
        split += [name] * n
    s["gold_split"] = (split + ["unused"] * len(s))[: len(s)]
    s = s[s.gold_split != "unused"]
    assert not s.duplicated("event_id").any(), "a filing may appear in only one gold partition"
    return s


def export_templates(sample: pd.DataFrame, qb: dict, chunks: pd.DataFrame | None, out_dir: Path,
                     prefill: dict | None = None, n_passages: int = 200, seed: int = 0) -> dict:
    """Filings x fields template; passage template (supports-label); gold-test never gets LLM pre-fill."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sample.itertuples(index=False):
        for f, spec in qb["fields"].items():
            pf = ""
            if prefill and r.gold_split in ("gold_dev", "gold_val"):
                pf = json.dumps((prefill.get(r.event_id) or {}).get(f, {}))
            rows.append({"gold_split": r.gold_split, "item_id": f"{r.event_id}:{f}", "event_id": r.event_id,
                         "accession": r.accessions[0], "form": r.forms[0], "filing_date": str(r.filing_date),
                         "section_id": "", "field": f, "question": spec["question"], "present": "", "direction": "",
                         "magnitude": "", "unit": "", "certainty": "", "evidence_quote": "", "evidence_chunk_id": "",
                         "annotator": "", "notes": "", "llm_prefill_json": pf})
    tmpl = pd.DataFrame(rows, columns=TEMPLATE_COLS)
    paths = {}
    for split, g in tmpl.groupby("gold_split"):
        p = out_dir / f"gold_{split}_filings_template.csv"
        g.to_csv(p, index=False)
        paths[f"{split}_filings"] = str(p)
    try:
        tmpl.to_excel(out_dir / "gold_filings_template.xlsx", index=False)
        paths["xlsx"] = str(out_dir / "gold_filings_template.xlsx")
    except Exception:
        pass
    if chunks is not None and len(chunks):
        rng = np.random.default_rng(seed)
        prow = []
        for split in ("gold_val", "gold_test"):
            ids = set(sample[sample.gold_split == split].event_id)
            c = chunks[chunks.event_id.isin(ids)]
            if c.empty:
                continue
            c = c.iloc[rng.choice(len(c), size=min(n_passages, len(c)), replace=False)]
            for x in c.itertuples(index=False):
                prow.append({"gold_split": split, "item_id": f"{x.event_id}:{x.section_id}:{x.chunk_no}",
                             "event_id": x.event_id, "chunk_id": f"{x.event_id}:{x.section_id}:{x.chunk_no}",
                             "assigned_label": "", "passage_text": x.text[:3000], "supports_label": "",
                             "annotator": "", "notes": ""})
        pt = pd.DataFrame(prow, columns=PASSAGE_COLS)
        p = out_dir / "gold_passages_template.csv"
        pt.to_csv(p, index=False)
        paths["passages"] = str(p)
    (out_dir / "ANNOTATION_GUIDE.md").write_text(annotation_guide(qb), encoding="utf-8")
    paths["guide"] = str(out_dir / "ANNOTATION_GUIDE.md")
    return paths


def annotation_guide(qb: dict) -> str:
    lines = ["# Annotation guide (gold set v1)", "",
             "Label each filing x field from the filing text only. Never look at prices or returns: price outcomes "
             "never validate semantic labels (SPEC.md §12).", "",
             "- `present`: TRUE / FALSE.", "- `direction`: one of the listed values, or blank.",
             "- `magnitude` + `unit`: the stated number and unit (e.g. 250, USD millions), or blank.",
             "- `certainty`: low / medium / high.", "- `evidence_quote`: a verbatim quote of <= 30 words supporting the label.", "",
             "**gold-test is labelled blind**: its template has no LLM pre-fill. gold-dev and gold-val may show a pre-fill "
             "column; treat it as a suggestion and correct it (anchoring risk noted in the report).", "",
             "A second annotator should label a stratified >= 30% of gold-test independently (columns `annotator`), so "
             "Cohen's kappa can be reported. Without one, all content-validity results are labelled single-rater, provisional.",
             "", "## Fields"]
    for k, f in qb["fields"].items():
        lines.append(f"- **{k}** - {f['question']} Directions: {', '.join(f['directions'])}.")
    lines += ["", "Return the completed CSV files to `DATA_ROOT/inputs/gold/` with the same file names."]
    return "\n".join(lines)


def import_gold(folder: Path, sample: pd.DataFrame, qb: dict) -> tuple[pd.DataFrame, list[str]]:
    errs = []
    frames = []
    for p in sorted(Path(folder).glob("gold_*_filings_template*.csv")):
        d = pd.read_csv(p, dtype=str).fillna("")
        missing = set(TEMPLATE_COLS) - set(d.columns)
        if missing:
            errs.append(f"{p.name}: missing columns {sorted(missing)}")
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=TEMPLATE_COLS), ["no completed gold files found"]
    g = pd.concat(frames, ignore_index=True)
    g = g[g["present"].astype(str).str.strip() != ""]
    valid_ids = set(sample.event_id)
    bad = ~g.event_id.isin(valid_ids)
    if bad.any():
        errs.append(f"{int(bad.sum())} rows reference events outside the frozen gold sample")
        g = g[~bad]
    split_of = dict(zip(sample.event_id, sample.gold_split))
    mism = g.gold_split != g.event_id.map(split_of)
    if mism.any():
        errs.append(f"{int(mism.sum())} rows changed gold partition (rejected)")
        g = g[~mism]
    for f, spec in qb["fields"].items():
        m = (g.field == f) & (g.direction != "") & ~g.direction.isin(spec["directions"])
        if m.any():
            errs.append(f"{int(m.sum())} invalid directions for {f}")
    g["present_b"] = g["present"].str.strip().str.lower().isin(["true", "1", "yes", "y"])
    return g, errs


def cohen_kappa(a, b) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    if len(a) < 5:
        return None
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        return None


class GoldTestLock:
    """gold-test may be evaluated once, after the pipeline freeze (addendum P1.6)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def used(self) -> dict | None:
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def consume(self, frozen_hash: str) -> None:
        if self.path.exists():
            raise RuntimeError(f"gold-test already used: {self.path.read_text()}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"used_at": pd.Timestamp.utcnow().isoformat(), "frozen_hash": frozen_hash}))
