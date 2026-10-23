"""Write the repository documents the spec places under docs/ (SPEC.md §14): one stage card per stage (purpose, inputs,
outputs, assertions, how to read) and the annotation guide. Run-specific cards (with run summaries) are still written next
to the artifacts in DATA_ROOT/<profile>/stage_cards/.

    python tools/write_docs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secprice import pipeline  # noqa: E402,F401  (registers the stages)
from secprice.config import load_config, load_side_config  # noqa: E402
from secprice.gold.tools import annotation_guide  # noqa: E402
from secprice.runner import REGISTRY, _stage_key  # noqa: E402


def main() -> None:
    cards = ROOT / "docs" / "stage_cards"
    cards.mkdir(parents=True, exist_ok=True)
    for sid in sorted(REGISTRY, key=_stage_key):
        s = REGISTRY[sid]
        lines = [f"# {s.id} {s.name}", "", f"- Part: {s.part}", f"- Depends on: {', '.join(s.deps) or 'none'}",
                 f"- Config keys in the stage hash: {', '.join(f'`{k}`' for k in s.config_keys) or 'none'}",
                 f"- Outputs: {', '.join(f'`{o}`' for o in s.outputs) or 'see the stage folder'}",
                 f"- Assertions: {'; '.join(s.assertions) or 'none'}", "", "## How to read", s.card or "-", "",
                 "Run-specific summaries: `DATA_ROOT/<profile>/stage_cards/" + s.id + ".md`."]
        (cards / f"{s.id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    qb = load_side_config(load_config("smoke"), "question_bank_v1.yaml")
    (ROOT / "docs" / "annotation_guide.md").write_text(annotation_guide(qb), encoding="utf-8", newline="\n")
    print(f"wrote {len(REGISTRY)} stage cards and docs/annotation_guide.md")


if __name__ == "__main__":
    main()
