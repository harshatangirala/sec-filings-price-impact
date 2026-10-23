"""Versioned 8-K item taxonomy: raw / legacy item -> category by effective date, with a catch-all
(SPEC.md §5.2, §18.1 #11)."""
from __future__ import annotations

import re

import pandas as pd


class Taxonomy8K:
    def __init__(self, spec: dict):
        self.spec = spec
        self.version = spec["version"]
        self.modern = {str(k): v for k, v in spec["modern"].items()}
        self.legacy = {str(k): v for k, v in spec["legacy"].items()}
        self.modern_effective = pd.Timestamp(spec["modern_effective"])
        self.item12_effective = pd.Timestamp(spec["legacy_item12_effective"])
        self.catch_all = spec["catch_all"]
        self.legacy_earn = {str(x) for x in spec["legacy_earnings_equivalents"]}
        self.title_patterns = {k: re.compile(v, re.I) for k, v in spec.get("title_patterns", {}).items()}

    @staticmethod
    def split_items(items_raw) -> list[str]:
        if items_raw is None or (isinstance(items_raw, float) and pd.isna(items_raw)):
            return []
        return [x.strip() for x in str(items_raw).replace(";", ",").split(",") if x.strip()]

    def regime(self, filing_date) -> str:
        return "modern" if pd.Timestamp(filing_date) >= self.modern_effective else "legacy"

    def map_item(self, item: str, filing_date) -> str:
        item = str(item).strip()
        if "." in item:
            # modern codes; zero-pad the decimal part ("2.2" -> "2.02")
            a, b = item.split(".", 1)
            item = f"{int(a)}.{int(b):02d}" if a.isdigit() and b.isdigit() else item
            return self.modern.get(item, self.catch_all)
        if item.isdigit():
            if self.regime(filing_date) == "modern":
                return self.catch_all
            if item == "12" and pd.Timestamp(filing_date) < self.item12_effective:
                return self.catch_all
            return self.legacy.get(str(int(item)), self.catch_all)
        return self.catch_all

    def categories(self, items: list[str], filing_date, titles: list[str] | None = None) -> list[str]:
        cats = [self.map_item(i, filing_date) for i in items]
        if not items and titles:
            for t in titles:
                hit = [k for k, rx in self.title_patterns.items() if rx.search(t)]
                cats += hit or [self.catch_all]
        cats = sorted(set(cats)) or [self.catch_all]
        return cats

    def earnings_candidate(self, items: list[str], filing_date) -> tuple[bool, bool]:
        """(is_modern_or_item12_earnings, is_legacy_equivalent_needing_content_check)."""
        modern = any(self.map_item(i, filing_date) == "earnings" for i in items)
        legacy_eq = (not modern and self.regime(filing_date) == "legacy"
                     and any(str(i).strip() in self.legacy_earn for i in items))
        return modern, legacy_eq
