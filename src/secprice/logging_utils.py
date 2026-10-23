"""Structured logs per stage, errors.log and error-rate tracking (SPEC.md §19, §13.3)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    log = logging.getLogger(f"secprice.{name}")
    if getattr(log, "_secprice_ready", False):
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    log.addHandler(sh)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        log.addHandler(fh)
        eh = logging.FileHandler(log_dir / "errors.log", encoding="utf-8")
        eh.setLevel(logging.ERROR)
        eh.setFormatter(logging.Formatter(_FMT))
        log.addHandler(eh)
    log._secprice_ready = True  # type: ignore[attr-defined]
    return log


class ErrorRateExceeded(RuntimeError):
    pass


class StageIncomplete(RuntimeError):
    """A resumable stage stopped cleanly at its wall-clock budget; run it again to continue (nothing is lost)."""


class ErrorTracker:
    """Counts successes/failures; raises when the failure share exceeds budget.max_error_rate after min_n items."""

    def __init__(self, max_rate: float, min_n: int = 50, name: str = ""):
        self.max_rate, self.min_n, self.name = float(max_rate), int(min_n), name
        self.ok = 0
        self.fail = 0
        self.reasons: dict[str, int] = {}

    def success(self) -> None:
        self.ok += 1

    def failure(self, reason: str) -> None:
        self.fail += 1
        key = str(reason)[:80]
        self.reasons[key] = self.reasons.get(key, 0) + 1
        n = self.ok + self.fail
        if n >= self.min_n and self.fail / n > self.max_rate:
            raise ErrorRateExceeded(f"{self.name}: error rate {self.fail}/{n} > {self.max_rate}; "
                                    f"top reasons {json.dumps(self.top())}")

    @property
    def rate(self) -> float:
        n = self.ok + self.fail
        return self.fail / n if n else 0.0

    def top(self, k: int = 5) -> dict:
        return dict(sorted(self.reasons.items(), key=lambda kv: -kv[1])[:k])

    def summary(self) -> dict:
        return {"ok": self.ok, "fail": self.fail, "rate": round(self.rate, 4), "top_reasons": self.top()}
