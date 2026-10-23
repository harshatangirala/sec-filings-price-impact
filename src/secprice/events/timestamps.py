"""The four timestamps per event (SPEC.md §5.1; addendum P0.3).

Verified 2026-10-23 against SGML headers (docs/decision_log.md D-003, D-047): the submissions-JSON
`acceptanceDateTime` (suffix 'Z') follows TWO conventions. About 70% of 8,789 pilot filings are the Eastern wall time
shifted by the ET->UTC offset twice (+8 h EDT / +10 h EST; two UTC->ET conversions reproduce the header), the other
~30% are true UTC; no field (agent, hour, form, year) predicts which. The SGML header ACCEPTANCE-DATETIME is therefore
read for every filing (8-K full submissions; a 20 KB prefix of the full submission for 10-K/10-Q). JSON-only times are
quality C. Pre-2002 filings carry midnight placeholders (no real time): quality B (date only).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ET = "America/New_York"
_HDR_RE = re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})|ACCEPTANCE-DATETIME:\s*(\d{14})")


def parse_header_acceptance(header_text: str) -> str | None:
    """Raw 14-digit ACCEPTANCE-DATETIME (Eastern) from an SGML header, or None."""
    if not header_text:
        return None
    m = _HDR_RE.search(header_text[:20000])
    if not m:
        return None
    return m.group(1) or m.group(2)


def header_to_et(raw14: str) -> pd.Timestamp:
    t = pd.Timestamp(f"{raw14[:4]}-{raw14[4:6]}-{raw14[6:8]} {raw14[8:10]}:{raw14[10:12]}:{raw14[12:14]}")
    return t.tz_localize(ET, ambiguous=True, nonexistent="shift_forward")


def json_to_et(raw: str) -> pd.Timestamp:
    """Corrected ET time from the submissions-JSON acceptanceDateTime (double UTC->ET conversion)."""
    t = pd.Timestamp(str(raw)[:19])
    once = t.tz_localize("UTC").tz_convert(ET).tz_localize(None)
    return once.tz_localize("UTC").tz_convert(ET)


def acceptance_from_sources(json_raw: str | None, header_raw: str | None, filing_date: str | None,
                            assumed_time: str = "17:30", json_rule_ok: bool = True) -> tuple:
    """Return (acceptance_time_et, ts_quality, ts_source, date_only).

    A = exact acceptance time (SGML header, or JSON value under the verified correction rule)
    B = date only (assume after the close at `assumed_time` ET)
    C = derived or uncertain (JSON rule failed its sample verification, or only a filing date with conflicts)
    """
    if header_raw:
        t = header_to_et(header_raw)
        return t, "A", "sgml_header", False
    if json_raw and str(json_raw).strip():
        t = json_to_et(json_raw)
        if t.hour == 0 and t.minute == 0 and t.second == 0:
            d = t.tz_localize(None).normalize()
            return _assumed(d, assumed_time), "B", "json_midnight_placeholder", True
        return t, ("A" if json_rule_ok else "C"), "submissions_json_corrected", False
    if filing_date:
        d = pd.Timestamp(filing_date).normalize()
        return _assumed(d, assumed_time), "B", "filing_date_only", True
    return pd.NaT, "C", "missing", True


def _assumed(d: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    hh, mm = hhmm.split(":")
    return (d + pd.Timedelta(hours=int(hh), minutes=int(mm))).tz_localize(ET)


def derive_times(acceptance_et: pd.Series, public_lag_min: float, decision_latency_min: float,
                 observed_public: pd.Series | None = None) -> pd.DataFrame:
    """public_available = observed if present, else acceptance + lag (flagged 'assumed'); feature_ready = public +
    latency; decision = feature_ready (SPEC.md §5.1)."""
    acc = pd.to_datetime(acceptance_et)
    pub = acc + pd.Timedelta(minutes=float(public_lag_min))
    src = pd.Series("assumed_lag", index=acc.index, dtype=object)
    if observed_public is not None:
        obs = pd.to_datetime(observed_public)
        has = obs.notna()
        pub = pub.where(~has, obs)
        src = src.where(~has, "observed")
    feat = pub + pd.Timedelta(minutes=float(decision_latency_min))
    return pd.DataFrame({"public_available_time": pub, "public_time_source": src,
                         "feature_ready_time": feat, "decision_time": feat})


def to_utc_ns(s: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(s)).tz_convert("UTC").asi8.astype(np.int64)


def timing_policy_label(cfg: dict) -> str:
    t = cfg["timing"]
    return (f"point-in-time evaluation under an assumed availability/processing-latency policy "
            f"(public = acceptance + {t['public_lag_minutes']} min [assumed unless observed], "
            f"decision = public + {t['decision_latency_minutes']} min)")
