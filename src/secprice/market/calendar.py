"""NYSE trading calendar with early closes; session lookups used by every clock (SPEC.md §5.1, §5.3, §5.4).

All lookups work on UTC nanoseconds so that DST, holidays and early closes are handled by the schedule itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"


def _to_utc_ns(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return int(t.tz_convert("UTC").value)


@dataclass
class TradingCalendar:
    sessions: pd.DatetimeIndex          # naive session dates
    open_utc: np.ndarray                # int64 ns
    close_utc: np.ndarray               # int64 ns
    source: str = "pandas_market_calendars:NYSE"

    @classmethod
    def nyse(cls, start: str, end: str) -> "TradingCalendar":
        import pandas_market_calendars as mcal
        sch = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
        o = pd.DatetimeIndex(sch["market_open"]).tz_convert("UTC")
        c = pd.DatetimeIndex(sch["market_close"]).tz_convert("UTC")
        ver = getattr(mcal, "__version__", "?")
        return cls(pd.DatetimeIndex(sch.index).tz_localize(None).normalize(), o.asi8.astype(np.int64),
                   c.asi8.astype(np.int64), f"pandas_market_calendars {ver}:NYSE")

    @classmethod
    def from_sessions(cls, dates, open_time="09:30", close_time="16:00", early: dict | None = None) -> "TradingCalendar":
        """Synthetic calendar for fixtures/tests. early = {date_str: 'HH:MM'} early closes."""
        early = early or {}
        d = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        opens, closes = [], []
        for x in d:
            ds = x.strftime("%Y-%m-%d")
            opens.append(pd.Timestamp(f"{ds} {open_time}").tz_localize(ET).tz_convert("UTC").value)
            closes.append(pd.Timestamp(f"{ds} {early.get(ds, close_time)}").tz_localize(ET).tz_convert("UTC").value)
        return cls(d, np.array(opens, dtype=np.int64), np.array(closes, dtype=np.int64), "synthetic")

    # ------------------------------------------------------------------ basic lookups
    def __len__(self) -> int:
        return len(self.sessions)

    def index_of_date(self, d) -> int:
        """Session index of a calendar date, or -1 if not a session."""
        d = pd.Timestamp(d).normalize()
        if d.tzinfo is not None:
            d = d.tz_localize(None)
        i = self.sessions.searchsorted(d)
        return int(i) if i < len(self.sessions) and self.sessions[i] == d else -1

    def bucket(self, ts) -> str:
        """PRE / INTRA / POST / NONTRADING from an ET-aware timestamp against the actual session open/close."""
        if ts is None or pd.isna(ts):
            return "UNKNOWN"
        t = pd.Timestamp(ts).tz_convert(ET)
        i = self.index_of_date(t.tz_localize(None).normalize())
        if i < 0:
            return "NONTRADING"
        v = _to_utc_ns(t)
        if v < self.open_utc[i]:
            return "PRE"
        if v < self.close_utc[i]:
            return "INTRA"
        return "POST"

    def last_close_at_or_before(self, ts) -> int:
        """Index of the last regular-session close <= ts (a close printed at the same instant counts as before)."""
        return int(np.searchsorted(self.close_utc, _to_utc_ns(ts), side="right") - 1)

    def first_close_after(self, ts) -> int:
        """Index of the first session whose close is strictly after ts."""
        return int(np.searchsorted(self.close_utc, _to_utc_ns(ts), side="right"))

    def first_open_at_or_after(self, ts) -> int:
        """Index of the first session whose open is at or after ts (T_trade entry rule)."""
        return int(np.searchsorted(self.open_utc, _to_utc_ns(ts), side="left"))

    def day0(self, public_ts) -> int:
        """Track A day 0: the session in which public time falls if before that session's close, else the next one."""
        return self.first_close_after(public_ts)

    def date(self, i: int):
        return self.sessions[i] if 0 <= i < len(self.sessions) else pd.NaT

    def close_time(self, i: int) -> pd.Timestamp:
        return pd.Timestamp(self.close_utc[i], tz="UTC").tz_convert(ET)

    def open_time(self, i: int) -> pd.Timestamp:
        return pd.Timestamp(self.open_utc[i], tz="UTC").tz_convert(ET)

    def is_early_close(self, i: int) -> bool:
        return self.close_time(i).hour < 16

    # ------------------------------------------------------------------ vectorised versions
    def v_bucket(self, ts_utc_ns: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.full(len(ts_utc_ns), "UNKNOWN", dtype=object)
        # session date in ET for each timestamp
        et_dates = pd.to_datetime(ts_utc_ns, utc=True).tz_convert(ET).tz_localize(None).normalize()
        idx = self.sessions.searchsorted(et_dates)
        idx_c = np.clip(idx, 0, len(self.sessions) - 1)
        is_sess = (idx < len(self.sessions)) & (self.sessions[idx_c] == et_dates)
        o, c = self.open_utc[idx_c], self.close_utc[idx_c]
        out[valid & ~is_sess] = "NONTRADING"
        m = valid & is_sess
        out[m & (ts_utc_ns < o)] = "PRE"
        out[m & (ts_utc_ns >= o) & (ts_utc_ns < c)] = "INTRA"
        out[m & (ts_utc_ns >= c)] = "POST"
        return out

    def v_last_close_at_or_before(self, ts_utc_ns: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.close_utc, ts_utc_ns, side="right") - 1

    def v_first_close_after(self, ts_utc_ns: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.close_utc, ts_utc_ns, side="right")

    def v_first_open_at_or_after(self, ts_utc_ns: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.open_utc, ts_utc_ns, side="left")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.sessions,
            "open_et": pd.to_datetime(self.open_utc, utc=True).tz_convert(ET),
            "close_et": pd.to_datetime(self.close_utc, utc=True).tz_convert(ET),
        }).assign(early_close=lambda d: d.close_et.dt.hour < 16)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, source: str = "artifact") -> "TradingCalendar":
        o = pd.DatetimeIndex(df["open_et"]).tz_convert("UTC").asi8.astype(np.int64)
        c = pd.DatetimeIndex(df["close_et"]).tz_convert("UTC").asi8.astype(np.int64)
        return cls(pd.DatetimeIndex(pd.to_datetime(df["date"])).normalize(), o, c, source)
