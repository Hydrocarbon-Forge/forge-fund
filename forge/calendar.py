"""Minimal NYSE calendar (weekdays minus the listed full-day holidays) - used only for scheduling.

If a date is missing from this list the worst case is a paper cycle on a holiday that fills at the
last close and is flagged ``price_source=last_close``; no real orders exist to be mis-sent.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_HOLIDAYS = {
    # 2025 (year-end only; backtests never use this calendar)
    "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05",
    "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04",
    "2028-11-23", "2028-12-25",
}  # extend yearly: a missing holiday only makes a paper cycle fill at the last close (flagged price_source)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _HOLIDAYS


def next_trading_day(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def decision_date(now: datetime | None = None) -> date:
    """The trading day whose 09:35 ET rebalance the current moment belongs to.

    Before the 16:00 ET close on a trading day -> that day; otherwise the next trading day.
    """
    now = (now or datetime.now(ET)).astimezone(ET)
    d = now.date()
    if is_trading_day(d) and now.hour < 16:
        return d
    return next_trading_day(d)


def market_open_now(now: datetime | None = None) -> bool:
    now = (now or datetime.now(ET)).astimezone(ET)
    mins = now.hour * 60 + now.minute
    return is_trading_day(now.date()) and 9 * 60 + 30 <= mins < 16 * 60
