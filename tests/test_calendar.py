"""forge/calendar.py: trading days, next trading day, decision date cut-off and market-open window (ET)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from forge.calendar import ET, decision_date, is_trading_day, market_open_now, next_trading_day


def _et(y, mo, d, h=0, mi=0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def test_weekends_are_not_trading_days():
    assert not is_trading_day(date(2026, 10, 10))              # Saturday
    assert not is_trading_day(date(2026, 10, 11))              # Sunday
    assert is_trading_day(date(2026, 10, 12))                  # Monday
    assert is_trading_day(date(2026, 10, 13))                  # Tuesday


def test_listed_holidays_are_not_trading_days():
    assert not is_trading_day(date(2026, 11, 26))              # Thanksgiving (Thu)
    assert not is_trading_day(date(2026, 7, 3))                # Independence Day observed (Fri)
    assert not is_trading_day(date(2027, 12, 24))              # Christmas observed (Fri)
    assert is_trading_day(date(2026, 11, 27))                  # day after Thanksgiving is open


def test_next_trading_day_skips_weekend():
    assert next_trading_day(date(2026, 10, 9)) == date(2026, 10, 12)   # Friday -> Monday
    assert next_trading_day(date(2026, 10, 12)) == date(2026, 10, 13)


def test_next_trading_day_skips_holidays_and_weekend():
    assert next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27)  # Wed -> Thanksgiving skipped
    assert next_trading_day(date(2026, 7, 2)) == date(2026, 7, 6)      # Thu -> Fri holiday, weekend skipped


def test_decision_date_before_close_is_same_day():
    assert decision_date(_et(2026, 10, 13, 9, 0)) == date(2026, 10, 13)
    assert decision_date(_et(2026, 10, 13, 15, 59)) == date(2026, 10, 13)


def test_decision_date_at_or_after_close_is_next_trading_day():
    assert decision_date(_et(2026, 10, 13, 16, 0)) == date(2026, 10, 14)
    assert decision_date(_et(2026, 10, 13, 17, 30)) == date(2026, 10, 14)


def test_decision_date_friday_evening_is_monday():
    assert decision_date(_et(2026, 10, 16, 17, 0)) == date(2026, 10, 19)


def test_decision_date_on_weekend_and_holiday_is_next_trading_day():
    assert decision_date(_et(2026, 10, 10, 11, 0)) == date(2026, 10, 12)   # Saturday
    assert decision_date(_et(2026, 11, 26, 10, 0)) == date(2026, 11, 27)   # Thanksgiving


def test_decision_date_converts_aware_utc_to_eastern():
    # 19:59 UTC in mid-October is 15:59 EDT -> still the same trading day
    assert decision_date(datetime(2026, 10, 13, 19, 59, tzinfo=timezone.utc)) == date(2026, 10, 13)
    # 20:00 UTC is 16:00 EDT -> the next trading day
    assert decision_date(datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc)) == date(2026, 10, 14)


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(9, 29, False), (9, 30, True), (15, 59, True), (16, 0, False)],
)
def test_market_open_boundaries_on_a_trading_day(hour, minute, expected):
    assert market_open_now(_et(2026, 10, 13, hour, minute)) is expected


def test_market_closed_on_weekend_and_holiday_during_session_hours():
    assert market_open_now(_et(2026, 10, 10, 11, 0)) is False  # Saturday
    assert market_open_now(_et(2026, 11, 26, 11, 0)) is False  # Thanksgiving
