"""Nautilus simulation core: accounting, fills, ordering, safety."""
import numpy as np
import pandas as pd
import pytest

from engine.sim import SimRunner
from forge.config import LIMITS, Limits
from forge.types import Plan

DATES = pd.bdate_range("2024-03-04", periods=30)
TICK = ["AAA", "BBB", "CCC"]


def panels():
    base = {"AAA": 100.0, "BBB": 50.0, "CCC": 200.0}
    close = pd.DataFrame({t: [p * (1 + 0.004 * i) for i in range(len(DATES))] for t, p in base.items()}, index=DATES)
    open_ = close.shift(1).fillna(close.iloc[0]) * 1.001
    return open_, close


def run(decide, events=None, cap=100_000.0, limits=None):
    o, c = panels()
    ev = events or [DATES[1].strftime("%Y-%m-%d"), DATES[8].strftime("%Y-%m-%d")]
    kw = {"limits": limits} if limits else {}
    return SimRunner(o, c, cap, decide, ev, tickers=TICK, **kw).run(), o, c


def test_accounting_identity_and_fill_prices():
    state = {"n": 0}

    def decide(date, view):
        state["n"] += 1
        return Plan({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25} if state["n"] == 1 else {"AAA": 0.10}, meta={})

    res, o, c = run(decide)
    assert res.errors == [] and res.rejected == []
    assert len(res.marks) == len(DATES)
    unreal = sum((p["last_px"] - p["avg_px"]) * p["qty"] for p in res.final_positions.values())
    assert res.final_equity == pytest.approx(100_000 + res.fills.realized_pnl.sum() + unreal, abs=0.05)
    first = res.fills[res.fills.date == DATES[1].strftime("%Y-%m-%d")]
    assert set(first.side) == {"BUY"}
    for r in first.itertuples():                     # buys fill at the open ask (open * 1.0001, cents)
        assert r.price == pytest.approx(round(o.loc[DATES[1], r.ticker] * 1.0001, 2), abs=0.011)
    assert (res.fills.fee > 0).all()


def test_partial_trim_and_full_exit():
    plans = iter([Plan({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25}), Plan({"CCC": 0.10})])
    res, *_ = run(lambda d, v: next(plans))
    day2 = res.fills[res.fills.date == DATES[8].strftime("%Y-%m-%d")]
    assert set(day2[day2.side == "SELL"].ticker) == {"AAA", "BBB", "CCC"}    # AAA/BBB exit, CCC trimmed to 10%
    assert set(res.final_positions) == {"CCC"}
    assert res.final_positions["CCC"]["value"] / res.final_equity == pytest.approx(0.10, abs=0.01)


def test_rotation_funded_by_same_event_sells():
    # fully invested -> the CCC buy is only possible with the cash released by selling BBB first
    lim = Limits(max_positions=10, max_single=0.5, max_gross=1.0, cash_floor=0.0, max_order_pct_equity=0.6)
    plans = iter([Plan({"AAA": 0.5, "BBB": 0.5}), Plan({"AAA": 0.5, "CCC": 0.5})])
    res, *_ = run(lambda d, v: next(plans), limits=lim)
    assert res.rejected == [] and res.errors == []
    assert set(res.final_positions) == {"AAA", "CCC"}
    assert res.final_positions["CCC"]["value"] / res.final_equity == pytest.approx(0.5, abs=0.02)
    assert res.final_cash >= 0


def test_limits_enforced_even_if_decider_asks_too_much():
    res, *_ = run(lambda d, v: Plan({"AAA": 0.9, "BBB": 0.9, "CCC": 0.9}))
    invested = sum(p["value"] for p in res.final_positions.values()) / res.final_equity
    assert invested <= LIMITS.max_gross + 1e-3
    assert all(p["weight"] <= LIMITS.max_single + 0.01 for p in res.final_positions.values())
    assert res.final_cash > 0


def test_halted_and_hold_plans_trade_nothing():
    res, *_ = run(lambda d, v: Plan(None, True, {"status": "halted"}))
    assert len(res.fills) == 0 and res.final_equity == pytest.approx(100_000)
    res2, *_ = run(lambda d, v: Plan(None))
    assert len(res2.fills) == 0


def test_decider_exception_is_captured_not_fatal():
    def boom(d, v):
        raise ValueError("pipeline bug")
    res, *_ = run(boom)
    assert len(res.errors) == 2 and "pipeline bug" in res.errors[0]
    assert len(res.marks) == len(DATES)                                  # engine kept running


def test_first_view_is_all_cash_and_decisions_are_chronological():
    seen = []
    def decide(date, view):
        seen.append((date, view.equity, view.cash, len(view.positions)))
        return Plan({"AAA": 0.2})
    run(decide)
    assert [s[0] for s in seen] == sorted(s[0] for s in seen)
    assert seen[0][1] == pytest.approx(100_000) and seen[0][3] == 0


def test_tiny_target_below_min_notional_is_skipped():
    res, *_ = run(lambda d, v: Plan({"AAA": 0.003}))                      # $300 < $500 min notional
    assert len(res.fills) == 0


def test_cash_account_never_goes_negative():
    res, *_ = run(lambda d, v: Plan({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25}), cap=3_000.0)
    assert res.final_cash >= -1e-6
    assert res.marks.cash.min() >= -1e-6


def test_closed_round_trips_have_pnl():
    plans = iter([Plan({"AAA": 0.2}), Plan({})])
    res, *_ = run(lambda d, v: next(plans))
    assert len(res.closed) == 1 and res.closed.iloc[0].ticker == "AAA"
    assert res.closed.iloc[0].pnl == pytest.approx(res.fills.realized_pnl.sum(), abs=0.01)


def test_fractional_crypto_at_tiny_capital():
    # Demonstrates that a $200 book can hold and trade fractional BTC-USD cleanly
    close = pd.DataFrame({
        "AAPL": [150.0 + i * 0.5 for i in range(len(DATES))],
        "BTC-USD": [60000.0 + i * 100.0 for i in range(len(DATES))],
    }, index=DATES)
    open_ = close.shift(1).fillna(close.iloc[0]) * 1.001
    ev = [DATES[1].strftime("%Y-%m-%d"), DATES[5].strftime("%Y-%m-%d")]

    plans = iter([
        Plan({"BTC-USD": 0.20}),  # $40 target notional -> ~0.000666 BTC
        Plan({}),                 # full exit -> close position
    ])

    runner = SimRunner(open_, close, 200.0, lambda d, v: next(plans), ev, tickers=["AAPL", "BTC-USD"])
    res = runner.run()

    assert res.errors == []
    assert res.rejected == []
    # Verify buys on day 1
    buys = res.fills[res.fills.side == "BUY"]
    assert len(buys) == 1
    buy_fill = buys.iloc[0]
    assert buy_fill.ticker == "BTC-USD"
    assert 0 < buy_fill.qty < 0.001  # fractional quantity
    assert buy_fill.notional == pytest.approx(buy_fill.qty * buy_fill.price, abs=0.01)
    assert 35.0 <= buy_fill.notional <= 41.0  # ~20% of $200

    # Verify sells on day 5
    sells = res.fills[res.fills.side == "SELL"]
    assert len(sells) == 1
    sell_fill = sells.iloc[0]
    assert sell_fill.ticker == "BTC-USD"
    assert sell_fill.qty == buy_fill.qty
    assert len(res.final_positions) == 0  # position closed
    assert len(res.closed) == 1
    assert res.closed.iloc[0].ticker == "BTC-USD"
    assert res.final_cash > 190.0

