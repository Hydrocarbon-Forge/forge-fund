"""engine/metrics.py, run/report.py and the pure helpers event_dates / spy_buy_hold in run/run_backtest.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.metrics import bootstrap_excess_ci, curve_stats, rank_ic, regress, trade_stats
from run.report import render_report
from run.run_backtest import event_dates, spy_buy_hold


def _bdays(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ----------------------------------------------------------------------------- curve_stats
def test_curve_stats_constant_daily_gain():
    idx = _bdays(100)
    eq = pd.Series(100_000.0 * 1.01 ** np.arange(100), index=idx)
    st = curve_stats(eq)
    assert st["days"] == 100
    assert st["total_return"] == pytest.approx(1.01 ** 99 - 1, rel=1e-9)
    years = (idx[-1] - idx[0]).days / 365.25
    assert st["cagr"] == pytest.approx((1.01 ** 99) ** (1 / years) - 1, rel=1e-9)
    assert st["cagr"] > 0
    assert st["max_drawdown"] == 0.0
    assert math.isnan(st["calmar"])                            # no drawdown -> undefined
    assert st["day_win_rate"] == 1.0
    assert st["best_day"] == pytest.approx(0.01, rel=1e-9)
    assert st["worst_day"] == pytest.approx(0.01, rel=1e-9)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="BUG engine/metrics.py:33: sharpe guards with std > 0, but a constant-return curve has std ~1e-15 "
           "float noise, so Sharpe comes out as ~1e15 instead of NaN",
)
def test_curve_stats_constant_gain_sharpe_is_undefined():
    eq = pd.Series(100_000.0 * 1.01 ** np.arange(100), index=_bdays(100))
    assert math.isnan(curve_stats(eq)["sharpe"])


def test_curve_stats_known_drawdown():
    idx = _bdays(5)
    eq = pd.Series([100.0, 110.0, 90.0, 95.0, 120.0], index=idx)
    st = curve_stats(eq)
    assert st["max_drawdown"] == pytest.approx(90.0 / 110.0 - 1.0)
    assert st["max_dd_date"] == str(idx[2].date())             # the trough, not the peak
    assert st["total_return"] == pytest.approx(0.20)
    assert st["calmar"] == pytest.approx(st["cagr"] / abs(st["max_drawdown"]))
    assert st["worst_day"] == pytest.approx(90.0 / 110.0 - 1.0)


def test_curve_stats_too_short_is_empty():
    assert curve_stats(pd.Series([1.0, 2.0], index=_bdays(2))) == {}


# ----------------------------------------------------------------------------- regress
def test_regress_recovers_half_beta_and_zero_alpha():
    rng = np.random.default_rng(5)
    rb = rng.normal(0.0004, 0.01, 250)
    idx = _bdays(251)
    bench = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + rb]), index=idx)
    fund = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + 0.5 * rb]), index=idx)
    out = regress(fund, bench)
    assert out["beta"] == pytest.approx(0.5, abs=1e-9)
    assert out["alpha_ann"] == pytest.approx(0.0, abs=1e-9)
    assert out["correlation"] == pytest.approx(1.0, abs=1e-9)
    # active return = 0.5*rb - rb = -0.5*rb, so tracking error is half the benchmark's vol
    assert out["tracking_error"] == pytest.approx(0.5 * rb.std(ddof=1) * math.sqrt(252), rel=1e-6)
    # IR = mean(active)/std(active)*sqrt(252) = (-0.5*m)/(0.5*s)*sqrt(252) = -m/s*sqrt(252)
    expected_ir = -rb.mean() / rb.std(ddof=1) * math.sqrt(252)
    assert out["information_ratio"] == pytest.approx(expected_ir, rel=1e-6)


def test_regress_short_or_flat_benchmark_is_empty():
    idx = _bdays(10)
    assert regress(pd.Series(np.arange(1.0, 11.0), index=idx), pd.Series(np.arange(1.0, 11.0), index=idx)) == {}
    flat = pd.Series(100.0, index=_bdays(300))
    assert regress(pd.Series(np.linspace(100, 120, 300), index=flat.index), flat) == {}


# ----------------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_is_ordered_and_seed_deterministic():
    rng = np.random.default_rng(9)
    idx = _bdays(301)
    bench = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.0004, 0.01, 300)]), index=idx)
    fund = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.0006, 0.009, 300)]), index=idx)
    a = bootstrap_excess_ci(fund, bench, n=500)
    b = bootstrap_excess_ci(fund, bench, n=500)
    assert a == b
    assert a["ci95_low"] <= a["ci95_high"]
    assert 0.0 <= a["p_excess_le_0"] <= 1.0
    assert set(a) == {"excess_ann_mean", "ci95_low", "ci95_high", "p_excess_le_0"}


def test_bootstrap_too_short_is_empty():
    idx = _bdays(40)
    s = pd.Series(np.linspace(100, 110, 40), index=idx)
    assert bootstrap_excess_ci(s, s * 1.0) == {}


# ----------------------------------------------------------------------------- trade_stats
def test_trade_stats_hand_example():
    fills = pd.DataFrame({"notional": [10_000.0, 10_000.0, 5_000.0, 5_000.0], "fee": [1.0, 1.0, 0.5, 0.5]})
    closed = pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -25.0], "ret": [0.01, -0.005, 0.02, -0.0025]})
    marks = pd.DataFrame({
        "date": ["2024-01-02", "2024-07-02", "2025-01-02"],
        "equity": [100_000.0, 110_000.0, 120_000.0],
        "positions_value": [50_000.0, 55_000.0, 60_000.0],
        "n_positions": [2, 3, 4],
    })
    ts = trade_stats(fills, closed, marks)
    assert ts["n_fills"] == 4
    assert ts["n_round_trips"] == 4
    assert ts["trade_win_rate"] == pytest.approx(0.5)
    assert ts["avg_win"] == pytest.approx(150.0)
    assert ts["avg_loss"] == pytest.approx(-37.5)
    assert ts["profit_factor"] == pytest.approx(300.0 / 75.0)
    assert ts["avg_trade_return"] == pytest.approx(np.mean([0.01, -0.005, 0.02, -0.0025]))
    years = (pd.Timestamp("2025-01-02") - pd.Timestamp("2024-01-02")).days / 365.25
    # one-way turnover = total notional / 2 / average equity / years
    assert ts["annual_turnover"] == pytest.approx(30_000.0 / 2 / 110_000.0 / years)
    assert ts["total_fees"] == pytest.approx(3.0)
    assert ts["fees_pct_of_start"] == pytest.approx(3.0 / 100_000.0)
    assert ts["avg_exposure"] == pytest.approx(0.5)
    assert ts["avg_positions"] == pytest.approx(3.0)


def test_trade_stats_without_round_trips_has_no_win_rate():
    fills = pd.DataFrame({"notional": [1000.0], "fee": [0.1]})
    marks = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "equity": [100e3, 100e3],
                          "positions_value": [0.0, 1000.0], "n_positions": [0, 1]})
    ts = trade_stats(fills, fills.iloc[:0], marks)
    assert ts["n_round_trips"] == 0
    assert "trade_win_rate" not in ts
    assert "profit_factor" not in ts


# ----------------------------------------------------------------------------- rank_ic
def test_rank_ic_perfectly_monotonic_is_one():
    pred = pd.Series(np.arange(20.0), index=[f"T{i}" for i in range(20)])
    realized = pd.Series(3.0 * np.arange(20.0) + 1.0, index=pred.index)
    assert rank_ic(pred, realized) == pytest.approx(1.0)
    assert rank_ic(-pred, realized) == pytest.approx(-1.0)


def test_rank_ic_too_few_points_is_nan():
    pred = pd.Series(np.arange(9.0))
    assert math.isnan(rank_ic(pred, pred))


# ----------------------------------------------------------------------------- event_dates
def _holiday_week_days() -> pd.DatetimeIndex:
    days = pd.bdate_range("2024-05-20", "2024-06-07")
    return days[days != pd.Timestamp("2024-05-27")]           # Memorial Day (a Monday) is not a trading day


def test_event_dates_weekly_picks_first_trading_day_of_each_iso_week():
    days = _holiday_week_days()
    assert event_dates(days, "weekly") == ["2024-05-20", "2024-05-28", "2024-06-03"]


def test_event_dates_daily_returns_every_day():
    days = _holiday_week_days()
    assert event_dates(days, "daily") == [d.strftime("%Y-%m-%d") for d in days]


def test_event_dates_integer_is_every_nth_day():
    days = _holiday_week_days()
    expected = [days[i].strftime("%Y-%m-%d") for i in range(0, len(days), 5)]
    assert event_dates(days, "5") == expected
    assert len(expected) == 3


# ----------------------------------------------------------------------------- spy_buy_hold
def test_spy_buy_hold_is_cash_before_first_event_then_invested():
    idx = _bdays(30)
    op = pd.Series(100.0, index=idx)
    cl = pd.Series(100.0, index=idx)
    first = idx[10].strftime("%Y-%m-%d")
    eq = spy_buy_hold(op, cl, first, 100_000.0, idx)
    assert (eq[eq.index < idx[10]] == 100_000.0).all()
    assert 99_500.0 < eq.iloc[-1] < 100_000.0                  # flat price: only spread, fee and buffer drag


def test_spy_buy_hold_tracks_price_after_entry():
    idx = _bdays(30)
    op = pd.Series(100.0, index=idx)
    cl = pd.Series(np.linspace(100.0, 200.0, 30), index=idx)
    eq = spy_buy_hold(op, cl, idx[5].strftime("%Y-%m-%d"), 100_000.0, idx)
    assert eq.iloc[-1] > 190_000.0
    assert eq.iloc[-1] > eq.iloc[10] > eq.iloc[6]


# ----------------------------------------------------------------------------- render_report
def _minimal_metrics(anonymized: bool) -> dict:
    idx = _bdays(60)
    rng = np.random.default_rng(4)
    fund = pd.Series(100_000.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.0005, 0.01, 59)]), index=idx)
    spy = pd.Series(100_000.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.0004, 0.009, 59)]), index=idx)
    stats = {k: curve_stats(fund) for k in ("fund", "spy_exposure_matched", "equal_weight_universe")}
    stats["spy_buy_hold"] = curve_stats(spy)
    return {
        "run_id": "bt_unit_test", "mode": "claude", "freq": "weekly",
        "start": stats["fund"]["start"], "end": stats["fund"]["end"],
        "n_rebalances": 12, "capital": 100_000.0,
        "stats": stats,
        "regression_vs_spy": regress(fund, spy),
        "excess_return_bootstrap": bootstrap_excess_ci(fund, spy, n=200),
        "trades": {"n_fills": 4, "n_round_trips": 2, "trade_win_rate": 0.5, "profit_factor": 1.5,
                   "avg_win": 300.0, "avg_loss": -200.0, "avg_exposure": 0.6, "avg_positions": 3.0,
                   "annual_turnover": 1.2, "total_fees": 12.0, "fees_pct_of_start": 0.00012},
        "kronos_signal": {"n_dates": 10, "mean_rank_ic": 0.03, "ic_tstat": 0.9,
                          "ic_positive_share": 0.6, "direction_hit_rate": 0.52},
        "claude": {"calls": 12, "decisions": 11, "fallback_holds": 1, "prompt_tokens": 1000,
                   "completion_tokens": 200, "est_cost_usd": 0.05},
        "halts": 0, "rejected_orders": 0, "decider_errors": 0,
        "assumptions": {"half_spread_bps": 1.0, "fee_bps_per_side": 1.0, "anonymized_prompts": anonymized},
    }


def test_render_report_contains_caveats_and_paper_banner():
    text = render_report(_minimal_metrics(anonymized=True))
    assert isinstance(text, str)
    assert "CAVEATS" in text
    assert "PAPER" in text
    assert "bt_unit_test" in text
    assert "ANONYMISED" in text
    assert "Kronos signal quality" in text


def test_render_report_flags_non_anonymised_prompts():
    text = render_report(_minimal_metrics(anonymized=False))
    assert "NOT anonymised" in text
