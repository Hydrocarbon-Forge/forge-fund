"""run_backtest end-to-end on a synthetic store: real Nautilus engine, rule baseline, real files."""
import json

import pandas as pd
import pytest

from forge.config import get_paths
from run import run_backtest
from tests.synth import write_store


def test_backtest_runs_end_to_end_and_saves_everything(capsys):
    write_store(get_paths().root, n_names=12, n_days=420, seed=2)
    rc = run_backtest.main(["--start", "2022-03-01", "--freq", "weekly", "--no-claude", "--no-kronos", "--run-id", "t1"])
    assert rc == 0
    out = get_paths().backtests / "t1"
    for f in ("metrics.json", "report.txt", "equity.csv", "trades.csv", "decisions.jsonl", "run.log"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    for k in ("cagr", "max_drawdown", "sharpe", "win_rate", "benchmark_cagr", "excess_cagr", "stats", "assumptions"):
        assert k in m
    assert m["max_drawdown"] <= 0 and m["stats"]["fund"]["days"] > 100
    assert m["trades"]["n_fills"] > 0 and m["decider_errors"] == 0
    eq = pd.read_csv(out / "equity.csv", index_col=0)
    assert eq["fund"].iloc[0] == 100_000 and (eq["fund"] > 0).all()
    printed = capsys.readouterr().out
    assert "forge-fund BACKTEST" in printed and "CAVEATS" in printed


def test_event_schedule_and_benchmark_alignment():
    idx = pd.bdate_range("2024-01-01", periods=30)
    ev = run_backtest.event_dates(idx, "weekly")
    assert ev[0] == "2024-01-01" and all(pd.Timestamp(e).weekday() == 0 for e in ev)
    op = pd.Series(100.0, index=idx)
    cl = pd.Series(range(100, 130), index=idx, dtype=float)
    spy = run_backtest.spy_buy_hold(op, cl, ev[1], 100_000, idx)
    assert (spy[spy.index < pd.Timestamp(ev[1])] == 100_000).all()
    assert spy.iloc[-1] == pytest.approx(100_000 * 129 / 100, rel=0.01)
