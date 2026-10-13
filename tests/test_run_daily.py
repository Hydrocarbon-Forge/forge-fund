"""run_daily CLI: idempotency, dry-run, halt exit code, resume - offline (no yfinance, no Claude, no Kronos)."""
import json

import pandas as pd
import pytest

from forge.config import get_paths
from run import run_daily
from tests.synth import write_store

DATE = None


@pytest.fixture()
def env(monkeypatch):
    names = write_store(get_paths().root, n_names=12, n_days=420, seed=4)
    from forge.data import load_store
    df = load_store()
    last = df["date"].max()
    last_close = df[df["date"] == last].set_index("ticker")["close"].to_dict()
    monkeypatch.setattr(run_daily, "update_store", lambda *a, **k: {"last_date": str(last.date()), "n_tickers": 13, "dropped": []})
    monkeypatch.setattr(run_daily, "latest_prices", lambda tickers: ({t: float(last_close[t]) for t in tickers if t in last_close}, "last_close"))
    return (last + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


ARGS = ["--no-claude", "--no-kronos"]


def test_cycle_writes_state_then_is_idempotent(env):
    assert run_daily.main(ARGS + ["--date", env]) == 0
    st = get_paths().state
    pf = json.loads((st / "portfolio.json").read_text())
    assert pf["inception"] == env and len(pf["positions"]) > 0 and pf["equity"] > 90_000
    before = (st / "ledger.jsonl").read_text()
    assert run_daily.main(ARGS + ["--date", env]) == 0                       # second run same day: no-op
    assert (st / "ledger.jsonl").read_text() == before
    assert run_daily.main(ARGS + ["--date", env, "--force"]) == 0            # explicit redo replaces, not duplicates
    assert len((st / "ledger.jsonl").read_text().splitlines()) == 1


def test_dry_run_writes_nothing(env):
    assert run_daily.main(ARGS + ["--date", env, "--dry-run"]) == 0
    st = get_paths().state
    assert not (st / "ledger.jsonl").exists() and not (st / "portfolio.json").exists()


def test_halt_blocks_trading_until_resume(env):
    st = get_paths().state
    (st / "halt.json").write_text(json.dumps({"ts": "x", "reason": "test halt", "loss_pct": -0.03}))
    assert run_daily.main(ARGS + ["--date", env]) == 3
    assert not (st / "ledger.jsonl").exists()
    assert run_daily.main(["--resume"]) == 0 and not (st / "halt.json").exists()
    assert run_daily.main(ARGS + ["--date", env]) == 0


def test_second_day_replays_and_extends(env):
    assert run_daily.main(ARGS + ["--date", env]) == 0
    nxt = (pd.Timestamp(env) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    first_trades = [json.loads(l) for l in (get_paths().state / "trades.jsonl").read_text().splitlines()]
    assert run_daily.main(ARGS + ["--date", nxt]) == 0
    trades = [json.loads(l) for l in (get_paths().state / "trades.jsonl").read_text().splitlines()]
    assert trades[: len(first_trades)] == first_trades
    assert len((get_paths().state / "ledger.jsonl").read_text().splitlines()) == 2


def test_no_secrets_in_engine_log(env):
    run_daily.main(ARGS + ["--date", env])
    log = (get_paths().logs / "engine.log").read_text()
    assert "test-token-NOT-REAL" not in log and len(log) > 100
