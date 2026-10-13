"""Paper cycle: event-sourced ledger replayed through Nautilus; state files derived from the replay."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from engine.paper import read_ledger, run_cycle
from forge.config import get_paths
from forge.types import Plan
from run.pipeline import Market
from tests.synth import make_long, panels_from_long, universe_names

NAMES = universe_names(8)


def market(n_rows):
    df = make_long(NAMES + ["SPY"], 300, seed=3)
    p = panels_from_long(df)
    return Market({k: v.iloc[:n_rows] for k, v in p.items()}, NAMES), p["close"].index


def prices(m):
    last = m.p["close"].iloc[-1]
    return {t: float(last[t]) for t in NAMES + ["SPY"]}


def cycle(n_rows, targets, offset=1, **kw):
    m, idx = market(n_rows)
    dd = (idx[n_rows - 1] + pd.offsets.BDay(offset)).strftime("%Y-%m-%d")
    plan = Plan(targets, meta={"status": "ok", "model": "haiku", "risk_view": "med", "cash": 0.5,
                               "notes": "n", "claude_holdings": {t: {"weight": w, "rationale": "r"} for t, w in (targets or {}).items()}})
    return run_cycle(paths=get_paths(), market=m, decision_date=dd, live_prices=prices(m), price_source="last_close",
                     live_decide=lambda d, v: plan, universe=NAMES, **kw), dd


def test_first_cycle_writes_consistent_state():
    s, dd = cycle(280, {"AAPL": 0.2, "ABBV": 0.2, "ABT": 0.2})
    st = get_paths().state
    led = read_ledger(get_paths())
    assert [r["date"] for r in led] == [dd] and led[0]["plan"]["targets"]["AAPL"] == 0.2
    pf = json.loads((st / "portfolio.json").read_text())
    assert pf["inception"] == dd and pf["mode"] == "paper" and not pf["halted"]
    assert pf["equity"] == pytest.approx(pf["cash"] + pf["positions_value"], abs=0.05)
    assert 99_000 < pf["equity"] <= 100_000 and len(pf["positions"]) == 3
    assert sum(p["weight"] for p in pf["positions"]) == pytest.approx(0.6, abs=0.02)
    trades = [json.loads(l) for l in (st / "trades.jsonl").read_text().splitlines()]
    assert len(trades) == 3 and all(t["side"] == "BUY" for t in trades)
    curve = [json.loads(l) for l in (st / "equity_curve.jsonl").read_text().splitlines()]
    assert len(curve) == 1 and curve[0]["benchmark_equity"] == pytest.approx(100_000, abs=1)
    dec = json.loads((st / "latest_decision.json").read_text())
    assert dec["model"] == "haiku" and dec["final_weights"]["AAPL"] == 0.2


def test_second_cycle_replays_first_exactly_and_extends_curve():
    cycle(280, {"AAPL": 0.2, "ABBV": 0.2, "ABT": 0.2})
    st = get_paths().state
    first_trades = (st / "trades.jsonl").read_text()
    s2, dd2 = cycle(281, {"AAPL": 0.1, "ACN": 0.25}, offset=1)
    trades = [json.loads(l) for l in (st / "trades.jsonl").read_text().splitlines()]
    old = [json.loads(l) for l in first_trades.splitlines()]
    assert trades[: len(old)] == old                             # day-1 fills reproduced bit-for-bit by the replay
    assert {t["date"] for t in trades} == {old[0]["date"], dd2}
    curve = [json.loads(l) for l in (st / "equity_curve.jsonl").read_text().splitlines()]
    assert [c["date"] for c in curve] == sorted(c["date"] for c in curve) and len(curve) == 2
    pf = json.loads((st / "portfolio.json").read_text())
    assert {p["ticker"] for p in pf["positions"]} == {"AAPL", "ACN"}
    assert len(read_ledger(get_paths())) == 2


def test_duplicate_date_refused_unless_replace():
    cycle(280, {"AAPL": 0.2})
    with pytest.raises(RuntimeError, match="already in ledger"):
        cycle(280, {"AAPL": 0.2})
    cycle(280, {"ABBV": 0.2}, replace=True)
    led = read_ledger(get_paths())
    assert len(led) == 1 and led[0]["plan"]["targets"] == {"ABBV": 0.2}


def test_deterministic_across_roots(tmp_path, monkeypatch):
    cycle(280, {"AAPL": 0.2, "ABBV": 0.2})
    cycle(281, {"AAPL": 0.15})
    a = json.loads((get_paths().state / "portfolio.json").read_text())
    other = tmp_path / "other"
    for d in ("data", "state", "logs", "backtests"):
        (other / d).mkdir(parents=True)
    monkeypatch.setenv("FORGE_ROOT", str(other))
    cycle(280, {"AAPL": 0.2, "ABBV": 0.2})
    cycle(281, {"AAPL": 0.15})
    b = json.loads((other / "state" / "portfolio.json").read_text())
    for k in ("equity", "cash", "positions"):
        assert a[k] == b[k]


def test_halted_plan_trades_nothing_and_is_flagged():
    m, idx = market(280)
    dd = (idx[279] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    run_cycle(paths=get_paths(), market=m, decision_date=dd, live_prices=prices(m), price_source="last_close",
              live_decide=lambda d, v: Plan(None, True, {"status": "halted", "error": "loss cap"}), universe=NAMES)
    pf = json.loads((get_paths().state / "portfolio.json").read_text())
    assert pf["halted"] is True and pf["equity"] == pytest.approx(100_000) and pf["positions"] == []
    assert (get_paths().state / "trades.jsonl").read_text() == ""


def test_dashboard_reads_the_state_files():
    cycle(280, {"AAPL": 0.2, "ABBV": 0.2, "ABT": 0.2})
    cycle(281, {"AAPL": 0.2, "ABBV": 0.2, "ABT": 0.1})
    c = TestClient(create_app(), base_url="http://127.0.0.1")
    for ep in ("summary", "equity", "positions", "decision", "pnl", "trades", "health"):
        r = c.get(f"/api/{ep}")
        assert r.status_code == 200, ep
    s = c.get("/api/summary").json()
    assert s["n_positions"] == 3 and s["equity"] > 90_000
    assert len(c.get("/api/equity").json()["rows"]) == 2
    assert c.get("/api/decision").json()["available"] is True
    assert c.get("/api/positions").json()["positions"][0]["ticker"]
