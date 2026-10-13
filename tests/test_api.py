"""Dashboard API + page + server guard. TestClient only: nothing binds a port.

Fixtures are written into the tmp FORGE_ROOT that tests/conftest.py provides, following
docs/CONTRACTS.md. Expected values are computed from the same files the API reads.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import server
from api.app import app as module_app
from api.app import create_app
from forge.config import STARTING_CAPITAL

ROOT = Path(__file__).resolve().parent.parent
END = date(2026, 10, 13)
START = STARTING_CAPITAL
DISCLAIMER = "Paper trading only. No guarantee of profit; performance is measured, not promised."


def _secret() -> str:
    return os.environ["CLAUDE_LOCAL_TOKEN"]  # set per test by conftest


def _sessions(end: date, n: int) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _position(ticker: str, qty: int, avg: float, last: float, equity: float) -> dict:
    mv = round(qty * last, 2)
    return {
        "ticker": ticker, "qty": qty, "avg_px": avg, "last_px": last, "market_value": mv,
        "unrealized_pnl": round((last - avg) * qty, 2),
        "unrealized_pct": round(last / avg - 1.0, 6),
        "weight": round(mv / equity, 6),
    }


def write_populated(root: Path, n_sessions: int = 30, n_calls: int = 60) -> dict:
    """Write every CONTRACTS.md file under ``root``; return the numbers the tests check."""
    rng = random.Random(42)
    state, logs, bt = root / "state", root / "logs", root / "backtests"
    for d in (state, logs, bt):
        d.mkdir(parents=True, exist_ok=True)

    sessions = _sessions(END, n_sessions)
    fund, spy = [START], [START]
    for _ in sessions[1:]:
        fund.append(fund[-1] * (1 + rng.gauss(0.0006, 0.009)))
        spy.append(spy[-1] * (1 + rng.gauss(0.0003, 0.006)))
    fund = [round(x, 2) for x in fund]
    spy = [round(x, 2) for x in spy]

    # equity_curve.jsonl: one line per session
    curve, peak = [], 0.0
    for day, eq, bench in zip(sessions, fund, spy):
        peak = max(peak, eq)
        curve.append({"date": day.isoformat(), "equity": eq, "cash": round(eq * 0.38, 2),
                      "benchmark_equity": bench, "drawdown": round(eq / peak - 1.0, 6)})
    (state / "equity_curve.jsonl").write_text(_jsonl(curve))

    # portfolio.json (positions deliberately unsorted in the file)
    equity = fund[-1]
    positions = [
        _position("XOM", 70, 110.00, 87.10, equity),
        _position("NVDA", 140, 128.20, 157.90, equity),
        _position("JPM", 45, 180.50, 186.70, equity),
        _position("AAPL", 100, 150.12, 152.30, equity),
        _position("MSFT", 30, 300.00, 326.60, equity),
    ]
    positions_value = round(sum(p["market_value"] for p in positions), 2)
    cash = round(equity - positions_value, 2)
    portfolio = {"as_of": f"{END.isoformat()}T15:58:02-04:00", "mode": "paper",
                 "inception": sessions[0].isoformat(), "starting_capital": START,
                 "cash": cash, "positions_value": positions_value, "equity": equity,
                 "price_source": "live_1m", "halted": False, "halt_reason": None,
                 "positions": positions}
    (state / "portfolio.json").write_text(json.dumps(portfolio, indent=2))

    # trades.jsonl: opening buys, then partial sells with realized pnl (append-only)
    trades = []
    for i, p in enumerate(positions):
        trades.append({"ts": f"{sessions[0].isoformat()}T09:35:{10 + i:02d}-04:00",
                       "date": sessions[0].isoformat(), "ticker": p["ticker"], "side": "BUY",
                       "qty": p["qty"], "price": p["avg_px"],
                       "notional": round(p["qty"] * p["avg_px"], 2), "fee": 1.0,
                       "realized_pnl": 0.0, "reason": "rebalance"})
    sells = [("AAPL", 20, 160.00, 150.12, sessions[10]),
             ("AAPL", 10, 158.00, 150.12, sessions[25]),
             ("JNJ", 40, 146.00, 150.00, sessions[20])]
    for ticker, qty, px, avg, day in sells:
        trades.append({"ts": f"{day.isoformat()}T11:02:45-04:00", "date": day.isoformat(),
                       "ticker": ticker, "side": "SELL", "qty": qty, "price": px,
                       "notional": round(qty * px, 2), "fee": 1.0,
                       "realized_pnl": round((px - avg) * qty, 2), "reason": "trim"})
    trades.sort(key=lambda t: t["ts"])
    (state / "trades.jsonl").write_text(_jsonl(trades))

    # latest_decision.json: Claude's raw weights next to what was executed
    claude = {
        "NVDA": {"weight": 0.22, "rationale": "Leadership intact; keep the largest weight."},
        "AAPL": {"weight": 0.12, "rationale": "Range-bound; hold near neutral."},
        "MSFT": {"weight": 0.13, "rationale": "Earnings revisions still rising."},
        "JPM": {"weight": 0.10, "rationale": "Steeper curve helps net interest income."},
        "XOM": {"weight": 0.06, "rationale": "Hedge against an energy spike."},
    }
    final = {"NVDA": 0.21, "AAPL": 0.11, "MSFT": 0.12, "JPM": 0.10, "XOM": 0.06}
    decision = {
        "date": END.isoformat(), "ts": f"{END.isoformat()}T09:36:44-04:00", "model": "haiku",
        "status": "ok", "risk_view": "med", "cash": round(1 - sum(final.values()), 4),
        "notes": "Breadth narrowing. <img src=x onerror=alert(1)> stays as text.",
        "prompt_hash": "ab12cd34ef56", "usage": {"prompt_tokens": 8000, "completion_tokens": 600},
        "cost_usd": 0.0123, "claude_holdings": claude, "final_weights": final,
        "error": f"upstream rejected the request (token {_secret()})",
    }
    (state / "latest_decision.json").write_text(json.dumps(decision, indent=2))

    # latest_forecasts.json: rows shuffled on disk, API must sort by score desc
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "PG", "KO", "WMT"]
    frows = []
    for t in tickers:
        exp_ret = rng.gauss(0.0, 0.015)
        frows.append({"ticker": t, "exp_ret": round(exp_ret, 5),
                      "p_up": round(min(0.95, max(0.05, 0.5 + exp_ret * 10)), 3),
                      "dispersion": round(abs(rng.gauss(0.011, 0.003)), 5),
                      "score": round(exp_ret / 0.015, 4)})
    rng.shuffle(frows)
    (state / "latest_forecasts.json").write_text(json.dumps({
        "asof": sessions[-2].isoformat(), "model": "Kronos-mini", "horizon_days": 5,
        "generated_at": f"{END.isoformat()}T07:02:19-04:00", "rows": frows}, indent=2))

    # decision_log.jsonl: routine calls, one failed repair with a secret in its error text
    log_rows = []
    for i in range(n_calls):
        day = sessions[i % len(sessions)]
        log_rows.append({"kind": "call", "ts": f"{day.isoformat()}T09:36:{i % 60:02d}-04:00",
                         "purpose": "decision", "model": "haiku", "prompt_hash": "ab12cd34ef56",
                         "prompt_chars": 31000, "status": "ok", "attempts": 1, "latency_s": 3.2,
                         "usage": {"prompt_tokens": 8000, "completion_tokens": 600},
                         "cost_usd": 0.012, "error": None})
    log_rows.append({"kind": "call", "ts": f"{END.isoformat()}T16:00:00-04:00", "purpose": "repair",
                     "model": "haiku", "prompt_hash": "00c0ffee0042", "prompt_chars": 9000,
                     "status": "error", "attempts": 2, "latency_s": 1.1,
                     "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "cost_usd": 0.0,
                     "error": f"HTTP 401 for token {_secret()} at /home/someone/.env"})
    log_rows.insert(3, {"kind": "decision", "ts": "2026-10-06T09:36:44-04:00",
                        "date": "2026-10-06", "prompt_hash": "ab12cd34ef56", "status": "ok",
                        "decision": {"risk_view": "low"}})
    (logs / "decision_log.jsonl").write_text(_jsonl(log_rows))

    # backtests: an older run and a newer run; the newer one must be chosen
    old, new = bt / "run-2026-06-01", bt / "run-2026-10-01"
    old.mkdir(parents=True, exist_ok=True)
    new.mkdir(parents=True, exist_ok=True)
    (old / "metrics.json").write_text(json.dumps({"cagr": 0.05, "run": "old"}))
    new_metrics = {"cagr": 0.187, "max_drawdown": -0.214, "sharpe": 1.12, "win_rate": 0.537,
                   "benchmark_cagr": 0.142, "excess_cagr": 0.045}
    (new / "metrics.json").write_text(json.dumps(new_metrics))
    (new / "report.txt").write_text("report\n")
    now = os.stat(new).st_mtime
    os.utime(old, (now - 3600, now - 3600))

    return {
        "sessions": sessions, "fund": fund, "spy": spy, "equity": equity, "cash": cash,
        "positions_value": positions_value, "positions": positions, "trades": trades,
        "forecast_rows": frows, "n_calls": n_calls + 1, "log_rows": log_rows,
        "backtest_run": new.name, "backtest_metrics": new_metrics,
    }


@pytest.fixture
def populated(tmp_path):
    return write_populated(tmp_path)


@pytest.fixture
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _get(client: TestClient, path: str):
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code, r.text[:300])
    return r


ENDPOINTS = ["/api/health", "/api/summary", "/api/equity", "/api/positions", "/api/decision",
             "/api/forecasts", "/api/pnl", "/api/trades", "/api/backtest", "/api/calls", "/"]


# ----------------------------------------------------------------- happy path
def test_every_endpoint_returns_200_and_expected_shape(client, populated):
    health = _get(client, "/api/health").json()
    assert health["ok"] is True and health["mode"] == "paper"
    assert health["last_run"] == f"{END.isoformat()}T15:58:02-04:00"
    assert health["halted"] is False and health["halt_reason"] is None
    assert "now" in health

    summary = _get(client, "/api/summary").json()
    for key in ("equity", "cash", "cash_pct", "positions_value", "n_positions", "starting_capital",
                "total_return_pct", "day_pnl", "day_return_pct", "benchmark_return_pct",
                "excess_return_pct", "max_drawdown_pct", "halted", "price_source", "as_of",
                "inception", "available"):
        assert key in summary, key
    assert summary["available"] is True

    eq = _get(client, "/api/equity").json()
    assert eq["available"] is True and len(eq["rows"]) == len(populated["sessions"])
    assert set(eq["rows"][0]) == {"date", "equity", "cash", "benchmark_equity", "drawdown"}

    pos = _get(client, "/api/positions").json()
    assert pos["available"] is True and {"positions", "cash", "equity"} <= set(pos)

    dec = _get(client, "/api/decision").json()
    assert dec["available"] is True and dec["model"] == "haiku"

    fc = _get(client, "/api/forecasts").json()
    assert fc["available"] is True and fc["horizon_days"] == 5 and len(fc["rows"]) == 12

    pnl = _get(client, "/api/pnl").json()
    assert pnl["available"] is True and {"rows", "total", "periods"} <= set(pnl)

    tr = _get(client, "/api/trades").json()
    assert tr["available"] is True and tr["limit"] == 100

    bt = _get(client, "/api/backtest").json()
    assert bt["available"] is True and bt["run_id"] == populated["backtest_run"]

    calls = _get(client, "/api/calls").json()
    assert calls["available"] is True and {"calls", "totals"} <= set(calls)

    html = _get(client, "/")
    assert "text/html" in html.headers["content-type"]


def test_summary_numbers_match_the_files(client, populated):
    s = _get(client, "/api/summary").json()
    fund, spy = populated["fund"], populated["spy"]
    eq = populated["equity"]
    assert s["equity"] == pytest.approx(eq)
    assert s["cash"] == pytest.approx(populated["cash"])
    assert s["cash_pct"] == pytest.approx(populated["cash"] / eq)
    assert s["positions_value"] == pytest.approx(populated["positions_value"])
    assert s["n_positions"] == 5
    assert s["total_return_pct"] == pytest.approx(eq / START - 1)
    assert s["benchmark_return_pct"] == pytest.approx(spy[-1] / START - 1)
    assert s["excess_return_pct"] == pytest.approx(s["total_return_pct"] - s["benchmark_return_pct"])
    assert s["day_pnl"] == pytest.approx(fund[-1] - fund[-2])
    assert s["day_return_pct"] == pytest.approx(fund[-1] / fund[-2] - 1)
    peak, worst = fund[0], 0.0
    for v in fund:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    assert s["max_drawdown_pct"] == pytest.approx(worst)
    assert s["max_drawdown_pct"] < 0
    assert s["price_source"] == "live_1m" and s["inception"] == populated["sessions"][0].isoformat()


def test_equity_is_capped_to_last_2000_rows(client, tmp_path):
    write_populated(tmp_path, n_sessions=2500)
    rows = _get(client, "/api/equity").json()["rows"]
    assert len(rows) == 2000
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates) and dates[-1] == END.isoformat()


def test_positions_sorted_by_market_value_desc(client, populated):
    pos = _get(client, "/api/positions").json()["positions"]
    mvs = [p["market_value"] for p in pos]
    assert mvs == sorted(mvs, reverse=True)
    assert pos[0]["ticker"] == "NVDA"
    assert set(pos[0]) == {"ticker", "qty", "avg_px", "last_px", "market_value",
                           "unrealized_pnl", "unrealized_pct", "weight"}


def test_decision_keeps_raw_and_final_weights_and_rationales(client, populated):
    dec = _get(client, "/api/decision").json()
    assert dec["claude_holdings"]["NVDA"]["weight"] == pytest.approx(0.22)
    assert dec["claude_holdings"]["NVDA"]["rationale"].startswith("Leadership")
    assert dec["final_weights"]["NVDA"] == pytest.approx(0.21)
    assert dec["usage"] == {"prompt_tokens": 8000, "completion_tokens": 600}
    assert dec["cost_usd"] == pytest.approx(0.0123)


def test_forecasts_sorted_by_score_desc(client, populated):
    rows = _get(client, "/api/forecasts").json()["rows"]
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert {r["ticker"] for r in rows} == {r["ticker"] for r in populated["forecast_rows"]}


def test_pnl_rows_totals_and_period_returns(client, populated):
    pnl = _get(client, "/api/pnl").json()
    by_t = {r["ticker"]: r for r in pnl["rows"]}
    aapl_realized = round((160.00 - 150.12) * 20, 2) + round((158.00 - 150.12) * 10, 2)
    assert by_t["AAPL"]["realized_pnl"] == pytest.approx(aapl_realized)
    assert by_t["JNJ"]["realized_pnl"] == pytest.approx(-160.0)   # sold out, no open position
    assert by_t["JNJ"]["unrealized_pnl"] == pytest.approx(0.0)
    unreal = sum(p["unrealized_pnl"] for p in populated["positions"])
    assert pnl["total"]["unrealized_pnl"] == pytest.approx(unreal)
    assert pnl["total"]["realized_pnl"] == pytest.approx(aapl_realized - 160.0)
    assert pnl["total"]["total_pnl"] == pytest.approx(unreal + aapl_realized - 160.0)

    periods = {p["key"]: p for p in pnl["periods"]}
    assert set(periods) == {"1d", "5d", "21d", "since_inception"}
    fund, spy = populated["fund"], populated["spy"]
    assert periods["1d"]["fund"] == pytest.approx(fund[-1] / fund[-2] - 1)
    assert periods["5d"]["fund"] == pytest.approx(fund[-1] / fund[-6] - 1)
    assert periods["21d"]["spy"] == pytest.approx(spy[-1] / spy[-22] - 1)
    assert periods["since_inception"]["fund"] == pytest.approx(fund[-1] / START - 1)
    assert periods["since_inception"]["spy"] == pytest.approx(spy[-1] / START - 1)


def test_trades_newest_first_and_limit_is_clamped(client, populated):
    trades = populated["trades"]
    body = _get(client, "/api/trades?limit=3").json()
    assert body["limit"] == 3 and len(body["trades"]) == 3
    assert body["trades"][0]["ts"] == trades[-1]["ts"]
    assert body["trades"][0]["ticker"] == trades[-1]["ticker"]
    assert set(body["trades"][0]) == {"ts", "date", "ticker", "side", "qty", "price", "notional",
                                      "fee", "realized_pnl", "reason"}

    assert _get(client, "/api/trades?limit=0").json()["limit"] == 1
    assert len(_get(client, "/api/trades?limit=0").json()["trades"]) == 1
    assert _get(client, "/api/trades?limit=-7").json()["limit"] == 1
    big = _get(client, "/api/trades?limit=100000").json()
    assert big["limit"] == 500 and len(big["trades"]) == len(trades)
    assert _get(client, "/api/trades").json()["limit"] == 100
    assert client.get("/api/trades?limit=abc").status_code < 500


def test_backtest_picks_most_recently_modified_run(client, populated):
    bt = _get(client, "/api/backtest").json()
    assert bt["run_id"] == "run-2026-10-01"
    assert bt["cagr"] == pytest.approx(0.187)
    assert bt["sharpe"] == pytest.approx(1.12)
    assert bt["available"] is True


def test_calls_last_50_totals_and_no_error_text(client, populated):
    body = _get(client, "/api/calls").json()
    assert body["totals"]["n_calls"] == populated["n_calls"]
    assert len(body["calls"]) == 50
    assert body["calls"][0]["purpose"] == "repair"        # newest first
    for c in body["calls"]:
        assert set(c) == {"ts", "purpose", "model", "status", "latency_s", "usage", "cost_usd"}
    calls = [r for r in populated["log_rows"] if r["kind"] == "call"]
    assert body["totals"]["total_cost_usd"] == pytest.approx(sum(c["cost_usd"] for c in calls))
    prompt = sum(c["usage"]["prompt_tokens"] for c in calls)
    completion = sum(c["usage"]["completion_tokens"] for c in calls)
    assert body["totals"]["prompt_tokens"] == prompt
    assert body["totals"]["completion_tokens"] == completion
    assert body["totals"]["total_tokens"] == prompt + completion
    # the failing call's error text (which holds a token) must never leave the server
    assert _secret() not in client.get("/api/calls").text
    assert "error" not in body["calls"][0]


def test_no_secret_or_filesystem_path_in_any_response(client, populated, tmp_path):
    bodies = [client.get(p).text for p in ENDPOINTS + ["/api/equity?x=1", "/api/nope"]]
    everything = "\n".join(bodies)
    assert _secret() not in everything
    assert "REDACTED" in everything                        # decision error was scrubbed, not dropped
    assert str(tmp_path) not in everything
    assert "claude.test" not in everything                 # CLAUDE_BASE_URL never exposed
    assert "/home/" not in everything


def test_decision_error_text_is_scrubbed_before_it_leaves(client, populated):
    dec = _get(client, "/api/decision").json()
    assert _secret() not in json.dumps(dec)
    assert "<img" in dec["notes"]   # LLM text is data: passed through, rendered as text by the page


def test_halt_file_and_portfolio_flag_are_reported(client, populated):
    path = Path(os.environ["FORGE_ROOT"]) / "state"
    (path / "halt.json").write_text(json.dumps({"ts": "x", "reason": "loss cap hit",
                                                "loss_pct": -0.031}))
    h = _get(client, "/api/health").json()
    assert h["halted"] is True and h["halt_reason"] == "loss cap hit"
    s = _get(client, "/api/summary").json()
    assert s["halted"] is True and s["halt_reason"] == "loss cap hit"


# ------------------------------------------------------------ degraded inputs
def test_missing_files_return_unavailable_not_500(client):
    h = _get(client, "/api/health").json()
    assert h["ok"] is True and h["last_run"] is None and h["halted"] is False
    assert _get(client, "/api/summary").json()["available"] is False
    assert _get(client, "/api/equity").json() == {"available": False, "rows": []}
    pos = _get(client, "/api/positions").json()
    assert pos == {"available": False, "positions": [], "cash": None, "equity": None}
    assert _get(client, "/api/decision").json()["available"] is False
    fc = _get(client, "/api/forecasts").json()
    assert fc["available"] is False and fc["rows"] == []
    pnl = _get(client, "/api/pnl").json()
    assert pnl["available"] is False and pnl["rows"] == []
    assert _get(client, "/api/trades").json()["trades"] == []
    assert _get(client, "/api/backtest").json() == {"available": False}
    calls = _get(client, "/api/calls").json()
    assert calls["available"] is False and calls["totals"]["n_calls"] == 0


def test_empty_files_are_unavailable(client, tmp_path):
    for d in ("state", "logs"):
        (tmp_path / d).mkdir(exist_ok=True)
    for rel in ("state/portfolio.json", "state/equity_curve.jsonl", "state/trades.jsonl",
                "state/latest_decision.json", "state/latest_forecasts.json",
                "logs/decision_log.jsonl"):
        (tmp_path / rel).write_bytes(b"")
    assert _get(client, "/api/summary").json()["available"] is False
    assert _get(client, "/api/decision").json()["available"] is False
    assert _get(client, "/api/forecasts").json()["available"] is False
    assert _get(client, "/api/calls").json()["available"] is False
    assert _get(client, "/api/trades").json()["available"] is False


def test_corrupt_files_do_not_500(client, tmp_path):
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "backtests" / "bad").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "portfolio.json").write_text("{not json at all")
    (tmp_path / "state" / "latest_decision.json").write_bytes(b"\xff\xfe\x00garbage")
    (tmp_path / "state" / "latest_forecasts.json").write_text('{"rows": [1, "x", null]')
    (tmp_path / "state" / "equity_curve.jsonl").write_text("{broken\n[1,2]\n")
    (tmp_path / "state" / "trades.jsonl").write_text("}}}\n")
    (tmp_path / "state" / "halt.json").write_text("<html>not json")
    (tmp_path / "logs" / "decision_log.jsonl").write_text('{"kind": "call", "usage": \n')
    (tmp_path / "backtests" / "bad" / "metrics.json").write_text("[oops")
    for path in ENDPOINTS + ["/api/summary", "/api/pnl", "/api/calls"]:
        r = client.get(path)
        assert r.status_code == 200, path
    h = _get(client, "/api/health").json()
    assert h["halted"] is True            # halt.json present => halted, even if unreadable
    assert _get(client, "/api/summary").json()["available"] is False
    assert _get(client, "/api/backtest").json() == {"available": False}


def test_partially_corrupt_curve_keeps_the_good_lines(client, tmp_path):
    (tmp_path / "state").mkdir(exist_ok=True)
    good = {"date": "2026-10-13", "equity": 100321.12, "cash": 12345.67,
            "benchmark_equity": 100110.4, "drawdown": -0.004}
    (tmp_path / "state" / "equity_curve.jsonl").write_text(
        json.dumps(good) + "\n" + '{"date": "2026-10-14", "equity": 10')  # half-written tail
    rows = _get(client, "/api/equity").json()
    assert rows["available"] is True and len(rows["rows"]) == 1
    assert rows["rows"][0]["equity"] == pytest.approx(100321.12)


def test_nan_and_wrong_types_are_sanitised(client, populated):
    state = Path(os.environ["FORGE_ROOT"]) / "state"
    (state / "portfolio.json").write_text(
        '{"as_of": "x", "equity": NaN, "cash": "lots", "positions": "nope", '
        '"positions_value": Infinity, "starting_capital": 100000.0}')
    body = _get(client, "/api/summary").json()
    assert body["equity"] == pytest.approx(populated["fund"][-1])   # falls back to the curve
    assert body["cash"] is None and body["cash_pct"] is None
    assert body["n_positions"] == 0
    assert "NaN" not in client.get("/api/summary").text
    assert "Infinity" not in client.get("/api/summary").text
    assert _get(client, "/api/positions").json()["positions"] == []


# ----------------------------------------------------------------------- page
def test_index_page_is_paper_labelled_offline_and_textcontent_only(client):
    html = _get(client, "/").text
    assert "PAPER" in html and "forge-fund" in html
    assert DISCLAIMER in html
    assert not re.search(r'(?:src|href)\s*=\s*["\']?https?:', html), "no external assets"
    assert "@import" not in html and "fonts.googleapis" not in html
    assert "innerHTML" not in html and "outerHTML" not in html and "insertAdjacentHTML" not in html
    assert "prefers-color-scheme: dark" in html
    csp = _get(client, "/").headers.get("content-security-policy", "")
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp


def test_untrusted_host_header_is_rejected(tmp_path):
    evil = TestClient(create_app(), base_url="http://evil.example")
    assert evil.get("/api/health").status_code == 400


def test_api_responses_are_not_cached_and_are_json(client, populated):
    r = _get(client, "/api/summary")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-type"].startswith("application/json")


def test_module_level_app_is_a_fastapi_app():
    assert isinstance(module_app, FastAPI)


# --------------------------------------------------------------- server config
def test_server_main_binds_lan_8905(monkeypatch):
    calls = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    server.main()
    server.main(host="192.168.1.201", port=8905)
    (a0, k0), (a1, k1) = calls
    assert a0 == ("api.app:app",) and k0["host"] == "192.168.1.201" and k0["port"] == 8905
    assert k0["log_level"] == "info"
    assert k1["host"] == "192.168.1.201" and k1["port"] == 8905


def test_server_main_rejects_bad_port(monkeypatch):
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(ValueError):
        server.main(port=0)


# ------------------------------------------------------------- demo helper
def _load_demo():
    spec = importlib.util.spec_from_file_location("make_demo_state", ROOT / "scripts" / "make_demo_state.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_helper_refuses_the_real_state_directory():
    demo = _load_demo()
    with pytest.raises(SystemExit):
        demo._refuse_real_dirs(demo.PROJECT_ROOT / "state")
    with pytest.raises(SystemExit):
        demo._refuse_real_dirs(demo.PROJECT_ROOT / "state" / "nested")
    with pytest.raises(SystemExit):
        demo._refuse_real_dirs(demo.PROJECT_ROOT)            # contains state/
    demo._refuse_real_dirs(Path(os.environ["FORGE_ROOT"]) / "demo")   # a tmp dir is fine


def test_demo_state_is_readable_by_the_api(tmp_path, monkeypatch):
    demo = _load_demo()
    target = tmp_path / "demo"
    assert demo.main(["make_demo_state.py", str(target)]) == 0
    monkeypatch.setenv("FORGE_ROOT", str(target))
    c = TestClient(create_app(), base_url="http://127.0.0.1")
    s = _get(c, "/api/summary").json()
    assert s["available"] is True and s["n_positions"] == 8
    assert len(_get(c, "/api/equity").json()["rows"]) == 30
    assert _get(c, "/api/decision").json()["status"] == "ok"
    assert _get(c, "/api/backtest").json()["available"] is True
    assert _secret() not in c.get("/api/calls").text
