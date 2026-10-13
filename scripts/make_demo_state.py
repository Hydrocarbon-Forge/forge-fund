#!/usr/bin/env python3
"""Write DEMO state files so the dashboard can be tried without a live engine.

    .venv/bin/python scripts/make_demo_state.py [DIR]
    FORGE_ROOT=DIR .venv/bin/python -m api.server      # then open http://127.0.0.1:8905

With no DIR a fresh temp directory is created and its path is printed. The script
refuses to write into the project's real state/, logs/, backtests/ or data/ folders.
Every number is synthetic and the notes say so.
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forge.config import STARTING_CAPITAL, estimate_cost_usd  # noqa: E402

PROTECTED = ("state", "logs", "backtests", "data")
SEED = 20261013
SESSIONS = 30
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "PG",
    "V", "KO", "WMT", "HD", "UNH", "BRK-B", "LLY", "AVGO", "COST", "MRK",
    "PEP", "CVX", "ABBV", "TMO", "ORCL", "CSCO", "MCD", "ACN", "ABT", "DHR",
]
# ticker -> target weight of equity (sums to ~0.86, leaving ~14% cash)
HOLDINGS = {
    "NVDA": 0.15, "MSFT": 0.13, "AAPL": 0.12, "JPM": 0.11,
    "LLY": 0.10, "XOM": 0.09, "COST": 0.09, "PG": 0.07,
}
RATIONALES = {
    "NVDA": "Momentum leader; keeping the largest weight but trimmed into strength.",
    "MSFT": "Steady earnings revisions; holds as a core quality name.",
    "AAPL": "Range-bound; weight held near neutral pending the next print.",
    "JPM": "Rate-sensitive financials; modest overweight while curve steepens.",
    "LLY": "Strong pipeline news; sized below the cap to limit gap risk.",
    "XOM": "Energy bid into the regime change; kept small as a hedge.",
    "COST": "Defensive grocer-retail; supports the book when breadth narrows.",
    "PG": "Low-beta staple; the ballast position.",
}


def _refuse_real_dirs(target: Path) -> None:
    """Compare resolved paths: never write where the engine keeps its real data."""
    resolved = target.resolve()
    for name in PROTECTED:
        real = (PROJECT_ROOT / name).resolve()
        if resolved == real or real in resolved.parents or (resolved / name) == real:
            raise SystemExit(
                f"refusing to write demo state into {resolved}: that is (or contains) "
                f"the project's real {name}/ directory. Pick another DIR."
            )


def _sessions(end: date) -> list[date]:
    day = end
    while day.weekday() >= 5:  # roll back over the weekend
        day -= timedelta(days=1)
    out: list[date] = []
    while len(out) < SESSIONS:
        if day.weekday() < 5:
            out.append(day)
        day -= timedelta(days=1)
    return sorted(out)


def _ts(day: date, hhmm: str = "09:35:00") -> str:
    return f"{day.isoformat()}T{hhmm}-04:00"


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build(root: Path, end: date, rng: random.Random) -> None:
    state, logs, backtests = root / "state", root / "logs", root / "backtests"
    for d in (state, logs, backtests):
        d.mkdir(parents=True, exist_ok=True)

    sessions = _sessions(end)
    inception = sessions[0]

    # --- equity curve vs SPY, both scaled to the same starting capital
    fund, spy = [STARTING_CAPITAL], [STARTING_CAPITAL]
    for _ in sessions[1:]:
        fund.append(fund[-1] * (1 + rng.gauss(0.0007, 0.0085)))
        spy.append(spy[-1] * (1 + rng.gauss(0.0004, 0.0065)))
    peak = 0.0
    curve = []
    for day, eq, bench in zip(sessions, fund, spy):
        peak = max(peak, eq)
        curve.append({
            "date": day.isoformat(),
            "equity": round(eq, 2),
            "cash": round(eq * 0.14, 2),
            "benchmark_equity": round(bench, 2),
            "drawdown": round(eq / peak - 1.0, 6),
        })
    _write_jsonl(state / "equity_curve.jsonl", curve)

    equity_now = curve[-1]["equity"]
    cash_now = round(equity_now * 0.14, 2)

    # --- prices and positions
    positions, trades = [], []
    for ticker, weight in HOLDINGS.items():
        price = rng.uniform(60, 420)
        path = [price]
        for _ in sessions[1:]:
            path.append(path[-1] * (1 + rng.gauss(0.0005, 0.017)))
        last = round(path[-1], 2)
        avg = round(last * (1 + rng.uniform(-0.12, 0.08)), 2)
        qty = int(weight * equity_now / last)
        mv = round(qty * last, 2)
        unreal = round((last - avg) * qty, 2)
        positions.append({
            "ticker": ticker, "qty": qty, "avg_px": avg, "last_px": last,
            "market_value": mv, "unrealized_pnl": unreal,
            "unrealized_pct": round(last / avg - 1.0, 6),
            "weight": round(mv / equity_now, 6),
        })
        trades.append({
            "ts": _ts(inception, "09:35:12"), "date": inception.isoformat(), "ticker": ticker,
            "side": "BUY", "qty": qty, "price": avg, "notional": round(qty * avg, 2),
            "fee": 1.0, "realized_pnl": 0.0, "reason": "rebalance",
        })
    # a couple of partial sells so the PnL table shows realized numbers
    for ticker, shares, px_mult in (("JNJ", 40, 1.06), ("HD", 25, 0.97)):
        px = round(rng.uniform(150, 330) * px_mult, 2)
        avg = round(px * 0.95, 2)
        day = sessions[len(sessions) // 2]
        trades.append({
            "ts": _ts(day, "10:14:41"), "date": day.isoformat(), "ticker": ticker,
            "side": "SELL", "qty": shares, "price": px, "notional": round(shares * px, 2),
            "fee": 1.0, "realized_pnl": round((px - avg) * shares, 2), "reason": "trim",
        })
    positions.sort(key=lambda p: -p["market_value"])
    positions_value = round(sum(p["market_value"] for p in positions), 2)
    portfolio = {
        "as_of": _ts(end, "15:58:02"), "mode": "paper", "inception": inception.isoformat(),
        "starting_capital": STARTING_CAPITAL, "cash": round(equity_now - positions_value, 2),
        "positions_value": positions_value, "equity": equity_now,
        "price_source": "last_close", "halted": False, "halt_reason": None,
        "positions": positions,
    }
    _write_json(state / "portfolio.json", portfolio)
    trades.sort(key=lambda t: t["ts"])
    _write_jsonl(state / "trades.jsonl", trades)

    # --- Kronos forecasts for the whole demo universe
    raw = {t: rng.gauss(0.0, 0.012) for t in UNIVERSE}
    mean = sum(raw.values()) / len(raw)
    sd = math.sqrt(sum((v - mean) ** 2 for v in raw.values()) / len(raw)) or 1.0
    frows = []
    for t in UNIVERSE:
        exp_ret = raw[t]
        p_up = min(0.95, max(0.05, 0.5 + exp_ret * 12 + rng.gauss(0, 0.05)))
        frows.append({
            "ticker": t, "exp_ret": round(exp_ret, 5), "p_up": round(p_up, 3),
            "dispersion": round(abs(rng.gauss(0.011, 0.003)), 5),
            "score": round((exp_ret - mean) / sd, 3),
        })
    frows.sort(key=lambda r: -r["score"])
    _write_json(state / "latest_forecasts.json", {
        "asof": sessions[-2].isoformat(), "model": "Kronos-mini", "horizon_days": 5,
        "generated_at": _ts(end, "07:02:19"), "rows": frows,
    })

    # --- last Claude decision (DEMO wording) and what was executed
    final = {p["ticker"]: round(HOLDINGS[p["ticker"]] - (0.01 if p["ticker"] == "NVDA" else 0.0), 4)
             for p in positions}
    claude = {}
    for t, w in final.items():
        claude[t] = {"weight": round(w + 0.01 if t in ("JPM", "LLY") else w, 4),
                     "rationale": RATIONALES[t]}
    usage = {"prompt_tokens": 7421, "completion_tokens": 612}
    decision = {
        "date": end.isoformat(), "ts": _ts(end, "09:36:44"), "model": "haiku", "status": "ok",
        "risk_view": "med", "cash": round(1 - sum(final.values()), 4),
        "notes": ("DEMO DATA, not a real decision. Leadership is narrowing toward mega-cap "
                  "tech while staples and energy get a bid. Keeping gross near 86 percent "
                  "and trimming the most extended winners."),
        "prompt_hash": "9f2c1ab04e7d", "usage": usage,
        "cost_usd": estimate_cost_usd("haiku", usage["prompt_tokens"], usage["completion_tokens"]),
        "claude_holdings": claude, "final_weights": final, "error": None,
    }
    _write_json(state / "latest_decision.json", decision)

    # --- decision log: one routine call per session, a weekly review, one failed repair
    log_rows: list[dict] = []
    for i, day in enumerate(sessions):
        p_tok = rng.randint(6900, 7600)
        c_tok = rng.randint(480, 680)
        log_rows.append({
            "kind": "call", "ts": _ts(day, "09:36:44"), "purpose": "decision", "model": "haiku",
            "prompt_hash": f"{rng.getrandbits(48):012x}", "prompt_chars": p_tok * 4,
            "status": "ok", "attempts": 1, "latency_s": round(rng.uniform(2.1, 6.4), 2),
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok},
            "cost_usd": estimate_cost_usd("haiku", p_tok, c_tok), "error": None,
        })
        if i % 5 == 4:
            p_tok, c_tok = rng.randint(14000, 16000), rng.randint(1100, 1500)
            log_rows.append({
                "kind": "call", "ts": _ts(day, "16:10:03"), "purpose": "review", "model": "sonnet",
                "prompt_hash": f"{rng.getrandbits(48):012x}", "prompt_chars": p_tok * 4,
                "status": "ok", "attempts": 1, "latency_s": round(rng.uniform(9, 22), 2),
                "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok},
                "cost_usd": estimate_cost_usd("sonnet", p_tok, c_tok), "error": None,
            })
        log_rows.append({
            "kind": "decision", "ts": _ts(day, "09:36:44"), "date": day.isoformat(),
            "prompt_hash": f"{rng.getrandbits(48):012x}", "status": "ok",
            "decision": {"risk_view": "med", "cash": 0.14},
        })
    log_rows.insert(len(log_rows) // 3, {
        "kind": "call", "ts": _ts(sessions[9], "09:41:20"), "purpose": "repair", "model": "haiku",
        "prompt_hash": "00c0ffee0042", "prompt_chars": 9100, "status": "error", "attempts": 3,
        "latency_s": 1.8, "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "cost_usd": 0.0,
        "error": "DEMO: upstream returned HTTP 502 (synthetic error text, never shown in the UI)",
    })
    _write_jsonl(logs / "decision_log.jsonl", log_rows)

    # --- one finished backtest run
    run = backtests / "demo-2020-12-to-2026-10"
    run.mkdir(parents=True, exist_ok=True)
    _write_json(run / "metrics.json", {
        "start": "2020-12-01", "end": end.isoformat(), "n_trades": 418, "rebalances": 1290,
        "cagr": 0.187, "max_drawdown": -0.214, "sharpe": 1.12, "sortino": 1.58,
        "volatility": 0.142, "win_rate": 0.537, "turnover_annual": 2.6,
        "benchmark_cagr": 0.142, "benchmark_max_drawdown": -0.337,
        "benchmark_sharpe": 0.71, "excess_cagr": 0.045, "final_equity": 226480.0,
    })
    (run / "report.txt").write_text(
        "DEMO backtest report (synthetic numbers for dashboard testing only).\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print("usage: make_demo_state.py [DIR]", file=sys.stderr)
        return 2
    target = Path(argv[1]) if len(argv) == 2 else Path(tempfile.mkdtemp(prefix="forge-demo-"))
    _refuse_real_dirs(target)
    end = datetime.now(timezone.utc).date()
    build(target.resolve(), end, random.Random(SEED))
    print(f"demo state written to: {target.resolve()}")
    print(f"try it: FORGE_ROOT={target.resolve()} {PROJECT_ROOT / '.venv/bin/python'} -m api.server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
