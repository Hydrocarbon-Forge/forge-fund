"""Paper-trading cycle: event-sourced on top of the shared Nautilus simulation (engine/sim.py).

State of record = ``state/ledger.jsonl``: one line per daily cycle holding the prices used for the
09:35 ET fills and the Plan (target weights) that was executed. Every cycle replays the whole ledger
through a fresh Nautilus engine (deterministic: fills depend only on ledger prices + plan weights) and
runs the live decision pipeline for today's date. Everything the dashboard reads
(portfolio.json, equity_curve.jsonl, trades.jsonl, latest_*.json) is DERIVED from that replay, so the
state files can never drift out of sync with the engine.

Structurally paper-only: the engine is Nautilus' backtest engine with one simulated venue; there is no
live data client, execution client or broker adapter anywhere (tests/test_no_live.py enforces this).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.sim import SimResult, SimRunner
from forge.config import BENCHMARK, MODE, STARTING_CAPITAL, Paths
from forge.types import Plan, PortfolioView

log = logging.getLogger("forge.paper")


# ------------------------------------------------------------------------------------ ledger io
def ledger_path(paths: Paths) -> Path:
    return paths.state / "ledger.jsonl"


def read_ledger(paths: Paths) -> list[dict]:
    p = ledger_path(paths)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["date"])


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_ledger(paths: Paths, rows: list[dict]) -> None:
    _atomic_write(ledger_path(paths), "".join(json.dumps(r, default=str) + "\n" for r in sorted(rows, key=lambda r: r["date"])))


def plan_to_json(plan: Plan) -> dict:
    return {"targets": plan.targets, "halted": plan.halted, "meta": plan.meta}


def plan_from_json(d: dict) -> Plan:
    return Plan(d.get("targets"), bool(d.get("halted")), d.get("meta") or {})


# ------------------------------------------------------------------------------------ cycle
def run_cycle(*, paths: Paths, market, decision_date: str, live_prices: dict[str, float], price_source: str,
              live_decide, universe: list[str], capital: float = STARTING_CAPITAL, replace: bool = False,
              persist: bool = True) -> dict:
    """Replay the ledger + run today's decision. Returns a summary dict (and writes state/ when persist)."""
    ledger = [r for r in read_ledger(paths) if r["date"] != decision_date or not replace]
    if any(r["date"] == decision_date for r in ledger):
        raise RuntimeError(f"cycle for {decision_date} already in ledger (use replace=True / --force)")
    past = [r for r in ledger if r["date"] < decision_date]
    inception = past[0]["date"] if past else decision_date

    stored = {r["date"]: plan_from_json(r["plan"]) for r in past}
    overrides = {r["date"]: r["prices"] for r in past}
    overrides[decision_date] = live_prices
    stored_dates = set(stored)

    def decide(date: str, view: PortfolioView) -> Plan:
        if date in stored_dates:
            return stored[date]                          # replay: no Claude, no Kronos
        return live_decide(date, view)                   # today: the live pipeline

    close = market.p["close"]
    open_ = market.p["open"]
    win = close.index[(close.index >= pd.Timestamp(inception)) & (close.index < pd.Timestamp(decision_date))]
    runner = SimRunner(open_.loc[win, universe], close.loc[win, universe], capital, decide,
                       [r["date"] for r in past] + [decision_date], price_overrides=overrides,
                       live_date=decision_date, tickers=universe)
    res = runner.run()
    if res.errors:
        raise RuntimeError(f"engine errors: {res.errors[:3]}")
    plan = res.plans[decision_date]

    entry = {"date": decision_date, "ts": datetime.now(timezone.utc).isoformat(), "mode": MODE,
             "price_source": price_source, "prices": {k: float(v) for k, v in live_prices.items()},
             "plan": plan_to_json(plan)}
    summary = {"date": decision_date, "inception": inception, "equity": res.final_equity, "cash": res.final_cash,
               "n_positions": len(res.final_positions), "status": plan.meta.get("status"), "halted": plan.halted,
               "n_fills_today": int((res.fills["date"] == decision_date).sum()) if len(res.fills) else 0}
    if persist:
        write_ledger(paths, ledger + [entry])
        # benchmark reference = SPY price the fund 'paid' on inception day (ledger price), else last close before it
        first_prices = past[0]["prices"] if past else live_prices
        ref = first_prices.get(BENCHMARK)
        if not ref:
            prior = market.p["close"][BENCHMARK].loc[: pd.Timestamp(inception) - pd.Timedelta(days=1)].dropna()
            ref = float(prior.iloc[-1]) if len(prior) else None
        write_state(paths, res, plan, entry, inception, capital, market, live_prices, ref)
    summary["result"] = res
    return summary


# ------------------------------------------------------------------------------------ state files
def write_state(paths: Paths, res: SimResult, plan: Plan, entry: dict, inception: str, capital: float,
                market, live_prices: dict[str, float], ref_px: float | None) -> None:
    st = paths.state
    today = entry["date"]
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    equity = res.final_equity
    positions = []
    for t, p in sorted(res.final_positions.items(), key=lambda kv: -kv[1]["value"]):
        positions.append({"ticker": t, "qty": int(p["qty"]), "avg_px": round(p["avg_px"], 4), "last_px": round(p["last_px"], 4),
                          "market_value": round(p["value"], 2), "unrealized_pnl": round((p["last_px"] - p["avg_px"]) * p["qty"], 2),
                          "unrealized_pct": round(p["last_px"] / p["avg_px"] - 1, 6) if p["avg_px"] else 0.0,
                          "weight": round(p["value"] / equity, 6) if equity else 0.0})
    halted = bool(plan.halted) or (st / "halt.json").exists()
    halt_reason = plan.meta.get("error") if plan.halted else None
    if halted and not halt_reason and (st / "halt.json").exists():
        try:
            halt_reason = json.loads((st / "halt.json").read_text()).get("reason")
        except (OSError, json.JSONDecodeError):
            halt_reason = "halt.json present"
    pos_value = sum(p["market_value"] for p in positions)
    _atomic_write(st / "portfolio.json", json.dumps({
        "as_of": now_iso, "mode": MODE, "inception": inception, "starting_capital": capital,
        "cash": round(res.final_cash, 2), "positions_value": round(pos_value, 2), "equity": round(equity, 2),
        "price_source": entry["price_source"], "halted": halted, "halt_reason": halt_reason, "positions": positions}, indent=1))

    # equity curve: replayed 16:00 marks + today's live snapshot; SPY buy&hold scaled from inception reference
    spy = market.p["close"][BENCHMARK]
    curve = {}
    for _, r in res.marks.iterrows():
        curve[r["date"]] = {"date": r["date"], "equity": round(r["equity"], 2), "cash": round(r["cash"], 2)}
    curve[today] = {"date": today, "equity": round(equity, 2), "cash": round(res.final_cash, 2)}
    rows, peak = [], 0.0
    for d in sorted(curve):
        row = curve[d]
        px = live_prices.get(BENCHMARK) if d == today else (float(spy.loc[:pd.Timestamp(d)].iloc[-1]) if len(spy.loc[:pd.Timestamp(d)]) else None)
        row["benchmark_equity"] = round(capital * px / ref_px, 2) if px and ref_px else capital
        peak = max(peak, row["equity"])
        row["drawdown"] = round(row["equity"] / peak - 1, 6) if peak else 0.0
        rows.append(row)
    _atomic_write(st / "equity_curve.jsonl", "".join(json.dumps(r) + "\n" for r in rows))

    trades = []
    for f in res.fills.itertuples():
        trades.append({"ts": f.ts, "date": f.date, "ticker": f.ticker, "side": f.side, "qty": int(f.qty),
                       "price": round(f.price, 4), "notional": round(f.notional, 2), "fee": round(f.fee, 4),
                       "realized_pnl": round(f.realized_pnl, 2), "reason": "rebalance"})
    _atomic_write(st / "trades.jsonl", "".join(json.dumps(t) + "\n" for t in trades))

    m = plan.meta
    _atomic_write(st / "latest_decision.json", json.dumps({
        "date": today, "ts": entry["ts"], "model": m.get("model"), "status": m.get("status"),
        "risk_view": m.get("risk_view"), "cash": m.get("cash"), "notes": m.get("notes"),
        "prompt_hash": m.get("prompt_hash"), "usage": m.get("usage"), "cost_usd": m.get("cost_usd"),
        "claude_holdings": m.get("claude_holdings") or {}, "final_weights": plan.targets or {},
        "optimizer": m.get("optimizer"), "error": m.get("error")}, indent=1, default=str))
