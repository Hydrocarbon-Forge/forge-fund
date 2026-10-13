"""Historical backtest of the FULL pipeline (Kronos -> Claude -> skfolio -> risk -> Nautilus execution).

    .venv/bin/python -m run.run_backtest --start 2023-01-03 --freq weekly
    .venv/bin/python -m run.run_backtest --phase kronos ...      # precompute Kronos forecasts only (CPU heavy)
    .venv/bin/python -m run.run_backtest --no-claude --freq weekly   # rule baseline ablation (no LLM calls)

Prints real metrics and saves metrics.json / report.txt / equity.csv / trades.csv / decisions.jsonl under
backtests/<run_id>/. Claude prompts are anonymised (opaque ids, no dates) so the LLM cannot recall history.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from brain.claude_client import ClaudeClient
from brain.decision import ClaudeBrain
from engine.metrics import bootstrap_excess_ci, curve_stats, rank_ic, regress, trade_stats
from engine.sim import CASH_BUFFER, FEE_RATE, HALF_SPREAD_BPS, SimRunner
from forge.config import BENCHMARK, KRONOS, LIMITS, STARTING_CAPITAL, claude_settings, get_paths
from forge.logging_setup import setup_logging
from kronos.forecast import KronosForecaster, forecast_with_cache, load_cached
from run.pipeline import DecisionPipeline, Market, PipelineConfig
from run.report import render_report

log = logging.getLogger("forge.backtest")


# ------------------------------------------------------------------------------------ schedule
def event_dates(dates: pd.DatetimeIndex, freq: str) -> list[str]:
    """Rebalance days: 'daily', 'weekly' (first trading day of each ISO week) or an integer N (every Nth day)."""
    if freq == "daily":
        pick = list(dates)
    elif freq == "weekly":
        iso = dates.isocalendar()
        key = iso.year.astype(int) * 100 + iso.week.astype(int)
        pick = [d for i, d in enumerate(dates) if i == 0 or key.iloc[i] != key.iloc[i - 1]]
    else:
        pick = list(dates[:: int(freq)])
    return [d.strftime("%Y-%m-%d") for d in pick]


# ------------------------------------------------------------------------------------ benchmarks
def spy_buy_hold(op: pd.Series, cl: pd.Series, first_event: str, cap: float, base_idx: pd.DatetimeIndex) -> pd.Series:
    """SPY bought at the first rebalance open (same spread + fee as the fund), held, marked at closes."""
    h = HALF_SPREAD_BPS / 1e4
    ask = float(op.loc[first_event]) * (1 + h)
    sh = np.floor(cap * CASH_BUFFER / (ask * (1 + float(FEE_RATE))))
    cash = cap - sh * ask * (1 + float(FEE_RATE))
    eq = pd.Series(cash + sh * cl.reindex(base_idx).ffill() * (1 - h), index=base_idx)
    eq[eq.index < pd.Timestamp(first_event)] = cap
    return eq


def kronos_ic(dates: list[str], mkt: Market) -> dict:
    """Rank IC of cached Kronos forecasts vs realised forward return over the forecast horizon."""
    close = mkt.p["close"]
    ics, hits, n = [], [], 0
    for d in dates:
        asof = mkt.prev_trading_day(d)
        if asof is None:
            continue
        f = load_cached(asof, KRONOS)
        if f is None:
            continue
        pos = close.index.get_loc(asof)
        if pos + KRONOS.pred_len >= len(close):
            continue
        real = close.iloc[pos + KRONOS.pred_len] / close.iloc[pos] - 1
        f = f.drop_duplicates("ticker").set_index("ticker")
        ic = rank_ic(f["exp_ret"], real.reindex(f.index))
        if np.isfinite(ic):
            ics.append(ic)
            r = real.reindex(f.index)
            hits.append(float(((f["p_up"] > 0.5) == (r > 0)).mean()))
            n += 1
    if not ics:
        return {}
    a = np.array(ics)
    return {"n_dates": n, "mean_rank_ic": float(a.mean()), "ic_tstat": float(a.mean() / (a.std(ddof=1) / np.sqrt(n))) if n > 2 else float("nan"),
            "ic_positive_share": float((a > 0).mean()), "direction_hit_rate": float(np.mean(hits))}


# ------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-01-03")
    ap.add_argument("--end", default=None, help="last date (default: latest bar)")
    ap.add_argument("--freq", default="weekly", help="weekly | daily | N (every Nth trading day)")
    ap.add_argument("--capital", type=float, default=STARTING_CAPITAL)
    ap.add_argument("--phase", choices=["full", "kronos"], default="full")
    ap.add_argument("--no-claude", action="store_true", help="rule-based baseline instead of Claude")
    ap.add_argument("--no-kronos", action="store_true")
    ap.add_argument("--no-anonymize", action="store_true", help="show real tickers/dates to Claude (leakage risk!)")
    ap.add_argument("--model", default=None, help="force one Claude model for every call (default: sonnet weekly review / haiku routine)")
    ap.add_argument("--await-kronos", type=float, default=0.0, metavar="SECONDS", help="wait for a Kronos precompute process")
    ap.add_argument("--wait-for-limit-hours", type=float, default=8.0,
                    help="if the Claude endpoint reports a usage limit, sleep until its reset (up to this many hours)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-events", type=int, default=None, help="smoke-test: stop after N rebalances")
    args = ap.parse_args(argv)

    paths = get_paths().ensure()
    mode = "rules" if args.no_claude else "claude"
    run_id = args.run_id or f"bt_{args.start}_{args.freq}_{mode}_{datetime.now():%Y%m%d_%H%M%S}"
    out = paths.backtests / run_id
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "run.log", console=True)
    log.info("run_id=%s freq=%s claude=%s kronos=%s anonymize=%s", run_id, args.freq, not args.no_claude,
             not args.no_kronos, not args.no_anonymize)

    mkt = Market.from_store()
    dates = mkt.dates[(mkt.dates >= pd.Timestamp(args.start)) & (mkt.dates <= pd.Timestamp(args.end or mkt.dates[-1]))]
    ev = event_dates(dates, args.freq)
    if args.max_events:
        ev = ev[: args.max_events]
        dates = dates[dates <= pd.Timestamp(ev[-1]) + pd.Timedelta(days=10)]
    log.info("window %s..%s | %d trading days | %d rebalances", dates[0].date(), dates[-1].date(), len(dates), len(ev))

    kronos = None if args.no_kronos else KronosForecaster()
    if args.phase == "kronos":
        t0 = time.time()
        for i, d in enumerate(ev, 1):
            asof = mkt.prev_trading_day(d)
            tickers = [t for t in mkt.universe if mkt.p["close"][t].loc[:asof].notna().sum() >= 61]
            forecast_with_cache(kronos, mkt.histories(tickers, asof, KRONOS.lookback + 5), asof)
            log.info("kronos %d/%d %s  (%.0fs elapsed)", i, len(ev), asof.date(), time.time() - t0)
        log.info("kronos IC: %s", kronos_ic(ev, mkt))
        return 0

    brain = None
    if not args.no_claude:
        settings = replace(claude_settings(), limit_wait_cap_s=args.wait_for_limit_hours * 3600)
        brain = ClaudeBrain(ClaudeClient(settings=settings, log_path=out / "calls.jsonl"), use_cache=True)
    cfg = PipelineConfig(mode="backtest", anonymize=not args.no_anonymize, use_claude=not args.no_claude,
                         use_kronos=not args.no_kronos, forced_model=args.model, sim_start=str(dates[0].date()),
                         decisions_path=out / "decisions.jsonl", kronos_wait_s=args.await_kronos)
    pipe = DecisionPipeline(mkt, cfg, brain, kronos)

    op = mkt.p["open"].loc[dates[0]: dates[-1], mkt.universe]
    cl = mkt.p["close"].loc[dates[0]: dates[-1], mkt.universe]
    t0 = time.time()
    res = SimRunner(op, cl, args.capital, pipe.decide, ev).run()
    log.info("simulation finished in %.0fs", time.time() - t0)
    if res.errors:
        log.error("decider errors (%d): %s", len(res.errors), res.errors[:3])

    # ---------------- metrics
    marks = res.marks.copy()
    marks["date"] = pd.to_datetime(marks["date"])
    eq = marks.set_index("date")["equity"]
    first = pd.Timestamp(ev[0])
    prior = mkt.prev_trading_day(ev[0])          # all-cash starting point: last close before the first rebalance
    start_pt = pd.Series([args.capital], index=[prior])
    fund = pd.concat([start_pt, eq[eq.index >= first]]).rename("fund")
    # series on the common calendar starting the day before the first rebalance
    idx = fund.index
    spy_op, spy_cl = mkt.p["open"][BENCHMARK], mkt.p["close"][BENCHMARK]
    spy = spy_buy_hold(spy_op, spy_cl, ev[0], args.capital, idx).rename("spy_buy_hold")
    spy_ret = spy_cl.reindex(idx).pct_change().fillna(0)
    expo = float((marks["positions_value"] / marks["equity"]).mean())
    matched = (args.capital * (1 + expo * spy_ret).cumprod()).rename("spy_exposure_matched")
    ew_ret = mkt.p["close"][mkt.universe].reindex(idx).pct_change(fill_method=None).mean(axis=1).fillna(0)
    ew = (args.capital * (1 + ew_ret).cumprod()).rename("equal_weight_universe")

    stats = {"fund": curve_stats(fund), "spy_buy_hold": curve_stats(spy),
             "spy_exposure_matched": curve_stats(matched), "equal_weight_universe": curve_stats(ew)}
    reg = regress(fund, spy)
    ci = bootstrap_excess_ci(fund, spy)
    ts = trade_stats(res.fills, res.closed, marks.assign(date=marks.date.astype(str)))
    kic = kronos_ic(ev, mkt) if not args.no_kronos else {}
    sc = pipe.stats
    f, b = stats["fund"], stats["spy_buy_hold"]
    metrics = {
        "run_id": run_id, "mode": mode, "freq": args.freq, "start": f["start"], "end": f["end"],
        "n_rebalances": len(ev), "capital": args.capital,
        # flat keys the dashboard reads
        "cagr": f["cagr"], "max_drawdown": f["max_drawdown"], "sharpe": f["sharpe"],
        "win_rate": ts.get("trade_win_rate"), "benchmark_cagr": b["cagr"], "excess_cagr": f["cagr"] - b["cagr"],
        "total_return": f["total_return"], "benchmark_total_return": b["total_return"],
        "stats": stats, "regression_vs_spy": reg, "excess_return_bootstrap": ci, "trades": ts,
        "kronos_signal": kic,
        "claude": {"calls": sc["calls"], "decisions": sc["decisions"], "fallback_holds": sc["fallback_holds"],
                   "prompt_tokens": sc["prompt_tokens"], "completion_tokens": sc["completion_tokens"],
                   "est_cost_usd": round(sc["cost_usd"], 4)},
        "validity": {"decisions_attempted": sc["decisions"] + sc["fallback_holds"], "fallback_holds": sc["fallback_holds"],
                     "fallback_share": sc["fallback_holds"] / max(sc["decisions"] + sc["fallback_holds"], 1),
                     "limit_hits": sc.get("limit_hits", 0)},
        "halts": pipe.halts, "rejected_orders": len(res.rejected), "decider_errors": len(res.errors),
        "assumptions": {"half_spread_bps": HALF_SPREAD_BPS, "fee_bps_per_side": float(FEE_RATE) * 1e4,
                        "risk_free_rate": 0.0, "limits": LIMITS.__dict__, "anonymized_prompts": cfg.anonymize,
                        "universe_size": len(mkt.universe), "kronos_model": KRONOS.model_name},
    }
    metrics["valid_run"] = (not args.no_claude) and metrics["validity"]["fallback_share"] <= 0.10 or args.no_claude
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    curve = pd.concat([fund, spy, matched, ew], axis=1)
    curve.index.name = "date"
    curve.to_csv(out / "equity.csv")
    res.fills.to_csv(out / "trades.csv", index=False)
    res.closed.to_csv(out / "round_trips.csv", index=False)
    text = render_report(metrics)
    (out / "report.txt").write_text(text)
    print(text)
    log.info("saved -> %s", out)
    if not metrics["valid_run"]:
        log.error("INVALID RUN: %.0f%% of decisions fell back to HOLD - results do not reflect the strategy", 100 * metrics["validity"]["fallback_share"])
        return 4
    return 0 if not res.errors else 2


if __name__ == "__main__":
    sys.exit(main())
