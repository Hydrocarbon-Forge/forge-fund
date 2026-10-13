"""One PAPER trading cycle: data -> Kronos -> Claude -> skfolio -> risk limits -> Nautilus (simulated venue).

    .venv/bin/python -m run.run_daily                 # normal daily run (cron: 09:35 America/New_York, Mon-Fri)
    .venv/bin/python -m run.run_daily --prepare       # evening prep: refresh data + Kronos forecasts only
    .venv/bin/python -m run.run_daily --dry-run       # compute the decision, change nothing on disk
    .venv/bin/python -m run.run_daily --force         # re-run today's cycle (replaces today's ledger entry)
    .venv/bin/python -m run.run_daily --resume        # clear a loss-cap halt (after you reviewed it)

Paper only: there is no broker/live code path in this repository. Exit codes: 0 ok, 2 error, 3 halted.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from brain.claude_client import ClaudeClient
from brain.decision import ClaudeBrain
from engine.paper import read_ledger, run_cycle
from forge.calendar import decision_date, market_open_now
from forge.config import BENCHMARK, KRONOS, STARTING_CAPITAL, get_paths
from forge.data import latest_prices, update_store
from forge.logging_setup import setup_logging
from forge.universe import SP100
from kronos.forecast import KronosForecaster, forecast_with_cache
from run.pipeline import DecisionPipeline, Market, PipelineConfig

log = logging.getLogger("forge.daily")


def _write_forecasts(paths, kdf: pd.DataFrame, asof) -> None:
    rows = [{"ticker": r.ticker, "exp_ret": round(float(r.exp_ret), 6), "p_up": round(float(r.p_up), 4),
             "dispersion": round(float(r.dispersion), 6), "score": round(float(r.score), 4)} for r in kdf.itertuples()]
    rows.sort(key=lambda r: -r["score"])
    payload = {"asof": str(pd.Timestamp(asof).date()), "model": KRONOS.model_name, "horizon_days": KRONOS.pred_len,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows}
    tmp = paths.state / "latest_forecasts.json.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(paths.state / "latest_forecasts.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="refresh data + Kronos forecasts only (no decision)")
    ap.add_argument("--dry-run", action="store_true", help="decide but write nothing")
    ap.add_argument("--force", action="store_true", help="replace an existing ledger entry for today")
    ap.add_argument("--resume", action="store_true", help="clear a loss-cap halt and exit")
    ap.add_argument("--no-fetch", action="store_true", help="skip the data refresh")
    ap.add_argument("--no-claude", action="store_true", help="rule baseline instead of Claude")
    ap.add_argument("--no-kronos", action="store_true", help="skip the Kronos signal")
    ap.add_argument("--model", default=None, help="force one Claude model")
    ap.add_argument("--date", default=None, help="override decision date (testing)")
    args = ap.parse_args(argv)

    paths = get_paths().ensure()
    setup_logging(paths.logs / "engine.log", console=True)
    lock = open(paths.state / "run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("another forge-fund cycle is already running; exiting")
        return 2

    halt = paths.state / "halt.json"
    if args.resume:
        if halt.exists():
            log.warning("clearing halt: %s", halt.read_text()[:300])
            halt.unlink()
        else:
            log.info("no halt to clear")
        return 0

    try:
        if not args.no_fetch:
            meta = update_store()
            log.info("data refreshed: last bar %s, %d tickers, dropped=%s", meta["last_date"], meta["n_tickers"], meta["dropped"])
        mkt = Market.from_store()
        asof_last = mkt.dates[-1]
        dd = args.date or decision_date().isoformat()

        # ---- evening prep: Kronos only (CPU heavy, run when nothing else is running)
        if args.prepare:
            t0 = time.time()
            kf = KronosForecaster()
            feats_t = [t for t in mkt.universe if mkt.p["close"][t].notna().sum() >= 61]
            kdf = forecast_with_cache(kf, mkt.histories(feats_t, asof_last, KRONOS.lookback + 5), asof_last)
            _write_forecasts(paths, kdf, asof_last)
            log.info("prepared Kronos forecasts for %s (%d tickers, %.0fs)", asof_last.date(), len(kdf), time.time() - t0)
            return 0

        if halt.exists():
            log.error("HALTED: %s -- review, then run with --resume", halt.read_text()[:300])
            return 3

        ledger = read_ledger(paths)
        if any(r["date"] == dd for r in ledger) and not args.force:
            log.info("cycle for %s already completed; nothing to do (use --force to redo)", dd)
            return 0

        # ---- live prices for the simulated 09:35 fill (stored closes when the market is closed)
        want = list(dict.fromkeys(mkt.universe + [BENCHMARK]))
        prices, source = latest_prices(want)
        if not market_open_now():
            source = "last_close"
        log.info("decision date %s | price source: %s | %d prices", dd, source, len(prices))

        brain = None if args.no_claude else ClaudeBrain(ClaudeClient(log_path=paths.logs / "decision_log.jsonl"), use_cache=False)
        cfg = PipelineConfig(mode="paper", anonymize=False, use_claude=not args.no_claude, use_kronos=not args.no_kronos, forced_model=args.model,
                             persist_halt=not args.dry_run, state_dir=paths.state,
                             decisions_path=None if args.dry_run else paths.logs / "decision_log.jsonl",
                             prior_event_dates=[r["date"] for r in ledger if r["date"] != dd])
        pipe = DecisionPipeline(mkt, cfg, brain, None if args.no_kronos else KronosForecaster())

        summary = run_cycle(paths=paths, market=mkt, decision_date=dd, live_prices=prices, price_source=source,
                            live_decide=pipe.decide, universe=mkt.universe, capital=STARTING_CAPITAL,
                            replace=args.force, persist=not args.dry_run)
        if pipe.last_kdf is not None and not args.dry_run:
            _write_forecasts(paths, pipe.last_kdf, pipe.last_asof)

        res = summary.pop("result")
        log.info("cycle %s: status=%s equity=$%.2f cash=$%.2f positions=%d fills_today=%d halted=%s",
                 dd, summary["status"], summary["equity"], summary["cash"], summary["n_positions"],
                 summary["n_fills_today"], summary["halted"])
        for t, p in sorted(res.final_positions.items(), key=lambda kv: -kv[1]["value"]):
            log.info("  %-6s qty=%5d  avg=%9.2f  last=%9.2f  w=%5.1f%%", t, p["qty"], p["avg_px"], p["last_px"], 100 * p["weight"])
        log.info("Claude usage this run: %s", pipe.stats)
        if args.dry_run:
            log.info("dry-run: nothing written")
        return 3 if summary["halted"] else 0
    except Exception as e:  # noqa: BLE001 - top-level: log (redacted) and signal failure to cron
        log.exception("run_daily failed: %s", e)
        return 2
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    sys.exit(main())
