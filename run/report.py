"""Plain-text backtest report (printed and saved next to metrics.json)."""
from __future__ import annotations

import math


def _p(x, d=1):
    return "   n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f}%"


def _u(x, d=1):
    return "   n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def _n(x, d=2):
    return "  n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def render_report(m: dict) -> str:
    st = m["stats"]
    cols = [("Fund", "fund"), ("SPY buy&hold", "spy_buy_hold"), ("SPY x avg-exposure", "spy_exposure_matched"),
            ("Equal-wt S&P100", "equal_weight_universe")]
    w = 20
    L = []
    L.append("=" * 92)
    L.append(f"forge-fund BACKTEST  |  {m['run_id']}")
    L.append(f"period {m['start']} -> {m['end']}  |  {m['n_rebalances']} rebalances ({m['freq']})  |  "
             f"mode: {m['mode']}  |  start capital ${m['capital']:,.0f}")
    L.append("PAPER SIMULATION - hypothetical results, not investment advice. See caveats below.")
    v = m.get("validity") or {}
    if v and m.get("mode") == "claude" and v.get("fallback_share", 0) > 0.10:
        L.append(f"!!! INVALID RUN: {v['fallback_share']:.0%} of decisions ({v['fallback_holds']}/{v['decisions_attempted']}) fell back to HOLD "
                 f"(Claude failures / usage limit). These numbers do NOT reflect the strategy. !!!")
    L.append("=" * 92)
    L.append(f"{'':<22}" + "".join(f"{c[0]:>{w}}" for c in cols))
    rows = [("Total return", "total_return", _p), ("CAGR", "cagr", _p), ("Annualised vol", "ann_vol", _u),
            ("Sharpe (rf=0)", "sharpe", _n), ("Sortino", "sortino", _n), ("Max drawdown", "max_drawdown", _p),
            ("Calmar", "calmar", _n), ("Daily win rate", "day_win_rate", _u), ("Worst day", "worst_day", _p)]
    for label, key, fmt in rows:
        L.append(f"{label:<22}" + "".join(f"{fmt(st[c[1]].get(key)):>{w}}" for c in cols))
    f, b = st["fund"], st["spy_buy_hold"]
    L.append("-" * 92)
    L.append(f"Excess CAGR vs SPY buy&hold: {_p(f['cagr'] - b['cagr'])}   |   Excess total return: {_p(f['total_return'] - b['total_return'])}")
    r = m.get("regression_vs_spy") or {}
    if r:
        L.append(f"CAPM vs SPY: beta {_n(r['beta'])}  alpha {_p(r['alpha_ann'])}/yr (t={_n(r['alpha_tstat'])})  "
                 f"corr {_n(r['correlation'])}  info ratio {_n(r['information_ratio'])}  tracking error {_u(r['tracking_error'])}")
    ci = m.get("excess_return_bootstrap") or {}
    if ci:
        L.append(f"Excess return vs SPY, annualised mean {_p(ci['excess_ann_mean'])}  95% bootstrap CI "
                 f"[{_p(ci['ci95_low'])}, {_p(ci['ci95_high'])}]  P(<=0)={_n(ci['p_excess_le_0'])}")
    t = m.get("trades") or {}
    L.append("-" * 92)
    L.append(f"Trades: {t.get('n_fills', 0)} fills, {t.get('n_round_trips', 0)} closed round trips | "
             f"TRADE win rate {_u(t.get('trade_win_rate'))} | profit factor {_n(t.get('profit_factor'))} | "
             f"avg win ${t.get('avg_win', 0):,.0f} / avg loss ${t.get('avg_loss', 0):,.0f}")
    L.append(f"Exposure: avg invested {_u(t.get('avg_exposure'), 0)} | avg positions {_n(t.get('avg_positions'), 1)} | "
             f"annual one-way turnover {_u(t.get('annual_turnover'), 0)} | fees ${t.get('total_fees', 0):,.0f} "
             f"({_u(t.get('fees_pct_of_start'), 2)} of capital)")
    k = m.get("kronos_signal") or {}
    if k:
        L.append(f"Kronos signal quality ({k['n_dates']} dates): mean rank IC {k['mean_rank_ic']:+.3f} "
                 f"(t={_n(k['ic_tstat'])}, positive on {_u(k['ic_positive_share'], 0)} of dates) | "
                 f"direction hit rate {_u(k['direction_hit_rate'])}")
    c = m.get("claude") or {}
    L.append(f"Claude: {c.get('calls', 0)} new calls (cached replies not counted), {c.get('decisions', 0)} usable decisions, {c.get('fallback_holds', 0)} fallback-holds | "
             f"tokens in/out {c.get('prompt_tokens', 0):,}/{c.get('completion_tokens', 0):,} | est. list-price cost ${c.get('est_cost_usd', 0):.2f}")
    L.append(f"Risk: {m.get('halts', 0)} loss-cap halts | {m.get('rejected_orders', 0)} rejected orders | {m.get('decider_errors', 0)} decider errors")
    a = m.get("assumptions") or {}
    L.append("-" * 92)
    L.append("CAVEATS (read these):")
    L.append(f" * Costs modelled: {a.get('half_spread_bps')}bp half-spread + {a.get('fee_bps_per_side')}bp fee per side; no market impact, borrow, or taxes.")
    L.append(" * Universe = TODAY's S&P 100 constituents -> survivorship/look-ahead bias in the universe itself (flatters all columns).")
    L.append(f" * Claude prompts were {'ANONYMISED (opaque ids, no dates)' if a.get('anonymized_prompts') else 'NOT anonymised - the LLM may recall history'};"
             " residual leakage via recognisable price patterns cannot be ruled out.")
    L.append(" * One path, one parameter set, few independent decisions: wide confidence intervals; this is NOT proof of an edge.")
    L.append(" * Fund is capped at 80% invested (cash drag) - compare against the exposure-matched column, not only SPY.")
    L.append("=" * 92)
    return "\n".join(L)
