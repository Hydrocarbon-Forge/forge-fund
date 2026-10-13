"""Hard risk limits, enforced in code AFTER optimization (never trusted to the prompt).

* ``clamp_weights``      - top-N, per-name cap, gross cap, cash floor, min notional.
* ``check_limits``       - audit a weight vector; returns a list of violations (empty == compliant).
* ``order_sanity``       - per-order sanity checks before anything is sent to the engine.
* ``loss_cap_breached``  - single-run loss cap (default 2% of equity) => HALT + log.

The portfolio is long-only, so gross exposure == invested fraction; the 80% gross cap therefore
subsumes the 10% cash floor (cash >= 20%). Both are still enforced so changing one limit stays safe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from forge.config import LIMITS, Limits

EPS = 1e-9


def _clean(weights: dict[str, float]) -> dict[str, float]:
    out = {}
    for t, w in weights.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if math.isfinite(w) and w > EPS:
            out[str(t).upper()] = w
    return out


def _cap_and_redistribute(w: dict[str, float], total: float, cap: float) -> dict[str, float]:
    """Scale ``w`` to sum to ``total`` with each entry <= ``cap`` (water-filling)."""
    if not w:
        return {}
    free = dict(w)
    fixed: dict[str, float] = {}
    budget = total
    for _ in range(len(w) + 1):
        s = sum(free.values())
        if s <= EPS:
            break
        scale = budget / s
        over = {t: v for t, v in free.items() if v * scale > cap + EPS}
        if not over:
            fixed.update({t: v * scale for t, v in free.items()})
            free = {}
            break
        for t in over:
            fixed[t] = cap
            budget -= cap
            del free[t]
        if budget <= EPS:
            break
    return {t: v for t, v in fixed.items() if v > EPS}


def max_investable(limits: Limits = LIMITS) -> float:
    """Largest invested fraction allowed by gross cap and cash floor."""
    return min(limits.max_gross, 1.0 - limits.cash_floor)


def clamp_weights(weights: dict[str, float], limits: Limits = LIMITS,
                  equity: float | None = None) -> dict[str, float]:
    """Return a compliant long-only weight vector.

    Order: drop invalid -> keep top ``max_positions`` -> cap total at gross/cash limits (only ever
    scales DOWN, never up) -> cap each name at ``max_single`` with redistribution -> drop names whose
    notional (``weight * equity``) is below ``min_notional`` (when ``equity`` is given).
    """
    w = _clean(weights)
    if not w:
        return {}
    top = sorted(w.items(), key=lambda kv: kv[1], reverse=True)[: limits.max_positions]
    w = dict(top)
    total = min(sum(w.values()), max_investable(limits))
    w = _cap_and_redistribute(w, total, limits.max_single)
    if equity is not None and equity > 0:
        floor = limits.min_notional(equity)
        w = {t: v for t, v in w.items() if v * equity >= floor}
    # floor (not round) to 6dp so rounding can never push a sum over a cap
    return {t: math.floor(v * 1e6) / 1e6 for t, v in w.items()}


def check_limits(weights: dict[str, float], limits: Limits = LIMITS, equity: float | None = None) -> list[str]:
    """List human-readable limit violations (empty list == compliant). Used as a post-condition."""
    v: list[str] = []
    w = _clean(weights)
    if len(w) > limits.max_positions:
        v.append(f"positions {len(w)} > {limits.max_positions}")
    for t, x in w.items():
        if x > limits.max_single + 1e-5:
            v.append(f"{t} weight {x:.4f} > {limits.max_single}")
        if equity and x * equity < limits.min_notional(equity) - 1e-6:
            v.append(f"{t} notional {x * equity:.0f} < {limits.min_notional(equity):.0f}")
    gross = sum(w.values())
    if gross > limits.max_gross + 1e-5:
        v.append(f"gross {gross:.4f} > {limits.max_gross}")
    if 1.0 - gross < limits.cash_floor - 1e-5:
        v.append(f"cash {1.0 - gross:.4f} < floor {limits.cash_floor}")
    return v


@dataclass(frozen=True)
class OrderCheck:
    ok: bool
    reason: str = ""


def order_sanity(ticker: str, side: str, qty: float, price: float, equity: float,
                 ref_close: float | None = None, limits: Limits = LIMITS) -> OrderCheck:
    """Reject nonsensical orders (bad qty/price, oversized notional, price far from last close)."""
    if side not in ("BUY", "SELL"):
        return OrderCheck(False, f"bad side {side!r}")
    if not (isinstance(qty, (int, float)) and math.isfinite(qty) and qty > 0 and float(qty).is_integer()):
        return OrderCheck(False, f"{ticker}: qty must be a positive whole number, got {qty!r}")
    if not (math.isfinite(price) and price > 0):
        return OrderCheck(False, f"{ticker}: bad price {price!r}")
    if equity <= 0:
        return OrderCheck(False, "non-positive equity")
    notional = qty * price
    if notional > limits.max_order_pct_equity * equity + 1e-6:
        return OrderCheck(False, f"{ticker}: order notional {notional:,.0f} > "
                                 f"{limits.max_order_pct_equity:.0%} of equity")
    floor = limits.min_notional(equity)
    if notional < floor - 1e-6 and side == "BUY":
        return OrderCheck(False, f"{ticker}: order notional {notional:,.0f} < min {floor:,.0f}")
    if ref_close and ref_close > 0 and abs(price / ref_close - 1.0) > limits.price_sanity_pct:
        return OrderCheck(False, f"{ticker}: price {price:.2f} deviates >{limits.price_sanity_pct:.0%} "
                                 f"from last close {ref_close:.2f}")
    return OrderCheck(True)


def loss_cap_breached(prev_equity: float | None, curr_equity: float, limits: Limits = LIMITS) -> tuple[bool, float]:
    """(breached?, run P&L as a fraction of previous equity). Breach when the run lost > loss_cap."""
    if not prev_equity or prev_equity <= 0:
        return False, 0.0
    pnl = curr_equity / prev_equity - 1.0
    return pnl < -limits.loss_cap, pnl
