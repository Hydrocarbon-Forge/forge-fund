"""Nautilus execution/accounting core shared by BACKTEST and PAPER trading.

Paper trading is the Nautilus *backtest engine* fed with a simulated venue: there is no live data
client, no execution client, and no broker adapter anywhere in this codebase (Nautilus' live-trading
package is never imported - enforced by tests/test_no_live.py). The venue is a long-only CASH account, so the
engine itself rejects any order that would need leverage or borrowing.

Timeline per trading day (America/New_York), driven by synthetic quote ticks:
    09:35:00  quote = day's open (+/- half spread)      <- the decision uses data through PRIOR close
    09:35:01  stage 1: ask the decider for a Plan, submit SELLs   (fill at bid)
    09:35:02  stage 2: submit BUYs with the cash the sells released (fill at ask)
    16:00:00  quote = day's close
    16:00:01  mark-to-market equity snapshot

Because decisions only see bars strictly before the decision date and fills happen at that day's open,
there is no look-ahead. ``decide(date, view) -> Plan`` is the only extension point: the backtest passes
the full Kronos->Claude->skfolio pipeline; paper mode replays stored plans for past days (event-sourced
from ``state/ledger.jsonl``) and runs the live pipeline for today.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from forge.config import LIMITS, Limits
from forge.risk import check_limits, clamp_weights, order_sanity
from forge.types import Plan, PortfolioView

log = logging.getLogger("forge.engine")

VENUE = Venue("SIM")           # simulated venue - the ONLY venue in the system
ET = ZoneInfo("America/New_York")
HALF_SPREAD_BPS = 1.0          # quote = px * (1 -/+ 1bp)  => 2 bps quoted spread
FEE_RATE = Decimal("0.0001")   # 1 bp commission per side (maker == taker)
CASH_BUFFER = 0.998            # keep 0.2% of free cash unspent so buys never get denied for rounding


# ------------------------------------------------------------------------------ data classes
@dataclass
class SimResult:
    marks: pd.DataFrame                  # date, cash, positions_value, equity  (16:00 marks)
    fills: pd.DataFrame                  # ts, date, ticker, side, qty, price, fee, notional, realized_pnl
    closed: pd.DataFrame                 # closed round trips: ticker, opened, closed, pnl, ret
    plans: dict[str, Plan]               # date -> executed plan
    final_positions: dict[str, dict]
    final_cash: float
    final_equity: float
    rejected: list[dict]
    errors: list[str]


# ------------------------------------------------------------------------------ helpers
def make_equity(ticker: str) -> Equity:
    iid = InstrumentId(Symbol(ticker), VENUE)
    return Equity(instrument_id=iid, raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
                  price_increment=Price.from_str("0.01"), lot_size=Quantity.from_int(1),
                  ts_event=0, ts_init=0, maker_fee=FEE_RATE, taker_fee=FEE_RATE)


def _ts_ns(dates: pd.DatetimeIndex, hour: int, minute: int, second: int = 0) -> np.ndarray:
    off = pd.Timedelta(hours=hour, minutes=minute, seconds=second)
    # pandas>=3 keeps the index's native unit (s/us), so force nanoseconds before taking int64 values
    return (pd.DatetimeIndex(dates).as_unit("ns") + off).tz_localize(ET).tz_convert("UTC").asi8


def _date_of(ts_ns: int) -> str:
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(ET).strftime("%Y-%m-%d")


def _quote(iid: InstrumentId, px: float, ts: int) -> QuoteTick:
    h = HALF_SPREAD_BPS / 1e4
    return QuoteTick(iid, Price(px * (1 - h), 2), Price(px * (1 + h), 2),
                     Quantity.from_int(1_000_000), Quantity.from_int(1_000_000), ts, ts)


# ------------------------------------------------------------------------------ strategy
class ForgeStrategy(Strategy):
    """Executes whatever Plan the decider returns - long-only, whole shares, with sanity checks."""

    def __init__(self, instruments: dict[str, Equity], event_dates: list[str], mark_dates: list[str],
                 decide: Callable[[str, PortfolioView], Plan], starting_capital: float,
                 limits: Limits = LIMITS):
        super().__init__(StrategyConfig(strategy_id="ForgeFund-001"))
        self.insts = instruments
        self.by_iid = {i.id: t for t, i in instruments.items()}
        self.event_dates = event_dates
        self.mark_dates = mark_dates
        self.decide = decide
        self.start_cap = starting_capital
        self.limits = limits
        # outputs
        self.marks: list[dict] = []
        self.fills: list[dict] = []
        self.plans: dict[str, Plan] = {}
        self.rejected: list[dict] = []
        self.errors: list[str] = []
        # state
        self._pending: dict[str, tuple[Plan, float]] = {}
        self._prev_equity: float | None = None
        self._n_runs = 0
        self._last_close: dict[str, float] = {}
        self._avg_cost: dict[str, float] = {}
        self._qty: dict[str, float] = {}
        self._opened_at: dict[str, str] = {}
        self._cost_basis_total: dict[str, float] = {}
        self._pos_pnl: dict[str, float] = {}
        self.closed_rows: list[dict] = []

    # ---- lifecycle
    def on_start(self):
        for d in self.event_dates:
            self.clock.set_time_alert_ns(f"s1-{d}", int(_ts_ns(pd.DatetimeIndex([d]), 9, 35, 1)[0]), self._stage1)
            self.clock.set_time_alert_ns(f"s2-{d}", int(_ts_ns(pd.DatetimeIndex([d]), 9, 35, 2)[0]), self._stage2)
        for d in self.mark_dates:
            self.clock.set_time_alert_ns(f"mk-{d}", int(_ts_ns(pd.DatetimeIndex([d]), 16, 0, 1)[0]), self._mark)

    # ---- accounting
    def _snapshot(self) -> tuple[float, float, dict[str, dict]]:
        acct = self.portfolio.account(VENUE)
        cash = acct.balance_total(USD).as_double()
        pos: dict[str, dict] = {}
        mv = 0.0
        for p in self.cache.positions_open():
            t = self.by_iid[p.instrument_id]
            q = self.cache.quote_tick(p.instrument_id)
            qty = float(p.signed_decimal_qty())
            px = (q.bid_price.as_double() + q.ask_price.as_double()) / 2 if q else p.avg_px_open
            pos[t] = {"qty": qty, "avg_px": float(p.avg_px_open), "last_px": px, "value": qty * px}
            mv += qty * px
        eq = cash + mv
        for t, d in pos.items():
            d["weight"] = d["value"] / eq if eq > 0 else 0.0
            d["unrealized_pct"] = d["last_px"] / d["avg_px"] - 1.0 if d["avg_px"] else 0.0
        return cash, mv, pos

    def _view(self, date: str) -> PortfolioView:
        cash, mv, pos = self._snapshot()
        return PortfolioView(date=date, equity=cash + mv, cash=cash, prev_equity=self._prev_equity,
                             starting_capital=self.start_cap, positions=pos,
                             equity_history=[(m["date"], m["equity"]) for m in self.marks[-260:]],
                             n_runs=self._n_runs)

    # ---- stage 1: decide + sells
    def _stage1(self, event):
        date = _date_of(event.ts_event)
        view = self._view(date)
        try:
            plan = self.decide(date, view)
        except Exception as e:  # noqa: BLE001 - never let a pipeline bug crash the engine mid-run
            log.exception("decider failed on %s", date)
            self.errors.append(f"{date}: {type(e).__name__}: {e}")
            plan = Plan(None, False, {"status": "error", "error": f"{type(e).__name__}: {e}"})
        self.plans[date] = plan
        self._n_runs += 1
        if plan.halted or plan.targets is None:
            self._prev_equity = view.equity
            self._pending.pop(date, None)
            return
        targets = clamp_weights(plan.targets, self.limits, view.equity)
        bad = check_limits(targets, self.limits, view.equity)
        if bad:                      # impossible after clamp; fail safe = hold
            self.errors.append(f"{date}: post-clamp violations {bad}")
            self._prev_equity = view.equity
            return
        plan.meta["executed_targets"] = targets
        self._pending[date] = (plan, view.equity)
        self._prev_equity = view.equity
        for t, qty, px in self._deltas(targets, view, side=OrderSide.SELL):
            self._submit(t, OrderSide.SELL, qty, px, view.equity, date)

    # ---- stage 2: buys with released cash
    def _stage2(self, event):
        date = _date_of(event.ts_event)
        if date not in self._pending:
            return
        plan, eq0 = self._pending.pop(date)
        view = self._view(date)
        free = self.portfolio.account(VENUE).balance_free(USD).as_double() * CASH_BUFFER
        buys = self._deltas(plan.meta["executed_targets"], view, side=OrderSide.BUY, equity=eq0)
        for t, qty, px in sorted(buys, key=lambda x: -x[1] * x[2]):
            qty = min(qty, math.floor(free / px)) if px > 0 else 0
            if qty <= 0:
                continue
            if self._submit(t, OrderSide.BUY, qty, px, eq0, date):
                free -= qty * px

    def _deltas(self, targets: dict[str, float], view: PortfolioView, side: OrderSide,
                equity: float | None = None) -> list[tuple[str, int, float]]:
        eq = equity if equity is not None else view.equity
        out = []
        for t in sorted(set(targets) | set(view.positions)):
            if t not in self.insts:
                continue
            q = self.cache.quote_tick(self.insts[t].id)
            if q is None:
                continue
            ask, bid = q.ask_price.as_double(), q.bid_price.as_double()
            cur = int(round(view.positions.get(t, {}).get("qty", 0.0)))
            tgt_w = targets.get(t, 0.0)
            tgt = int(math.floor(tgt_w * eq / ask)) if tgt_w > 0 else 0
            delta = tgt - cur
            if delta == 0:
                continue
            # rebalance band: skip small drifts on names we keep (full exits / new entries always trade)
            if cur > 0 and tgt > 0 and abs(delta) * ask / eq < self.limits.rebalance_band:
                continue
            if abs(delta) * ask < self.limits.min_notional and tgt > 0:
                continue          # sub-minimum top-up / trim: not worth trading
            if side == OrderSide.SELL and delta < 0:
                out.append((t, -delta, bid))
            elif side == OrderSide.BUY and delta > 0:
                out.append((t, delta, ask))
        return out

    def _submit(self, t: str, side: OrderSide, qty: int, px: float, equity: float, date: str) -> bool:
        chk = order_sanity(t, side.name, qty, px, equity, self._last_close.get(t), self.limits)
        if not chk.ok and not (side == OrderSide.SELL and "min" in chk.reason):
            self.rejected.append({"date": date, "ticker": t, "side": side.name, "qty": qty, "reason": chk.reason})
            log.warning("order rejected: %s", chk.reason)
            return False
        order = self.order_factory.market(self.insts[t].id, side, Quantity.from_int(int(qty)))
        self.submit_order(order)
        return True

    # ---- fills -> trade log with realized pnl (running average cost, long-only)
    def on_order_filled(self, event):
        t = self.by_iid[event.instrument_id]
        qty, px = float(event.last_qty), float(event.last_px)
        fee = event.commission.as_double() if event.commission else 0.0
        side = event.order_side
        date = _date_of(event.ts_event)
        realized = 0.0
        if side == OrderSide.BUY:
            q0 = self._qty.get(t, 0.0)
            c0 = self._avg_cost.get(t, 0.0)
            self._avg_cost[t] = (q0 * c0 + qty * px) / (q0 + qty)
            if q0 == 0:
                self._opened_at[t] = date
                self._cost_basis_total[t] = 0.0
            self._qty[t] = q0 + qty
            self._cost_basis_total[t] = self._cost_basis_total.get(t, 0.0) + qty * px + fee
            realized = -fee
            self._pos_pnl[t] = self._pos_pnl.get(t, 0.0) - fee
        else:
            avg = self._avg_cost.get(t, px)
            realized = (px - avg) * qty - fee
            self._qty[t] = self._qty.get(t, 0.0) - qty
            self._pos_pnl[t] = self._pos_pnl.get(t, 0.0) + realized
            if self._qty[t] <= 1e-9:                       # round trip closed
                basis = self._cost_basis_total.get(t, 0.0)
                pnl = self._pos_pnl.pop(t, 0.0)
                self.closed_rows.append({"ticker": t, "opened": self._opened_at.get(t), "closed": date,
                                         "pnl": pnl, "ret": pnl / basis if basis else 0.0})
                self._qty.pop(t, None)
                self._avg_cost.pop(t, None)
        self.fills.append({"ts": pd.Timestamp(event.ts_event, unit="ns", tz="UTC").isoformat(), "date": date,
                           "ticker": t, "side": side.name, "qty": qty, "price": px, "fee": fee,
                           "notional": qty * px, "realized_pnl": realized})

    # ---- 16:00 marks
    def _mark(self, event):
        date = _date_of(event.ts_event)
        cash, mv, pos = self._snapshot()
        self.marks.append({"date": date, "cash": cash, "positions_value": mv, "equity": cash + mv,
                           "n_positions": len(pos)})
        for t in self.insts:
            q = self.cache.quote_tick(self.insts[t].id)
            if q is not None:
                self._last_close[t] = (q.bid_price.as_double() + q.ask_price.as_double()) / 2


# ------------------------------------------------------------------------------ runner
class SimRunner:
    """Build a Nautilus backtest engine from daily open/close panels and run it."""

    def __init__(self, open_px: pd.DataFrame, close_px: pd.DataFrame, starting_capital: float,
                 decide: Callable[[str, PortfolioView], Plan], event_dates: list[str],
                 price_overrides: dict[str, dict[str, float]] | None = None,
                 live_date: str | None = None, limits: Limits = LIMITS, tickers: list[str] | None = None):
        """
        open_px / close_px : wide frames (index=trading dates, cols=tickers) for the simulated window
        event_dates        : dates on which the decider is invoked (09:35 ET)
        price_overrides    : date -> {ticker: price} replacing that date's 09:35 quote (paper ledger fills)
        live_date          : paper mode only - last event date; gets NO 16:00 mark (the day isn't over)
        """
        self.open_px, self.close_px = open_px.sort_index(), close_px.sort_index()
        self.cap = starting_capital
        self.decide = decide
        self.event_dates = sorted(event_dates)
        self.overrides = price_overrides or {}
        self.live_date = live_date
        self.limits = limits
        self.tickers = tickers

    def run(self) -> SimResult:
        tickers = list(self.tickers) if self.tickers else [c for c in self.close_px.columns if self.close_px[c].notna().any()]
        instruments = {t: make_equity(t) for t in tickers}
        idx = self.close_px.index
        mark_dates = [d.strftime("%Y-%m-%d") for d in idx if d.strftime("%Y-%m-%d") != self.live_date]
        # every event date needs a 09:35 quote even if not in the price index (live day)
        quote_days = sorted(set(idx.strftime("%Y-%m-%d")) | set(self.event_dates))

        quotes: list[QuoteTick] = []
        open_ts = dict(zip(quote_days, _ts_ns(pd.DatetimeIndex(quote_days), 9, 35)))
        close_ts = dict(zip(quote_days, _ts_ns(pd.DatetimeIndex(quote_days), 16, 0)))
        o_arr = self.open_px.reindex(columns=tickers)
        c_arr = self.close_px.reindex(columns=tickers)
        o_map = {d.strftime("%Y-%m-%d"): row for d, row in zip(o_arr.index, o_arr.to_numpy())}
        c_map = {d.strftime("%Y-%m-%d"): row for d, row in zip(c_arr.index, c_arr.to_numpy())}
        for day in quote_days:
            ov = self.overrides.get(day, {})
            orow = o_map.get(day)
            crow = c_map.get(day)
            for j, t in enumerate(tickers):
                iid = instruments[t].id
                px_o = ov.get(t, orow[j] if orow is not None else np.nan)
                if px_o is not None and np.isfinite(px_o) and px_o > 0:
                    quotes.append(_quote(iid, float(px_o), int(open_ts[day])))
                if crow is not None and day != self.live_date:
                    px_c = crow[j]
                    if np.isfinite(px_c) and px_c > 0:
                        quotes.append(_quote(iid, float(px_c), int(close_ts[day])))
        # sentinel tick 10 min after the last quote so the final timers (09:35:02 buys / 16:00:01 mark) fire
        if quotes:
            last = max(quotes, key=lambda q: q.ts_event)
            end_ts = int(last.ts_event) + 600 * 10**9
            quotes.append(QuoteTick(last.instrument_id, last.bid_price, last.ask_price, last.bid_size,
                                    last.ask_size, end_ts, end_ts))
        quotes.sort(key=lambda q: (q.ts_event, str(q.instrument_id)))

        engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR", bypass_logging=True)))
        try:
            engine.add_venue(venue=VENUE, oms_type=OmsType.NETTING, account_type=AccountType.CASH, base_currency=USD,
                             starting_balances=[Money(self.cap, USD)], bar_execution=False, trade_execution=False)
            for inst in instruments.values():
                engine.add_instrument(inst)
            engine.add_data(quotes)
            strat = ForgeStrategy(instruments, self.event_dates, mark_dates, self.decide, self.cap, self.limits)
            engine.add_strategy(strat)
            engine.run()
            cash, mv, pos = strat._snapshot()
        finally:
            engine.dispose()

        marks = pd.DataFrame(strat.marks, columns=["date", "cash", "positions_value", "equity", "n_positions"])
        fills = pd.DataFrame(strat.fills, columns=["ts", "date", "ticker", "side", "qty", "price", "fee", "notional", "realized_pnl"])
        closed = pd.DataFrame(strat.closed_rows, columns=["ticker", "opened", "closed", "pnl", "ret"])
        return SimResult(marks=marks, fills=fills, closed=closed, plans=strat.plans, final_positions=pos,
                         final_cash=cash, final_equity=cash + mv,
                         rejected=strat.rejected, errors=strat.errors)
