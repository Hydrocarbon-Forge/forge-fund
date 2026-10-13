"""Read-only FastAPI dashboard for forge-fund (paper trading).

The engine writes the state files listed in ``docs/CONTRACTS.md``; this module only
reads them. Each file is parsed on demand and re-parsed only when its mtime or size
changes. A missing, empty or corrupt file never produces a 500: the endpoint answers
with the empty structure and ``"available": false``.

Output rules: GET only; documented fields only; NaN/Infinity are sent as null; every
string passes through ``forge.config.redact``; no file paths and no error text from
the decision log are ever returned.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from forge.config import MODE, STARTING_CAPITAL, Paths, get_paths, redact

log = logging.getLogger("forge.dashboard")

STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
# DNS-rebinding guard. The dashboard now binds the box's LAN IP so Gavin can
# open it from his device, so the allowlist carries that host alongside loopback.
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "192.168.1.201"]
CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
EQUITY_ROWS_MAX = 2000
CALLS_SHOWN = 50
TRADES_DEFAULT, TRADES_MIN, TRADES_MAX = 100, 1, 500
PERIODS = (("1d", "1 day", 1), ("5d", "5 days", 5), ("21d", "21 days", 21))
ERROR_CHARS = 300

_EMPTY_TOTALS = {
    "n_calls": 0,
    "total_cost_usd": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


# --------------------------------------------------------------------- coercion
def _num(x: Any) -> float | None:
    """Finite float, or None. Strings and bools are not numbers in these files."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _int(x: Any) -> int:
    f = _num(x)
    return int(f) if f is not None else 0


def _text(x: Any) -> str | None:
    return x if isinstance(x, str) and x else None


def _as_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def _weights(x: Any) -> dict[str, float]:
    if not isinstance(x, dict):
        return {}
    out: dict[str, float] = {}
    for ticker, w in x.items():
        v = _num(w)
        if v is not None:
            out[str(ticker)] = v
    return out


def _sanitize(obj: Any) -> Any:
    """JSON-safe copy: plain containers only, no NaN/Infinity, strings redacted."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return redact(obj)
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- parsers
def _parse_obj(raw: bytes) -> dict | None:
    """A non-empty JSON object, or None (missing/empty/corrupt/not an object)."""
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) and obj else None


def _parse_jsonl(raw: bytes) -> list[dict]:
    """JSON-object lines; blank and corrupt lines (e.g. a half-written tail) are skipped."""
    rows: list[dict] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _parse_curve(raw: bytes) -> list[dict] | None:
    by_date: dict[str, dict] = {}
    for row in _parse_jsonl(raw):  # last write wins per date
        date, equity = _text(row.get("date")), _num(row.get("equity"))
        if date is None or equity is None:
            continue
        by_date[date] = {
            "date": date,
            "equity": equity,
            "cash": _num(row.get("cash")),
            "benchmark_equity": _num(row.get("benchmark_equity")),
            "drawdown": _num(row.get("drawdown")),
        }
    if not by_date:
        return None
    rows = [by_date[d] for d in sorted(by_date)]
    peak: float | None = None
    for r in rows:  # fill drawdown from the full curve where the engine left it out
        peak = r["equity"] if peak is None else max(peak, r["equity"])
        if r["drawdown"] is None and peak > 0:
            r["drawdown"] = r["equity"] / peak - 1.0
    return rows


def _parse_trades(raw: bytes) -> list[dict] | None:
    out: list[dict] = []
    for row in _parse_jsonl(raw):
        ticker = _text(row.get("ticker"))
        if ticker is None:
            continue
        out.append(
            {
                "ts": _text(row.get("ts")),
                "date": _text(row.get("date")),
                "ticker": ticker,
                "side": _text(row.get("side")),
                "qty": _num(row.get("qty")),
                "price": _num(row.get("price")),
                "notional": _num(row.get("notional")),
                "fee": _num(row.get("fee")),
                "realized_pnl": _num(row.get("realized_pnl")),
                "reason": _text(row.get("reason")),
            }
        )
    return out or None


def _parse_decision(raw: bytes) -> dict | None:
    d = _parse_obj(raw)
    if d is None:
        return None
    holdings: dict[str, dict] = {}
    raw_h = d.get("claude_holdings")
    for ticker, h in raw_h.items() if isinstance(raw_h, dict) else []:
        if isinstance(h, dict):
            holdings[str(ticker)] = {
                "weight": _num(h.get("weight")),
                "rationale": _text(h.get("rationale")),
            }
    usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
    error = _text(d.get("error"))
    return {
        "date": _text(d.get("date")),
        "ts": _text(d.get("ts")),
        "model": _text(d.get("model")),
        "status": _text(d.get("status")),
        "risk_view": _text(d.get("risk_view")),
        "cash": _num(d.get("cash")),
        "notes": _text(d.get("notes")),
        "prompt_hash": _text(d.get("prompt_hash")),
        "usage": {
            "prompt_tokens": _num(usage.get("prompt_tokens")),
            "completion_tokens": _num(usage.get("completion_tokens")),
        },
        "cost_usd": _num(d.get("cost_usd")),
        "claude_holdings": holdings,
        "final_weights": _weights(d.get("final_weights")),
        "error": error[:ERROR_CHARS] if error else None,
    }


def _parse_forecasts(raw: bytes) -> dict | None:
    d = _parse_obj(raw)
    if d is None:
        return None
    rows: list[dict] = []
    for r in _as_list(d.get("rows")):
        if not isinstance(r, dict):
            continue
        ticker = _text(r.get("ticker"))
        if ticker is None:
            continue
        rows.append(
            {
                "ticker": ticker,
                "exp_ret": _num(r.get("exp_ret")),
                "p_up": _num(r.get("p_up")),
                "dispersion": _num(r.get("dispersion")),
                "score": _num(r.get("score")),
            }
        )
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))  # score desc
    horizon = _num(d.get("horizon_days"))
    return {
        "asof": _text(d.get("asof")),
        "model": _text(d.get("model")),
        "horizon_days": int(horizon) if horizon is not None else None,
        "generated_at": _text(d.get("generated_at")),
        "rows": rows,
    }


def _parse_calls(raw: bytes) -> dict | None:
    calls: list[dict] = []
    for row in _parse_jsonl(raw):
        if row.get("kind") != "call":
            continue
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        # error text is deliberately not carried over: the dashboard shows status only
        calls.append(
            {
                "ts": _text(row.get("ts")),
                "purpose": _text(row.get("purpose")),
                "model": _text(row.get("model")),
                "status": _text(row.get("status")),
                "latency_s": _num(row.get("latency_s")),
                "usage": {
                    "prompt_tokens": _int(usage.get("prompt_tokens")),
                    "completion_tokens": _int(usage.get("completion_tokens")),
                },
                "cost_usd": _num(row.get("cost_usd")),
            }
        )
    if not calls:
        return None
    prompt = sum(c["usage"]["prompt_tokens"] for c in calls)
    completion = sum(c["usage"]["completion_tokens"] for c in calls)
    return {
        "recent": list(reversed(calls[-CALLS_SHOWN:])),  # newest first
        "totals": {
            "n_calls": len(calls),
            "total_cost_usd": round(sum(c["cost_usd"] or 0.0 for c in calls), 6),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


# ------------------------------------------------------------------- file cache
class _FileCache:
    """Parsed results keyed by path; reused only while (mtime_ns, size) is unchanged."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[tuple[int, int], Any]] = {}

    def read(self, path: Path, parse: Callable[[bytes], Any]) -> Any:
        try:
            st = path.stat()
        except OSError:
            return None
        sig = (st.st_mtime_ns, st.st_size)
        key = str(path)
        with self._lock:
            hit = self._entries.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        try:
            value = parse(path.read_bytes())
        except OSError:
            return None
        with self._lock:
            self._entries[key] = (sig, value)
        return value


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


class _Data:
    """Normalised views over the state files. Each accessor returns None when unusable."""

    def __init__(self, paths: Paths | None) -> None:
        self._paths = paths
        self._cache = _FileCache()

    @property
    def p(self) -> Paths:
        # resolved per call so FORGE_ROOT changes (tests, demo) are honoured
        return self._paths if self._paths is not None else get_paths()

    def portfolio(self) -> dict | None:
        return self._cache.read(self.p.state / "portfolio.json", _parse_obj)

    def curve(self) -> list[dict] | None:
        return self._cache.read(self.p.state / "equity_curve.jsonl", _parse_curve)

    def trades(self) -> list[dict] | None:
        return self._cache.read(self.p.state / "trades.jsonl", _parse_trades)

    def decision(self) -> dict | None:
        return self._cache.read(self.p.state / "latest_decision.json", _parse_decision)

    def forecasts(self) -> dict | None:
        return self._cache.read(self.p.state / "latest_forecasts.json", _parse_forecasts)

    def calls(self) -> dict | None:
        return self._cache.read(self.p.logs / "decision_log.jsonl", _parse_calls)

    def halt_state(self, port: dict | None) -> tuple[bool, str | None]:
        halt_file = self.p.state / "halt.json"  # present only while halted
        present = halt_file.exists()
        halted = (port or {}).get("halted") is True or present
        reason = _text((port or {}).get("halt_reason"))
        if reason is None and present:
            info = self._cache.read(halt_file, _parse_obj) or {}
            reason = _text(info.get("reason"))
        return halted, reason

    def backtest(self) -> tuple[str, dict] | None:
        base = self.p.backtests
        try:
            runs = [d for d in base.iterdir() if d.is_dir() and (d / "metrics.json").is_file()]
        except OSError:
            return None
        if not runs:
            return None
        newest = max(runs, key=_mtime)
        metrics = self._cache.read(newest / "metrics.json", _parse_obj)
        if metrics is None:
            return None
        return newest.name, metrics


# ------------------------------------------------------------- derived figures
def _positions(port: dict) -> list[dict]:
    equity = _num(port.get("equity"))
    out: list[dict] = []
    for x in _as_list(port.get("positions")):
        if not isinstance(x, dict):
            continue
        ticker = _text(x.get("ticker"))
        if ticker is None:
            continue
        mv = _num(x.get("market_value"))
        weight = _num(x.get("weight"))
        if weight is None and mv is not None and equity:
            weight = mv / equity
        out.append(
            {
                "ticker": ticker,
                "qty": _num(x.get("qty")),
                "avg_px": _num(x.get("avg_px")),
                "last_px": _num(x.get("last_px")),
                "market_value": mv,
                "unrealized_pnl": _num(x.get("unrealized_pnl")),
                "unrealized_pct": _num(x.get("unrealized_pct")),
                "weight": weight,
            }
        )
    out.sort(key=lambda r: (r["market_value"] is None, -(r["market_value"] or 0.0)))
    return out


def _max_drawdown(rows: list[dict]) -> float | None:
    if not rows:
        return None
    peak = rows[0]["equity"]
    worst = 0.0
    for r in rows:
        peak = max(peak, r["equity"])
        if peak > 0:
            worst = min(worst, r["equity"] / peak - 1.0)
    return worst


def _ret(now: float | None, base: float | None) -> float | None:
    if now is None or base is None or base == 0:
        return None
    return now / base - 1.0


def _period_return(vals: list[float | None], n: int) -> float | None:
    if len(vals) <= n:
        return None
    return _ret(vals[-1], vals[-1 - n])


def _empty_summary() -> dict:
    return {
        "available": False, "equity": None, "cash": None, "cash_pct": None,
        "positions_value": None, "n_positions": 0, "starting_capital": STARTING_CAPITAL,
        "total_return_pct": None, "day_pnl": None, "day_return_pct": None,
        "benchmark_return_pct": None, "excess_return_pct": None, "max_drawdown_pct": None,
        "halted": False, "halt_reason": None, "price_source": None, "as_of": None,
        "inception": None,
    }


def _empty_decision() -> dict:
    return {
        "available": False, "date": None, "ts": None, "model": None, "status": None,
        "risk_view": None, "cash": None, "notes": None, "prompt_hash": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None}, "cost_usd": None,
        "claude_holdings": {}, "final_weights": {}, "error": None,
    }


def _empty_forecasts() -> dict:
    return {
        "available": False, "asof": None, "model": None, "horizon_days": None,
        "generated_at": None, "rows": [],
    }


# ----------------------------------------------------------------------- the app
def create_app(paths: Paths | None = None) -> FastAPI:
    """Build the dashboard. ``paths`` defaults to ``get_paths()``, read per request."""
    data = _Data(paths)
    app = FastAPI(title="forge-fund dashboard", version="0.1.0")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.middleware("http")
    async def _headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    def guarded(fallback: Callable[[], dict]):
        """A bad file must never become a 500: log the class name and answer empty."""

        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return _sanitize(fn(*args, **kwargs))
                except Exception as exc:  # noqa: BLE001 - deliberate degrade-to-empty
                    log.warning("dashboard %s degraded (%s)", fn.__name__, type(exc).__name__)
                    return _sanitize(fallback())

            return wrapper

        return deco

    @app.get("/", response_class=HTMLResponse)
    def index():
        try:
            html = STATIC_INDEX.read_text(encoding="utf-8")
        except OSError:
            return HTMLResponse("<!doctype html><title>forge-fund</title>dashboard page missing",
                                status_code=404)
        return HTMLResponse(html, headers={"Content-Security-Policy": CSP})

    @app.get("/api/health")
    @guarded(lambda: {"ok": True, "now": _now(), "mode": MODE, "last_run": None,
                      "halted": False, "halt_reason": None})
    def health():
        port = data.portfolio()
        halted, reason = data.halt_state(port)
        return {
            "ok": True,
            "now": _now(),
            "mode": MODE,
            "last_run": _text((port or {}).get("as_of")),
            "halted": halted,
            "halt_reason": reason,
        }

    @app.get("/api/summary")
    @guarded(_empty_summary)
    def summary():
        port = data.portfolio()
        rows = data.curve() or []
        halted, reason = data.halt_state(port)
        port = port or {}
        positions = _positions(port) if port else []

        equity = _num(port.get("equity"))
        if equity is None and rows:
            equity = rows[-1]["equity"]
        cash = _num(port.get("cash"))
        starting = _num(port.get("starting_capital")) or STARTING_CAPITAL

        bench_points = [r["benchmark_equity"] for r in rows if r["benchmark_equity"] is not None]
        bench_last = bench_points[-1] if bench_points else None
        total_ret = _ret(equity, starting)
        bench_ret = _ret(bench_last, starting)
        excess = total_ret - bench_ret if total_ret is not None and bench_ret is not None else None

        day_pnl = day_ret = None
        if len(rows) >= 2:
            prev, last = rows[-2]["equity"], rows[-1]["equity"]
            day_pnl = last - prev
            day_ret = _ret(last, prev)

        positions_value = _num(port.get("positions_value"))
        if positions_value is None and positions:
            mvs = [p["market_value"] for p in positions if p["market_value"] is not None]
            positions_value = sum(mvs) if mvs else None

        return {
            "available": bool(port),
            "equity": equity,
            "cash": cash,
            "cash_pct": cash / equity if cash is not None and equity else None,
            "positions_value": positions_value,
            "n_positions": len(positions),
            "starting_capital": starting,
            "total_return_pct": total_ret,
            "day_pnl": day_pnl,
            "day_return_pct": day_ret,
            "benchmark_return_pct": bench_ret,
            "excess_return_pct": excess,
            "max_drawdown_pct": _max_drawdown(rows),
            "halted": halted,
            "halt_reason": reason,
            "price_source": _text(port.get("price_source")),
            "as_of": _text(port.get("as_of")),
            "inception": _text(port.get("inception")),
        }

    @app.get("/api/equity")
    @guarded(lambda: {"available": False, "rows": []})
    def equity():
        rows = data.curve() or []
        return {"available": bool(rows), "rows": rows[-EQUITY_ROWS_MAX:]}

    @app.get("/api/positions")
    @guarded(lambda: {"available": False, "positions": [], "cash": None, "equity": None})
    def positions():
        port = data.portfolio()
        if port is None:
            return {"available": False, "positions": [], "cash": None, "equity": None}
        return {
            "available": True,
            "positions": _positions(port),
            "cash": _num(port.get("cash")),
            "equity": _num(port.get("equity")),
        }

    @app.get("/api/decision")
    @guarded(_empty_decision)
    def decision():
        d = data.decision()
        if d is None:
            return _empty_decision()
        return {**d, "available": True}

    @app.get("/api/forecasts")
    @guarded(_empty_forecasts)
    def forecasts():
        f = data.forecasts()
        if f is None:
            return _empty_forecasts()
        return {**f, "available": bool(f["rows"])}

    @app.get("/api/pnl")
    @guarded(lambda: {"available": False, "rows": [], "total": None, "periods": []})
    def pnl():
        port = data.portfolio()
        rows = data.curve() or []
        trades = data.trades() or []
        held = {p["ticker"]: p for p in _positions(port)} if port else {}

        realized: dict[str, float] = {}
        for t in trades:
            if t["realized_pnl"] is not None:
                realized[t["ticker"]] = realized.get(t["ticker"], 0.0) + t["realized_pnl"]

        table = []
        for ticker in set(held) | set(realized):
            pos = held.get(ticker)
            unreal = pos["unrealized_pnl"] if pos else 0.0
            real = realized.get(ticker, 0.0)
            table.append({
                "ticker": ticker,
                "qty": pos["qty"] if pos else 0.0,
                "unrealized_pnl": unreal,
                "realized_pnl": real,
                "total_pnl": (unreal or 0.0) + real,
            })
        table.sort(key=lambda r: -r["total_pnl"])

        total = {
            "ticker": "Total",
            "unrealized_pnl": sum(r["unrealized_pnl"] or 0.0 for r in table),
            "realized_pnl": sum(r["realized_pnl"] for r in table),
            "total_pnl": sum(r["total_pnl"] for r in table),
        }

        fund = [r["equity"] for r in rows]
        spy = [r["benchmark_equity"] for r in rows]
        starting = _num(port.get("starting_capital")) if port else None
        starting = starting or STARTING_CAPITAL
        periods = []
        for key, label, n in PERIODS:
            periods.append({"key": key, "label": label,
                            "fund": _period_return(fund, n), "spy": _period_return(spy, n)})
        since_fund = _ret(_num((port or {}).get("equity")) or (fund[-1] if fund else None), starting)
        since_spy = _ret(next((v for v in reversed(spy) if v is not None), None), starting)
        periods.append({"key": "since_inception", "label": "since inception",
                        "fund": since_fund, "spy": since_spy})

        return {
            "available": bool(port or trades or rows),
            "rows": table,
            "total": total,
            "periods": periods,
        }

    @app.get("/api/trades")
    @guarded(lambda: {"available": False, "limit": TRADES_DEFAULT, "trades": []})
    def trades(limit: int = Query(TRADES_DEFAULT, description="Rows to return; clamped to 1..500.")):
        n = max(TRADES_MIN, min(TRADES_MAX, int(limit)))
        rows = data.trades() or []
        return {"available": bool(rows), "limit": n, "trades": list(reversed(rows))[:n]}

    @app.get("/api/backtest")
    @guarded(lambda: {"available": False})
    def backtest():
        found = data.backtest()
        if found is None:
            return {"available": False}
        run_id, metrics = found
        return {**metrics, "available": True, "run_id": run_id}

    @app.get("/api/calls")
    @guarded(lambda: {"available": False, "calls": [], "totals": dict(_EMPTY_TOTALS)})
    def calls():
        c = data.calls()
        if c is None:
            return {"available": False, "calls": [], "totals": dict(_EMPTY_TOTALS)}
        return {"available": True, "calls": c["recent"], "totals": c["totals"]}

    return app


app = create_app()
