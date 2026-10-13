# forge-fund — module contracts & on-disk schemas

Python 3.14, venv at `.venv/` (`.venv/bin/python`, `.venv/bin/pytest`). Paper trading only.
All code imports shared settings from `forge.config` (paths via `get_paths()`, limits via `LIMITS`,
secrets via `claude_settings()` / `redact()`). **Never print or log the Claude token.**
Tests must never touch real `data/ state/ logs/`: `tests/conftest.py` points `FORGE_ROOT` at a tmp dir.

## On-disk files (the engine writes, the dashboard reads)

All under `get_paths().state` unless noted. All timestamps ISO-8601 strings; dates `YYYY-MM-DD`.

### `state/portfolio.json`  (current paper portfolio, overwritten each cycle)
```json
{
  "as_of": "2026-10-13T09:35:04-04:00", "mode": "paper", "inception": "2026-10-13",
  "starting_capital": 100000.0, "cash": 12345.67, "positions_value": 87975.45, "equity": 100321.12,
  "price_source": "live_1m | last_close",
  "halted": false, "halt_reason": null,
  "positions": [
    {"ticker": "AAPL", "qty": 100, "avg_px": 150.12, "last_px": 152.3, "market_value": 15230.0,
     "unrealized_pnl": 218.0, "unrealized_pct": 0.0145, "weight": 0.152}
  ]
}
```

### `state/equity_curve.jsonl`  (one line per date, last write wins per date)
`{"date": "2026-10-13", "equity": 100321.12, "cash": 12345.67, "benchmark_equity": 100110.4, "drawdown": -0.004}`
`benchmark_equity` = SPY buy&hold scaled to the same starting capital from the inception date.

### `state/trades.jsonl`  (append-only fills)
`{"ts": "...", "date": "2026-10-13", "ticker": "AAPL", "side": "BUY|SELL", "qty": 100, "price": 150.12,
  "notional": 15012.0, "fee": 1.5, "realized_pnl": 0.0, "reason": "rebalance"}`

### `state/latest_decision.json`  (last brain decision + what was actually executed)
```json
{
  "date": "2026-10-13", "ts": "...", "model": "haiku", "status": "ok | fallback_hold | halted | no_claude",
  "risk_view": "low|med|high", "cash": 0.2, "notes": "free text regime notes",
  "prompt_hash": "ab12cd34ef56", "usage": {"prompt_tokens": 8000, "completion_tokens": 600}, "cost_usd": 0.0123,
  "claude_holdings": {"AAPL": {"weight": 0.12, "rationale": "..."}},
  "final_weights":   {"AAPL": 0.11},
  "error": null
}
```

### `state/latest_forecasts.json`  (Kronos signal panel)
```json
{
  "asof": "2026-10-12", "model": "Kronos-mini", "horizon_days": 5, "generated_at": "...",
  "rows": [{"ticker": "AAPL", "exp_ret": 0.0123, "p_up": 0.66, "dispersion": 0.011, "score": 1.12}]
}
```
(`exp_ret` = mean forecast cumulative return over the horizon, `p_up` = share of samples with positive
cumulative return, `dispersion` = std of sample returns, `score` = cross-sectional z-score of exp_ret.)

### `state/halt.json`  (only present while halted)  `{"ts": "...", "reason": "...", "loss_pct": -0.031}`

### `logs/decision_log.jsonl` — one JSON object per line, two kinds
* `{"kind": "call", "ts": "...", "purpose": "decision|review|repair", "model": "haiku", "prompt_hash": "...",
   "prompt_chars": 31000, "status": "ok|error", "attempts": 1, "latency_s": 3.2,
   "usage": {"prompt_tokens": 8000, "completion_tokens": 600}, "cost_usd": 0.012, "error": null}`
* `{"kind": "decision", "ts": "...", "date": "...", "prompt_hash": "...", "status": "...", "decision": {...}}`

### `data/ohlcv_daily.parquet`  — long format, columns
`date (datetime64[ns], tz-naive), ticker (str), open, high, low, close, volume` (split/dividend adjusted,
yfinance `auto_adjust=True`); includes the benchmark `SPY`. `data/meta.json`:
`{"fetched_at": "...", "start": "2020-12-01", "last_date": "2026-10-12", "n_tickers": 99, "dropped": ["XYZ"]}`

### `backtests/<run_id>/` — `metrics.json`, `report.txt`, `equity.csv`, `trades.csv`, `decisions.jsonl`

## Module APIs

### `forge/universe.py`
`SP100: list[str]` (yfinance symbols, e.g. `BRK-B`), `BENCHMARK = "SPY"`.

### `forge/data.py`
* `fetch_history(tickers, start="2020-12-01", end=None) -> pd.DataFrame` (long format above; drops tickers with
  no rows; cleans non-positive prices / NaN OHLC rows; sorted by ticker, date).
* `update_store(tickers=None, start="2020-12-01", store_dir=None) -> dict` fetches universe + SPY, writes
  `ohlcv_daily.parquet` + `meta.json`, returns meta dict. If the download fails entirely, keep the old store and raise.
* `load_store() -> pd.DataFrame`; `load_panel(field="close", tickers=None, start=None, end=None) -> pd.DataFrame`
  (wide: index=date, columns=ticker, forward-filled max 3 days, NO backfill).
* `latest_prices(tickers) -> tuple[dict[str, float], str]` live-ish prices via yfinance 1-minute bars for today;
  falls back per-ticker to last stored close. Second return value is the source: `"live_1m"` or `"last_close"`.

### `brain/schema.py`
pydantic v2 models `Holding(weight, rationale)`, `Decision(holdings, cash, notes, risk_view)`,
`DecisionParseError(Exception)`, `parse_decision(text: str, valid_tickers: set[str]) -> Decision`.
### `brain/claude_client.py`
`ClaudeClient(settings=None, transport=None, log_path=None, sleep=time.sleep, clock=time.monotonic)`,
`.chat(messages, model=None, max_tokens=4096, temperature=0.2, purpose="decision") -> ChatResult`
(`text, model, usage, latency_s, cost_usd, prompt_hash, attempts`), `ClaudeError`, `extract_json(text)`.

### `api/app.py`
`create_app(paths=None) -> FastAPI`; module-level `app = create_app()`; `python -m api.server` runs uvicorn on
`127.0.0.1:8905` (SPEC says 8904, but that port is held by another local service; host/port from `forge.config.DASHBOARD_HOST/PORT`).
