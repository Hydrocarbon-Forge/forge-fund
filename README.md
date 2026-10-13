# forge-fund — a paper-first AI hedge fund

An end-to-end, **paper-trading-only** equity fund that runs entirely on one CPU-only machine
(4 cores, ~7 GB RAM, no GPU, Python 3.14). Claude is the portfolio manager, Kronos supplies a
forecasting signal, skfolio sizes the positions, hard-coded risk limits clip everything, and
Nautilus Trader simulates execution and keeps the books. A FastAPI dashboard shows the result.

## Honest framing (read this first)

* **No strategy can guarantee profits, and this project does not claim one.** It delivers a working
  backtest pipeline, a live *paper* loop, enforced risk limits and a dashboard with real PnL numbers.
  Whether the strategy has an edge is an open research question that this repo *measures and reports*.
* Everything is **paper**. There is no broker adapter, no live data/execution client and no API key for
  one anywhere in the code; `tests/test_no_live.py` enforces that structurally. Going live is a separate
  decision that needs a human, broker credentials and the work listed in
  [Going live later](#going-live-later-documented-not-built).
* Backtests are one historical path with known biases (survivorship in the universe, modelled costs,
  a small number of independent decisions). See [Results](#backtest-results) and
  [Known limitations](#known-limitations).

## What it does

Universe: S&P 100 (US large caps) + SPY as benchmark. Long-only, paper capital **$100,000**, whole shares.

```
yfinance daily bars ─► data/ohlcv_daily.parquet
        │
        ├─► Kronos-mini (CPU)  ─► per-ticker 5-day forecast paths ─► exp_ret / p_up / dispersion / z-score
        │                                                              (data/forecasts_YYYYMMDD.parquet)
        ▼
 market snapshot per ticker (returns, vol, RSI, MA distance, 52w position, volume, 60d path)
        │      + Kronos columns + current portfolio & PnL + the previous review's notes
        ▼
 Claude (local OpenAI-compatible endpoint) ─► STRICT JSON {holdings{id:{weight,rationale}},cash,notes,risk_view}
        │     schema-validated; one repair retry; on any failure => HOLD the current portfolio (never guess)
        ▼
 skfolio MeanRisk (max utility, Ledoit-Wolf covariance, long-only, budget = invested fraction,
        │     alpha = Grinold-style IC·σ·z from Kronos z-score + Claude conviction)
        ▼
 forge.risk.clamp_weights  ─► max 10 names · ≤25% per name · ≤80% gross · ≥10% cash · ≥$500 per position
        ▼
 Nautilus simulated venue (CASH account): sells at 09:35:01 ET, buys at 09:35:02, mark at 16:00 ET
        ▼
 state/*.json(l) ─► FastAPI dashboard on 127.0.0.1:8905
```

### Components

| Path | Role |
|---|---|
| `forge/` | config (paths, limits, secrets/redaction), universe, data store (yfinance→parquet), risk limits, calendar, shared types |
| `kronos/forecast.py` | Kronos-mini (4.1M params) + `Kronos-Tokenizer-2k`, CPU only; per-sample paths, seeded, cached per as-of date |
| `brain/` | `claude_client.py` (OpenAI-compatible, 429/Retry-After handling, 9 req/min client-side limiter, cost log), `schema.py` (strict validation), `features.py`, `decision.py` (prompt, anonymisation, repair, response cache) |
| `optimize/portfolio.py` | skfolio wrapper + inverse-vol fallback |
| `engine/sim.py` | Nautilus execution/accounting core shared by backtest and paper |
| `engine/paper.py` | event-sourced paper cycle: `state/ledger.jsonl` replayed through Nautilus each day |
| `engine/metrics.py` | CAGR, vol, Sharpe, Sortino, max DD, win rates, CAPM alpha/beta, bootstrap CI, rank-IC |
| `run/` | `pipeline.py` (the decider), `run_backtest.py`, `run_daily.py`, `report.py` |
| `api/` | dashboard (`app.py`, `server.py`, `static/index.html`; read-only, loopback-only, no CDN) |
| `scripts/` | `setup.sh` (bootstrap), `run_daily.sh` (cron wrapper), `make_demo_state.py` |
| `tests/` | pytest suite (unit + smoke + end-to-end on synthetic data; a real-model Kronos smoke test) |

### Not used, on purpose
* **Freqtrade** is a crypto framework — not applicable to US equities.
* **Hummingbot** ("hummifund") is crypto market-making — not applicable to a US-equity daily-rebalance
  fund. Both are skipped entirely.

## Quick start

```bash
scripts/setup.sh                 # venv, deps (CPU torch), pinned Kronos source, Kronos-mini weights
# .env must contain CLAUDE_BASE_URL / CLAUDE_LOCAL_TOKEN / CLAUDE_MODEL (chmod 600, gitignored, never printed)

.venv/bin/python -m pytest -q                                   # unit + smoke + e2e tests
.venv/bin/python -m run.run_backtest --start 2023-01-03 --freq weekly   # full pipeline backtest, prints real metrics
.venv/bin/python -m run.run_daily                                # one paper cycle (data→Kronos→Claude→skfolio→Nautilus)
.venv/bin/python -m api.server                                   # dashboard on http://127.0.0.1:8905
```

Useful flags: `run_backtest --no-claude` (rule baseline), `--no-kronos`, `--freq daily|weekly|N`,
`--phase kronos` (precompute forecasts in a separate process), `--model haiku`;
`run_daily --dry-run | --prepare | --force | --resume`.

### Scheduling (cron-ready)

```cron
CRON_TZ=America/New_York
30 18 * * 1-5  /path/to/forge-fund/scripts/run_daily.sh --prepare   # evening: refresh data + Kronos forecasts
35  9 * * 1-5  /path/to/forge-fund/scripts/run_daily.sh             # 09:35 ET: decide + paper-trade
```
Exit codes: `0` ok · `2` error · `3` halted by the loss cap (review, then `run_daily.sh --resume`).
Only one cycle can run at a time (file lock). CPU budget: Kronos (heavy, ~25 s) runs before Nautilus
(light) and never concurrently with another cycle.

### Claude usage
* Endpoint: `CLAUDE_BASE_URL` (OpenAI `/chat/completions`), bearer token from `.env` only. The token is
  never logged: every log line and exception passes through `forge.config.redact()`.
* One batched call per rebalance. **Sonnet** on the first rebalance of each ISO week (deep review),
  **haiku** otherwise. 10 req/min per IP is respected by a client-side limiter plus `Retry-After` handling.
* Every call is appended to `logs/decision_log.jsonl` (prompt hash, model, tokens, latency, estimated
  list-price cost, status). Prompt hashes identify a prompt without storing it.

## Risk controls (enforced in code, after the optimizer)

| Control | Value | Where |
|---|---|---|
| Max positions | 10 | `forge.risk.clamp_weights` |
| Max single name | 25% of equity | same |
| Max gross exposure | 80% (long-only ⇒ cash ≥ 20%; the 10% cash floor is subsumed but still enforced) | same |
| Min position notional | $500 | clamp + engine skip |
| Order sanity | whole shares, price within 25% of last close, order ≤ 30% of equity | `forge.risk.order_sanity` |
| Single-run loss cap | equity down > 2% since the previous run ⇒ **HALT**, log, no orders | `forge.risk.loss_cap_breached`, `run/pipeline.py` |
| Cash account | Nautilus rejects any order needing leverage/borrowing | `engine/sim.py` |
| Parse/API failure | **hold** current portfolio — never guess | `brain/decision.py` |
| Paper-only | no live client/adapter exists in the codebase | `tests/test_no_live.py` |

In paper mode a halt writes `state/halt.json` and persists until you run `run_daily --resume`. In backtests
a halt skips that rebalance and trading resumes at the next one (otherwise one bad week would end the
simulation); halts are counted in the report. Note the cap is per *run*, so with weekly rebalances it is
effectively a weekly loss cap.

## Backtest methodology (what is and is not controlled)

* Decisions at 09:35 ET use bars **strictly before** that day; fills are at that day's open ± 1 bp half-spread
  with a 1 bp fee per side; equity is marked at the close. A test (`test_no_lookahead_...`) corrupts all
  future bars and asserts the plan and the prompt do not change.
* **Anonymised prompts**: in backtest mode Claude sees opaque ids (`S001…`) and "sim-day N" instead of real
  tickers/dates, so it cannot simply recall what happened in 2023–2026. This is a mitigation, not a proof:
  recognisable price patterns can still leak information. Live/paper mode uses real tickers.
* Kronos forecasts for each rebalance are computed from data up to the prior close, seeded, cached.
* Reported against SPY buy-and-hold, an exposure-matched SPY (the fund is ≤80% invested), and an
  equal-weight universe baseline, with a CAPM regression, a block-bootstrap CI on excess return, trade-level
  win rate and the measured rank-IC of the Kronos signal.

<<RESULTS>>

## State, logs and files

| File | Content |
|---|---|
| `state/ledger.jsonl` | **source of truth**: per-day fill prices + executed Plan; replayed by Nautilus every cycle |
| `state/portfolio.json`, `equity_curve.jsonl`, `trades.jsonl` | derived from the replay (never hand-edited) |
| `state/latest_decision.json`, `latest_forecasts.json` | last Claude decision (+executed weights), Kronos signal panel |
| `state/halt.json` | present only while halted |
| `logs/decision_log.jsonl`, `logs/engine.log` | every Claude call + decisions; engine log (secrets redacted) |
| `data/ohlcv_daily.parquet`, `data/forecasts_YYYYMMDD.parquet` | market data; cached Kronos forecasts |
| `backtests/<run_id>/` | `metrics.json`, `report.txt`, `equity.csv`, `trades.csv`, `decisions.jsonl`, `calls.jsonl` |

Dashboard (`http://127.0.0.1:8905`, loopback only, GET-only): equity curve vs SPY, positions, last Claude decision
with rationales, Kronos signal panel, PnL table, trade log, backtest summary, Claude usage/cost.

## Going live later (documented, **not built**)

Do **not** attempt this without a human decision, a funded broker account and an explicit review. Suggested path:

1. **Prove it on paper first**: months of daily paper results, compared with the exposure-matched SPY, with
   the Kronos IC and Claude fallback/halt counts reviewed. A good backtest is not evidence of a live edge.
2. **Stage 1 – real-time data, simulated fills**: Nautilus ships a `sandbox` adapter (live market data +
   simulated execution) — run the same strategy on a live clock before any broker is involved.
3. **Stage 2 – broker**: replace `engine/sim.py`'s backtest engine with a Nautilus *live* `TradingNode` and the
   `interactive_brokers` adapter (data + execution clients) for US equities. The decider
   (`run/pipeline.py: decide(date, view) -> Plan`) is already engine-agnostic; what must be written is the
   live strategy wrapper, position/cash reconciliation from the broker (replacing the ledger replay),
   partial-fill/rejection handling, market-hours and halt calendars, and an intraday kill switch.
4. **Controls to add before real money**: broker-side hard limits (max order size, daily loss), a manual
   approval step for the first N orders, alerting, secrets in a proper store (not `.env`), audit retention of
   prompts/decisions, and a documented rollback. Taxes, compliance and data-licensing are on you.
5. **Credentials**: the owner supplies broker keys at that time; nothing in this repo reads or stores them.

## Known limitations

* Universe is **today's** S&P 100 constituents (survivorship / look-ahead in membership) — flatters results.
* Daily bars only; fills at the open are idealised (no market impact; 2 bp quoted spread + 1 bp fee/side).
* yfinance adjusted prices (splits/dividends) are used for history; live paper fills use the recorded price, so a
  held name's ex-dividend date can shift paper PnL by up to its dividend yield.
* One backtest path, weekly cadence (a daily-cadence run would need ~960 Claude calls and ~8 h of Kronos CPU), so
  confidence intervals are wide; the LLM component is non-deterministic across runs (responses are cached by
  prompt hash for resumability, not for reproducibility of a fresh run).
* Kronos-mini is a very small model; its measured IC is reported, not assumed.
* The 80% gross cap structurally lags a fully-invested benchmark in rising markets.
* Dashboard port 8905 must be free; a different local service may hold it (see `docs/`/final notes).

## Layout & conventions
Python 3.14 only. `FORGE_ROOT` redirects `data/ state/ logs/ backtests/` (tests use a tmp dir).
Spec: `SPEC.md`. Module contracts and on-disk schemas: `docs/CONTRACTS.md`.
