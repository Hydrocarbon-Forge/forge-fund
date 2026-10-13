# forge-fund — AI Hedge Fund (SPEC)

Owner: Gavin. Machine: 4 CPU cores, ~7GB RAM, NO GPU, Python 3.14.7 (only python on host).
Everything runs on THIS machine. Paper trading FIRST; live real-money is a later, separate
decision (Gavin approves + supplies broker keys).

## Honest framing (state this in README, never hide it)
- No strategy can guarantee profits. This project delivers: a working backtest pipeline,
  a live PAPER trading loop, risk limits, and a dashboard with real PnL numbers.
  Profitability is an open research question, measured and reported, not promised.
- The fund trades US-listed equities (S&P 100 universe), daily rebalance, paper capital $100,000.

## Stack (all real, verified)
1. **nautilus_trader** (1.231.0, cp314 wheel verified) — backtesting + paper trading engine,
   execution, positions, PnL. Paper = Nautilus backtest node running on a live clock.
2. **Kronos** (shiyu-coder/Kronos, HF: NeoQuasar/Kronos) — candlestick foundation model.
   Use the SMALL model (tiny ~400MB base). CPU-only, forecast on a schedule (nightly),
   NEVER per-tick. Predicts next-candle paths for the universe; outputs become signals.
3. **skfolio** — portfolio optimization: takes returns history + forecast signal →
   position weights (risk parity or mean-variance with constraints). skfolio datasets
   provide S&P data, or yfinance as backup data source.
4. **Claude via local OpenAI-compatible endpoint** — the strategy/positioning brain.
   - base_url `http://192.168.1.201:8910/v1`, bearer key in project `.env`
     (CLAUDE_LOCAL_TOKEN, chmod 600, gitignored — NEVER print it in logs/code/commits).
   - OpenAI `/v1/chat/completions` format, models: `haiku`/`sonnet`/`opus` (real ids
     claude-sonnet-4-6 / claude-haiku-4-5 / claude-opus-4-6).
   - Rate limit: 10 req/min per IP → batch decisions, use `haiku` for cheap routine calls,
     `sonnet` for weekly deep reviews. Retry with Retry-After on 429.
   - Claude's role: given (a) market snapshot per ticker (last 60d OHLCV + indicators),
     (b) Kronos forecast summary, (c) current portfolio + PnL, (d) regime notes →
     output STRICT JSON: {"holdings": {"AAPL": {"weight": 0.12, "rationale": "..."}},
     "cash": 0.1, "notes": "...", "risk_view": "low|med|high"}. Schema-validated; on
     parse failure → hold current portfolio (never guess).
5. **Freqtrade** — crypto framework; NOT used for equities. Skip entirely.
6. **Hummingbot** (Gavin's "hummifund") — crypto market-making; not applicable to US
   equities paper fund. Skip (note in README why).
7. **Dashboard**: FastAPI + small HTML page on port **8904** (localhost only): equity
   curve, positions, last Claude decision + rationales, Kronos signal panel, PnL table,
   trade log. (forge-console ports 8901-8903 are taken — do NOT touch them.)

## Architecture
```
data/ (parquet)   yfinance bars, S&P100, daily, 3y+ history
  └─ nightly: fetch → data/*.parquet
kronos/          small model; nightly forecast per ticker → data/forecasts_YYYYMMDD.parquet
brain/           claude_client.py (OpenAI-compatible, .env token, 429 retry, JSON schema
                 validation) + decision.py (snapshot builder, prompt builder)
optimize/        skfolio wrapper: returns + kronos signal + claude targets → weights
                 with constraints (max 10 positions, max 25% single, max 80% gross
                 exposure, cash floor 10%, min $500 notional)
engine/          nautilus paper node: rebalance daily 09:35 ET (after open), executes
                 target weights, tracks equity; state/ holds portfolio state.json
api/             FastAPI dashboard :8904
run/             run_daily.py (orchestrates: data → kronos → claude → optimize → engine),
                 run_backtest.py (historical: same pipeline on 2023-2026 daily bars,
                 prints metrics: CAGR, max DD, Sharpe, win rate, vs SPY buy-hold)
logs/            decision_log.jsonl (every claude call: prompt hash, decision, cost),
                 engine.log
```

## Hard requirements
- `run_backtest.py` MUST run end-to-end on historical data and print real metrics
  (this is the proof the pipeline works; no fabricated numbers).
- `run_daily.py` MUST run once successfully in paper mode and update state/ + dashboard.
- All Claude calls logged; token/cost tracked; 429 handled; no secrets in logs.
- Risk limits in code (not just prompt): weights clamped post-optimization, order size
  sanity checks, single-run loss cap 2% of equity → halt + log.
- CPU budget: one heavy process at a time (kronos then nautilus, never concurrent).
- Every component has unit-level smoke tests in tests/ (pytest).
- Git: commit as you go; identity "Hydrocarbon Forge" / forge@hydrocarbon.local (repo-local).
- README.md: what it does, honest framing, how to run, how to switch to live broker LATER
  (nautilus live node + broker adapter — documented, not built yet).
- No live orders, no broker keys, no real money — paper only, structurally enforced
  (engine config has no live data-venue wired).

## Environment
- venv: `python3.14 -m venv .venv`; install nautilus_trader, skfolio, yfinance, pandas,
  fastapi, uvicorn, pydantic, Kronos (from shiyu-coder/Kronos, torch CPU).
- Kronos model download: HuggingFace `NeoQuasar/Kronos` smallest variant → models/ (gitignored).
- .env already exists with CLAUDE_BASE_URL / CLAUDE_LOCAL_TOKEN / CLAUDE_MODEL (do not print token).

## Deliverables (definition of done)
1. Backtest report printed + saved to backtests/ (metrics JSON + text).
2. Paper engine running daily via cron-ready script; state/ consistent.
3. Dashboard live on :8904 with real data.
4. tests/ passing.
5. README honest + complete.
6. Final summary: what ran, metrics, what's left for live mode.

## Do NOT
- No real orders, no broker keys, no withdrawals, no crypto.
- No touching stock-playground (Gavin's existing paper-trader) — separate project.
- No ports other than 8904. No GPU assumptions. No secrets in code/logs/commits.
