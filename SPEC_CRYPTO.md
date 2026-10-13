# Task: Add crypto to forge-fund (paper)

**Harness/model:** You are running on **antigravity** with **gemini-3.8-flash-high** (the user's
choice; the default Claude decider is rate-limited until Oct 16). Build autonomously; no live
trading; paper only.

## Goal
Add **crypto** as tradeable assets to the existing **forge-fund** paper system, alongside the
current S&P-100 stocks, in the SAME paper engine and daily cycle. Keep everything else
working. This is paper-only — there is and must remain **no live-order code path**
(`tests/test_no_live.py` enforces this; never import a Nautilus live client / broker adapter).

## The real blocker (read first)
The engine is built around **integer whole-share** accounting:
- `engine/sim.py::make_equity()` → `Equity(..., lot_size=Quantity.from_int(1), price_precision=2)`.
- `engine/sim.py::ForgeStrategy._deltas()` and `_stage2()` round order size with
  `int(math.floor(tgt_w * eq / ask))` and `math.floor(free / px)`.
- `forge/risk.py::order_sanity()` / `check_limits()` assume integer `qty`.

At the fund's **$200** paper capital (`PAPER_CAPITAL` in `forge/config.py`), an integer lot makes
**whole BTC / ETH / SOL untradeable** (one SOL ≈ >$200). So simply adding crypto symbols to the
universe is NOT enough — crypto must be **fractionally sizeable** (or the book must hold a
fractional exposure). Decide and implement the cleanest correct mechanism. Guidance (verify it,
note if the real constraint differs):
- Give crypto instruments a **fractional lot** (e.g. Nautilus `lot_size` with a fine
  `price_precision` / lot of 1e-6, or a fractional quantity) so a $200 book can hold, say, 0.001
  BTC. Keep **stocks integer** (lot=1) — do not make stocks fractional.
- Make the sizing/`qty` math work for both integer and fractional lots. If Nautilus' `Equity`
  type cannot express fractional lot cleanly, use the appropriate Nautilus instrument type for
  the crypto legs (e.g. a fractional/`FX`-like or `Crypto` instrument on the same `SIM` venue)
  while keeping `Equity` for stocks. **The backtest and paper paths must share the same sizing
  code** (that is the invariant: one `engine/sim.py` for both).
- Whatever you pick: fills, marks, realized-PnL, cost-basis (`on_order_filled`) and the ledger
  must be correct for fractional quantities.

## Concrete scope
1. **Universe** (`forge/universe.py`): add a `CRYPTO` list. Start with a small curated set,
   e.g. `BTC-USD`, `ETH-USD`, `SOL-USD` (yfinance daily symbols). Keep `SP100` and `BENCHMARK="SPY"`
   unchanged. Provide a combined default investable universe = `SP100 + CRYPTO`.
2. **Data** (`forge/data.py`): the store is yfinance-based and already handles `BTC-USD`-style
   daily bars — confirm crypto flows through `fetch_history` / `update_store` / `load_panel` /
   `latest_prices` unchanged (they key off symbol, so this should be free). If not, extend
   minimally. Crypto trades 24/7 but the fund rebalances **daily at 09:35 ET** on the daily bar —
   that is fine; do not build an intraday crypto feed.
3. **Sizing** (`engine/sim.py`): fractional lot for crypto, integer for stocks (see blocker).
4. **Risk** (`forge/risk.py`, `forge/config.py`): limits are fraction-based and already
   asset-agnostic. Add a **crypto aggregate weight cap** (e.g. ≤ 30% of equity across all crypto
   combined) so the fund can't tilt into a single risky asset class; enforce it in the same
   `clamp_weights`/`check_limits` flow. Per-name `max_weight` should also apply to crypto.
5. **Pipeline / decider** (`run/pipeline.py`, `brain/decision.py`, `run/run_daily.py`): the
   pipeline and the Claude decider must treat crypto names like any other asset in the universe
   (features, Kronos signal, skfolio optimization, `valid_ids`). The decider prompt should make
   clear which names are crypto vs equities so it can size sensibly. Keep the "stocks + crypto,
   one fund" framing.
6. **Config** (`forge/config.py`): a `CRYPTO_CAP` (default 0.30) and a way to list the crypto
   names the risk layer knows about (so the aggregate cap is enforceable). `PAPER_CAPITAL` stays
   $200; `STARTING_CAPITAL` stays $100k for backtests.

## Constraints (non-negotiable)
- **Paper only.** No live client, no broker, no real orders. `tests/test_no_live.py` must still
  pass and still assert the no-live invariants (and that the dashboard binds `192.168.1.201:8905`).
- **No look-ahead.** Decisions see only bars strictly before the decision date; fills at that
  day's open. Do not change the timeline.
- **One sizing engine** for backtest and paper (`engine/sim.py`).
- **No secrets in the repo.** `.env` stays `chmod 600` and gitignored; never print API keys.
- **Backtests** (`STARTING_CAPITAL=100k`) and the **$200 paper book** must both work with the
  new crypto+stock universe.

## Definition of done
- `update_store` pulls crypto daily bars alongside stocks; `load_panel`/`latest_prices` return
  crypto.
- A $200 book can hold a **fractional** crypto position (demonstrate with a unit test at a tiny
  capital).
- Crypto aggregate cap enforced; per-name caps apply to crypto.
- Decider can target crypto names; the pipeline treats them uniformly.
- **Full test suite passes** (`.venv/bin/python -m pytest -q`). Add regression tests for:
  fractional crypto sizing at small capital, the crypto aggregate cap, and a no-live invariant.
- **Git:** this is a git repo — commit as you go (the git rule). Clear per-item commits.
- A short `docs/CRYPTO.md` (or a section in the README) describing what changed and the
  fractional-sizing design decision, so a future reader knows why crypto is fractional and stocks
  are not.

## Verify before claiming done
Run the real test suite; run one **paper** cycle on the combined universe (it will HOLD if the
decider is rate-limited — that's expected today, the Claude CLI is on weekly limit until Oct 16;
a HOLD is a PASSING outcome here, not a failure). Do NOT run a full long backtest unless it is
fast; a short-window backtest or unit-level check is enough to prove crypto sizes correctly.

## Note on the current "halted" book
The $200 book is **not loss-halted** (no `state/halt.json`). It has been sitting in all-cash
because the decider was returning 403 (bridge down) and is now rate-limited (Claude CLI weekly
limit, resets Oct 16 4pm UTC). The bridge (192.168.1.201:8910) has been restarted and is
healthy. No `--resume` is needed. Just don't treat the all-cash state as a bug in this task.
