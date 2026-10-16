# SPEC: Stocks-only fund (drop crypto from the live book)

## Why
The fund is presented and benchmarked as a **US-equity** fund (SPY benchmark, S&P 100
universe). Crypto was bolted in earlier, and the live book now holds a fractional
BTC-USD position (~$21, 10% of a $200 book) which shows up on the stock dashboard.
Gavin: *"for the stock thing, you can't have crypto."* → the live fund must be
**stocks-only**, and the held crypto must be **liquidated**, not left stranded.

## Root cause
`forge/universe.py` builds `UNIVERSE = [*SP100, *CRYPTO]` unconditionally, and every
consumer (Market.from_store, Kronos, decider, sim) reads `UNIVERSE`. The decider prompt
(`brain/decision.py`) also hardcodes "US large caps and crypto" text and a crypto asset
class. There is no switch.

## The subtle bug to solve (not just a flag)
If crypto is removed from `universe`, `run_cycle(..., universe=mkt.universe)` will not
build a Nautilus instrument for BTC-USD, so `sim._deltas` hits `if t not in self.insts:
continue` and the held BTC is **never sold** — it stays marked-to-market forever.
"Dropping from the universe" alone silently strands the position. The change must **drain**
existing crypto exposure before it disappears from the instrument set.

## Design (reversible, backtests unaffected)
1. **`forge/config.py`** — add a single switch, read from env at import:
   `STOCKS_ONLY = os.environ.get("FORGE_STOCKS_ONLY", "true").strip().lower() in ("1","true","yes","on")`
   Default **true** (stocks-only) — matches the current policy; `FORGE_STOCKS_ONLY=false`
   re-enables crypto (preserves the earlier crypto backtests / SPEC_CRYPTO.md).

2. **`forge/universe.py`** — expose both:
   - `UNIVERSE` = tradeable universe. If `STOCKS_ONLY`: `list(SP100)` only. Else `[*SP100, *CRYPTO]`.
   - Keep `CRYPTO` exported so a one-off drain step can reference it.

3. **Drain (the important part)** — held crypto must sell. Add a helper in
   `engine/paper.py` (or `run/run_daily.py`) that, when `STOCKS_ONLY`, augments the cycle's
   instrument/universe list with **any crypto tickers currently held in the latest ledger
   positions**, so the engine builds a valid instrument + quote for them and the decider
   targets 0 → the engine sells. Once the position is flat, the next cycle no longer
   includes it. Implementation notes:
   - The decider is told the book is **equity-only** (no crypto in its UNIVERSE table), so it
     will not *re-buy* crypto. The held crypto is passed only to the *instrument builder*,
     not to the decider's tradeable table.
   - `latest_prices(...)` must fetch a price for held crypto tickers too (they're in the
     `want` list) so the sell fill is valid.
   - Net effect across 1-2 cycles: BTC-USD sells, position gone, fund is pure equity.

4. **`brain/decision.py`** — make the crypto prompt text **conditional** on whether any
   crypto ticker is actually in the tradeable universe (it isn't when STOCKS_ONLY). When
   stocks-only: drop the "and crypto" / crypto asset-class / "fractional crypto" lines and
   the crypto cap; the mandate reads as a pure US-equity fund. Keep the exact current text
   when crypto is enabled, so the crypto path is unchanged.

5. **No-op guarantees** — when `STOCKS_ONLY` is true, crypto backtests are *not* run by the
   daily runner (it's stocks-only); the crypto code path stays intact and is only disabled
   at the universe boundary. `forge/risk.py` crypto cap logic is untouched (just not
   triggered).

## Acceptance
- [ ] `STOCKS_ONLY` default true; `forge.universe.UNIVERSE` == S&P100 only (no BTC/ETH/SOL).
- [ ] The daily runner, on the first cycle after the change, **sells the held BTC-USD**
      (visible in `state/trades.jsonl` as a BTC-USD SELL) and does not re-buy crypto.
- [ ] `state/portfolio.json` positions contain **no -USD crypto tickers** after the drain.
- [ ] Decider prompt (stocks-only) has no crypto mandate lines; (crypto mode) text is byte-identical to before.
- [ ] `test_no_live.py` and the full suite pass; no live-order path introduced.
- [ ] Dashboard `http://192.168.1.201:8905` Positions shows only equity names.
- [ ] `FORGE_STOCKS_ONLY=false` restores the prior crypto-inclusive behavior (regression guard).

## Out of scope
- Live/real-money orders (structurally absent — `test_no_live.py`).
- Changing backtest starting capital or the SPY benchmark.
- Removing the crypto code entirely (kept, gated).
