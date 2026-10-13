# Crypto in Forge-Fund (Paper)

## Overview
Crypto assets (`BTC-USD`, `ETH-USD`, `SOL-USD`) are integrated as tradeable instruments alongside the S&P 100 stock universe within the unified forge-fund paper trading engine and daily cycle ("stocks + crypto, one fund").

Trading is **strictly paper-only** via simulated execution in Nautilus Trader on `Venue("SIM")` using `AccountType.CASH`. No broker or live trading adapters are imported or used.

## Fractional Sizing Design Decision

### The Capital Problem
The paper book operates with $200 of capital (`PAPER_CAPITAL = 200.0`). At whole-unit share sizes:
- 1 BTC ≈ $60,000 - $85,000 (exceeds total fund capital by >300x)
- 1 ETH ≈ $2,500 - $3,500 (exceeds total fund capital by >12x)
- 1 SOL ≈ $100 - $200 (exceeds the 25% single-position risk limit)

Restricting trading to whole integer lots made crypto completely untradeable for the paper fund.

### The Solution: Fractional Lots for Crypto, Integer for Stocks
- **Stocks remain integer:** Equities use Nautilus `Equity` with `lot_size = 1` and `size_precision = 0` (whole shares).
- **Crypto is fractional:** Crypto uses `size_precision = 6` with `lot_size = 0.000001` (1e-6), allowing the fund to size positions like 0.0005 BTC (~$30) or 0.01 ETH (~$25) within risk caps.

### Nautilus Trader Instrument Choice
1. Nautilus `Equity` hardcodes `size_precision = 0` in Cython, disallowing fractional quantities.
2. Nautilus `CurrencyPair` requires a multi-currency cash account (holding balances in both base and quote currencies) or a margin account, which conflicts with our single-currency USD `AccountType.CASH` simulated venue.
3. Nautilus `Commodity` with `asset_class = AssetClass.CRYPTOCURRENCY`, `quote_currency = USD`, and `size_precision = 6` provides seamless spot cash trading against USD in a single-currency `AccountType.CASH` netting account.
4. Synthetic quote ticks (`_quote`) pass bid and ask sizes with precision matching each instrument's `size_precision` (0 for stocks, 6 for crypto), ensuring the simulated matching engine fills orders cleanly without precision rejection.

## Risk Management

1. **Crypto Aggregate Cap:** Total allocation across all crypto assets combined is capped at `crypto_cap = 0.30` (30% of portfolio equity).
2. **Per-Name Cap:** The standard `max_single = 0.25` (25% of equity) applies to individual crypto holdings just like equities.
3. **Clamping & Enforcement:** `clamp_weights` caps aggregate crypto exposure and redistributes or scales down weights appropriately. `check_limits` audits and flags violations if aggregate crypto exceeds the cap.
4. **Order Sanity:** `order_sanity` validates that crypto order quantities are positive floats, while enforcing that stock order quantities remain positive whole numbers.

## Universe & Data Flow

- `forge/universe.py`: `CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]`, `UNIVERSE = SP100 + CRYPTO`.
- `forge/data.py`: `update_store` downloads daily bars for `UNIVERSE + [BENCHMARK]` via yfinance. `latest_prices` pulls 1-minute intraday closes (or stored daily closes when the market is closed).
- `run/pipeline.py` & `brain/decision.py`: The Claude decision prompt explicitly identifies crypto vs. equity assets and states the 30% aggregate crypto cap, allowing the decider to select and weight names with full awareness of the asset classes.
