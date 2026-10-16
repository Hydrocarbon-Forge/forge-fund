"""Tests for stocks-only universe (FORGE_STOCKS_ONLY) and held-crypto drain."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

import forge.config as cfg
import forge.universe as universe
from brain.decision import Masker, build_messages, system_prompt
from brain.features import compute_features
from engine.paper import held_crypto_tickers, plan_from_json, read_ledger, run_cycle, write_ledger
from forge.config import LIMITS, Limits, Paths, get_paths
from forge.risk import is_crypto
from forge.types import Plan
from run.pipeline import Market
from tests.synth import make_long, panels_from_long
from tests.synth_small import make_close, make_view, make_volume


def test_stocks_only_universe_has_no_crypto_tickers():
    assert cfg.STOCKS_ONLY is True
    assert universe.UNIVERSE == universe.SP100
    assert len(universe.UNIVERSE) == 101
    assert not any(is_crypto(t) for t in universe.UNIVERSE)
    assert not any(t.endswith("-USD") for t in universe.UNIVERSE)
    assert set(universe.CRYPTO) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    assert "CRYPTO" in universe.__all__
    assert "SP100" in universe.__all__
    assert "UNIVERSE" in universe.__all__


def test_env_flag_disables_stocks_only(monkeypatch):
    monkeypatch.setenv("FORGE_STOCKS_ONLY", "false")
    importlib.reload(cfg)
    importlib.reload(universe)
    try:
        assert cfg.STOCKS_ONLY is False
        assert len(universe.UNIVERSE) == 104
        assert set(universe.CRYPTO).issubset(set(universe.UNIVERSE))
    finally:
        monkeypatch.delenv("FORGE_STOCKS_ONLY", raising=False)
        importlib.reload(cfg)
        importlib.reload(universe)


def test_system_prompt_byte_identity_when_crypto_enabled():
    # When has_crypto=True, prompt must match the exact prior prompt text byte-for-byte
    prompt_crypto = system_prompt(LIMITS, has_crypto=True)
    expected = (
        "You are the portfolio manager of a small PAPER-TRADING fund investing in US large caps and crypto "
        "(stocks + crypto, one fund). Once per rebalance you output target portfolio weights.\n"
        "HARD RULES (code enforces them again, so violations are simply clipped):\n"
        f"- Long-only. At most {LIMITS.max_positions} holdings. At most {LIMITS.max_single:.0%} in any one name.\n"
        f"- Total invested at most {LIMITS.max_gross:.0%} of equity (so cash >= {1 - LIMITS.max_gross:.0%}). Weights are fractions of TOTAL equity.\n"
        f"- Total crypto combined at most {LIMITS.crypto_cap:.0%} of equity.\n"
        "- Only use ids that appear in the UNIVERSE table. Never invent ids.\n"
        "- Integer-share equity is whole lots, crypto is fractional, and weights for names the book cannot afford are clipped to 0 by the engine — plan to the equity shown.\n"
        "MANDATE: this is a long-only equity fund (with select crypto), not a market-timing vehicle. Its job is to hold a diversified set of "
        "the best-looking large caps and select crypto. The normal state is roughly 60-80% invested across 6-10 names. Crypto is fractionally "
        f"sizeable and capped at {LIMITS.crypto_cap:.0%} in aggregate. Hold more than 50% "
        "cash ONLY when the INDEX line shows concrete stress RIGHT NOW (drawdown from its 52-week high worse than -10%, "
        "or 20-day volatility above ~25%). Sitting mostly in cash for an extended period is a failure of the mandate, "
        "not a conservative choice. Higher cash is a response to stress visible in today's data, never a standing preference.\n"
        "GUIDANCE:\n"
        "- Trading costs money: do not churn. Keep current holdings unless the evidence changed.\n"
        "- The kr_* columns are forecasts from a small statistical foundation model (Kronos). They are noisy and "
        "often wrong; treat them as ONE weak input next to trend, momentum, volatility and the market context. "
        "Do not wait for Kronos to be 'proven' before investing: if you distrust it, give it little weight and decide "
        "on trend, momentum, volatility and the market context.\n"
        "- Do not assume any information beyond what is in the prompt.\n"
        "OUTPUT: reply with ONE JSON object and nothing else (no markdown, no prose) with exactly this shape:\n"
        '{"holdings": {"<ID>": {"weight": 0.12, "rationale": "<=25 words"}}, "cash": 0.20, "notes": "<=120 words", "risk_view": "low|med|high"}\n'
        "holdings weights + cash must sum to 1.0 (within 0.02). rationale: <=25 words each. "
        "notes: <=120 words - your regime view, written for the human audit log only (you will NOT see it again; "
        "every review is decided fresh from the data and the current holdings). "
        "risk_view: your overall market risk assessment."
    )
    assert prompt_crypto == expected
    # Default argument has_crypto must also return the byte-identical text
    assert system_prompt(LIMITS) == expected


def test_system_prompt_stocks_only_drops_crypto_mandate():
    prompt_stocks = system_prompt(LIMITS, has_crypto=False)
    assert "crypto" not in prompt_stocks.lower()
    assert "stocks + crypto" not in prompt_stocks
    assert "Total crypto combined" not in prompt_stocks
    assert "crypto is fractional" not in prompt_stocks
    assert "(with select crypto)" not in prompt_stocks
    assert "Crypto is fractionally sizeable" not in prompt_stocks
    assert "investing in US large caps. Once per rebalance" in prompt_stocks
    assert "MANDATE: this is a long-only equity fund, not a market-timing vehicle." in prompt_stocks
    assert "- Integer-share equity is whole lots, and weights for names the book cannot afford" in prompt_stocks


def test_build_messages_prompt_conditional():
    tickers_stocks = ["AAPL", "MSFT", "PFE"]
    close = make_close(tickers_stocks, n_days=300, seed=1)
    feats = compute_features(close, make_volume(close, seed=1), close.index[-1])
    view = make_view(equity=200.0)
    market = {"r1": 0.01, "r5": 0.02, "vol20": 0.15}
    prices = {"AAPL": 150.0, "MSFT": 300.0, "PFE": 25.0}

    # 1. Stocks-only universe
    msgs_stocks = build_messages("2026-10-16", feats, None, view, Masker(tickers_stocks, enabled=False), market, None, None, prices=prices)
    sys_content = msgs_stocks[0]["content"]
    user_content = msgs_stocks[1]["content"]
    assert "crypto" not in sys_content.lower()
    assert "ASSET CLASSES:" not in user_content
    assert "Fractional crypto" not in user_content
    assert "plan holdings using ONLY affordable whole-share names." in user_content

    # 2. Universe with crypto
    tickers_crypto = ["AAPL", "BTC-USD"]
    close_c = make_close(tickers_crypto, n_days=300, seed=2)
    feats_c = compute_features(close_c, make_volume(close_c, seed=2), close_c.index[-1])
    msgs_crypto = build_messages("2026-10-16", feats_c, None, view, Masker(tickers_crypto, enabled=False), market, None, None, prices={"AAPL": 150.0, "BTC-USD": 80000.0})
    sys_c = msgs_crypto[0]["content"]
    user_c = msgs_crypto[1]["content"]
    assert "crypto" in sys_c.lower()
    assert "ASSET CLASSES:" in user_c
    assert "BTC-USD" in user_c
    assert "Fractional crypto (BTC-USD) is always sizeable" in user_c


def test_held_crypto_tickers_helper(tmp_path, monkeypatch):
    root = tmp_path / "forge_test"
    paths = Paths(
        root=root,
        data=root / "data",
        state=root / "state",
        logs=root / "logs",
        backtests=root / "backtests",
        models=root / "models",
        vendor=root / "vendor",
    ).ensure()

    # 1. Empty state
    assert held_crypto_tickers(paths) == []

    # 2. Portfolio has crypto position
    pf = {
        "positions": [
            {"ticker": "AAPL", "qty": 10},
            {"ticker": "BTC-USD", "qty": 0.00025},
        ]
    }
    (paths.state / "portfolio.json").write_text(json.dumps(pf))
    assert held_crypto_tickers(paths) == ["BTC-USD"]

    # 3. Portfolio has zero crypto qty
    pf_zero = {
        "positions": [
            {"ticker": "AAPL", "qty": 10},
            {"ticker": "BTC-USD", "qty": 0.0},
        ]
    }
    (paths.state / "portfolio.json").write_text(json.dumps(pf_zero))
    assert held_crypto_tickers(paths) == []

    # 4. Portfolio only equities
    pf_eq = {
        "positions": [
            {"ticker": "AAPL", "qty": 10},
            {"ticker": "MSFT", "qty": 5},
        ]
    }
    (paths.state / "portfolio.json").write_text(json.dumps(pf_eq))
    assert held_crypto_tickers(paths) == []

    # 5. portfolio.json missing, ledger has crypto target
    (paths.state / "portfolio.json").unlink()
    write_ledger(paths, [
        {"date": "2026-10-14", "plan": {"targets": {"AAPL": 0.5, "BTC-USD": 0.2}}},
    ])
    assert held_crypto_tickers(paths) == ["BTC-USD"]


def test_held_crypto_drain_produces_a_sell(tmp_path, monkeypatch):
    root = tmp_path / "drain_test"
    paths = Paths(
        root=root,
        data=root / "data",
        state=root / "state",
        logs=root / "logs",
        backtests=root / "backtests",
        models=root / "models",
        vendor=root / "vendor",
    ).ensure()
    monkeypatch.setenv("FORGE_ROOT", str(root))

    # Create market with stocks + crypto
    names = ["AAPL", "MSFT", "BTC-USD", "SPY"]
    df = make_long(names, 300, seed=7)
    p = panels_from_long(df)
    m = Market(p, universe=["AAPL", "MSFT"])  # decider's universe is stocks only!

    close = m.p["close"]
    idx = close.index
    d1 = idx[-3].strftime("%Y-%m-%d")
    d2 = idx[-2].strftime("%Y-%m-%d")
    d3 = idx[-1].strftime("%Y-%m-%d")

    px1 = {t: float(close.loc[idx[-3], t]) for t in names}
    px2 = {t: float(close.loc[idx[-2], t]) for t in names}
    px3 = {t: float(close.loc[idx[-1], t]) for t in names}

    # Day 1: bought BTC-USD and AAPL (historical cycle)
    p1 = Plan({"AAPL": 0.4, "BTC-USD": 0.2})
    run_cycle(paths=paths, market=m, decision_date=d1, live_prices=px1, price_source="test",
              live_decide=lambda d, v: p1, universe=["AAPL", "MSFT", "BTC-USD"], capital=100_000.0)

    # Verify BTC-USD is currently held
    assert held_crypto_tickers(paths) == ["BTC-USD"]
    pf1 = json.loads((paths.state / "portfolio.json").read_text())
    assert any(pos["ticker"] == "BTC-USD" and pos["qty"] > 0 for pos in pf1["positions"])

    # Day 2: DRAIN CYCLE under STOCKS_ONLY!
    # Decider trades ONLY equities (AAPL, MSFT), targeting 0 crypto.
    decide_called_with_tickers = []
    def decider_stocks_only(date, view):
        decide_called_with_tickers.extend(view.positions.keys())
        return Plan({"AAPL": 0.5, "MSFT": 0.3})

    summary2 = run_cycle(paths=paths, market=m, decision_date=d2, live_prices=px2, price_source="test",
                         live_decide=decider_stocks_only, universe=["AAPL", "MSFT"], capital=100_000.0)

    # 1. Decider was called and saw BTC-USD in portfolio view
    assert "BTC-USD" in decide_called_with_tickers

    # 2. Portfolio now has NO crypto positions!
    pf2 = json.loads((paths.state / "portfolio.json").read_text())
    pos_tickers2 = [pos["ticker"] for pos in pf2["positions"]]
    assert "BTC-USD" not in pos_tickers2
    assert not any(is_crypto(t) for t in pos_tickers2)

    # 3. trades.jsonl contains a SELL fill for BTC-USD on Day 2
    trades = [json.loads(l) for l in (paths.state / "trades.jsonl").read_text().splitlines()]
    btc_sells = [t for t in trades if t["ticker"] == "BTC-USD" and t["side"] == "SELL" and t["date"] == d2]
    assert len(btc_sells) == 1
    assert btc_sells[0]["qty"] > 0

    # 4. held_crypto_tickers is now empty!
    assert held_crypto_tickers(paths) == []

    # Day 3: Next cycle - no crypto held, only equities trade
    summary3 = run_cycle(paths=paths, market=m, decision_date=d3, live_prices=px3, price_source="test",
                         live_decide=lambda d, v: Plan({"AAPL": 0.5, "MSFT": 0.3}),
                         universe=["AAPL", "MSFT"], capital=100_000.0)

    pf3 = json.loads((paths.state / "portfolio.json").read_text())
    pos_tickers3 = [pos["ticker"] for pos in pf3["positions"]]
    assert not any(is_crypto(t) for t in pos_tickers3)
