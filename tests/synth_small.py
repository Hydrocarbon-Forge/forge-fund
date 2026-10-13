"""Small, seeded synthetic fixtures for the unit tests (no I/O, no network, no real state)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.claude_client import ChatResult
from forge.types import PortfolioView


def make_close(tickers, n_days: int = 300, seed: int = 7, start: str = "2024-01-02",
               daily_vol: float = 0.012, drift: float = 0.0003) -> pd.DataFrame:
    """Wide close-price frame (index = business days, columns = tickers): seeded geometric random walks."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    steps = rng.normal(drift, daily_vol, size=(n_days, len(tickers)))
    px = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(px, index=idx, columns=list(tickers))


def make_volume(close: pd.DataFrame, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vals = rng.integers(500_000, 2_000_000, size=close.shape).astype(float)
    return pd.DataFrame(vals, index=close.index, columns=close.columns)


def make_returns(n_days: int = 300, n_tickers: int = 12, seed: int = 3, prefix: str = "T") -> pd.DataFrame:
    """Wide daily simple-return frame (index = business days) for the optimiser."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.025, n_tickers)
    rets = rng.normal(0.0003, vols, size=(n_days, n_tickers))
    idx = pd.bdate_range("2023-06-01", periods=n_days)
    return pd.DataFrame(rets, index=idx, columns=[f"{prefix}{i:02d}" for i in range(n_tickers)])


def make_view(equity: float = 103_456.78, positions: dict | None = None) -> PortfolioView:
    return PortfolioView(
        date="2024-06-03",
        equity=equity,
        cash=equity * 0.5,
        prev_equity=102_000.0,
        starting_capital=100_000.0,
        positions=positions or {},
        equity_history=[("2024-05-31", 101_000.0), ("2024-06-03", 102_000.0)],
        n_runs=3,
    )


def chat_result(text: str, model: str = "haiku", prompt_tokens: int = 1200, completion_tokens: int = 150,
                cost_usd: float = 0.0021) -> ChatResult:
    """A ChatResult with the same shape ClaudeClient.chat() returns."""
    return ChatResult(
        text=text, model=model, usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        latency_s=0.01, cost_usd=cost_usd, prompt_hash="0" * 12, attempts=1, request_id=None,
    )


class FakeClaude:
    """Stand-in for ClaudeClient. Returns scripted replies (the last one repeats) or raises ``exc``.

    Every request is recorded in ``requests`` so tests can inspect exactly what would have been sent.
    """

    def __init__(self, replies: list[str] | None = None, exc: Exception | None = None):
        self.replies = list(replies or [])
        self.exc = exc
        self.requests: list[dict] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def chat(self, messages, model, max_tokens, temperature, purpose):
        self.requests.append({"messages": [dict(m) for m in messages], "model": model,
                              "max_tokens": max_tokens, "temperature": temperature, "purpose": purpose})
        if self.exc is not None:
            raise self.exc
        text = self.replies[min(len(self.requests), len(self.replies)) - 1]
        return chat_result(text, model=model)
