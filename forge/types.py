"""Plain dataclasses shared by the engine, brain and runners (no heavy imports)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plan:
    """What the decider wants. ``targets=None`` means HOLD (no trading); ``{}`` means go to cash."""

    targets: dict[str, float] | None
    halted: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class PortfolioView:
    date: str
    equity: float
    cash: float
    prev_equity: float | None            # equity at the previous run (before its trades)
    starting_capital: float
    positions: dict[str, dict]           # ticker -> qty, avg_px, last_px, value, weight, unrealized_pct
    equity_history: list[tuple[str, float]]
    n_runs: int = 0
