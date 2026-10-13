"""Central configuration: paths, risk limits, endpoint settings, secret handling.

Design rules
------------
* The Claude bearer token is read from ``.env`` / the process environment only.
  It is never logged; ``redact()`` scrubs it from any text that might reach a log.
* ``get_paths()`` honours ``FORGE_ROOT`` so tests can run against a tmp directory
  and never touch real state/ logs/ data/.
* Risk limits live here (code), not in the prompt. See ``forge/risk.py`` for enforcement.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dashboard binds the box's LAN IP so Gavin can open http://192.168.1.201:8905
# from his phone/laptop. Paper-trading data only — no secrets, GET-only endpoints.
DASHBOARD_HOST = "192.168.1.201"
DASHBOARD_PORT = 8905

BENCHMARK = "SPY"
STARTING_CAPITAL = 100_000.0
MODE = "paper"  # structurally the only mode this codebase knows about


# --------------------------------------------------------------------------- env
def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (without overriding existing)."""
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _secret_values() -> list[str]:
    vals = []
    for k in ("CLAUDE_LOCAL_TOKEN",):
        v = os.environ.get(k)
        if v and len(v) >= 6:
            vals.append(v)
    return vals


def redact(text: object) -> str:
    """Remove any known secret from ``text``. Use on anything headed for a log/exception."""
    s = str(text)
    for v in _secret_values():
        s = s.replace(v, "***REDACTED***")
    return s


# ------------------------------------------------------------------------- paths
@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    state: Path
    logs: Path
    backtests: Path
    models: Path
    vendor: Path

    def ensure(self) -> "Paths":
        for p in (self.data, self.state, self.logs, self.backtests):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    root = Path(os.environ.get("FORGE_ROOT", PROJECT_ROOT)).resolve()
    # models/ and vendor/ always live in the real project (large, shared, read-only)
    return Paths(
        root=root,
        data=root / "data",
        state=root / "state",
        logs=root / "logs",
        backtests=root / "backtests",
        models=Path(os.environ.get("FORGE_MODELS", PROJECT_ROOT / "models")),
        vendor=PROJECT_ROOT / "vendor",
    )


# ------------------------------------------------------------------- risk limits
@dataclass(frozen=True)
class Limits:
    """Hard limits, enforced in code after optimization (see forge/risk.py)."""

    max_positions: int = 10
    max_single: float = 0.25        # max weight of one name (fraction of equity)
    max_gross: float = 0.80         # long-only => gross == invested fraction
    cash_floor: float = 0.10        # min cash fraction (subsumed by max_gross for long-only)
    min_notional: float = 500.0     # drop / ignore positions & trades smaller than this
    loss_cap: float = 0.02          # single-run loss > 2% of equity => HALT
    max_order_pct_equity: float = 0.30   # sanity: no single order > 30% of equity
    price_sanity_pct: float = 0.25  # reject fill price >25% away from last close
    rebalance_band: float = 0.01    # ignore weight changes smaller than 1% of equity


LIMITS = Limits()


# ------------------------------------------------------------- claude / pipeline
@dataclass(frozen=True)
class ClaudeSettings:
    base_url: str
    token: str = ""            # never print; use redact()
    default_model: str = "sonnet"   # CLAUDE_MODEL from .env (weekly deep review)
    routine_model: str = "haiku"    # cheap daily calls
    max_requests_per_min: int = 9   # endpoint limit is 10/min per IP
    timeout_s: float = 180.0
    max_retries: int = 6
    limit_wait_cap_s: float = 0.0   # >0: if the endpoint reports a usage-limit, sleep until reset (<= this many s)

    def __repr__(self) -> str:  # never leak the token via repr()
        return (f"ClaudeSettings(base_url={self.base_url!r}, default_model={self.default_model!r}, "
                f"routine_model={self.routine_model!r}, token={'set' if self.token else 'MISSING'})")


def claude_settings() -> ClaudeSettings:
    load_env()
    return ClaudeSettings(
        base_url=os.environ.get("CLAUDE_BASE_URL", "http://192.168.1.201:8910/v1").rstrip("/"),
        token=os.environ.get("CLAUDE_LOCAL_TOKEN", ""),
        default_model=os.environ.get("CLAUDE_MODEL", "sonnet"),
        routine_model=os.environ.get("CLAUDE_ROUTINE_MODEL", "haiku"),
    )


# List-price USD per million tokens (input, output). ESTIMATES for the cost log only —
# the local wrapper may bill differently (e.g. subscription). Real model ids per SPEC:
# haiku=claude-haiku-4-5, sonnet=claude-sonnet-4-6, opus=claude-opus-4-6.
PRICE_PER_MTOK = {
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (5.0, 25.0),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["sonnet"])
    return round((prompt_tokens * pin + completion_tokens * pout) / 1_000_000, 6)


# ----------------------------------------------------------- model / signal cfg
@dataclass(frozen=True)
class KronosSettings:
    model_name: str = "Kronos-mini"            # smallest variant (4.1M params)
    tokenizer_name: str = "Kronos-Tokenizer-2k"
    max_context: int = 512
    lookback: int = 128        # daily bars fed to the model (128 ~ 22s/101 tickers on 4 CPU cores)
    pred_len: int = 5          # forecast horizon (trading days)
    sample_count: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    batch_size: int = 25
    seed: int = 7


KRONOS = KronosSettings()
