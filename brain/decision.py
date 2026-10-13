"""Prompt construction + Claude call + strict parsing for the portfolio decision.

* ``Masker``     - backtest anonymisation: real tickers -> stable opaque ids (S001...), real dates hidden.
                   Stops the LLM from "remembering" 2023-2026 outcomes; live mode uses real tickers.
* ``build_messages`` - system + user prompt (market snapshot table, Kronos columns, portfolio, notes).
* ``ClaudeBrain.decide`` - one batched call per rebalance; one repair retry on schema failure;
                   any remaining failure => status "fallback_hold" (the caller holds the portfolio).
* Responses are cached on disk by prompt hash so interrupted backtests resume without re-spending calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from brain.claude_client import ClaudeClient, ClaudeError, ClaudeLimitError, detect_limit_message
from brain.schema import Decision, DecisionParseError, parse_decision
from forge.config import LIMITS, Limits, get_paths, redact
from forge.types import PortfolioView

log = logging.getLogger("forge.brain")

SCHEMA_EXAMPLE = ('{"holdings": {"<ID>": {"weight": 0.12, "rationale": "<=25 words"}}, '
                  '"cash": 0.20, "notes": "<=120 words", "risk_view": "low|med|high"}')


def system_prompt(limits: Limits = LIMITS) -> str:
    return (
        "You are the portfolio manager of a small PAPER-TRADING equity fund investing in US large caps. "
        "Once per rebalance you output target portfolio weights.\n"
        "HARD RULES (code enforces them again, so violations are simply clipped):\n"
        f"- Long-only. At most {limits.max_positions} holdings. At most {limits.max_single:.0%} in any one name.\n"
        f"- Total invested at most {limits.max_gross:.0%} of equity (so cash >= {1 - limits.max_gross:.0%}). Weights are fractions of TOTAL equity.\n"
        "- Only use ids that appear in the UNIVERSE table. Never invent ids.\n"
        "GUIDANCE:\n"
        "- Trading costs money: do not churn. Keep current holdings unless the evidence changed.\n"
        "- The kr_* columns are forecasts from a small statistical foundation model (Kronos). They are noisy and "
        "often wrong; treat them as ONE weak input next to trend, momentum, volatility and the market context.\n"
        "- Holding more cash is a valid decision when the market looks fragile. Be calibrated, not heroic.\n"
        "- Do not assume any information beyond what is in the prompt.\n"
        "OUTPUT: reply with ONE JSON object and nothing else (no markdown, no prose) with exactly this shape:\n"
        f"{SCHEMA_EXAMPLE}\n"
        "holdings weights + cash must sum to 1.0 (within 0.02). rationale: <=25 words each. "
        "notes: <=120 words - your regime view and what would change your mind (it is shown to you at the next review). "
        "risk_view: your overall market risk assessment."
    )


# --------------------------------------------------------------------------------- anonymisation
class Masker:
    """Stable ticker <-> opaque id mapping. ``enabled=False`` is the identity (live mode)."""

    def __init__(self, tickers: list[str], enabled: bool, salt: str = "forge-bt-v1"):
        self.enabled = enabled
        if enabled:
            order = sorted(tickers, key=lambda t: hashlib.sha256(f"{salt}:{t}".encode()).hexdigest())
            self.fwd = {t: f"S{i + 1:03d}" for i, t in enumerate(order)}
        else:
            self.fwd = {t: t for t in tickers}
        self.rev = {v: k for k, v in self.fwd.items()}

    def to_id(self, t: str) -> str:
        if self.enabled and t not in self.fwd:           # never leak a real ticker into an anonymised prompt
            raise KeyError(f"ticker {t!r} is not in the anonymisation map")
        return self.fwd.get(t, t)

    def to_ticker(self, i: str) -> str:
        return self.rev.get(i, i)

    def ids(self) -> set[str]:
        return set(self.rev)


# --------------------------------------------------------------------------------- prompt
def _pct(x, d=1) -> str:
    return "na" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f}"


def _row(i: str, r: pd.Series, k: pd.Series | None) -> str:
    kr = "|".join([f"{k['exp_ret'] * 100:+.2f}", f"{k['p_up']:.2f}", f"{k['score']:+.1f}"]) if k is not None else "na|na|na"
    path = ",".join(str(int(p)) for p in r["path"])
    return (f"{i}|{_pct(r['r1'])}|{_pct(r['r5'])}|{_pct(r['r20'])}|{_pct(r['r60'])}|{r['vol20'] * 100:.0f}|"
            f"{r['rsi14']:.0f}|{_pct(r['d50'])}|{_pct(r['d200'])}|{r['p52']:.2f}|{r['volx']:.2f}|{kr}|{path}")


def build_messages(label: str, features: pd.DataFrame, kronos: pd.DataFrame | None, view: PortfolioView,
                   masker: Masker, market: dict[str, float], prev_notes: str | None, prev_risk: str | None,
                   limits: Limits = LIMITS, horizon: int = 5) -> list[dict]:
    kidx = kronos.drop_duplicates("ticker").set_index("ticker") if kronos is not None and len(kronos) else None
    rows = []
    ex = ([masker.to_id(t) for t in list(features.index[:2])] + ["AAA", "BBB"])[:2]
    # table order is opaque (hash order) in anonymised mode, alphabetical otherwise
    tickers = sorted(features.index, key=lambda t: masker.to_id(t))
    for t in tickers:
        k = kidx.loc[t] if kidx is not None and t in kidx.index else None
        rows.append(_row(masker.to_id(t), features.loc[t], k))

    eq_hist = [e for _, e in view.equity_history] or [view.equity]
    peak = max(eq_hist + [view.equity])
    dd = view.equity / peak - 1 if peak else 0.0
    since = view.equity / view.starting_capital - 1
    period = (view.equity / view.prev_equity - 1) if view.prev_equity else 0.0
    cash_pct = view.cash / view.equity if view.equity else 1.0
    if view.positions:
        hold = "; ".join(
            f"{masker.to_id(t)} {p['weight'] * 100:.1f}% (cost->now {_pct(p['unrealized_pct'])}%)"
            for t, p in sorted(view.positions.items(), key=lambda kv: -kv[1]["value"]))
    else:
        hold = "none (100% cash)"
    mk = market
    market_line = (f"INDEX r1 {_pct(mk.get('r1'))}% r5 {_pct(mk.get('r5'))}% r20 {_pct(mk.get('r20'))}% "
                   f"r60 {_pct(mk.get('r60'))}% | vol20 {mk.get('vol20', float('nan')) * 100:.0f}% | "
                   f"52w-range position {mk.get('p52', float('nan')):.2f} | drawdown from 52w high "
                   f"{_pct(mk.get('dd_from_high'))}%") if mk else "INDEX: unavailable"
    user = (
        f"AS OF: {label} (decision for the next open; you only see data through the prior close)\n"
        f"FUND: equity ${view.equity:,.0f} | cash {cash_pct * 100:.0f}% | since inception {_pct(since, 2)}% | "
        f"last period {_pct(period, 2)}% | drawdown from peak {_pct(dd, 2)}% | previous risk_view: {prev_risk or 'none'}\n"
        f"CURRENT HOLDINGS: {hold}\n"
        f"YOUR NOTES FROM THE PREVIOUS REVIEW: {prev_notes or 'none (first review)'}\n"
        f"MARKET CONTEXT: {market_line}\n"
        f"UNIVERSE ({len(rows)} names). Columns, pipe-separated: id|r1|r5|r20|r60 (% price change over 1/5/20/60 trading days)|"
        f"vol (annualised 20d volatility %)|rsi (14d)|d50|d200 (% vs 50/200-day average)|p52 (position in 52-week range 0-1)|"
        f"volx (20d vs 60d avg volume)|kr_ret (Kronos mean forecast % return over next {horizon} trading days)|"
        f"kr_pup (share of Kronos sample paths that end up)|kr_z (cross-sectional z-score of kr_ret)|"
        f"path (12 points: % change vs 60 trading days ago, sampled every 5 days, oldest first)\n"
        + "\n".join(rows) +
        f"\n\nChoose target weights for the next rebalance.\n"
        f"Reply with ONLY a JSON object using EXACTLY these keys (no markdown fences, no other keys, no prose). Example shape:\n"
        f'{{"holdings": {{"{ex[0]}": {{"weight": 0.15, "rationale": "short reason"}}, "{ex[1]}": {{"weight": 0.10, "rationale": "short reason"}}}}, '
        f'"cash": 0.75, "notes": "regime view and what would change your mind", "risk_view": "med"}}\n'
        f"Rules: \"holdings\" maps id -> {{weight, rationale}}; weights are fractions of total equity; holdings weights + cash = 1.0; "
        f"risk_view is exactly one of: low, med, high."
    )
    return [{"role": "system", "content": system_prompt(limits)}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------------- the brain
@dataclass
class BrainResult:
    status: str                     # ok | fallback_hold | claude_limit
    decision: Decision | None = None
    model: str = ""
    prompt_hash: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None
    calls: int = 0
    cached: bool = False


class ClaudeBrain:
    def __init__(self, client: ClaudeClient | None, use_cache: bool = True, cache_dir: Path | None = None,
                 max_tokens: int = 2500):
        self.client = client
        self.use_cache = use_cache
        self.cache_dir = cache_dir or (get_paths().data / "claude_cache")
        self.max_tokens = max_tokens

    def _key(self, model: str, messages: list[dict]) -> str:
        return hashlib.sha256(json.dumps([model, messages], sort_keys=True).encode()).hexdigest()[:16]

    def _cache_get(self, key: str) -> dict | None:
        p = self.cache_dir / f"{key}.json"
        if self.use_cache and p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _cache_put(self, key: str, payload: dict) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(payload))

    def decide(self, messages: list[dict], model: str, valid_ids: set[str]) -> BrainResult:
        orig_key = self._key(model, messages)
        res = BrainResult(status="fallback_hold", model=model, prompt_hash=orig_key[:12])
        convo = list(messages)
        for attempt in range(2):                       # original + one repair
            cached = self._cache_get(self._key(model, convo))
            if cached and detect_limit_message(cached.get("text", ""))[0]:
                cached = None                          # never trust a poisoned entry
            try:
                if cached:
                    text, usage, cost, was_cached = cached["text"], cached["usage"], cached["cost_usd"], True
                else:
                    if self.client is None:
                        res.error = "no claude client"
                        return res
                    out = self.client.chat(convo, model=model, max_tokens=self.max_tokens, temperature=0.2,
                                           purpose="decision" if attempt == 0 else "repair")
                    text, usage, cost, was_cached = out.text, out.usage, out.cost_usd, False
                    res.calls += 1
            except ClaudeLimitError as e:
                res.status, res.error = "claude_limit", redact(f"claude usage limit: {e}")
                log.error(res.error)
                return res
            except ClaudeError as e:
                res.error = redact(f"claude call failed: {e}")
                log.error(res.error)
                return res
            res.usage = {k: res.usage.get(k, 0) + usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
            res.cost_usd += cost
            res.cached = was_cached
            try:
                res.decision = parse_decision(text, valid_ids)
                res.status = "ok"
                res.error = None
                # cache only VALID answers, under the original prompt key (so a resumed run is deterministic)
                self._cache_put(orig_key, {"text": text, "usage": res.usage, "cost_usd": res.cost_usd, "model": model})
                return res
            except DecisionParseError as e:
                res.error = f"parse failure: {e}"
                log.warning("decision parse failure (attempt %d): %s", attempt + 1, e)
                convo = list(messages) + [
                    {"role": "assistant", "content": text[:4000]},
                    {"role": "user", "content": f"Your reply was rejected by the validator: {e}. "
                                                 "Reply again with ONLY the corrected JSON object."}]
        return res
