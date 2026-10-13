"""The decision pipeline: data -> Kronos -> Claude -> skfolio -> hard risk limits -> Plan.

``DecisionPipeline.decide(date, view)`` is the single decider used by BOTH the backtest and the paper
engine (see engine/sim.py). It only ever reads bars strictly before ``date`` (decision at 09:35 ET uses
data through the prior close). Every failure path degrades to HOLD - the pipeline never guesses.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from brain.claude_client import ClaudeClient
from brain.decision import BrainResult, ClaudeBrain, Masker, build_messages
from brain.features import compute_features, index_context, universe_regime
from forge.config import BENCHMARK, KRONOS, LIMITS, Limits, claude_settings, get_paths, redact
from forge.data import load_panel
from forge.risk import clamp_weights, loss_cap_breached, max_investable
from forge.types import Plan, PortfolioView
from forge.universe import SP100, UNIVERSE
from kronos.forecast import KronosForecaster, forecast_with_cache
from optimize.portfolio import optimize_weights

log = logging.getLogger("forge.pipeline")


# ------------------------------------------------------------------------------------- market data
class Market:
    """Wide OHLCV panels (index=date, cols=ticker) plus as-of helpers."""

    def __init__(self, panels: dict[str, pd.DataFrame], universe: list[str], benchmark: str = BENCHMARK):
        self.p = panels
        self.universe = [t for t in universe if t in panels["close"].columns]
        self.benchmark = benchmark
        self.dates = panels["close"].index

    @classmethod
    def from_store(cls, start: str | None = None, universe: list[str] | None = None) -> "Market":
        fields = ("open", "high", "low", "close", "volume")
        panels = {f: load_panel(f, start=start) for f in fields}
        return cls(panels, universe or UNIVERSE)


    def prev_trading_day(self, date: str) -> pd.Timestamp | None:
        prior = self.dates[self.dates < pd.Timestamp(date)]
        return prior[-1] if len(prior) else None

    def histories(self, tickers: list[str], asof: pd.Timestamp, n: int) -> dict[str, pd.DataFrame]:
        sl = slice(None, asof)
        out = {}
        for t in tickers:
            df = pd.DataFrame({f: self.p[f][t].loc[sl] for f in ("open", "high", "low", "close", "volume")}).dropna().tail(n)
            out[t] = df
        return out

    def returns(self, asof: pd.Timestamp) -> pd.DataFrame:
        c = self.p["close"].loc[:asof, self.universe]
        return c.pct_change(fill_method=None).iloc[1:]


# ------------------------------------------------------------------------------------- rule baseline
def rule_weights(features: pd.DataFrame, kronos: pd.DataFrame | None, limits: Limits = LIMITS) -> dict[str, float]:
    """No-Claude baseline: rank by 60d momentum (ex last week) + Kronos z-score - volatility; hold top N."""
    z = lambda s: ((s - s.mean()) / (s.std(ddof=0) or 1.0)).clip(-3, 3)  # noqa: E731
    mom = z(features["r60"] - features["r5"])
    low_vol = -z(features["vol20"])
    score = 0.5 * mom + 0.2 * low_vol
    if kronos is not None and len(kronos):
        k = kronos.drop_duplicates("ticker").set_index("ticker")["score"].reindex(features.index).fillna(0.0)
        score = score + 0.3 * k.clip(-3, 3)
    top = score.sort_values(ascending=False).head(limits.max_positions)
    return {t: max_investable(limits) / len(top) for t in top.index}


# ------------------------------------------------------------------------------------- pipeline
@dataclass
class PipelineConfig:
    mode: str                                   # "backtest" | "paper"
    anonymize: bool                             # mask tickers + dates in prompts (backtest)
    use_claude: bool = True
    use_kronos: bool = True
    forced_model: str | None = None             # override the haiku/sonnet schedule
    persist_halt: bool = False                  # paper: a loss-cap halt survives until --resume
    state_dir: Path | None = None
    sim_start: str | None = None                # backtest: for the "sim-day N" label
    decisions_path: Path | None = None          # backtest: per-decision jsonl
    prior_event_dates: list[str] = field(default_factory=list)
    kronos_wait_s: float = 0.0                  # wait for a separate Kronos precompute process


class DecisionPipeline:
    def __init__(self, market: Market, cfg: PipelineConfig, brain: ClaudeBrain | None = None,
                 kronos: KronosForecaster | None = None, limits: Limits = LIMITS):
        self.m, self.cfg, self.brain, self.kronos, self.limits = market, cfg, brain, kronos, limits
        self.masker = Masker(market.universe, enabled=cfg.anonymize)
        self.settings = claude_settings()
        self.prev_notes: str | None = None
        self.prev_risk: str | None = None
        self.halts = 0
        self.limit_blocked = False                  # backtest circuit breaker once the Claude limit cannot be waited out
        self.last_kdf: pd.DataFrame | None = None
        self.last_asof: pd.Timestamp | None = None
        self.stats = {"calls": 0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                      "fallback_holds": 0, "decisions": 0}
        self._weeks_seen: set[tuple[int, int]] = set()
        for d in cfg.prior_event_dates:
            self._weeks_seen.add(self._week(d))
        if cfg.state_dir and (cfg.state_dir / "latest_decision.json").exists():
            try:
                last = json.loads((cfg.state_dir / "latest_decision.json").read_text())
                self.prev_notes, self.prev_risk = last.get("notes"), last.get("risk_view")
            except (OSError, json.JSONDecodeError):
                pass

    @staticmethod
    def _week(date: str) -> tuple[int, int]:
        iso = pd.Timestamp(date).isocalendar()
        return int(iso.year), int(iso.week)

    def _model_for(self, date: str) -> tuple[str, bool]:
        """sonnet for the first rebalance of each ISO week (deep review), haiku otherwise."""
        wk = self._week(date)
        deep = wk not in self._weeks_seen
        self._weeks_seen.add(wk)
        model = self.cfg.forced_model or (self.settings.default_model if deep else self.settings.routine_model)
        return model, deep

    # ------------------------------------------------------------------------------ the decider
    def decide(self, date: str, view: PortfolioView) -> Plan:
        meta: dict = {"date": date, "status": "init"}
        cfg, st = self.cfg, self.cfg.state_dir
        asof = self.m.prev_trading_day(date)
        if asof is None:
            return Plan(None, meta={**meta, "status": "no_data"})

        # 0) persisted halt (paper) -------------------------------------------------------------
        if cfg.persist_halt and st and (st / "halt.json").exists():
            return Plan(None, True, {**meta, "status": "halted", "error": "halt.json present (use --resume to clear)"})

        # 1) single-run loss cap ----------------------------------------------------------------
        breached, pnl = loss_cap_breached(view.prev_equity, view.equity, self.limits)
        if breached:
            self.halts += 1
            reason = f"run loss {pnl:.2%} exceeds cap {self.limits.loss_cap:.0%} of equity"
            log.warning("HALT %s: %s", date, reason)
            if cfg.persist_halt and st:
                (st / "halt.json").write_text(json.dumps(
                    {"ts": datetime.now(timezone.utc).isoformat(), "reason": reason, "loss_pct": round(pnl, 5)}))
            return Plan(None, True, {**meta, "status": "halted", "error": reason, "loss_pct": pnl})

        # 2) features -------------------------------------------------------------------------
        feats = compute_features(self.m.p["close"][self.m.universe], self.m.p["volume"][self.m.universe], asof)
        if len(feats) < max(3, min(20, len(self.m.universe) // 2)):
            return Plan(None, meta={**meta, "status": "no_data", "error": f"only {len(feats)} tickers with history"})
        market = index_context(self.m.p["close"][self.m.benchmark], asof)

        # 3) Kronos signals (cached per as-of date) -----------------------------------------------
        kdf = None
        if cfg.use_kronos and self.kronos is not None:
            try:
                kdf = forecast_with_cache(self.kronos, self.m.histories(list(feats.index), asof, KRONOS.lookback + 5), asof,
                                        wait_s=cfg.kronos_wait_s)
                kdf = kdf[kdf["ticker"].isin(feats.index)]
            except Exception as e:  # noqa: BLE001 - signals are optional; never block trading logic
                log.error("kronos failed: %s", redact(e))
                meta["kronos_error"] = redact(f"{type(e).__name__}: {e}")
        meta["kronos_n"] = 0 if kdf is None else int(len(kdf))
        self.last_kdf, self.last_asof = kdf, asof

        # 4) brain: Claude (or rule baseline) -----------------------------------------------------
        label = (f"sim-day {int(np.searchsorted(self.m.dates, pd.Timestamp(date)) - np.searchsorted(self.m.dates, pd.Timestamp(cfg.sim_start)))}"
                 if cfg.anonymize and cfg.sim_start else date)
        claude_w: dict[str, float] = {}
        br: BrainResult | None = None
        if cfg.use_claude:
            model, deep = self._model_for(date)
            msgs = build_messages(label, feats, kdf, view, self.masker, market, self.prev_notes, self.prev_risk, self.limits,
                                  KRONOS.pred_len, regime=universe_regime(feats))
            if self.limit_blocked:
                self.stats["fallback_holds"] += 1
                meta.update(status="claude_limit", error="Claude usage limit not cleared; decision skipped")
                self._record(date, meta, None)
                return Plan(None, meta=meta)
            br = self.brain.decide(msgs, model, self.masker.ids())
            self.stats["calls"] += br.calls
            self.stats["cost_usd"] += br.cost_usd
            self.stats["prompt_tokens"] += br.usage.get("prompt_tokens", 0)
            self.stats["completion_tokens"] += br.usage.get("completion_tokens", 0)
            meta.update(model=model, deep_review=deep, prompt_hash=br.prompt_hash, usage=br.usage,
                        cost_usd=round(br.cost_usd, 6), cached=br.cached)
            if br.status != "ok" or br.decision is None:
                self.stats["fallback_holds"] += 1
                if br.status == "claude_limit":
                    self.stats["limit_hits"] = self.stats.get("limit_hits", 0) + 1
                    self.limit_blocked = cfg.mode == "backtest"
                meta.update(status=br.status if br.status == "claude_limit" else "fallback_hold", error=br.error)
                self._record(date, meta, None)
                return Plan(None, meta=meta)
            d = br.decision
            claude_w = {self.masker.to_ticker(i): h.weight for i, h in d.holdings.items()}
            meta.update(risk_view=d.risk_view, cash=d.cash, notes=d.notes,
                        claude_holdings={self.masker.to_ticker(i): {"weight": h.weight, "rationale": h.rationale}
                                         for i, h in d.holdings.items()})
            self.prev_notes, self.prev_risk = d.notes, d.risk_view
        else:
            claude_w = rule_weights(feats, kdf, self.limits)
            meta.update(status="no_claude", model="rules", risk_view="med", cash=round(1 - sum(claude_w.values()), 4),
                        notes="rule baseline (no Claude)",
                        claude_holdings={t: {"weight": w, "rationale": "rule baseline"} for t, w in claude_w.items()})

        # 5) optimise / size, then re-impose hard limits ------------------------------------------------
        if not claude_w:                                   # Claude chose 100% cash
            final: dict[str, float] = {}
            meta["optimizer"] = "all_cash"
        else:
            opt = optimize_weights(claude_w, self.m.returns(asof), kdf, self.limits, view.equity)
            meta["optimizer"] = opt.method
            if opt.notes:
                meta["optimizer_notes"] = opt.notes
            if opt.method == "empty":
                self.stats["fallback_holds"] += 1
                meta.update(status="fallback_hold", error="optimizer produced no weights: " + "; ".join(opt.notes))
                self._record(date, meta, None)
                return Plan(None, meta=meta)
            final = clamp_weights(opt.weights, self.limits, view.equity)
        meta["final_weights"] = final
        if meta["status"] in ("init",):
            meta["status"] = "ok"
        self.stats["decisions"] += 1
        log.info("%s status=%s model=%s positions=%d invested=%.0f%% risk=%s cost=$%.4f", date, meta["status"],
                 meta.get("model"), len(final), 100 * sum(final.values()), meta.get("risk_view"), meta.get("cost_usd", 0.0))
        self._record(date, meta, final)
        return Plan(final, False, meta)

    # ------------------------------------------------------------------------------ persistence
    def _record(self, date: str, meta: dict, final: dict | None) -> None:
        rec = {"kind": "decision", "ts": datetime.now(timezone.utc).isoformat(), "date": date,
               "prompt_hash": meta.get("prompt_hash"), "status": meta.get("status"), "decision": meta}
        if self.cfg.decisions_path:
            self.cfg.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.decisions_path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
