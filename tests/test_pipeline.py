"""DecisionPipeline: limits, fallbacks, loss-cap halt, no look-ahead, model schedule."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from brain.decision import BrainResult
from brain.schema import Decision, Holding
from forge.config import get_paths
from forge.risk import check_limits
from forge.types import PortfolioView
from run.pipeline import DecisionPipeline, Market, PipelineConfig
from tests.synth import make_long, panels_from_long, universe_names

NAMES = universe_names(12)


def mkt(n=420, seed=5):
    return Market(panels_from_long(make_long(NAMES + ["SPY"], n, seed)), NAMES)


def view(eq=100_000.0, prev=None):
    return PortfolioView("x", eq, eq, prev, 100_000.0, {}, [], 0)


class FakeBrain:
    def __init__(self, decision=None, status="ok"):
        self.decision, self.status, self.calls = decision, status, []

    def decide(self, messages, model, valid_ids):
        self.calls.append((model, messages))
        if self.status != "ok":
            return BrainResult(status="fallback_hold", model=model, error="boom")
        # map real names -> opaque ids when anonymised
        d = self.decision(valid_ids) if callable(self.decision) else self.decision
        return BrainResult(status="ok", decision=d, model=model, prompt_hash="abc", usage={"prompt_tokens": 10, "completion_tokens": 5}, calls=1)


def decision_for(ids_sorted, n=4, w=0.15, cash=0.4):
    ids = sorted(ids_sorted)[:n]
    return Decision(holdings={i: Holding(weight=w, rationale="r") for i in ids}, cash=cash, notes="note", risk_view="med")


def pipe(brain=None, anonymize=True, **cfg_kw):
    m = mkt()
    cfg = PipelineConfig(mode="backtest", anonymize=anonymize, use_kronos=False, sim_start=str(m.dates[300].date()), **cfg_kw)
    return DecisionPipeline(m, cfg, brain, None), m


def test_plan_respects_limits_and_unmasks_tickers():
    p, m = pipe(FakeBrain(lambda ids: decision_for(ids, n=4, w=0.15, cash=0.4)))
    d = m.dates[350].strftime("%Y-%m-%d")
    plan = p.decide(d, view())
    assert plan.meta["status"] == "ok" and plan.targets
    assert set(plan.targets) <= set(NAMES)                           # real tickers, not S001 ids
    assert check_limits(plan.targets, equity=100_000) == []
    assert plan.meta["claude_holdings"] and set(plan.meta["claude_holdings"]) <= set(NAMES)


def test_no_lookahead_future_bars_do_not_change_the_plan():
    brain_a, brain_b = FakeBrain(lambda ids: decision_for(ids)), FakeBrain(lambda ids: decision_for(ids))
    pa, ma = pipe(brain_a)
    pb, mb = pipe(brain_b)
    d = ma.dates[350]
    for f in mb.p:                                                      # corrupt everything on/after the decision date
        mb.p[f].loc[d:] = mb.p[f].loc[d:] * 7.0 + 123
    plan_a, plan_b = pa.decide(d.strftime("%Y-%m-%d"), view()), pb.decide(d.strftime("%Y-%m-%d"), view())
    assert brain_a.calls[0][1] == brain_b.calls[0][1]                  # identical prompts
    assert plan_a.targets == plan_b.targets


def test_prompt_is_anonymised_in_backtest_mode():
    brain = FakeBrain(lambda ids: decision_for(ids))
    p, m = pipe(brain)
    p.decide(m.dates[350].strftime("%Y-%m-%d"), view())
    text = json.dumps(brain.calls[0][1])
    assert not any(t in text for t in NAMES)
    assert "sim-day" in text and "2024" not in text and "2023" not in text


def test_brain_failure_means_hold_not_guess():
    p, m = pipe(FakeBrain(status="fail"))
    plan = p.decide(m.dates[350].strftime("%Y-%m-%d"), view())
    assert plan.targets is None and plan.meta["status"] == "fallback_hold" and not plan.halted
    assert p.stats["fallback_holds"] == 1


def test_all_cash_decision_goes_to_cash():
    dec = Decision(holdings={}, cash=1.0, notes="risk off", risk_view="high")
    p, m = pipe(FakeBrain(dec))
    plan = p.decide(m.dates[350].strftime("%Y-%m-%d"), view())
    assert plan.targets == {} and plan.meta["optimizer"] == "all_cash"


def test_loss_cap_halts_and_paper_halt_persists():
    st = get_paths().state
    p, m = pipe(FakeBrain(lambda ids: decision_for(ids)), persist_halt=True, state_dir=st)
    d = m.dates[350].strftime("%Y-%m-%d")
    plan = p.decide(d, view(eq=97_000, prev=100_000))                  # -3% > 2% cap
    assert plan.halted and plan.targets is None and (st / "halt.json").exists()
    assert json.loads((st / "halt.json").read_text())["loss_pct"] == pytest.approx(-0.03)
    again = p.decide(m.dates[351].strftime("%Y-%m-%d"), view(eq=100_000, prev=100_000))   # even a flat run stays halted
    assert again.halted and p.halts == 1


def test_backtest_halt_does_not_persist():
    p, m = pipe(FakeBrain(lambda ids: decision_for(ids)))
    d = m.dates[350].strftime("%Y-%m-%d")
    assert p.decide(d, view(eq=97_000, prev=100_000)).halted
    assert not p.decide(m.dates[351].strftime("%Y-%m-%d"), view(eq=100_000, prev=100_000)).halted


def test_model_schedule_sonnet_first_of_week_haiku_after():
    brain = FakeBrain(lambda ids: decision_for(ids))
    p, m = pipe(brain)
    days = [d for d in m.dates if d >= m.dates[350]][:7]
    for d in days:
        p.decide(d.strftime("%Y-%m-%d"), view())
    models = [c[0] for c in brain.calls]
    weeks = [d.isocalendar().week for d in days]
    for i, (mod, wk) in enumerate(zip(models, weeks)):
        first_of_week = i == 0 or weeks[i - 1] != wk
        assert mod == ("sonnet" if first_of_week else "haiku")


def test_forced_model_and_rule_baseline():
    brain = FakeBrain(lambda ids: decision_for(ids))
    p, m = pipe(brain, forced_model="opus")
    p.decide(m.dates[350].strftime("%Y-%m-%d"), view())
    assert brain.calls[0][0] == "opus"
    cfg = PipelineConfig(mode="backtest", anonymize=True, use_claude=False, use_kronos=False)
    rp = DecisionPipeline(m, cfg, None, None)
    plan = rp.decide(m.dates[350].strftime("%Y-%m-%d"), view())
    assert plan.meta["status"] == "no_claude" and 0 < len(plan.targets) <= 10
    assert check_limits(plan.targets, equity=100_000) == []
