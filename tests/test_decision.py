"""brain/decision.py (Masker, build_messages, ClaudeBrain.decide), brain/schema.py and forge/types.py.

Everything runs against FakeClaude: no HTTP, no real endpoint, no real token (conftest sets a fake one).
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd
import pytest

from brain.claude_client import ClaudeError
from brain.decision import ClaudeBrain, Masker, build_messages, system_prompt
from brain.features import compute_features
from tests.synth_small import FakeClaude, make_close, make_view, make_volume

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]     # tickers that must never leak in masked mode
FAKE_TOKEN = "test-token-NOT-REAL-123456"
LABEL = "sim-day 12"
MARKET = {"r1": 0.004, "r5": 0.011, "r20": 0.021, "r60": -0.012, "vol20": 0.14, "p52": 0.71,
          "dd_from_high": -0.031}

VALID_REPLY = json.dumps({
    "holdings": {"S001": {"weight": 0.2, "rationale": "trend up"},
                 "S002": {"weight": 0.1, "rationale": "momentum"}},
    "cash": 0.7, "notes": "calm tape, would cut if breadth fails", "risk_view": "low",
})
BAD_SUM_REPLY = json.dumps({
    "holdings": {"S001": {"weight": 0.2, "rationale": "trend up"},
                 "S002": {"weight": 0.1, "rationale": "momentum"}},
    "cash": 0.5, "notes": "forgot to add up", "risk_view": "med",
})
UNKNOWN_ID_REPLY = json.dumps({
    "holdings": {"S999": {"weight": 0.2, "rationale": "invented id"}},
    "cash": 0.8, "notes": "", "risk_view": "high",
})


@pytest.fixture
def features():
    close = make_close(UNIVERSE, n_days=300, seed=21)
    volume = make_volume(close, seed=22)
    return compute_features(close, volume, close.index[-1])


@pytest.fixture
def kronos_df():
    return pd.DataFrame({
        "ticker": UNIVERSE,
        "exp_ret": [0.01, -0.005, 0.02, 0.0, -0.01],
        "p_up": [0.6, 0.4, 0.7, 0.5, 0.35],
        "dispersion": [0.02] * 5,
        "score": [1.2, -0.8, 2.0, 0.0, -1.5],
    })


@pytest.fixture
def view():
    return make_view(
        equity=103_456.78,
        positions={"AAPL": {"qty": 100, "avg_px": 150.0, "last_px": 160.0, "value": 16_000.0,
                            "weight": 0.1547, "unrealized_pct": 0.0667}},
    )


def _messages(features, kronos_df, view, masker):
    return build_messages(LABEL, features, kronos_df, view, masker, MARKET, prev_notes=None, prev_risk=None)


# ----------------------------------------------------------------------------- (a) Masker
def test_masked_mapping_is_deterministic_bijective_and_opaque():
    m1 = Masker(UNIVERSE, enabled=True)
    m2 = Masker(list(reversed(UNIVERSE)), enabled=True)      # input order must not matter
    assert m1.fwd == m2.fwd
    assert len(set(m1.fwd.values())) == len(UNIVERSE)          # injective
    assert set(m1.rev) == set(m1.fwd.values())                 # bijective with rev
    assert all(re.fullmatch(r"S\d{3}", i) for i in m1.fwd.values())
    assert sorted(m1.fwd.values()) == [f"S{i:03d}" for i in range(1, len(UNIVERSE) + 1)]
    assert not set(UNIVERSE) & set(m1.fwd.values())            # real tickers are hidden


def test_disabled_mapping_is_identity():
    m = Masker(UNIVERSE, enabled=False)
    assert m.fwd == {t: t for t in UNIVERSE}
    assert all(m.to_id(t) == t for t in UNIVERSE)


@pytest.mark.parametrize("enabled", [True, False])
def test_to_id_to_ticker_round_trip(enabled):
    m = Masker(UNIVERSE, enabled=enabled)
    for t in UNIVERSE:
        assert m.to_ticker(m.to_id(t)) == t
    assert m.ids() == set(m.rev)


# ----------------------------------------------------------------------------- (b) build_messages
def test_masked_prompt_hides_tickers_and_years(features, kronos_df, view):
    msgs = _messages(features, kronos_df, view, Masker(UNIVERSE, enabled=True))
    user = msgs[1]["content"]
    full = json.dumps(msgs)
    for t in UNIVERSE:
        assert t not in user, f"real ticker {t} leaked into the user prompt"
        assert t not in full
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", user)
    assert years == [], f"calendar years leaked into the prompt: {years}"
    assert f"AS OF: {LABEL} " in user
    assert "2024" not in user                                  # view dates are not printed


def test_live_prompt_shows_real_tickers(features, kronos_df, view):
    msgs = _messages(features, kronos_df, view, Masker(UNIVERSE, enabled=False))
    user = msgs[1]["content"]
    for t in UNIVERSE:
        assert f"\n{t}|" in user
    assert "CURRENT HOLDINGS: AAPL 15.5%" in user


def test_system_prompt_states_the_limits():
    sys_text = system_prompt()
    assert "At most 10 holdings" in sys_text
    assert "25%" in sys_text
    assert "80%" in sys_text
    assert "cash >= 20%" in sys_text


def test_json_example_uses_ids_from_the_table(features, kronos_df, view):
    masker = Masker(UNIVERSE, enabled=True)
    user = _messages(features, kronos_df, view, masker)[1]["content"]
    example_ids = set(re.findall(r'"(S\d{3})"', user))
    assert example_ids, "no example ids found in the prompt"
    assert example_ids <= masker.ids()
    table_ids = {line.split("|", 1)[0] for line in user.splitlines() if re.fullmatch(r"S\d{3}\|.*", line)}
    assert table_ids == masker.ids()


def test_prompt_never_contains_the_token(features, kronos_df, view, monkeypatch):
    monkeypatch.setenv("CLAUDE_LOCAL_TOKEN", FAKE_TOKEN)
    for masker in (Masker(UNIVERSE, enabled=True), Masker(UNIVERSE, enabled=False)):
        assert FAKE_TOKEN not in json.dumps(_messages(features, kronos_df, view, masker))


# ----------------------------------------------------------------------------- (c) ClaudeBrain.decide
def _brain(client, tmp_path, use_cache=False):
    return ClaudeBrain(client, use_cache=use_cache, cache_dir=tmp_path / "claude_cache")


def _decide_inputs(features, kronos_df, view):
    masker = Masker(UNIVERSE, enabled=True)
    return _messages(features, kronos_df, view, masker), masker.ids()


def test_valid_json_is_accepted_with_one_call(features, kronos_df, view, tmp_path):
    msgs, ids = _decide_inputs(features, kronos_df, view)
    client = FakeClaude([VALID_REPLY])
    res = _brain(client, tmp_path).decide(msgs, "haiku", ids)
    assert res.status == "ok"
    assert res.error is None
    assert res.calls == 1 and client.calls == 1
    assert res.decision is not None
    assert res.decision.holdings["S001"].weight == pytest.approx(0.2)
    assert res.decision.risk_view == "low"
    assert res.usage == {"prompt_tokens": 1200, "completion_tokens": 150}
    assert res.cost_usd == pytest.approx(0.0021)
    assert client.requests[0]["purpose"] == "decision"


def test_invalid_then_valid_repairs_and_feeds_error_back(features, kronos_df, view, tmp_path):
    msgs, ids = _decide_inputs(features, kronos_df, view)
    client = FakeClaude([BAD_SUM_REPLY, VALID_REPLY])
    res = _brain(client, tmp_path).decide(msgs, "haiku", ids)
    assert res.status == "ok"
    assert res.calls == 2 and client.calls == 2
    second = client.requests[1]
    assert second["purpose"] == "repair"
    assert second["messages"][-2] == {"role": "assistant", "content": BAD_SUM_REPLY}
    assert "must sum to 1.0" in second["messages"][-1]["content"]
    assert "rejected by the validator" in second["messages"][-1]["content"]


def test_two_invalid_replies_fall_back_to_hold(features, kronos_df, view, tmp_path):
    msgs, ids = _decide_inputs(features, kronos_df, view)
    client = FakeClaude([BAD_SUM_REPLY, BAD_SUM_REPLY])
    res = _brain(client, tmp_path).decide(msgs, "haiku", ids)
    assert res.status == "fallback_hold"
    assert res.decision is None
    assert res.calls == 2 and client.calls == 2
    assert "must sum to 1.0" in (res.error or "")


def test_claude_error_falls_back_and_never_reports_the_token(features, kronos_df, view, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_LOCAL_TOKEN", FAKE_TOKEN)
    msgs, ids = _decide_inputs(features, kronos_df, view)
    leaky = ClaudeError(f"HTTP 503 from endpoint; echoed credential {FAKE_TOKEN}")
    client = FakeClaude(exc=leaky)
    res = _brain(client, tmp_path).decide(msgs, "haiku", ids)
    assert res.status == "fallback_hold"
    assert res.decision is None
    assert res.error and "claude call failed" in res.error
    assert FAKE_TOKEN not in res.error


def test_response_cache_prevents_a_second_paid_call(features, kronos_df, view, tmp_path):
    msgs, ids = _decide_inputs(features, kronos_df, view)
    client = FakeClaude([VALID_REPLY])
    brain = _brain(client, tmp_path, use_cache=True)
    first = brain.decide(msgs, "haiku", ids)
    assert first.status == "ok" and first.cached is False and client.calls == 1
    second = brain.decide(msgs, "haiku", ids)
    assert second.status == "ok"
    assert second.cached is True
    assert second.calls == 0
    assert client.calls == 1                                    # no new request reached the client
    assert second.decision == first.decision
    assert any((tmp_path / "claude_cache").glob("*.json"))


def test_unknown_ticker_id_falls_back_after_repair(features, kronos_df, view, tmp_path):
    msgs, ids = _decide_inputs(features, kronos_df, view)
    client = FakeClaude([UNKNOWN_ID_REPLY, UNKNOWN_ID_REPLY])
    res = _brain(client, tmp_path).decide(msgs, "haiku", ids)
    assert res.status == "fallback_hold"
    assert res.decision is None
    assert client.calls == 2
    assert "unknown ticker" in (res.error or "") and "S999" in (res.error or "")


def test_single_ticker_feature_table_builds_a_prompt(kronos_df, view):
    close = make_close(["AAPL"], n_days=300, seed=5)
    feats = compute_features(close, make_volume(close, seed=5), close.index[-1])
    msgs = build_messages(LABEL, feats, None, view, Masker(["AAPL"], enabled=True), MARKET, None, None)
    assert "S001" in msgs[1]["content"]


def test_prompt_never_feeds_back_model_authored_notes_but_shows_computed_regime(features, kronos_df, view):
    """v3: the previous review's notes (model-written, rule-hardening) must NOT reach the next prompt."""
    from brain.features import universe_regime
    msgs = build_messages(LABEL, features, kronos_df, view, Masker(list(features.index), True), MARKET,
                          prev_notes="SECRET-RULE: stay in cash until breadth > 50", prev_risk="high",
                          regime=universe_regime(features))
    text = msgs[1]["content"]
    assert "SECRET-RULE" not in text and "PREVIOUS REVIEW" not in text
    assert "UNIVERSE REGIME (computed by code" in text and "above 50d avg" in text
    assert "MANDATE" in msgs[0]["content"] and "long-only equity fund" in msgs[0]["content"]
