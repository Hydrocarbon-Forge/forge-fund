"""Strict validation of brain decisions and the JSON extractor. Pure logic, no I/O."""
from __future__ import annotations

import json

import pytest

from brain import claude_client, schema
from brain.schema import Decision, DecisionParseError, extract_json, parse_decision

TICKERS = {"AAPL", "MSFT", "NVDA", "BRK-B"}


def _decision(**overrides) -> dict:
    base = {
        "holdings": {
            "AAPL": {"weight": 0.5, "rationale": "Strong momentum and earnings revisions."},
            "MSFT": {"weight": 0.4, "rationale": "Quality compounder with stable margins."},
        },
        "cash": 0.1,
        "notes": "Calm tape, low realised volatility.",
        "risk_view": "low",
    }
    base.update(overrides)
    return base


def _text(obj: dict) -> str:
    return json.dumps(obj)


# ----------------------------------------------------------------- happy paths
def test_valid_decision_parses():
    d = parse_decision(_text(_decision()), TICKERS)
    assert isinstance(d, Decision)
    assert set(d.holdings) == {"AAPL", "MSFT"}
    assert d.holdings["AAPL"].weight == 0.5
    assert d.holdings["AAPL"].rationale.startswith("Strong momentum")
    assert d.cash == 0.1
    assert d.risk_view == "low"
    assert d.notes == "Calm tape, low realised volatility."


def test_markdown_fenced_json_parses():
    text = "Here is my decision:\n```json\n" + _text(_decision()) + "\n```\nThanks."
    assert parse_decision(text, TICKERS).holdings["MSFT"].weight == 0.4


def test_fence_without_language_tag_parses():
    text = "```\n" + _text(_decision()) + "\n```"
    assert parse_decision(text, TICKERS).cash == 0.1


def test_ticker_keys_are_upper_cased_and_stripped():
    obj = _decision(holdings={" aapl ": {"weight": 0.6, "rationale": "x"}}, cash=0.4)
    d = parse_decision(_text(obj), TICKERS)
    assert set(d.holdings) == {"AAPL"}


def test_notes_default_to_empty_string():
    obj = _decision()
    del obj["notes"]
    assert parse_decision(_text(obj), TICKERS).notes == ""


def test_empty_holdings_is_all_cash():
    d = parse_decision(_text(_decision(holdings={}, cash=1.0)), TICKERS)
    assert d.holdings == {}
    assert d.cash == 1.0


@pytest.mark.parametrize(
    ("given", "expected"),
    [("medium", "med"), ("Medium", "med"), ("MID", "med"), ("Low", "low"), ("HIGH", "high"), ("med", "med")],
)
def test_risk_view_aliases(given, expected):
    assert parse_decision(_text(_decision(risk_view=given)), TICKERS).risk_view == expected


def test_long_rationale_is_stripped_and_truncated():
    long = "  " + "x" * 500 + "  "
    obj = _decision(holdings={"AAPL": {"weight": 0.5, "rationale": long}}, cash=0.5)
    d = parse_decision(_text(obj), TICKERS)
    assert len(d.holdings["AAPL"].rationale) == 400


def test_long_notes_are_truncated():
    d = parse_decision(_text(_decision(notes="n" * 2000)), TICKERS)
    assert len(d.notes) == 1500


# ----------------------------------------------------------------- rejections
def test_unknown_ticker_rejected_and_named():
    obj = _decision(holdings={"AAPL": {"weight": 0.5, "rationale": "x"}, "ZZZZ": {"weight": 0.4, "rationale": "y"}})
    with pytest.raises(DecisionParseError, match="unknown ticker.*ZZZZ"):
        parse_decision(_text(obj), TICKERS)


def test_weights_plus_cash_off_by_more_than_tolerance_rejected():
    obj = _decision(holdings={"AAPL": {"weight": 0.5, "rationale": "x"}, "MSFT": {"weight": 0.3, "rationale": "y"}}, cash=0.1)
    with pytest.raises(DecisionParseError, match="sum to 1.0"):
        parse_decision(_text(obj), TICKERS)


@pytest.mark.parametrize(
    ("w_aapl", "w_msft", "cash", "ok"),
    [
        (0.5, 0.4, 0.1, True),    # exact
        (0.5, 0.4, 0.14, True),   # +0.04 is inside the 0.05 tolerance
        (0.5, 0.4, 0.2, False),   # +0.10 outside
        (0.5, 0.3, 0.1, False),   # -0.10 outside
    ],
)
def test_sum_tolerance_boundaries(w_aapl, w_msft, cash, ok):
    obj = _decision(
        holdings={"AAPL": {"weight": w_aapl, "rationale": "x"}, "MSFT": {"weight": w_msft, "rationale": "y"}},
        cash=cash,
    )
    if ok:
        assert parse_decision(_text(obj), TICKERS).cash == cash
    else:
        with pytest.raises(DecisionParseError):
            parse_decision(_text(obj), TICKERS)


def test_negative_weight_rejected_even_if_sum_is_one():
    obj = _decision(holdings={"AAPL": {"weight": -0.1, "rationale": "x"}, "MSFT": {"weight": 0.6, "rationale": "y"}}, cash=0.5)
    with pytest.raises(DecisionParseError, match="greater than or equal to 0"):
        parse_decision(_text(obj), TICKERS)


def test_weight_above_one_rejected():
    obj = _decision(holdings={"AAPL": {"weight": 1.5, "rationale": "x"}}, cash=-0.5)
    with pytest.raises(DecisionParseError):
        parse_decision(_text(obj), TICKERS)


def test_duplicate_ticker_key_in_json_rejected():
    # json.loads would silently keep the last value, so the raw duplicate must be refused.
    text = '{"holdings": {"AAPL": {"weight": 0.5, "rationale": "a"}, "AAPL": {"weight": 0.4, "rationale": "b"}}, "cash": 0.1, "risk_view": "low"}'
    with pytest.raises(DecisionParseError, match="duplicate"):
        parse_decision(text, TICKERS)


def test_case_colliding_tickers_rejected():
    text = _text({
        "holdings": {"AAPL": {"weight": 0.5, "rationale": "a"}, "aapl": {"weight": 0.4, "rationale": "b"}},
        "cash": 0.1,
        "risk_view": "low",
    })
    with pytest.raises(DecisionParseError, match="duplicate ticker AAPL"):
        parse_decision(text, TICKERS)


def test_more_than_thirty_holdings_rejected():
    tickers = {f"T{i:02d}" for i in range(31)}
    holdings = {t: {"weight": 0.01, "rationale": "x"} for t in tickers}
    obj = {"holdings": holdings, "cash": 0.69, "risk_view": "med"}
    with pytest.raises(DecisionParseError, match="too many holdings"):
        parse_decision(_text(obj), tickers)


def test_invalid_risk_view_rejected():
    with pytest.raises(DecisionParseError, match="risk_view"):
        parse_decision(_text(_decision(risk_view="extreme")), TICKERS)


def test_missing_risk_view_rejected():
    obj = _decision()
    del obj["risk_view"]
    with pytest.raises(DecisionParseError, match="risk_view"):
        parse_decision(_text(obj), TICKERS)


def test_empty_rationale_rejected():
    obj = _decision(holdings={"AAPL": {"weight": 0.5, "rationale": "   "}}, cash=0.5)
    with pytest.raises(DecisionParseError, match="rationale"):
        parse_decision(_text(obj), TICKERS)


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "   ",
        "I would buy AAPL and MSFT, probably.",
        "[1, 2, 3]",
        "{not json at all}",
        '{"holdings": {"AAPL": {"weight": 0.5, "rationale": "x"',
    ],
)
def test_garbage_text_rejected(garbage):
    with pytest.raises(DecisionParseError):
        parse_decision(garbage, TICKERS)


def test_error_message_is_human_readable_and_lists_all_problems():
    obj = _decision(holdings={"ZZZZ": {"weight": 0.5, "rationale": "x"}}, risk_view="extreme")
    with pytest.raises(DecisionParseError) as info:
        parse_decision(_text(obj), TICKERS)
    message = str(info.value)
    assert "ZZZZ" in message and "risk_view" in message


# ----------------------------------------------------------------- extract_json
def test_extract_json_handles_braces_and_escapes_inside_strings():
    payload = {"notes": 'use {braces} and } here "ok" \\ slash', "x": [1, {"y": 2}]}
    text = "Sure. " + json.dumps(payload) + " done."
    assert extract_json(text) == payload


def test_extract_json_takes_first_balanced_object():
    assert extract_json('{"a": 1} then {"b": 2}') == {"a": 1}


def test_extract_json_strips_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", '{"a": 1,}', '{"a": {"b": 1}', ""])
def test_extract_json_rejects_missing_or_malformed(text):
    with pytest.raises(DecisionParseError):
        extract_json(text)


def test_extract_json_rejects_duplicate_keys():
    with pytest.raises(DecisionParseError, match="duplicate key"):
        extract_json('{"a": 1, "a": 2}')


def test_claude_client_reexports_extract_json():
    assert claude_client.extract_json is schema.extract_json
