"""The local endpoint returns usage-limit notices as HTTP 200 text. They must never be treated as answers."""
import json
from datetime import datetime, timezone

import httpx
import pytest

from brain.claude_client import ClaudeClient, ClaudeError, ClaudeLimitError, detect_limit_message
from brain.decision import ClaudeBrain
from forge.config import claude_settings
from dataclasses import replace

LIMIT_TEXT = "You've hit your session limit · resets 4:50am (UTC)"
VALID = json.dumps({"holdings": {"S001": {"weight": 0.2, "rationale": "r"}}, "cash": 0.8, "notes": "n", "risk_view": "low"})
NOW = datetime(2026, 10, 13, 3, 30, tzinfo=timezone.utc)


def reply(text, tokens=12):
    return httpx.Response(200, json={"id": "x", "choices": [{"message": {"role": "assistant", "content": text}}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": tokens}})


@pytest.mark.parametrize("text,is_limit", [
    (LIMIT_TEXT, True),
    ("You've reached your usage limit. Try again later.", True),
    ("You have exceeded the rate limit", True),
    ('{"holdings": {}, "cash": 1.0, "notes": "we hit the gross limit", "risk_view": "low"}', False),   # JSON never matches
    ("```json\n{\"notes\": \"hit your limit\"}\n```", False),
    ("Sure, here is the portfolio.", False),
    ("", False),
])
def test_detection(text, is_limit):
    assert detect_limit_message(text, NOW)[0] is is_limit


def test_reset_time_parsing_rolls_over_midnight():
    assert detect_limit_message(LIMIT_TEXT, NOW)[1] == datetime(2026, 10, 13, 4, 50, tzinfo=timezone.utc)
    late = datetime(2026, 10, 13, 23, 0, tzinfo=timezone.utc)
    assert detect_limit_message(LIMIT_TEXT, late)[1] == datetime(2026, 10, 14, 4, 50, tzinfo=timezone.utc)
    assert detect_limit_message("You've hit your session limit · resets 2:15pm (UTC)", NOW)[1].hour == 14
    assert detect_limit_message("You've hit your limit · resets 4:50am (PDT)", NOW)[1] is None     # untrusted tz


def client(handler, cap=0.0, sleeps=None):
    s = replace(claude_settings(), limit_wait_cap_s=cap)
    return ClaudeClient(settings=s, transport=httpx.MockTransport(handler),
                        sleep=(sleeps.append if sleeps is not None else (lambda x: None)))


def test_limit_notice_raises_typed_error_not_a_reply():
    c = client(lambda r: reply(LIMIT_TEXT))
    with pytest.raises(ClaudeLimitError) as e:
        c.chat([{"role": "user", "content": "hi"}], model="haiku")
    assert "usage limit" in str(e.value)
    log = [json.loads(l) for l in c._log_path.read_text().splitlines()]
    assert log[-1]["status"] == "error" and log[-1]["error"]


def test_waits_for_reset_then_succeeds():
    calls, sleeps = [], []

    def handler(r):
        calls.append(1)
        return reply(LIMIT_TEXT) if len(calls) == 1 else reply(VALID, 90)

    c = client(handler, cap=24 * 3600, sleeps=sleeps)
    out = c.chat([{"role": "user", "content": "hi"}], model="haiku")
    assert out.text == VALID and len(calls) == 2
    assert len(sleeps) == 1 and sleeps[0] >= 60                     # slept until the reset (+ buffer)


def test_wait_longer_than_cap_raises():
    c = client(lambda r: reply("You've hit your limit · resets 11:59pm (UTC)"), cap=5.0)
    with pytest.raises(ClaudeLimitError):
        c.chat([{"role": "user", "content": "hi"}], model="haiku")


def test_brain_reports_claude_limit_and_never_caches_failures(tmp_path):
    msgs = [{"role": "user", "content": "decide"}]
    brain = ClaudeBrain(client(lambda r: reply(LIMIT_TEXT)), use_cache=True, cache_dir=tmp_path / "c")
    res = brain.decide(msgs, "haiku", {"S001"})
    assert res.status == "claude_limit" and res.decision is None and "usage limit" in res.error
    assert not list((tmp_path / "c").glob("*.json"))                # nothing cached

    bad = ClaudeBrain(client(lambda r: reply("not json at all, just chatter")), use_cache=True, cache_dir=tmp_path / "c2")
    assert bad.decide(msgs, "haiku", {"S001"}).status == "fallback_hold"
    assert not list((tmp_path / "c2").glob("*.json"))               # parse failures are not cached either


def test_brain_cache_hit_after_valid_answer_and_poisoned_entry_ignored(tmp_path):
    msgs = [{"role": "user", "content": "decide"}]
    n = []
    c = client(lambda r: (n.append(1), reply(VALID, 90))[1])
    brain = ClaudeBrain(c, use_cache=True, cache_dir=tmp_path / "c")
    assert brain.decide(msgs, "haiku", {"S001"}).status == "ok"
    again = brain.decide(msgs, "haiku", {"S001"})
    assert again.status == "ok" and again.cached and len(n) == 1
    # poison the entry with a limit notice: it must be ignored and re-requested
    key = brain._key("haiku", msgs)
    (tmp_path / "c" / f"{key}.json").write_text(json.dumps({"text": LIMIT_TEXT, "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "cost_usd": 0}))
    third = brain.decide(msgs, "haiku", {"S001"})
    assert third.status == "ok" and len(n) == 2
