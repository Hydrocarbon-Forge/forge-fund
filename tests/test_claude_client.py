"""brain.claude_client against httpx.MockTransport with a virtual clock.

No network and no real sleeping: ``sleep`` advances a fake clock and records the request.
The fake token from conftest is the only secret in play.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import httpx
import pytest

from brain.claude_client import ChatResult, ClaudeClient, ClaudeError, extract_json
from forge.config import ClaudeSettings, estimate_cost_usd

TOKEN = "test-token-NOT-REAL-123456"
PROMPT_MARKER = "PROMPT-MARKER-7f3a"
MESSAGES = [
    {"role": "system", "content": "You are a risk manager."},
    {"role": "user", "content": PROMPT_MARKER},
]
LOG_KEYS = {
    "kind", "ts", "purpose", "model", "prompt_hash", "prompt_chars", "status",
    "attempts", "latency_s", "usage", "cost_usd", "error",
}


class FakeTime:
    """Virtual clock. sleep() advances it instantly and records the requested duration."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


class Script:
    """Transport handler that plays the given responses in order (the last one repeats)."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses[min(len(self.requests), len(self.responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def _settings(**overrides) -> ClaudeSettings:
    values = dict(
        base_url="http://claude.test/v1",
        token=TOKEN,
        default_model="sonnet",
        routine_model="haiku",
        max_requests_per_min=9,
        timeout_s=5.0,
        max_retries=3,
    )
    values.update(overrides)
    return ClaudeSettings(**values)


def _ok_body(text: str = '{"ok": true}', usage: bool = True) -> dict:
    body: dict = {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 1000, "completion_tokens": 200}
    return body


def _client(handler, tmp_path, timer: FakeTime, **overrides) -> ClaudeClient:
    return ClaudeClient(
        settings=_settings(**overrides),
        transport=httpx.MockTransport(handler),
        log_path=tmp_path / "logs" / "decision_log.jsonl",
        sleep=timer.sleep,
        clock=timer.clock,
    )


def _log_path(tmp_path):
    return tmp_path / "logs" / "decision_log.jsonl"


def _log_records(tmp_path) -> list[dict]:
    return [json.loads(line) for line in _log_path(tmp_path).read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------- success path
def test_success_returns_all_result_fields(tmp_path):
    timer = FakeTime()
    script = Script(httpx.Response(200, json=_ok_body('{"ok": true}')))
    with _client(script, tmp_path, timer) as client:
        result = client.chat(MESSAGES, purpose="decision")

    assert isinstance(result, ChatResult)
    assert result.text == '{"ok": true}'
    assert result.model == "sonnet"
    assert result.usage == {"prompt_tokens": 1000, "completion_tokens": 200}
    assert result.attempts == 1
    assert result.request_id == "chatcmpl-test-1"
    assert result.cost_usd == estimate_cost_usd("sonnet", 1000, 200)
    assert re.fullmatch(r"[0-9a-f]{12}", result.prompt_hash)
    assert result.latency_s >= 0
    assert timer.sleeps == []


def test_request_shape_and_path(tmp_path):
    script = Script(httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, FakeTime()) as client:
        client.chat(MESSAGES)

    request = script.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "sonnet",
        "messages": MESSAGES,
        "max_tokens": 4096,
        "temperature": 0.2,
    }


def test_model_and_parameter_overrides(tmp_path):
    script = Script(httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, FakeTime()) as client:
        result = client.chat(MESSAGES, model="haiku", max_tokens=321, temperature=0.0, purpose="repair")

    body = json.loads(script.requests[0].content)
    assert (body["model"], body["max_tokens"], body["temperature"]) == ("haiku", 321, 0.0)
    assert result.model == "haiku"
    assert result.cost_usd == estimate_cost_usd("haiku", 1000, 200)


def test_prompt_hash_is_deterministic_and_model_sensitive(tmp_path):
    script = Script(httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, FakeTime()) as client:
        first = client.chat(MESSAGES, model="sonnet")
        again = client.chat(MESSAGES, model="sonnet")
        other = client.chat(MESSAGES, model="haiku")
    assert first.prompt_hash == again.prompt_hash
    assert other.prompt_hash != first.prompt_hash


def test_list_content_parts_are_joined(tmp_path):
    parts = [{"type": "text", "text": "Hello "}, {"type": "image", "url": "x"}, {"type": "text", "text": "world"}]
    script = Script(httpx.Response(200, json={"choices": [{"message": {"content": parts}}]}))
    with _client(script, tmp_path, FakeTime()) as client:
        assert client.chat(MESSAGES).text == "Hello world"


def test_missing_usage_is_estimated_from_chars(tmp_path):
    text = "x" * 400
    script = Script(httpx.Response(200, json=_ok_body(text=text, usage=False)))
    with _client(script, tmp_path, FakeTime()) as client:
        result = client.chat(MESSAGES)
    assert result.usage["completion_tokens"] == 100
    assert result.usage["prompt_tokens"] == len(json.dumps(MESSAGES, ensure_ascii=False)) // 4


def test_non_json_success_body_raises(tmp_path):
    script = Script(httpx.Response(200, text="<html>gateway page</html>"))
    with _client(script, tmp_path, FakeTime()) as client:
        with pytest.raises(ClaudeError):
            client.chat(MESSAGES)
    assert len(script.requests) == 1


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_content_raises_without_retry(tmp_path, content):
    body = {"choices": [{"message": {"content": content}}]}
    script = Script(httpx.Response(200, json=body))
    with _client(script, tmp_path, FakeTime()) as client:
        with pytest.raises(ClaudeError):
            client.chat(MESSAGES)
    assert len(script.requests) == 1
    assert _log_records(tmp_path)[0]["status"] == "error"


def test_context_manager_closes_http_client(tmp_path):
    client = _client(Script(httpx.Response(200, json=_ok_body())), tmp_path, FakeTime())
    with client:
        pass
    assert client._http.is_closed


def test_extract_json_is_reexported():
    assert extract_json('pre {"a": [1, {"b": "}"}]} post') == {"a": [1, {"b": "}"}]}


# ------------------------------------------------------------------- auth/secret
def test_auth_header_is_sent(tmp_path):
    script = Script(httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, FakeTime()) as client:
        client.chat(MESSAGES)
    assert script.requests[0].headers["authorization"] == f"Bearer {TOKEN}"


def test_token_never_appears_in_exception_log_or_repr(tmp_path):
    echoed = httpx.Response(400, text=f"invalid credentials: {TOKEN}")
    timer = FakeTime()
    with _client(Script(echoed), tmp_path, timer) as client:
        with pytest.raises(ClaudeError) as info:
            client.chat(MESSAGES)
        assert TOKEN not in repr(client)

    message = str(info.value)
    assert TOKEN not in message
    assert "***REDACTED***" in message
    assert TOKEN not in repr(info.value)
    assert TOKEN not in _log_path(tmp_path).read_text(encoding="utf-8")


def test_token_never_leaks_from_transport_errors(tmp_path):
    handler = Script(httpx.ConnectError(f"refused while using {TOKEN}"))
    with _client(handler, tmp_path, FakeTime(), max_retries=0) as client:
        with pytest.raises(ClaudeError) as info:
            client.chat(MESSAGES)
    assert TOKEN not in str(info.value)
    assert TOKEN not in _log_path(tmp_path).read_text(encoding="utf-8")


def test_refuses_to_send_without_token(tmp_path):
    script = Script(httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, FakeTime(), token="") as client:
        with pytest.raises(ClaudeError, match="token"):
            client.chat(MESSAGES)
    assert script.requests == []
    assert _log_records(tmp_path)[0]["status"] == "error"


# ------------------------------------------------------------------ retry policy
def test_429_honours_retry_after_then_succeeds(tmp_path):
    timer = FakeTime()
    script = Script(
        httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow down"}),
        httpx.Response(200, json=_ok_body()),
    )
    with _client(script, tmp_path, timer) as client:
        result = client.chat(MESSAGES)
    assert timer.sleeps == [7.0]
    assert result.attempts == 2
    assert len(script.requests) == 2


def test_retry_after_is_capped_at_120_seconds(tmp_path):
    timer = FakeTime()
    script = Script(httpx.Response(429, headers={"Retry-After": "600"}), httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, timer) as client:
        client.chat(MESSAGES)
    assert timer.sleeps == [120.0]


def test_retry_after_http_date_in_past_means_no_wait(tmp_path):
    timer = FakeTime()
    script = Script(
        httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        httpx.Response(200, json=_ok_body()),
    )
    with _client(script, tmp_path, timer) as client:
        client.chat(MESSAGES)
    assert timer.sleeps == [0.0]


def test_429_without_header_uses_exponential_backoff(tmp_path):
    timer = FakeTime()
    script = Script(httpx.Response(429), httpx.Response(429), httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, timer) as client:
        result = client.chat(MESSAGES)
    assert result.attempts == 3
    first, second = timer.sleeps
    assert 5.0 <= first <= 5.0 * 1.2 + 1e-9
    assert 10.0 <= second <= 10.0 * 1.2 + 1e-9


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_is_retried(tmp_path, status):
    timer = FakeTime()
    script = Script(httpx.Response(status, text="busy"), httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, timer) as client:
        result = client.chat(MESSAGES)
    assert result.attempts == 2
    assert len(timer.sleeps) == 1
    assert 5.0 <= timer.sleeps[0] <= 6.0


def test_transport_error_is_retried(tmp_path):
    timer = FakeTime()
    script = Script(httpx.ConnectError("refused"), httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, timer) as client:
        result = client.chat(MESSAGES)
    assert result.attempts == 2
    assert len(timer.sleeps) == 1


def test_timeout_is_retried(tmp_path):
    timer = FakeTime()
    script = Script(httpx.ReadTimeout("slow"), httpx.Response(200, json=_ok_body()))
    with _client(script, tmp_path, timer) as client:
        assert client.chat(MESSAGES).attempts == 2


def test_retries_exhausted_raises_with_status(tmp_path):
    timer = FakeTime()
    script = Script(httpx.Response(503, text="overloaded"))
    with _client(script, tmp_path, timer, max_retries=2) as client:
        with pytest.raises(ClaudeError, match="503"):
            client.chat(MESSAGES)
    assert len(script.requests) == 3  # first attempt + 2 retries
    assert len(timer.sleeps) == 2
    record = _log_records(tmp_path)[0]
    assert record["status"] == "error"
    assert record["attempts"] == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_other_4xx_is_not_retried_and_includes_body_snippet(tmp_path, status):
    timer = FakeTime()
    script = Script(httpx.Response(status, text="bad request: messages too long"))
    with _client(script, tmp_path, timer) as client:
        with pytest.raises(ClaudeError) as info:
            client.chat(MESSAGES)
    assert len(script.requests) == 1
    assert timer.sleeps == []
    assert str(status) in str(info.value)
    assert "messages too long" in str(info.value)


def test_unexpected_exception_is_logged_and_wrapped(tmp_path):
    script = Script(RuntimeError("kaboom"))
    with _client(script, tmp_path, FakeTime()) as client:
        with pytest.raises(ClaudeError, match="kaboom"):
            client.chat(MESSAGES)
    assert _log_records(tmp_path)[0]["status"] == "error"


# ------------------------------------------------------------------ rate limiter
def test_rate_limiter_makes_tenth_call_wait_for_the_window(tmp_path):
    timer = FakeTime()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body())

    with _client(handler, tmp_path, timer, max_requests_per_min=9) as client:
        for _ in range(9):
            client.chat(MESSAGES)
        assert timer.sleeps == []
        client.chat(MESSAGES)  # 10th inside the same 60s window
    assert len(timer.sleeps) == 1
    assert timer.sleeps[0] == pytest.approx(60.0)


def test_rate_limiter_counts_retries_as_requests(tmp_path):
    timer = FakeTime()
    # 8 successes, then one 429 (retry) and a success: the 10th request (the retry) must wait.
    responses = [httpx.Response(200, json=_ok_body()) for _ in range(8)]
    responses += [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json=_ok_body())]
    script = Script(*responses)
    with _client(script, tmp_path, timer, max_requests_per_min=9) as client:
        for _ in range(9):
            client.chat(MESSAGES)
    assert len(script.requests) == 10
    # Retry-After wait (1s) then the limiter forces the 10th send to wait out the window.
    assert timer.sleeps[0] == 1.0
    assert timer.sleeps[1] == pytest.approx(59.0)


# ------------------------------------------------------------------- call log
def test_log_lines_are_json_with_expected_keys_and_no_message_text(tmp_path):
    timer = FakeTime()
    script = Script(
        httpx.Response(200, json=_ok_body("RESPONSE-MARKER-9c1e")),
        httpx.Response(400, text="nope"),
    )
    with _client(script, tmp_path, timer) as client:
        client.chat(MESSAGES, purpose="decision")
        with pytest.raises(ClaudeError):
            client.chat(MESSAGES, purpose="review")

    raw = _log_path(tmp_path).read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    for record in records:
        assert set(record) == LOG_KEYS
        assert record["kind"] == "call"
        assert datetime.fromisoformat(record["ts"]).utcoffset().total_seconds() == 0
        assert set(record["usage"]) == {"prompt_tokens", "completion_tokens"}

    ok, failed = records
    assert ok["status"] == "ok"
    assert ok["purpose"] == "decision"
    assert ok["attempts"] == 1
    assert ok["usage"] == {"prompt_tokens": 1000, "completion_tokens": 200}
    assert ok["error"] is None
    assert ok["prompt_chars"] > 0
    assert re.fullmatch(r"[0-9a-f]{12}", ok["prompt_hash"])

    assert failed["status"] == "error"
    assert failed["purpose"] == "review"
    assert "400" in failed["error"]

    for forbidden in (PROMPT_MARKER, "RESPONSE-MARKER", "Authorization", "Bearer", TOKEN):
        assert forbidden not in raw


def test_log_directory_is_created_when_missing(tmp_path):
    log_path = tmp_path / "nested" / "logs" / "decision_log.jsonl"
    client = ClaudeClient(
        settings=_settings(),
        transport=httpx.MockTransport(Script(httpx.Response(200, json=_ok_body()))),
        log_path=log_path,
        sleep=FakeTime().sleep,
        clock=FakeTime().clock,
    )
    client.chat(MESSAGES)
    client.close()
    assert log_path.exists()
