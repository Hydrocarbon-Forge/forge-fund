"""OpenAI-compatible ``/chat/completions`` client for the local Claude endpoint.

Built on httpx. Provides a client-side sliding-window rate limiter, a retry policy
(429 / 5xx / transport errors), and one append-only JSON line per ``chat()`` call.

Secret handling: the bearer token only ever goes into the Authorization header.
Every string that reaches an exception message or the log passes through ``_scrub``.
Message contents and headers are never logged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import httpx

from brain.schema import extract_json
from forge.config import ClaudeSettings, claude_settings, estimate_cost_usd, get_paths, redact

__all__ = ["ChatResult", "ClaudeClient", "ClaudeError", "ClaudeLimitError", "detect_limit_message", "extract_json"]

log = logging.getLogger(__name__)

CHAT_PATH = "/chat/completions"
RATE_WINDOW_S = 60.0
RETRY_AFTER_CAP_S = 120.0
BACKOFF_BASE_S = 5.0
JITTER_FRAC = 0.2
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
BODY_SNIPPET_CHARS = 300
_REDACTED = "***REDACTED***"


class ClaudeError(Exception):
    """Raised when a chat call fails for good. The message is redacted and never holds the token."""


class ClaudeLimitError(ClaudeError):
    """The endpoint answered HTTP 200 with a *usage-limit notice* instead of a model reply.

    The local endpoint fronts a subscription, so when the session limit is hit it returns text such as
    "You've hit your session limit · resets 4:50am (UTC)" as ordinary content. That must never be parsed
    as a model answer (and never cached): it is surfaced as this typed error with the reset time.
    """

    def __init__(self, message: str, reset_at: datetime | None = None):
        super().__init__(message)
        self.reset_at = reset_at


_LIMIT_RE = re.compile(r"(hit|reached|exceeded)\s+(your|the)\b[^.\n]{0,40}\blimit", re.I)
_RESET_RE = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*\(?\s*(utc|gmt)?", re.I)


def detect_limit_message(text: str, now: datetime | None = None) -> tuple[bool, datetime | None]:
    """(is_limit_notice, reset_at_utc). Only short texts that look like a limit notice qualify."""
    t = (text or "").strip()
    if not t or len(t) > 400 or t.startswith(("{", "[", "```")) or not _LIMIT_RE.search(t):
        return False, None
    now = now or datetime.now(timezone.utc)
    m = _RESET_RE.search(t)
    if not m or not m.group(4):                      # only trust an explicit UTC/GMT reset time
        return True, None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return True, None
    reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return True, reset


@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict[str, int]
    latency_s: float
    cost_usd: float
    prompt_hash: str
    attempts: int
    request_id: str | None


# ------------------------------------------------------------------- helpers
def _serialise(messages: Any) -> str:
    return json.dumps(messages, ensure_ascii=False, default=str)


def _prompt_hash(messages: Any, model: str) -> str:
    blob = json.dumps({"model": model, "messages": messages}, ensure_ascii=False, default=str, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait given a Retry-After header (integer seconds or HTTP-date), else None."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _extract_text(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ClaudeError("response has no choices[0].message.content") from None
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise ClaudeError("model returned empty content")
    return content


def _snippet(resp: httpx.Response) -> str:
    try:
        return resp.text[:BODY_SNIPPET_CHARS]
    except Exception:  # undecodable body: report nothing rather than fail the error path
        return ""


# -------------------------------------------------------------------- client
class ClaudeClient:
    def __init__(
        self,
        settings: ClaudeSettings | None = None,
        transport: httpx.BaseTransport | None = None,
        log_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings if settings is not None else claude_settings()
        self._token = self.settings.token
        if log_path is None:
            logs_dir = get_paths().logs
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "decision_log.jsonl"
        self._log_path = Path(log_path)
        self._sleep = sleep
        self._clock = clock
        self._rng = random.Random()
        self._sent_at: deque[float] = deque()
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        self._http = httpx.Client(
            base_url=self.settings.base_url,
            transport=transport,
            timeout=self.settings.timeout_s,
            headers=headers,
        )

    # lifecycle ----------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ClaudeClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ClaudeClient(base_url={self.settings.base_url!r}, token={'set' if self._token else 'MISSING'})"

    # public API ---------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        purpose: str = "decision",
    ) -> ChatResult:
        model = model or self.settings.default_model
        prompt_hash = _prompt_hash(messages, model)
        prompt_chars = len(_serialise(messages))
        started = self._clock()
        attempts = 0
        retries = 0
        limit_waits = 0
        try:
            if not self._token:
                raise ClaudeError("Claude token is not configured (CLAUDE_LOCAL_TOKEN); refusing to send request")
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            while True:
                attempts += 1
                self._wait_for_slot()
                try:
                    resp = self._http.post(CHAT_PATH, json=payload)
                except httpx.TransportError as exc:
                    failure = f"network error ({type(exc).__name__})"
                    wait = self._backoff(retries)
                else:
                    if resp.is_success:
                        result = self._build_result(resp, messages, model, prompt_hash, attempts, started)
                        is_limit, reset_at = detect_limit_message(result.text)
                        if is_limit:
                            wait_s = max((reset_at - datetime.now(timezone.utc)).total_seconds() + 90, 60.0) if reset_at else 1800.0
                            if limit_waits < 3 and wait_s <= self.settings.limit_wait_cap_s:
                                limit_waits += 1
                                log.warning("Claude usage limit reported by endpoint; waiting %.0f min for reset (%s)",
                                            wait_s / 60, reset_at.isoformat(timespec="minutes") if reset_at else "unknown")
                                self._sleep(wait_s)
                                continue
                            raise ClaudeLimitError(
                                "endpoint reported a usage limit" + (f"; resets {reset_at.isoformat(timespec='minutes')}" if reset_at else ""),
                                reset_at)
                        break
                    status = resp.status_code
                    if status == 429:
                        failure = "HTTP 429 (rate limited)"
                        retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                        if retry_after is not None:
                            wait = min(retry_after, RETRY_AFTER_CAP_S)
                        else:
                            wait = self._backoff(retries)
                    elif status in RETRYABLE_STATUS:
                        failure = f"HTTP {status}"
                        wait = self._backoff(retries)
                    else:
                        raise ClaudeError(
                            self._scrub(f"HTTP {status} from endpoint (not retried): {_snippet(resp)}")
                        )
                if retries >= self.settings.max_retries:
                    raise ClaudeError(f"giving up after {attempts} attempt(s): {failure}")
                retries += 1
                self._sleep(wait)
        except ClaudeError as exc:
            self._log_call(
                purpose=purpose, model=model, prompt_hash=prompt_hash, prompt_chars=prompt_chars,
                status="error", attempts=attempts, latency_s=round(self._clock() - started, 3),
                usage={"prompt_tokens": 0, "completion_tokens": 0}, cost_usd=0.0,
                error=self._scrub(exc),
            )
            raise
        except Exception as exc:  # unexpected: still record the call, then surface as ClaudeError
            message = self._scrub(f"unexpected {type(exc).__name__}: {exc}")
            self._log_call(
                purpose=purpose, model=model, prompt_hash=prompt_hash, prompt_chars=prompt_chars,
                status="error", attempts=attempts, latency_s=round(self._clock() - started, 3),
                usage={"prompt_tokens": 0, "completion_tokens": 0}, cost_usd=0.0, error=message,
            )
            raise ClaudeError(message) from None

        self._log_call(
            purpose=purpose, model=model, prompt_hash=prompt_hash, prompt_chars=prompt_chars,
            status="ok", attempts=attempts, latency_s=result.latency_s, usage=result.usage,
            cost_usd=result.cost_usd, error=None,
        )
        return result

    # internals ----------------------------------------------------------
    def _scrub(self, text: object) -> str:
        out = redact(text)
        if self._token:
            out = out.replace(self._token, _REDACTED)
        return out

    def _wait_for_slot(self) -> None:
        """Block (via the injected sleep) until fewer than N requests were sent in the last 60s."""
        limit = max(1, self.settings.max_requests_per_min)
        while True:
            now = self._clock()
            while self._sent_at and now - self._sent_at[0] >= RATE_WINDOW_S:
                self._sent_at.popleft()
            if len(self._sent_at) < limit:
                break
            self._sleep(max(0.0, RATE_WINDOW_S - (now - self._sent_at[0])))
        self._sent_at.append(self._clock())

    def _backoff(self, retries: int) -> float:
        """Exponential backoff 5s, 10s, 20s, ... with up to 20% upward jitter."""
        base = BACKOFF_BASE_S * (2 ** retries)
        return base * (1.0 + self._rng.uniform(0.0, JITTER_FRAC))

    def _build_result(
        self,
        resp: httpx.Response,
        messages: list[dict[str, Any]],
        model: str,
        prompt_hash: str,
        attempts: int,
        started: float,
    ) -> ChatResult:
        try:
            data = resp.json()
        except ValueError:
            raise ClaudeError("endpoint returned a non-JSON response body") from None
        if not isinstance(data, dict):
            raise ClaudeError("endpoint returned an unexpected JSON shape")
        text = _extract_text(data)

        usage_raw = data.get("usage")
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        prompt_tokens = usage_raw.get("prompt_tokens")
        completion_tokens = usage_raw.get("completion_tokens")
        if not _is_int(prompt_tokens):
            prompt_tokens = len(_serialise(messages)) // 4
        if not _is_int(completion_tokens):
            completion_tokens = len(text) // 4

        request_id = resp.headers.get("x-request-id")
        if not request_id and isinstance(data.get("id"), str):
            request_id = data["id"]

        return ChatResult(
            text=text,
            model=model,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            latency_s=round(self._clock() - started, 3),
            cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
            prompt_hash=prompt_hash,
            attempts=attempts,
            request_id=request_id,
        )

    def _log_call(
        self,
        *,
        purpose: str,
        model: str,
        prompt_hash: str,
        prompt_chars: int,
        status: str,
        attempts: int,
        latency_s: float,
        usage: dict[str, int],
        cost_usd: float,
        error: str | None,
    ) -> None:
        record = {
            "kind": "call",
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": purpose,
            "model": model,
            "prompt_hash": prompt_hash,
            "prompt_chars": prompt_chars,
            "status": status,
            "attempts": attempts,
            "latency_s": latency_s,
            "usage": usage,
            "cost_usd": cost_usd,
            "error": error,
        }
        line = json.dumps(record, ensure_ascii=False)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            # A logging failure must not discard a successful model response.
            log.warning("could not append to decision log: %s", self._scrub(exc))
