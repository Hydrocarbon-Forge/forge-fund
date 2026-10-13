"""Brain output schema: strict parsing and validation of the Claude decision JSON.

Nothing here guesses. Any deviation from the contract raises ``DecisionParseError``,
whose message is human-readable so it can be fed back to the model in a repair prompt.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

MAX_HOLDINGS = 30
MAX_RATIONALE_CHARS = 400
MAX_NOTES_CHARS = 1500
WEIGHT_SUM_TOLERANCE = 0.05

RiskView = Literal["low", "med", "high"]
_RISK_ALIASES = {"medium": "med", "mid": "med"}


class DecisionParseError(Exception):
    """Model output could not be parsed or failed validation. ``str(exc)`` is human-readable."""


# --------------------------------------------------------------------- JSON extraction
_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """object_pairs_hook: duplicate JSON keys are ambiguous, so refuse them."""
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise DecisionParseError(f"duplicate key {key!r} in JSON object")
        obj[key] = value
    return obj


def _balanced_end(s: str, start: int) -> int:
    """Index of the '}' closing the object opened at ``s[start]``, or -1. String/escape aware."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_json(text: str) -> dict[str, Any]:
    """Return the first balanced top-level JSON object in ``text`` (``` fences tolerated).

    Raises DecisionParseError if there is no complete object or it is not valid JSON.
    """
    if not isinstance(text, str) or not text.strip():
        raise DecisionParseError("model returned no text; expected a JSON object")
    fenced = [m.group(1) for m in _FENCE_RE.finditer(text) if "{" in m.group(1)]
    for body in [*fenced, text]:
        start = body.find("{")
        if start == -1:
            continue
        end = _balanced_end(body, start)
        if end == -1:
            continue
        raw = body[start : end + 1]
        try:
            return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise DecisionParseError(f"JSON object is malformed: {exc}") from None
    raise DecisionParseError("no complete JSON object found in model output (expected {...})")


# ----------------------------------------------------------------------- models
class Holding(BaseModel):
    weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _clean_rationale(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rationale must be non-empty")
        return value[:MAX_RATIONALE_CHARS]


class Decision(BaseModel):
    holdings: dict[str, Holding] = Field(default_factory=dict)
    cash: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    notes: str = ""
    risk_view: RiskView

    @field_validator("holdings", mode="before")
    @classmethod
    def _normalise_tickers(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value  # pydantic reports the type error
        out: dict[str, Any] = {}
        for raw_key, holding in value.items():
            ticker = str(raw_key).strip().upper()
            if not ticker:
                raise ValueError("a holding has an empty ticker key")
            if ticker in out:
                raise ValueError(f"duplicate ticker {ticker} (ticker keys are case-insensitive)")
            out[ticker] = holding
        if len(out) > MAX_HOLDINGS:
            raise ValueError(f"too many holdings: {len(out)} (max {MAX_HOLDINGS})")
        return out

    @field_validator("notes", mode="before")
    @classmethod
    def _clean_notes(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("notes must be a string")
        return value.strip()[:MAX_NOTES_CHARS]

    @field_validator("risk_view", mode="before")
    @classmethod
    def _risk_alias(cls, value: Any) -> Any:
        if isinstance(value, str):
            key = value.strip().lower()
            return _RISK_ALIASES.get(key, key)
        return value

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Decision":
        total = sum(h.weight for h in self.holdings.values()) + self.cash
        if round(abs(total - 1.0), 9) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"holding weights + cash must sum to 1.0 (tolerance {WEIGHT_SUM_TOLERANCE}); got {total:.3f}"
            )
        return self


# ---------------------------------------------------------------------- parsing
def _describe(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "decision"
        msg = err["msg"]
        prefix = "Value error, "
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
        problems.append(f"{loc}: {msg}")
    return problems


def parse_decision(text: str, valid_tickers: set[str]) -> Decision:
    """Parse and strictly validate a brain decision. Raises DecisionParseError on any problem."""
    obj = extract_json(text)
    problems: list[str] = []

    raw_holdings = obj.get("holdings")
    if isinstance(raw_holdings, dict):
        allowed = {str(t).strip().upper() for t in valid_tickers}
        unknown = sorted({str(k).strip().upper() for k in raw_holdings} - allowed)
        if unknown:
            problems.append("unknown ticker(s) not in the allowed universe: " + ", ".join(unknown))

    decision: Decision | None = None
    try:
        decision = Decision.model_validate(obj)
    except ValidationError as exc:
        problems.extend(_describe(exc))

    if problems or decision is None:
        raise DecisionParseError("; ".join(problems) or "decision failed validation")
    return decision
