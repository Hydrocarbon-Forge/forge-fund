# Task: Claude→Gemini decider fallback (forge-fund)

## Goal
The fund's decider currently talks to the **local Claude** endpoint (192.168.1.201:8910,
OpenAI-compatible) and, when that fails, falls back to a safe **HOLD**. Add a **Gemini fallback
brain**: keep Claude as the **primary** ("ranked in"), and **when Claude does not work** fall
back to **Gemini** (via the local `agy` CLI, model `gemini-3.8-flash-high`) instead of going
straight to HOLD. If Claude works, use Claude and never touch Gemini. If both fail, HOLD.

This is the reason the book has been sitting all-cash: the Claude CLI subscription is on a
**weekly usage limit** (resets Oct 16, 4pm UTC), so today's decider returns the limit notice
and the fund HOLDs. With the Gemini fallback, the fund can make real decisions **now**.

## The single hook (verify, note if different)
`brain/decision.py::ClaudeBrain.decide(messages, model, valid_ids) -> BrainResult` is the ONLY
decider path. Today:
- `except ClaudeLimitError` (line ~226) → `status="claude_limit"` → HOLD.
- `except ClaudeError` (line ~230) → `status` stays `fallback_hold` → HOLD.
- after 2 parse failures → `fallback_hold` → HOLD.

Insert the fallback at these failure points: **before** returning a HOLD, if the Claude attempt
failed (limit error, any `ClaudeError`, or 2x parse failure) AND a Gemini fallback is
configured, try Gemini with the SAME `messages` and the SAME `valid_ids`, parse with the SAME
`brain.schema.parse_decision`, and return `BrainResult(status="ok", ...)` if it yields a valid
decision. Only if Gemini ALSO fails do we return HOLD.

## Mechanism for Gemini (already verified working)
One-shot CLI, no interactive TUI:
```
agy --model gemini-3.8-flash-high --dangerously-skip-permissions -p '<prompt text>'
```
returns the model's text on stdout (verified: `OK`). `agy` is at `/home/bsmt-pc/.local/bin/agy`.
It also supports `--json-schema` for structured output, but reusing the existing text prompt +
`parse_decision` is the least-surprising path. Construct the prompt text by concatenating the
`messages` (system + user) — the prompt already carries the full instruction set and the exact
JSON shape. Shell out with `subprocess` (timeout, capture stdout, scrub/redact any error). Parse
the returned text with `brain.schema.parse_decision(text, valid_ids)`. If the model wraps the
JSON in prose, `parse_decision`/`brain.schema.extract_json` already handles extraction.

Add `brain/gemini_client.py` (or a function `decide_with_gemini(messages, valid_ids, model)`):
- shells out to `agy`, returns the text.
- raises a clear error on timeout/empty/non-JSON so the caller can HOLD.

## Config (`forge/config.py`)
- `GEMINI_MODEL` env, default `gemini-3.8-flash-high`.
- `GEMINI_BIN` env, default `agy` (allow override to a full path).
- `GEMINI_FALLBACK` env, default `true` (allow disabling the fallback).
- Wire these into `decide()` so the fallback only fires when enabled.

## Audit / honesty (required)
The decider's result and the decision log must say **which brain answered**. Add a field to
`BrainResult` (e.g. `brain: str = "claude"` → `"gemini"` → `"hold"`), and log/record it so a
human can see "this decision was made by Gemini because Claude was rate-limited" vs Claude vs
HOLD. The paper `latest_decision.json` / ledger should surface `brain` alongside `status` so the
dashboard/DM line can reflect it. Do NOT silently swap brains without a traceable record.

## Constraints
- **Claude stays primary.** Never route to Gemini when Claude succeeds. Never reorder.
- **Paper only** — no live/broker code (keep `test_no_live.py` passing).
- **No secrets** in the repo or in error messages (`agy` one-shot needs no key; if you touch the
  Claude path, keep the existing `_scrub`/redact discipline).
- **No high-frequency loops** — the decider is called once per rebalance; a single `agy` call per
  fallback is fine.
- Cache: cache Gemini responses separately from Claude (key includes the brain) so a resumed
  backtest is deterministic per brain.

## Definition of done
- `ClaudeBrain.decide()` tries Claude first; on Claude failure (limit/403/any error/2x
  parse-fail) with `GEMINI_FALLBACK` on, tries Gemini once; returns `ok` if Gemini yields a
  valid decision, else HOLD. When Claude works, Gemini is never called.
- `BrainResult.brain` reflects the brain that answered; it is recorded in the decision log and
  surfaced in `latest_decision.json`.
- Unit tests: (a) Claude ok → brain "claude", no agy call; (b) Claude `ClaudeLimitError` →
  Gemini ok → brain "gemini", status "ok"; (c) Claude error AND Gemini error → HOLD; (d)
  `GEMINI_FALLBACK=false` → Claude error → HOLD (no agy). **Mock the `agy` shell call** in tests
  (do not spawn a real agy in the suite).
- **Full test suite passes** (`.venv/bin/python -m pytest -q`).
- **Live proof:** run one real paper cycle (`run/run_daily --force` on a fresh date or the
  current one) with the Claude limit active and confirm the decision is produced by **Gemini**
  (brain="gemini" in `latest_decision.json` / the log), NOT a HOLD. If the book was all-cash
  before and now holds a real allocation, that is the proof the fallback works end-to-end.
- **Git:** commit as you go (clear per-item commits).
- A short section in `docs/CRYPTO.md` or a new `docs/DECIDER.md` documenting the Claude→Gemini
  fallback (when it fires, how to disable it, how to read the `brain` field).

## Note
Do NOT modify the crypto work or the sizing — that's done. This task is purely the decider
fallback. The Claude bridge (8910) is currently UP and healthy; the reason for the HOLD is the
CLI's **weekly limit**, which the limit-notice detection already catches (`ClaudeLimitError`).
