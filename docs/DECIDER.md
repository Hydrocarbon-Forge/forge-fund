# Decider Architecture & Fallback (Claude → Gemini)

## Overview

Forge-Fund uses an LLM-driven decision pipeline to generate portfolio target weights across equities and crypto.

The decider architecture implements a primary/fallback hierarchy designed for resilience:
1. **Primary Decider: Claude** (OpenAI-compatible bridge at `http://192.168.1.201:8910/v1`).
2. **Fallback Decider: Gemini** (one-shot local CLI via `agy --model gemini-3.8-flash-high`).
3. **Safe Default: HOLD** (0% rebalance, keeping existing positions intact).

Claude remains the primary decider ("ranked in"). Whenever Claude is healthy and within quota, Claude generates the decisions and Gemini is never called. When Claude fails (usage limit, API error, or parse failure) and fallback is enabled, the pipeline seamlessly falls back to Gemini to avoid leaving the fund stranded in all-cash or frozen in HOLD.

## When Gemini Fallback Triggers

The fallback fires inside `ClaudeBrain.decide()` (`brain/decision.py`) under any of the following Claude failure conditions, provided `GEMINI_FALLBACK=true`:

1. **Usage Limit / Quota Exceeded (`ClaudeLimitError`)**:
   - Triggered when Claude bridge reports weekly subscription limits, token exhaustion, or HTTP 429/403 limit notices.
2. **Bridge Errors / Network Failures (`ClaudeError`)**:
   - Triggered when the Claude bridge is offline, unreachable, returns 5xx server errors, or terminates unexpectedly.
3. **Parse Failures (2x Retry Exhaustion)**:
   - If Claude produces non-JSON or invalid schema output that fails `brain.schema.parse_decision` twice, the pipeline routes to Gemini with the same prompt and universe.

If Gemini also fails (or times out), the pipeline safely falls back to `status="fallback_hold"` (`brain="hold"`).

## Configuration

The decider behavior is controlled via environment variables in `forge/config.py`:

| Environment Variable | Default | Description |
|---|---|---|
| `GEMINI_FALLBACK` | `true` | Enable or disable the Gemini fallback. Set to `false` or `0` to disable. |
| `GEMINI_MODEL` | `gemini-3.8-flash-high` | Gemini model name passed to `agy --model`. |
| `GEMINI_BIN` | `agy` | Path or name of the `agy` CLI binary. |
| `GEMINI_TIMEOUT_S` | `300.0` | Subprocess timeout in seconds for `agy` execution (accommodates large universe prompts). |

### CLI Override
During daily cycle runs (`run.run_daily`), fallback can be disabled on the command line:
```bash
python -m run.run_daily --no-gemini
```

## Traceability & Audit Trail (`brain` field)

To ensure full transparency and auditability, the fund never silently swaps brains without recording it. Every decision records which brain answered:

- **`BrainResult.brain`**: Set to `"claude"`, `"gemini"`, or `"hold"`.
- **`state/latest_decision.json`**:
  ```json
  {
    "date": "2026-10-14",
    "brain": "gemini",
    "status": "ok",
    "model": "gemini-3.8-flash-high",
    "target_weights": { ... },
    "rationale": "...",
    "error": null
  }
  ```
- **Cycle Summary & Logging**:
  The daily runner logs decider usage clearly, indicating whether the decision originated from Claude or Gemini fallback, and tracking token costs according to each model's pricing.
- **Cache Isolation**:
  The decider cache key includes the brain identity (`_key(..., brain="gemini")`), preventing cache collisions between Claude and Gemini decisions and enabling deterministic backtesting.

## Testing & Safety Constraints

- **Strictly Mocked Unit Tests**: Unit tests in `tests/test_gemini_client.py` and `tests/test_decision.py` mock the `agy` shell call (`call_gemini` or `subprocess.run`). Real `agy` commands are never spawned during `pytest`.
- **Paper-Trading Only**: The fallback decider only feeds target weights into the paper-trading engine (`engine/paper.py`) via simulated execution. No broker or live exchange adapters exist.
