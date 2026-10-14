# Task: make the decider BUDGET-AWARE (forge-fund)

## Problem (verified)
The $200 paper book is sitting ~97% cash with only a fractional BTC position. Root cause,
confirmed: `brain/decision.py::build_messages(...)` is told `FUND: equity $200` but is **never
shown a per-name price**, so it cannot know that MSFT/V/XOM/etc. (~$700–1,100) are unbuyable
as whole shares on a $200 book. It plans weights for big names; the engine (integer lots for
equities) clips them to zero → the fund can only fill the fractional crypto. The decider is
blind to its own budget.

## Goal
Tell the decider the book's real equity AND which names it can actually afford, so its plan
gets filled and the fund deploys real money. When equity is large (backtests, $100k) the
budget note is trivial (everything affordable) and must not distort the plan.

## What to build

### 1. Prices into the prompt
`pipeline.decide()` (run/pipeline.py) has last-close prices via `self.m.p["close"]` at `asof`.
Thread a `prices` mapping (ticker -> last close as of asof) into `build_messages(...)`. Add a
`px` column (last close, $) to the UNIVERSE table rows (next to the existing id|r1|...|path
columns) so the decider sees each name's price. Keep the existing table format and column
legend accurate (update the legend line that describes the columns).

### 2. A budget block in the prompt
Compute it in code from `view.equity` + `prices` (do NOT ask the model to count). Add a
concise `BUDGET:` line(s) near the FUND line, e.g. for a $200 book:
- `BUDGET: equity $200 -> you can buy whole shares of N names (price <= $X).`
- If the book is small enough that few names are affordable, say so plainly and remind the
  decider that fractional crypto (BTC-USD/ETH-USD/SOL-USD) is always sizeable, while any
  equity weight above what the book can afford in whole shares is clipped to 0 by the engine.
- Guard the note so it only becomes prominent when the book is genuinely constrained
  (e.g. when the number of affordable whole-share names is small); when most names are
  affordable (large equity) keep it a short single line so the backtest prompt is unaffected.
Keep it deterministic (no randomness) so the decision cache stays stable.

### 3. System prompt
In `system_prompt(limits)` (brain/decision.py) add one sentence: integer-share equity is
whole lots, crypto is fractional, and weights for names the book cannot afford are clipped to
0 by the engine — plan to the equity shown. Keep the existing mandate/rules intact.

## Constraints
- **Both brains** benefit (Claude primary, Gemini fallback share `build_messages`) — no brain
  divergence; do not fork prompts per brain.
- **Backtest unchanged in behavior**: at $100k (STARTING_CAPITAL) nearly all names are
  affordable, so the budget note must be a short no-op line there. Verify a backtest-era
  prompt diff is minimal.
- **Paper only** — no live/broker code. Keep `test_no_live.py` green.
- **Cache**: prompt now depends on prices+equity; that is fine (still deterministic). Update the
  prompt-hash/caching tests accordingly (the key already hashes messages).
- **No secrets** in prompts or logs.

## Definition of done
- `build_messages` takes `prices` and emits the `px` column + the `BUDGET:` line(s) computed
  from equity/prices.
- New unit tests: (a) small equity ($200) + a price mix → budget line lists the affordable
  whole-share names and flags the unaffordable ones; (b) large equity ($100k) → the budget
  line is the short no-op form; (c) `px` column present and correct in the table. **No real
  model calls in tests** (mock the brain / build_messages is pure).
- Existing prompt/decision tests updated to the new prompt shape.
- **Full suite passes** (`.venv/bin/python -m pytest -q`).
- **Live proof:** run one real paper cycle (`run/run_daily --force`) on the $200 book. Confirm
  the decision (Gemini, since Claude is on its weekly limit) now names at least the
  affordable names and the resulting paper book holds **more than just the single fractional
  BTC** position (i.e. it deployed to whole-share affordable names or fractional crypto
  beyond one position). Capture before/after position counts.
- **Git:** commit as you go.
- Short note in `docs/DECIDER.md` (budget-awareness: what the model sees, the clip rule,
  how the backtest stays unaffected).

## Note
Do NOT touch the crypto sizing, the Claude→Gemini fallback, or the risk caps — those are done.
This task is purely "give the decider its budget."
