#!/usr/bin/env bash
# One-shot, idempotent bootstrap: venv + dependencies + pinned Kronos source + Kronos-mini weights (CPU only).
set -euo pipefail
cd "$(dirname "$0")/.."

KRONOS_SHA=67b630e67f6a18c9e9be918d9b4337c960db1e9a      # shiyu-coder/Kronos (pinned)

[ -d .venv ] || python3.14 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt

mkdir -p vendor models data state logs backtests
if [ ! -d vendor/Kronos/.git ]; then
  git clone -q https://github.com/shiyu-coder/Kronos vendor/Kronos
fi
git -C vendor/Kronos fetch -q origin "$KRONOS_SHA" 2>/dev/null || true
git -C vendor/Kronos checkout -q "$KRONOS_SHA"

# smallest Kronos variant (4.1M params) + its matching tokenizer
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for name in ("Kronos-Tokenizer-2k", "Kronos-mini"):
    snapshot_download(f"NeoQuasar/{name}", local_dir=f"models/{name}")
    print("ok", name)
PY

if [ ! -f .env ]; then
  cat > .env <<'ENV'
# Local OpenAI-compatible Claude endpoint. chmod 600, never commit.
CLAUDE_BASE_URL=http://192.168.1.201:8910/v1
CLAUDE_LOCAL_TOKEN=
CLAUDE_MODEL=sonnet
ENV
  chmod 600 .env
  echo "created .env template - put the bearer token in CLAUDE_LOCAL_TOKEN"
fi
echo "setup complete. next: .venv/bin/python -m pytest && .venv/bin/python -m run.run_backtest"
