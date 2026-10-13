#!/usr/bin/env bash
# Cron-ready wrapper for one PAPER trading cycle. Paper only: nothing here can place a real order.
#
#   crontab -e   (cron >= 3.0pl1 / cronie support CRON_TZ)
#   CRON_TZ=America/New_York
#   30 18 * * 1-5  /path/to/forge-fund/scripts/run_daily.sh --prepare    # evening: data + Kronos
#   35  9 * * 1-5  /path/to/forge-fund/scripts/run_daily.sh              # 09:35 ET: decide + paper-trade
#
# Exit codes: 0 ok | 2 error | 3 halted by the 2% loss cap (clear with: run_daily.sh --resume)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs state
exec .venv/bin/python -m run.run_daily "$@" >> logs/cron_daily.log 2>&1
