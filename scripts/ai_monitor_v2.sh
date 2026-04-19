#!/bin/bash
set -euo pipefail

BOT_DIR="${PM_BOT_DIR:-/opt/polymarket-bot}"
BOT_URL="${PM_BOT_URL:-http://localhost:8075}"
LOG_DIR="$BOT_DIR/data"
MONITOR_LOG="$LOG_DIR/ai_monitor.log"
CRON_LOG="$LOG_DIR/ai_monitor_cron.log"
SNIPER_LOG="$BOT_DIR/sniper_output.log"
CERTAINTY_LOG="$BOT_DIR/certainty_sniper_output.log"
UVICORN_LOG="$BOT_DIR/uvicorn.log"
mkdir -p "$LOG_DIR"

TS=$(date -u +"%Y-%m-%d %H:%M UTC")
echo "[$TS] AI Monitor v2 starting..."

BOT_HEALTH=$(curl -sf "$BOT_URL/health" 2>/dev/null || echo '{"status":"unreachable"}')
ALL_STATUS=$(curl -sf "$BOT_URL/api/all-status" 2>/dev/null || echo '{}')

UVICORN_PROC=$(ps aux 2>/dev/null | grep -E "uvicorn|src.main:app" | grep -v grep | head -1 || true)
SNIPER_PROC=$(ps aux 2>/dev/null | grep "run_sniper.py" | grep -v grep | head -1 || true)
CERTAINTY_PROC=$(ps aux 2>/dev/null | grep "run_certainty_sniper.py" | grep -v grep | head -1 || true)

SNIPER_TAIL=$(tail -20 "$SNIPER_LOG" 2>/dev/null || true)
CERTAINTY_TAIL=$(tail -20 "$CERTAINTY_LOG" 2>/dev/null || true)
UVICORN_TAIL=$(tail -20 "$UVICORN_LOG" 2>/dev/null || true)

PROMPT="You are monitoring a Polymarket trading stack.

Current time: $TS

Health endpoint:
$BOT_HEALTH

Unified status:
$ALL_STATUS

Processes:
uvicorn: ${UVICORN_PROC:-NOT RUNNING}
run_sniper.py: ${SNIPER_PROC:-NOT RUNNING}
run_certainty_sniper.py: ${CERTAINTY_PROC:-NOT RUNNING}

Recent sniper log:
$SNIPER_TAIL

Recent certainty log:
$CERTAINTY_TAIL

Recent uvicorn log:
$UVICORN_TAIL

Assess:
1. Are the API, sniper, and certainty bots running?
2. Are trades firing or being filtered too hard?
3. Any obvious errors, dead processes, or stale heartbeats?
4. Is there any urgent action needed?

Output EXACTLY:
STATUS: OK|WARNING|CRITICAL
SUMMARY: <2-3 sentences>
ISSUES: <comma-separated or NONE>"

RESULT=$(claude -p "$PROMPT" --model haiku --output-format text 2>/dev/null || cat <<'EOF'
STATUS: UNKNOWN
SUMMARY: Could not reach Claude API for health check.
ISSUES: claude_api_error
EOF
)

{
  echo "──────────────────────────────────────────"
  echo "[$TS]"
  echo "$RESULT"
  echo ""
} >> "$MONITOR_LOG"

STATUS_LINE=$(echo "$RESULT" | grep "^STATUS:" | head -1 || echo "STATUS: UNKNOWN")
echo "[$TS] $STATUS_LINE" | tee -a "$CRON_LOG"
echo "[$TS] Monitor v2 complete" >> "$CRON_LOG"
