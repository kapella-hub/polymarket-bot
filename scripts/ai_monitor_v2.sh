#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Polymarket Bot AI Monitor v2
# Analyzes the v4 live arb engine using Claude
# Crontab: 0 * * * * /opt/polymarket-bot/scripts/ai_monitor_v2.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

BOT_DIR="${PM_BOT_DIR:-/opt/polymarket-bot}"
LOG_DIR="$BOT_DIR/data"
MONITOR_LOG="$LOG_DIR/ai_monitor.log"
LIVE_LOG="$BOT_DIR/live_arb_output.log"
mkdir -p "$LOG_DIR"

TS=$(date -u +"%Y-%m-%d %H:%M UTC")

echo "[$TS] AI Monitor v2 starting..."

# ── 1. Gather state ─────────────────────────────────────────

# Recent log lines
LOG_TAIL=""
if [ -f "$LIVE_LOG" ]; then
    LOG_TAIL=$(tail -50 "$LIVE_LOG" 2>/dev/null || true)
fi

# Recent trades
TRADES=""
if [ -f "$LIVE_LOG" ]; then
    TRADES=$(grep -E "LIVE_FILL|LIVE_RESOLVED" "$LIVE_LOG" 2>/dev/null | tail -20 || true)
fi

# Latest cycle
LATEST=""
if [ -f "$LIVE_LOG" ]; then
    LATEST=$(grep "v4_cycle" "$LIVE_LOG" 2>/dev/null | tail -1 || true)
fi

# Process check
PROCESS=$(ps aux 2>/dev/null | grep run_live_arb | grep -v grep | head -1 || true)

# ── 2. Build prompt ──────────────────────────────────────────

PROMPT="You are monitoring a Polymarket 15-minute crypto latency arb bot (v4).

Current state:
Process: ${PROCESS:-NOT RUNNING}

Latest cycle:
${LATEST:-No cycle data}

Recent activity (last 50 lines):
${LOG_TAIL:-No log data}

Recent trades:
${TRADES:-No trades}

Analyze:
1. Is the bot healthy? (process running, WS connected, detecting signals)
2. Is it making trades? If not, why? (flat market, no cheap asks, too early in period)
3. Any losses? Is the bankroll growing or shrinking?
4. Any red flags? (disconnections, errors, high loss rate)

Output EXACTLY this format (no other text):
STATUS: OK|WARNING|CRITICAL
SUMMARY: <2-3 sentences>
ISSUES: <comma-separated or NONE>"

# ── 3. Call Claude ───────────────────────────────────────────

RESULT=$(claude -p "$PROMPT" --model haiku --output-format text 2>/dev/null || echo "STATUS: UNKNOWN
SUMMARY: Could not reach Claude API for health check.
ISSUES: claude_api_error")

# ── 4. Log verdict ───────────────────────────────────────────

{
    echo "──────────────────────────────────────────"
    echo "[$TS]"
    echo "$RESULT"
    echo ""
} >> "$MONITOR_LOG"

# ── 5. Report ────────────────────────────────────────────────

STATUS_LINE=$(echo "$RESULT" | grep "^STATUS:" | head -1 || echo "STATUS: UNKNOWN")
echo "[$TS] $STATUS_LINE"
echo "[$TS] Monitor v2 complete"
