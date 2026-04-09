#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Polymarket Bot AI Monitor
# Periodic health + profitability audit using Claude AI
# Inspired by Artemis AI monitor pattern
# ──────────────────────────────────────────────────────────────
set -euo pipefail

BOT_URL="${PM_BOT_URL:-http://localhost:8075}"
LOG_DIR="/opt/polymarket-bot/data"
MONITOR_LOG="$LOG_DIR/ai_monitor.log"
ARB_LOG="/opt/polymarket-bot/crypto_arb_output.log"
mkdir -p "$LOG_DIR"

TS=$(date -u +"%Y-%m-%d %H:%M UTC")

# ── 1. Gather state ─────────────────────────────────────────
echo "[$TS] AI Monitor starting..."

# Bot health
BOT_HEALTH=$(curl -sf "$BOT_URL/health" 2>/dev/null || echo '{"status":"unreachable"}')
BOT_STATUS=$(curl -sf "$BOT_URL/status" 2>/dev/null || echo '{}')
ARB_STATUS=$(curl -sf "$BOT_URL/api/crypto-arb" 2>/dev/null || echo '{}')

# Process check
BOT_PROC=$(ps aux | grep "uvicorn src.main:app" | grep -v grep | wc -l)
ARB_PROC=$(ps aux | grep -E "run_(crypto|fast)_arb" | grep python | grep -v grep | wc -l)

# Recent arb trades
ARB_TRADES=""
if [ -f "$ARB_LOG" ]; then
    ARB_TRADES=$(grep "paper_trade\|paper_resolved\|crypto_arb_cycle" "$ARB_LOG" | tail -20)
fi

# Previous verdict
PREV_VERDICT=""
if [ -f "$MONITOR_LOG" ]; then
    PREV_VERDICT=$(tail -30 "$MONITOR_LOG" | head -20)
fi

# Binance prices
BTC_PRICE=$(curl -sf "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT" | python3 -c "import sys,json; print(json.load(sys.stdin)['price'])" 2>/dev/null || echo "unavailable")
ETH_PRICE=$(curl -sf "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT" | python3 -c "import sys,json; print(json.load(sys.stdin)['price'])" 2>/dev/null || echo "unavailable")

# ── 2. Build prompt ──────────────────────────────────────────
PROMPT=$(cat <<PROMPT_EOF
You are an autonomous trading bot health monitor. Analyze the current state and produce a structured verdict.

## Current State ($TS)
- Bot process: $BOT_PROC instances running
- Arb engine: $ARB_PROC instances running
- Binance BTC: \$$BTC_PRICE | ETH: \$$ETH_PRICE

## Bot Health
$BOT_HEALTH

## Bot Status
$BOT_STATUS

## Crypto Arb Engine
$ARB_STATUS

## Recent Arb Activity
$ARB_TRADES

## Previous Verdict
$PREV_VERDICT

## Architecture Note
The system has TWO independent processes by design:
1. Main bot (uvicorn) — runs market discovery, signal generation, ensemble strategy (shadow mode = no live trades)
2. Crypto arb engine (run_crypto_arb.py) — separate paper/live trading engine for crypto latency arb
These are NOT duplicates. Both should be running simultaneously. The arb engine's paper trades are EXPECTED even when the main bot is in shadow mode.

## Rules — flag violations:
1. Bot process must be running (exactly 1 instance)
2. Kill switch should NOT be active unless manually set
3. If in shadow mode: no real trades should exist
4. Arb engine: check if trades make sense given current prices
5. Arb engine: flag if any trade has entry > 0.80 (low edge, high risk)
6. Arb engine: flag buy_yes trades where Binance price is now BELOW threshold (losing)
7. Arb engine: flag buy_no trades where Binance price is now ABOVE threshold (losing)
8. Check for duplicate processes
9. If arb engine is not running but should be: WARNING
10. If bankroll + open position value has dropped >30% from \$1000: WARNING
    Note: bankroll alone being low is NORMAL — cash is deployed into open trades.
    Low bankroll + open positions = capital deployed, not capital lost.

## Output format (follow exactly):
STATUS: OK|WARNING|CRITICAL
ISSUES: <comma-separated list, or NONE>
ACTION: NONE|ALERT|SHUTDOWN
PRICES: BTC=\$$BTC_PRICE ETH=\$$ETH_PRICE
ARB_SUMMARY: <bankroll, open trades, P&L>
POSITIONS: <per-trade assessment>
RISK: <risk assessment>
SUMMARY: <2-3 sentence overall judgment>
PROMPT_EOF
)

# ── 3. Call Claude ───────────────────────────────────────────
VERDICT=$(echo "$PROMPT" | claude -p --model haiku 2>/dev/null || echo "STATUS: UNKNOWN
ISSUES: claude_api_error
ACTION: NONE
SUMMARY: Could not reach Claude API for health check.")

# ── 4. Log verdict ───────────────────────────────────────────
{
    echo "──────────────────────────────────────────"
    echo "[$TS]"
    echo "$VERDICT"
    echo ""
} >> "$MONITOR_LOG"

# ── 5. Take action ───────────────────────────────────────────
# Strip markdown code fences if present
VERDICT_CLEAN=$(echo "$VERDICT" | sed 's/^```.*//g' | sed 's/^```//g')
STATUS_LINE=$(echo "$VERDICT_CLEAN" | grep "STATUS:" | head -1)
ACTION_LINE=$(echo "$VERDICT_CLEAN" | grep "ACTION:" | head -1)

if echo "$ACTION_LINE" | grep -q "SHUTDOWN"; then
    echo "[$TS] CRITICAL: AI monitor recommends SHUTDOWN"
    echo "[$TS] WARNING: Auto-shutdown disabled. Manual review required." >> "$MONITOR_LOG"
    # Only auto-kill if explicitly enabled
    if [ "${PM_AUTO_SHUTDOWN:-false}" = "true" ]; then
        curl -sf -X POST "$BOT_URL/admin/kill-switch" > /dev/null 2>&1 || true
        echo "[$TS] Kill switch activated" >> "$MONITOR_LOG"
    fi
fi

echo "[$TS] $STATUS_LINE"
echo "[$TS] Monitor complete"
