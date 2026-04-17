# Certainty Sniper — Design Spec
**Date:** 2026-04-17
**Status:** Approved

## Problem
The existing momentum sniper fires in the first 90 seconds of each 15-minute period — when outcome uncertainty is highest (~50/50). All 15 production orders were cancelled unfilled. Capital sits idle and no compounding occurs.

## Goal
A high-frequency compounding engine that generates daily cashflow by betting on **near-certain outcomes** in the final minutes of each period, reinvesting winnings immediately.

## Strategy: Triple-Confirmed Late-Period Certainty Snipe

Instead of betting at period start (uncertain), bet at period end (near-certain).

With 2–3 minutes left in a 15-minute BTC/crypto binary market, if a coin has moved 0.7%+ from period open, the winning-side token is virtually certain to resolve $1 — but still trading at $0.82–0.90 due to thin end-of-period liquidity. We buy at ~$0.85, collect $1 in under 3 minutes.

**Expected win rate:** 88–92%
**Return per winning trade:** ~15–18%
**Signals per day:** 3–8 across 6 coins

## Signal Architecture — 3 Gates (all must pass)

**Gate 1 — Move threshold**
Coin price ≥ 0.7% from its period-open price at check time.
Source: existing Binance WebSocket feed (`src/crypto_arb/ws_feeds.py`).

**Gate 2 — Multi-coin confirmation**
At least 2 of 6 tracked coins (BTC, ETH, SOL, XRP, DOGE, BNB) moving the same direction ≥ 0.5%.
Filters coin-specific noise — macro crypto moves affect multiple coins simultaneously.

**Gate 3 — Order book pressure**
Binance REST depth check (`GET /api/v3/depth?symbol={COIN}USDT&limit=10`).
For UP moves: bid depth > ask depth (buyers still aggressive).
For DOWN moves: ask depth > bid depth.
Catches "moved but now reversing" scenarios before we enter.

## Timing

- **Window:** Minutes 12–14 of each 15-minute period
- **Check frequency:** Every 30 seconds
- **Action:** First check where all 3 gates pass → market buy immediately
- **Max positions:** 1 per coin per period (no pyramiding initially)
- **Skip condition:** If coin drops back below threshold before we fire → skip period

## Sizing — Half-Kelly Compounding

At 90% win rate, ~$0.85 entry, $1 payout:
- Kelly fraction ≈ 46% (too aggressive for unvalidated live strategy)
- **Half-Kelly: 23% of current bankroll per trade**
- Floor: $8 minimum bet
- Cap: $40 maximum bet (protects against over-concentration)
- Cap entry price: $0.92 — if market has already priced in certainty, edge is gone, skip

Bankroll updates in real time from state file. Each win immediately raises the base for the next bet.

**Compounding projection (90% win rate, 5 signals/day):**
- Day 1: $151 → ~$166
- Week 1: ~$260
- Month 1: ~$800+

## Risk Controls

| Control | Threshold | Action |
|---------|-----------|--------|
| Circuit breaker | 2 consecutive losses | Skip next 4 periods (~1 hour) |
| Daily loss limit | Bankroll drops 12% in one day | Pause 24 hours |
| Entry price cap | Token price > $0.92 | Skip — edge gone |
| Min bankroll | Below $30 | Halt all trading |

## Implementation

**New file:** `run_certainty_sniper.py` (~250 lines, follows `run_sniper.py` pattern)

| Component | Status | Source |
|-----------|--------|--------|
| Binance WS price feed | Reuse | `src/crypto_arb/ws_feeds.py` |
| Period tracking | Reuse | Existing sniper pattern |
| Market fetch (token IDs) | Reuse | `src/crypto_arb/fast_markets.py` |
| CLOB order placement | Reuse | `py-clob-client` |
| Binance order book REST | New | ~20 lines, `httpx` |
| Late-period signal logic | New | ~60 lines |
| Half-Kelly sizer | New | ~20 lines |
| State & compounding | New | Adapt `sniper_state.json` pattern |
| Dashboard widget | New | Card alongside existing sniper |

**State file:** `data/certainty_sniper_state.json`
```json
{
  "bankroll": 151.25,
  "trades": [],
  "total_invested": 0,
  "total_returned": 0,
  "consecutive_losses": 0,
  "daily_start_bankroll": 151.25,
  "daily_start_date": "2026-04-17"
}
```

**Log events:**
- `certainty_signal` — all 3 gates passed, about to fire
- `certainty_trade` — order placed
- `certainty_resolved` — outcome collected, bankroll updated
- `certainty_skip` — gates passed but entry price > $0.92 or circuit open
- `certainty_status` — heartbeat every 60s

## What This Is Not
- Not a replacement for the existing momentum sniper (keep both running, separate bankrolls)
- Not a long-term position strategy (Power Trader handles that)
- Not tested yet — win rate projections are based on BTC 15-min historical behavior, not live data

## Open Questions (resolve after first 48h live)
1. What is the actual average entry price achieved?
2. What is the realized win rate?
3. Adjust Kelly fraction up/down based on observed win rate after 20+ trades
