# Sniper v2: Regime-Aware Signal Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix under-triggering in momentum sniper by making confirmation an entry gate (not a size boost), adding coin tiers, fixing certainty Kelly over-sizing, and preparing shadow infrastructure for continuation mode.

**Architecture:** Modify `run_sniper.py` and `run_certainty_sniper.py` to add config-gated correlated-burst logic, coin-tier filters, per-cycle skip-reason logging, and Kelly sizing cap. Continuation mode lands as shadow-only in Phase 2.

**Tech Stack:** Python 3.12, py-clob-client, httpx, structlog, Binance WS. No new deps.

**Rollout:** Phase 1 (live) + Phase 2 (shadow). Backtest validates thresholds before code ships.

---

## Pre-Flight: Backtest Validation

Run before any coding — calibrate thresholds against actual market data.

### Task 0: Fetch Binance kline data + simulate new logic

**Files:**
- Create: `scripts/backtest_sniper_v2.py`

- [ ] **Step 1: Fetch 72h of 1-minute klines**

For each of BTC, ETH, SOL, XRP, DOGE, BNB from Binance public API:
```python
import httpx, time
from datetime import datetime

def fetch_klines(symbol: str, start_ms: int, end_ms: int):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m",
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    # paginate until end_ms reached — returns [[open_ts, o, h, l, c, v, ...]]
```

Save to `data/backtest_klines_72h.json` keyed by symbol.

- [ ] **Step 2: Simulate correlated-burst logic for every 15-min period in the last 72h**

For each period boundary `ts_period`:
1. Compute 90-second moves for each coin: `(price_at_ts+90 - price_at_ts) / price_at_ts`
2. Apply new gate:
   - At least 2 coins aligned same direction
   - ≥1 of them is BTC or ETH
   - Each aligned coin moved ≥0.25%
   - Aggregate of aligned coins ≥0.60%
3. Record: would_fire, direction, confirming_coins, aggregate_score
4. Resolution at t+900: did actual price end up in the fired direction?

- [ ] **Step 3: Simulate solo-major logic**

BTC or ETH solo fire at ≥0.38%. Record same fields.

- [ ] **Step 4: Print calibration report**

```
=== CORRELATED BURST ===
Periods evaluated: 288 (72h × 4/hr)
Would-fire count: X
Win rate: Y%
Avg aggregate_score at fires: Z%

=== SOLO MAJOR ===
Would-fire: X, Win rate: Y%

=== BASELINE (current 0.40% solo any-coin) ===
Would-fire: X, Win rate: Y%
```

**Go/no-go gates:**
- Correlated burst fires ≤8/day avg and win rate ≥55% → PROCEED to Phase 1
- If fires >15/day OR win rate <52% → STOP, retune thresholds, re-run

- [ ] **Step 5: Commit backtest + report**

```bash
git add scripts/backtest_sniper_v2.py data/backtest_klines_72h.json
git commit -m "backtest: validate sniper v2 thresholds against 72h historical data"
```

---

## Phase 1: Live Deployment (Core Gate Logic)

### Task 1: Add coin tier constants and strategy mode enum

**Files:**
- Modify: `run_sniper.py:50-65` (constants section)

- [ ] **Step 1: Add tier and mode constants after existing constants**

```python
# Coin tiers — determines solo-trigger eligibility
TIER_1_COINS = frozenset(["btc", "eth"])
TIER_3_COINS = frozenset(["doge", "xrp", "bnb"])  # never solo

# Correlated burst gate (v2)
CORRELATED_MIN_MOVE_PCT     = 0.0025  # each confirming coin
CORRELATED_MIN_COINS        = 2
CORRELATED_MIN_AGGREGATE    = 0.006   # sum of aligned magnitudes
CORRELATED_REQUIRE_MAJOR    = True    # at least 1 must be TIER_1

# Solo major (TIER_1 only)
SOLO_MAJOR_MIN_MOVE_PCT     = 0.0038

# Sizing multipliers
BASE_SIZE_MULT              = 1.0
SOLO_MAJOR_SIZE_MULT        = 0.75
STRONG_CONFIRM_SIZE_MULT    = 1.25   # 3+ aligned, BTC+ETH both in
```

- [ ] **Step 2: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper-v2): add coin tier and gate constants"
```

### Task 2: Replace single-coin signal gate with tiered + correlated logic

**Files:**
- Modify: `run_sniper.py:283-322` (snipe window logic)

- [ ] **Step 1: Replace the signal detection block**

Current logic iterates per-coin with a single threshold and uses `confirm_count` only for size boost. Replace with:

```python
# Compute per-coin moves
period_prices = hist_start.get(current_period, {})
coin_moves = {}
for coin, market in current_markets.items():
    if coin in fired_this_period:
        continue
    asset = COIN_TO_ASSET[coin]
    cur = ws_feed.get(asset); start = period_prices.get(asset)
    if cur and start:
        coin_moves[coin] = (cur - start) / start

# Classify each coin's contribution for each direction
def aligned_coins(direction: str) -> list[tuple[str, float]]:
    sign = 1 if direction == "up" else -1
    return [
        (c, m) for c, m in coin_moves.items()
        if (m * sign) >= CORRELATED_MIN_MOVE_PCT
    ]

signals = []  # list of (coin, market, side, move, mode, size_mult)

for direction in ("up", "down"):
    aligned = aligned_coins(direction)
    if len(aligned) < CORRELATED_MIN_COINS:
        continue
    has_major = any(c in TIER_1_COINS for c, _ in aligned)
    if CORRELATED_REQUIRE_MAJOR and not has_major:
        logger.info("burst_skip_no_major", direction=direction,
                    aligned=[c for c, _ in aligned])
        continue
    aggregate = sum(abs(m) for _, m in aligned)
    if aggregate < CORRELATED_MIN_AGGREGATE:
        logger.info("burst_skip_aggregate_low", direction=direction,
                    aggregate=round(aggregate * 100, 3))
        continue

    # Determine size multiplier
    major_count = sum(1 for c, _ in aligned if c in TIER_1_COINS)
    if len(aligned) >= 3 and major_count >= 2:
        size_mult = STRONG_CONFIRM_SIZE_MULT
    else:
        size_mult = BASE_SIZE_MULT

    # Fire on each aligned coin, but never TIER_3 solo (here always confirmed)
    for coin, move in aligned:
        if coin in fired_this_period:
            continue
        side = "buy_up" if direction == "up" else "buy_down"
        signals.append((coin, current_markets[coin], side, move,
                        "correlated_burst", size_mult, len(aligned)))

# Solo major fallback — only if no correlated signal found
if not signals:
    for coin in TIER_1_COINS:
        if coin not in coin_moves or coin in fired_this_period:
            continue
        m = coin_moves[coin]
        if abs(m) >= SOLO_MAJOR_MIN_MOVE_PCT:
            side = "buy_up" if m > 0 else "buy_down"
            signals.append((coin, current_markets[coin], side, m,
                            "solo_major", SOLO_MAJOR_SIZE_MULT, 1))

if not signals:
    # periodic debug only — not every iteration
    if elapsed in (30, 60, 89):
        logger.debug("no_burst_signal",
                     elapsed=elapsed,
                     coin_moves={c: round(m * 100, 3) for c, m in coin_moves.items()})
```

- [ ] **Step 2: Update downstream `for (coin, market, side, move, ...) in signals:` loop**

Unpack extra fields from new tuple shape. Drop the old `boost = CROSS_BOOST if confirm_count >= 2 else 1.0` — now the `size_mult` comes from the signal.

Update size calculation:
```python
size = min(BET_MAX, max(BET_MIN, bankroll * BET_PCT * size_mult))
```

Add to trade record:
```python
"strategy_mode":   mode,  # "correlated_burst" or "solo_major"
"aligned_count":   aligned_count,
```

- [ ] **Step 3: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper-v2): correlated burst gate + solo major fallback"
```

### Task 3: Per-cycle structured logging

**Files:**
- Modify: `run_sniper.py` (in snipe window block, once per period elapsed==30s)

- [ ] **Step 1: Emit one structured evaluation log per period**

Once per period at elapsed~30s:
```python
if elapsed == 30 and current_period not in logged_eval:
    logged_eval.add(current_period)
    logger.info(
        "burst_cycle_eval",
        period=current_period,
        btc_move=round(coin_moves.get("btc", 0) * 100, 3),
        eth_move=round(coin_moves.get("eth", 0) * 100, 3),
        all_moves={c: round(m * 100, 3) for c, m in coin_moves.items()},
    )
```

Add `logged_eval: set[int] = set()` to main loop state.

- [ ] **Step 2: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper-v2): per-period evaluation logging"
```

### Task 4: Cap certainty sniper Kelly at 15% of bankroll

**Files:**
- Modify: `run_certainty_sniper.py:70-76` (kelly_size function)
- Modify: `run_certainty_sniper.py:60-62` (BET_MIN/MAX constants)

- [ ] **Step 1: Add hard cap constant**

```python
# Risk caps
BET_MIN             = 8.0
BET_MAX             = 40.0
MAX_BET_PCT_BANKROLL = 0.15   # Never exceed 15% of bankroll per trade
```

- [ ] **Step 2: Modify `kelly_size` to enforce the cap**

```python
def kelly_size(bankroll: float, win_rate: float = 0.90, avg_entry: float = 0.85) -> float:
    """Half-Kelly with hard 15% bankroll cap. Guards against over-optimistic win_rate."""
    b = (1.0 - avg_entry) / avg_entry
    kelly = (win_rate * b - (1.0 - win_rate)) / b
    size = bankroll * (kelly / 2.0)
    pct_cap = bankroll * MAX_BET_PCT_BANKROLL
    size = min(size, pct_cap)
    return min(BET_MAX, max(BET_MIN, round(size, 2)))
```

On a $50 bankroll this caps at $7.50 max (below BET_MIN=$8, so floor wins → $8). Safer than current $40 theoretical max.

- [ ] **Step 3: Unit test the cap**

```python
# tests/test_certainty_kelly.py
from run_certainty_sniper import kelly_size

def test_kelly_capped_at_15pct_of_small_bankroll():
    # $50 bankroll: 15% = $7.50, but BET_MIN=$8 → returns $8
    assert kelly_size(50.0) == 8.0

def test_kelly_capped_on_large_bankroll():
    # $500 bankroll: 15% = $75, but BET_MAX=$40 → returns $40
    assert kelly_size(500.0) == 40.0

def test_kelly_midrange():
    # $200 bankroll: 15% = $30, below BET_MAX
    assert 8.0 <= kelly_size(200.0) <= 30.0
```

Run: `venv/Scripts/python.exe -m pytest tests/test_certainty_kelly.py -v`

- [ ] **Step 4: Commit**

```bash
git add run_certainty_sniper.py tests/test_certainty_kelly.py
git commit -m "fix(certainty): cap Kelly at 15% of bankroll — prevent over-levering on optimistic win_rate"
```

### Task 5: Deploy Phase 1 to VPS

- [ ] **Step 1: Push to git**

```bash
cd E:/Documents/Projects/polymarket-bot
git push origin main
```

- [ ] **Step 2: On VPS — stop bots, pull, restart**

```bash
ssh root@72.62.78.141 "
  pkill -f 'run_sniper.py' || true
  pkill -f 'run_certainty_sniper.py' || true
  cd /opt/polymarket-bot && git pull origin main
  nohup venv/bin/python3 -u run_sniper.py 604800 >> sniper_output.log 2>&1 &
  nohup venv/bin/python3 -u run_certainty_sniper.py 604800 >> certainty_sniper_output.log 2>&1 &
  sleep 3
  ps aux | grep -E 'run_sniper|run_certainty' | grep -v grep
"
```

- [ ] **Step 3: Verify via log — look for v2 startup banner + first burst_cycle_eval**

```bash
ssh root@72.62.78.141 "sleep 120 && grep -E 'burst_cycle_eval|burst_skip|SNIPER_TRADE' /opt/polymarket-bot/sniper_output.log | tail -10"
```

---

## Phase 2: Continuation Mode — SHADOW ONLY

Defer for 3 days post-Phase-1 deploy. Revisit this section after reviewing Phase 1 fire rate and fill quality.

### Task 6: Continuation signal evaluator (no trading, log-only)

**Files:**
- Modify: `run_sniper.py` (add evaluator function + call in main loop)

- [ ] **Step 1: Add constants**

```python
# Continuation mode — SHADOW ONLY initially
ENABLE_CONTINUATION_LIVE    = False   # When True, place orders; else log only
CONT_LOOKBACK_SECONDS       = 420     # 7 minutes
CONT_MAJOR_MIN_MOVE_PCT     = 0.003
CONT_REVERSAL_BLOCK_SECONDS = 60
CONT_SIZE_MULT              = 0.50    # Conservative — data unproven
```

- [ ] **Step 2: Add `rolling_prices: dict[str, deque[tuple[float, float]]]` to track 8 min of price history per coin**

Push (ts, price) each loop iteration, prune older than `CONT_LOOKBACK_SECONDS + 60`.

- [ ] **Step 3: Evaluator function**

```python
def evaluate_continuation(coin_prices: dict[str, deque],
                          now_ts: float) -> dict | None:
    """Returns a signal dict if continuation conditions met, else None."""
    btc_hist = list(coin_prices.get("BTC", []))
    eth_hist = list(coin_prices.get("ETH", []))
    if len(btc_hist) < 10 or len(eth_hist) < 10:
        return None

    lookback_start = now_ts - CONT_LOOKBACK_SECONDS
    btc_old = next((p for t, p in btc_hist if t >= lookback_start), None)
    eth_old = next((p for t, p in eth_hist if t >= lookback_start), None)
    btc_now = btc_hist[-1][1]; eth_now = eth_hist[-1][1]
    if not all((btc_old, eth_old, btc_now, eth_now)):
        return None

    btc_move = (btc_now - btc_old) / btc_old
    eth_move = (eth_now - eth_old) / eth_old

    if (btc_move >= CONT_MAJOR_MIN_MOVE_PCT and
        eth_move >= CONT_MAJOR_MIN_MOVE_PCT):
        direction = "up"
    elif (btc_move <= -CONT_MAJOR_MIN_MOVE_PCT and
          eth_move <= -CONT_MAJOR_MIN_MOVE_PCT):
        direction = "down"
    else:
        return None

    # Recent reversal check — last 60s should not be against direction
    reversal_start = now_ts - CONT_REVERSAL_BLOCK_SECONDS
    btc_recent = [p for t, p in btc_hist if t >= reversal_start]
    if len(btc_recent) >= 2:
        recent_move = (btc_recent[-1] - btc_recent[0]) / btc_recent[0]
        if (direction == "up" and recent_move < -0.001) or \
           (direction == "down" and recent_move > 0.001):
            return None  # reversal

    return {
        "direction": direction,
        "btc_move": btc_move,
        "eth_move": eth_move,
        "mode": "continuation",
    }
```

- [ ] **Step 4: Call evaluator every 60s, log would-fire events**

```python
if int(now_ts) % 60 == 0 and int(now_ts) != last_cont_eval:
    last_cont_eval = int(now_ts)
    cont_signal = evaluate_continuation(rolling_prices, now_ts)
    if cont_signal:
        logger.info("CONT_SHADOW_FIRE",
                    period=current_period,
                    elapsed=elapsed,
                    direction=cont_signal["direction"],
                    btc_move=round(cont_signal["btc_move"] * 100, 3),
                    eth_move=round(cont_signal["eth_move"] * 100, 3),
                    would_trade=ENABLE_CONTINUATION_LIVE)
        # If ENABLE_CONTINUATION_LIVE, we'd place the order here.
        # For Phase 2, keep False → observation only.
```

- [ ] **Step 5: Commit and deploy shadow**

```bash
git add run_sniper.py
git commit -m "feat(sniper-v2): continuation mode shadow evaluator — log-only"
```

### Task 7: Analyze shadow data after 72h

**Files:**
- Create: `scripts/analyze_continuation_shadow.py`

- [ ] **Step 1: Parse `CONT_SHADOW_FIRE` events from log**

Count fires, compute would-be fill price at `raw + $0.01` (market take assumption), check 15-min resolution direction.

- [ ] **Step 2: Report**

```
Continuation shadow — 72h:
  Fires: X
  Hypothetical win rate: Y%
  Estimated EV per trade: $Z
```

**Go-live gate:** EV > $0.50/trade AND win rate ≥65% → enable live in Phase 3.

---

## Phase 3: Enable Continuation Live (conditional on Phase 2 data)

### Task 8: Flip `ENABLE_CONTINUATION_LIVE` to True + FAK entry

Only execute if Phase 2 metrics meet go-live gate.

---

## Observability Enhancements (applies throughout)

All SNIPER_TRADE log entries carry:
- `strategy_mode`: correlated_burst / solo_major / continuation / certainty
- `aligned_count`: number of confirming coins
- `size_mult`: multiplier applied

Dashboard `src/main.py` should surface:
- Trades-per-mode counts
- Win rate by mode
- PnL by mode

(Dashboard work deferred — not blocking trading logic.)

---

## Rollback

If Phase 1 produces >20 fires/day or win rate <50% after 72h:

```bash
ssh root@72.62.78.141 "cd /opt/polymarket-bot && git revert <phase-1-commit> && pkill -f run_sniper.py && nohup venv/bin/python3 -u run_sniper.py 604800 >> sniper_output.log 2>&1 &"
```

Pre-v2 behavior preserved by reverting; state files (bankroll, trade history) are not affected.

---

## Summary for Reviewer

| Change | File | Risk | Reversible |
|---|---|---|---|
| Coin tiers | run_sniper.py | Low | Yes |
| Correlated burst gate | run_sniper.py | Med — new logic path | Yes |
| Solo major fallback | run_sniper.py | Low | Yes |
| Per-cycle eval logging | run_sniper.py | None | Yes |
| Certainty Kelly cap 15% | run_certainty_sniper.py | Low — reduces risk | Yes |
| Continuation shadow | run_sniper.py (Phase 2) | None (no trades) | Yes |
| Continuation live | run_sniper.py (Phase 3) | High — gated on data | Yes |

**Backtest first. Phase 1 only. Shadow Phase 2. Data-gate Phase 3.**
