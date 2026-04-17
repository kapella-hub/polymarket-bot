# Continuation Mode v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Dispatch Team A, Team B in parallel. Team C waits for Team A to merge.

**Goal:** Capture mid-period drift moves the current 90s-boundary sniper misses. Shadow-first, conservatively sized, BTC+ETH only. Ship alongside certainty Kelly cap (urgent bug fix).

**Context:** Live Apr 17 evidence shows 2+ periods per hour where BTC drifts ≥0.30% mid-period but misses both the 90s snipe window and the certainty 12-13min gate. See `memory/project_continuation_evidence.md`.

**Architecture:** Add second signal path to `run_sniper.py` that runs elapsed 240-600s, requires BTC+ETH same-direction rolling-5min moves, guards against reversals, fires FAK at 0.55× base size. Gated behind ENABLE_CONTINUATION_SHADOW (default on) and ENABLE_CONTINUATION_MODE (default off).

---

## Team Split

| Team | Files Owned | Blocking | Target |
|---|---|---|---|
| **A — Continuation** | `run_sniper.py`, `tests/test_sniper_continuation.py` | None | ship shadow |
| **B — Kelly Cap** | `run_certainty_sniper.py`, `tests/test_certainty_kelly.py` | None | ship fix |
| **C — Coin Tiers** | `run_sniper.py` | Team A merge | Phase 2 |

Team A and B can dispatch in parallel — they touch different files. Team C waits.

---

# Team A — Continuation Mode v1

## Task A0: Backtest calibration (non-skippable)

**Files:**
- Create: `scripts/backtest_continuation_v1.py`
- Create: `data/backtest_klines_72h.json` (cached Binance data)

- [ ] **Step 1: Fetch 72h of 1-minute klines** for BTC, ETH, SOL from Binance public API. Save as `data/backtest_klines_72h.json`.

```python
import httpx, json, time
from pathlib import Path

def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    all_bars = []
    cur = start_ms
    while cur < end_ms:
        r = httpx.get("https://api.binance.com/api/v3/klines",
                      params={"symbol": symbol, "interval": "1m",
                              "startTime": cur, "endTime": end_ms, "limit": 1000})
        bars = r.json()
        if not bars:
            break
        all_bars.extend(bars)
        cur = bars[-1][0] + 60_000
    return all_bars

end_ms = int(time.time() * 1000)
start_ms = end_ms - 72 * 3600 * 1000
data = {s: fetch_klines(s, start_ms, end_ms) for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
Path("data/backtest_klines_72h.json").write_text(json.dumps(data))
```

- [ ] **Step 2: Simulate Continuation Mode v1 on every 15-min period in the dataset.**

For each period, for each candidate evaluation tick (elapsed 240-600s in 60s steps):
1. Compute 5-min rolling BTC and ETH moves from `price_now - price_5min_ago`
2. Check gates: BTC ≥0.30%, ETH ≥0.25%, same direction
3. Check reversal: `(max(prices_last_60s) - price_now) / price_now` for UP, mirror for DOWN; block if ≥0.12%
4. If all pass → record would-fire event with direction, entry_price estimate (use `close + 0.01` as take price)
5. Resolution: compare `price_at_period_end` vs `price_at_entry` → win if same direction as bet

- [ ] **Step 3: Report structured metrics**

```
=== CONTINUATION MODE v1 BACKTEST — 72h ===
Periods evaluated: 288
Would-fire events:   X
  UP direction:      X1
  DOWN direction:    X2
Fires per day:       X/3.0
Hypothetical win rate: Y%
Avg entry price:     $Z
Estimated EV/trade:  $W (after 2% fee)

Skip reasons breakdown:
  btc_move_too_small:      N1
  eth_move_too_small:      N2
  direction_mismatch:      N3
  reversal_block:          N4
```

**Go/no-go gates:**
- Fire rate ≤10/day AND win rate ≥60% → proceed to Task A1
- Fire rate >15/day OR win rate <55% → STOP, surface the data, discuss threshold retuning before writing any bot code

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest_continuation_v1.py data/backtest_klines_72h.json
git commit -m "backtest: validate Continuation Mode v1 thresholds on 72h klines"
```

---

## Task A1: Add config constants

**Files:**
- Modify: `run_sniper.py` (constants section, ~line 65)

- [ ] **Step 1: Append after existing constants**

```python
# --- Continuation Mode v1 ---
# Captures mid-period BTC+ETH same-direction drift missed by boundary sniper.
# Default: shadow log only (no trades). Flip ENABLE_CONTINUATION_MODE to True for live.
ENABLE_CONTINUATION_MODE            = False   # Live trading flag
ENABLE_CONTINUATION_SHADOW          = True    # Log-only evaluation
CONTINUATION_ACTIVE_START_SEC       = 240     # Minute 4
CONTINUATION_ACTIVE_END_SEC         = 600     # Minute 10 (≥5min hold before period end)
CONTINUATION_LOOKBACK_SEC           = 300     # 5-min rolling
CONTINUATION_BTC_MIN_MOVE_PCT       = 0.003   # 0.30%
CONTINUATION_ETH_MIN_MOVE_PCT       = 0.0025  # 0.25% (raised from 0.20% — ETH noisier than BTC)
CONTINUATION_REVERSAL_WINDOW_SEC    = 60
CONTINUATION_REVERSAL_BLOCK_PCT     = 0.0012  # 0.12% peak-to-current counter-move
CONTINUATION_SIZE_MULT              = 0.55
CONTINUATION_MAX_TRADES_PER_PERIOD  = 1
```

- [ ] **Step 2: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper): continuation mode v1 config constants"
```

## Task A2: Add rolling price history tracking

**Files:**
- Modify: `run_sniper.py` (main loop, add data structure and update logic)

- [ ] **Step 1: Add import at top of file**

```python
from collections import deque
```

- [ ] **Step 2: Initialize in main() before the while loop**

```python
# Rolling price history for continuation mode — (ts, price) tuples per asset
# Store enough for max lookback + reversal window + buffer
_rolling_max_age = CONTINUATION_LOOKBACK_SEC + CONTINUATION_REVERSAL_WINDOW_SEC + 30
rolling_prices: dict[str, deque] = {a: deque() for a in COIN_TO_ASSET.values()}

# Per-period continuation state
continuation_trades_this_period = 0
last_continuation_period = 0
```

- [ ] **Step 3: Update rolling_prices each iteration of the main loop (inside while)**

```python
# Update rolling price history
for asset in COIN_TO_ASSET.values():
    p = ws_feed.get(asset)
    if p:
        rolling_prices[asset].append((now_f, p))
        # Prune beyond max age
        cutoff = now_f - _rolling_max_age
        while rolling_prices[asset] and rolling_prices[asset][0][0] < cutoff:
            rolling_prices[asset].popleft()

# Reset per-period continuation counter at new period
if period_ts != last_continuation_period:
    continuation_trades_this_period = 0
    last_continuation_period = period_ts
```

- [ ] **Step 4: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper): rolling price history for continuation mode"
```

## Task A3: Helper functions for continuation evaluation

**Files:**
- Modify: `run_sniper.py` (add helpers above main())

- [ ] **Step 1: Add helpers**

```python
def _get_price_at(history: deque, target_ts: float) -> float | None:
    """Return price closest to but not after target_ts, or None if no data that old."""
    best = None
    for ts, price in history:
        if ts <= target_ts:
            best = price
        else:
            break
    return best


def _compute_move_pct(history: deque, now_ts: float, lookback_sec: int) -> float | None:
    """Rolling % move from lookback_sec ago to now. None if insufficient data."""
    if len(history) < 2:
        return None
    p_old = _get_price_at(history, now_ts - lookback_sec)
    p_now = history[-1][1]
    if p_old is None or p_old <= 0:
        return None
    return (p_now - p_old) / p_old


def _reversal_counter_move(history: deque, now_ts: float, window_sec: int,
                           direction: str) -> float:
    """Peak-to-current counter-move magnitude over window_sec.
    For UP direction: how far below recent peak we are — (max - now) / now.
    For DOWN direction: how far above recent trough we are — (now - min) / now.
    Returns non-negative magnitude (0.0 if no counter-move seen)."""
    if not history:
        return 0.0
    cutoff = now_ts - window_sec
    recent = [p for ts, p in history if ts >= cutoff]
    if not recent:
        return 0.0
    p_now = recent[-1]
    if direction == "up":
        p_peak = max(recent)
        return max(0.0, (p_peak - p_now) / p_now)
    else:  # down
        p_trough = min(recent)
        return max(0.0, (p_now - p_trough) / p_now)
```

- [ ] **Step 2: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper): continuation mode helper functions"
```

## Task A4: Continuation evaluator — main logic

**Files:**
- Modify: `run_sniper.py` (add evaluator call in main loop after existing burst logic)

- [ ] **Step 1: Add evaluator function above main()**

```python
async def evaluate_continuation(
    http: httpx.AsyncClient,
    clob: ClobClient,
    loop,
    ws_feed: BinanceWSFeed,
    rolling_prices: dict[str, deque],
    current_markets: dict[str, dict],
    period_ts: int,
    elapsed: int,
    now_f: float,
    bankroll: float,
    state: dict,
    open_trades: list,
    continuation_trades_this_period: int,
    skip_signals: int,
) -> tuple[int, float]:
    """Evaluate continuation mode. Returns (new_count, new_bankroll).

    Emits CONT_CHECK / CONT_SHADOW_FIRE / CONT_LIVE_FIRE / CONT_SKIP events.
    Only fires on BTC+ETH same-direction 5min rolling moves, past the 90s burst window.
    """
    if not (ENABLE_CONTINUATION_MODE or ENABLE_CONTINUATION_SHADOW):
        return continuation_trades_this_period, bankroll

    # Gate: active window
    if elapsed < CONTINUATION_ACTIVE_START_SEC or elapsed > CONTINUATION_ACTIVE_END_SEC:
        return continuation_trades_this_period, bankroll

    # Gate: per-period trade cap
    if continuation_trades_this_period >= CONTINUATION_MAX_TRADES_PER_PERIOD:
        logger.debug("CONT_SKIP", reason="already_traded_this_period",
                     period=period_ts, elapsed=elapsed)
        return continuation_trades_this_period, bankroll

    # Gate: circuit breaker
    if skip_signals > 0:
        logger.debug("CONT_SKIP", reason="circuit_breaker",
                     period=period_ts, elapsed=elapsed)
        return continuation_trades_this_period, bankroll

    btc_hist = rolling_prices.get("BTC", deque())
    eth_hist = rolling_prices.get("ETH", deque())
    btc_move = _compute_move_pct(btc_hist, now_f, CONTINUATION_LOOKBACK_SEC)
    eth_move = _compute_move_pct(eth_hist, now_f, CONTINUATION_LOOKBACK_SEC)

    if btc_move is None or eth_move is None:
        return continuation_trades_this_period, bankroll  # still warming up

    logger.debug("CONT_CHECK", period=period_ts, elapsed=elapsed,
                 btc_move=round(btc_move * 100, 3),
                 eth_move=round(eth_move * 100, 3))

    # Gate: BTC magnitude
    if abs(btc_move) < CONTINUATION_BTC_MIN_MOVE_PCT:
        logger.info("CONT_SKIP", reason="btc_move_too_small",
                    period=period_ts, elapsed=elapsed,
                    btc_move=round(btc_move * 100, 3))
        return continuation_trades_this_period, bankroll

    # Gate: ETH magnitude
    if abs(eth_move) < CONTINUATION_ETH_MIN_MOVE_PCT:
        logger.info("CONT_SKIP", reason="eth_move_too_small",
                    period=period_ts, elapsed=elapsed,
                    eth_move=round(eth_move * 100, 3))
        return continuation_trades_this_period, bankroll

    # Gate: same direction
    if (btc_move > 0) != (eth_move > 0):
        logger.info("CONT_SKIP", reason="direction_mismatch",
                    period=period_ts, elapsed=elapsed,
                    btc_move=round(btc_move * 100, 3),
                    eth_move=round(eth_move * 100, 3))
        return continuation_trades_this_period, bankroll

    direction = "up" if btc_move > 0 else "down"

    # Gate: reversal block
    btc_rev = _reversal_counter_move(btc_hist, now_f,
                                      CONTINUATION_REVERSAL_WINDOW_SEC, direction)
    if btc_rev >= CONTINUATION_REVERSAL_BLOCK_PCT:
        logger.info("CONT_SKIP", reason="reversal_block",
                    period=period_ts, elapsed=elapsed,
                    direction=direction,
                    reversal_pct=round(btc_rev * 100, 3))
        return continuation_trades_this_period, bankroll

    # We trade BTC for continuation (cleanest signal, deepest liquidity)
    coin = "btc"
    market = current_markets.get(coin)
    if not market:
        logger.info("CONT_SKIP", reason="no_market", period=period_ts, elapsed=elapsed)
        return continuation_trades_this_period, bankroll

    side = "buy_up" if direction == "up" else "buy_down"
    token_id = market["up_token_id"] if direction == "up" else market["down_token_id"]

    # Gate: CLOB depth + price cap
    best_ask, ask_usd_vol = await get_clob_ask(http, token_id)
    if ask_usd_vol < MIN_BOOK_VOLUME:
        logger.info("CONT_SKIP", reason="depth_fail",
                    period=period_ts, elapsed=elapsed,
                    ask_vol_usd=round(ask_usd_vol, 1))
        return continuation_trades_this_period, bankroll

    entry_price = best_ask if best_ask > 0 else (
        market["up_price"] if direction == "up" else market["down_price"])
    if entry_price > MAX_ENTRY_PRICE:
        logger.info("CONT_SKIP", reason="price_cap_fail",
                    period=period_ts, elapsed=elapsed,
                    entry_price=round(entry_price, 4))
        return continuation_trades_this_period, bankroll

    # Gate: max concurrent
    open_count = sum(1 for t in open_trades if t.get("status") == "placed")
    if open_count >= MAX_CONCURRENT:
        logger.info("CONT_SKIP", reason="max_concurrent_reached",
                    period=period_ts, elapsed=elapsed,
                    open_count=open_count)
        return continuation_trades_this_period, bankroll

    # Gate: size floor
    base_size = min(BET_MAX, max(BET_MIN, bankroll * BET_PCT))
    size = base_size * CONTINUATION_SIZE_MULT
    if size < BET_MIN:
        logger.info("CONT_SKIP", reason="size_below_min",
                    period=period_ts, elapsed=elapsed,
                    computed_size=round(size, 2))
        return continuation_trades_this_period, bankroll
    size = min(size, bankroll - 5)
    if size < BET_MIN:
        logger.info("CONT_SKIP", reason="bankroll_guard",
                    period=period_ts, elapsed=elapsed)
        return continuation_trades_this_period, bankroll

    tokens = size / entry_price
    available_tokens = ask_usd_vol / entry_price
    tokens = min(tokens, available_tokens)
    actual_size = round(tokens * entry_price, 2)

    # Shadow or live?
    if not ENABLE_CONTINUATION_MODE:
        logger.info("CONT_SHADOW_FIRE",
                    period=period_ts, elapsed=elapsed,
                    direction=direction, coin=coin,
                    btc_move=round(btc_move * 100, 3),
                    eth_move=round(eth_move * 100, 3),
                    entry_price=round(entry_price, 4),
                    size=actual_size,
                    ask_vol=round(ask_usd_vol, 1))
        return continuation_trades_this_period + 1, bankroll

    # LIVE: place FAK
    try:
        order_args = OrderArgs(
            token_id=token_id, price=round(entry_price, 4),
            size=round(tokens, 2), side=BUY,
        )
        signed = await loop.run_in_executor(None, clob.create_order, order_args)
        result = await loop.run_in_executor(
            None, functools.partial(clob.post_order, signed, OrderType.FAK))
        order_id = result.get("orderID", "")

        if order_id:
            bankroll -= actual_size
            state["bankroll"] = bankroll
            state["total_invested"] = state.get("total_invested", 0) + actual_size
            trade = {
                "period": period_ts, "coin": coin, "side": side,
                "token_id": token_id, "condition_id": market["condition_id"],
                "entry_price": round(entry_price, 4), "size_usd": actual_size,
                "tokens": round(tokens, 2), "order_id": order_id,
                "placed_at": now_f, "period_end": period_ts + 900,
                "move_pct": round(btc_move * 100, 3),
                "cross_confirm": 2,  # BTC + ETH
                "btc_at_entry": ws_feed.get("BTC"),
                "strategy_mode": "continuation",
                "status": "placed", "pnl": 0,
            }
            open_trades.append(trade)
            state["trades"].append(trade)
            save_state(state)
            logger.info("CONT_LIVE_FIRE",
                        period=period_ts, elapsed=elapsed,
                        direction=direction, coin=coin,
                        btc_move=round(btc_move * 100, 3),
                        eth_move=round(eth_move * 100, 3),
                        entry_price=round(entry_price, 4),
                        size=actual_size, order_id=order_id[:16],
                        bankroll=round(bankroll, 2))
            return continuation_trades_this_period + 1, bankroll
        else:
            logger.warning("CONT_SKIP", reason="no_order_id",
                           period=period_ts, resp=str(result)[:120])
    except Exception as e:
        logger.error("CONT_SKIP", reason="order_error",
                     period=period_ts, error=str(e)[:120])

    return continuation_trades_this_period, bankroll
```

- [ ] **Step 2: Invoke evaluator in main loop AFTER existing burst logic**

Place after the snipe window block and before the resolve block. Call every iteration (throttled by active-window gate internally).

```python
continuation_trades_this_period, bankroll = await evaluate_continuation(
    http, clob, loop, ws_feed, rolling_prices, current_markets,
    period_ts, elapsed, now_f, bankroll, state, open_trades,
    continuation_trades_this_period, skip_signals,
)
```

- [ ] **Step 3: Commit**

```bash
git add run_sniper.py
git commit -m "feat(sniper): continuation mode v1 evaluator — shadow + live gated paths"
```

## Task A5: Tests

**Files:**
- Create: `tests/test_sniper_continuation.py`

- [ ] **Step 1: Write tests using the helper functions directly**

```python
from collections import deque
from run_sniper import (
    _compute_move_pct, _reversal_counter_move,
    CONTINUATION_LOOKBACK_SEC, CONTINUATION_REVERSAL_WINDOW_SEC,
)


def _mk_history(ts_start: float, prices: list[float], step: float = 1.0) -> deque:
    return deque((ts_start + i * step, p) for i, p in enumerate(prices))


def test_move_pct_up_drift():
    # 100 → 100.4 over 300s = +0.40%
    now = 1000.0
    prices = [100.0] * 10 + [100.4] * 10  # older → newer
    # Build deque: timestamps span now-300 to now
    hist = deque()
    for i, p in enumerate(prices):
        ts = now - 300 + i * (300 / len(prices))
        hist.append((ts, p))
    move = _compute_move_pct(hist, now, 300)
    assert abs(move - 0.004) < 1e-4


def test_move_pct_insufficient_data():
    hist = deque([(1000.0, 100.0)])
    assert _compute_move_pct(hist, 1000.0, 300) is None


def test_reversal_up_peak_bounce():
    # Price went 100 → 100.3 → 100.0 in last 60s; intended UP entry
    # peak=100.3, now=100.0 → counter-move = 0.3%
    now = 2000.0
    hist = deque([(now - 60, 100.0), (now - 30, 100.3), (now, 100.0)])
    rev = _reversal_counter_move(hist, now, 60, "up")
    assert abs(rev - 0.003) < 1e-4


def test_reversal_down_trough_bounce():
    # Price went 100 → 99.7 → 100.0 in last 60s; intended DOWN entry
    # trough=99.7, now=100.0 → counter-move = 0.3%
    now = 2000.0
    hist = deque([(now - 60, 100.0), (now - 30, 99.7), (now, 100.0)])
    rev = _reversal_counter_move(hist, now, 60, "down")
    assert abs(rev - 0.003) < 1e-4


def test_reversal_no_counter_move():
    # Pure UP drift; no counter-move for UP direction
    now = 2000.0
    hist = deque([(now - 60, 100.0), (now - 30, 100.1), (now, 100.2)])
    rev = _reversal_counter_move(hist, now, 60, "up")
    assert rev == 0.0


def test_config_constants_sane():
    from run_sniper import (
        ENABLE_CONTINUATION_MODE, ENABLE_CONTINUATION_SHADOW,
        CONTINUATION_ACTIVE_START_SEC, CONTINUATION_ACTIVE_END_SEC,
        CONTINUATION_BTC_MIN_MOVE_PCT, CONTINUATION_ETH_MIN_MOVE_PCT,
        CONTINUATION_SIZE_MULT, CONTINUATION_MAX_TRADES_PER_PERIOD,
    )
    assert ENABLE_CONTINUATION_MODE is False, "live must default off"
    assert ENABLE_CONTINUATION_SHADOW is True
    assert CONTINUATION_ACTIVE_START_SEC == 240
    assert CONTINUATION_ACTIVE_END_SEC == 600
    assert CONTINUATION_BTC_MIN_MOVE_PCT == 0.003
    assert CONTINUATION_ETH_MIN_MOVE_PCT == 0.0025
    assert 0.4 <= CONTINUATION_SIZE_MULT <= 0.7
    assert CONTINUATION_MAX_TRADES_PER_PERIOD == 1
```

- [ ] **Step 2: Run tests**

```bash
cd E:/Documents/Projects/polymarket-bot
venv/Scripts/python.exe -m pytest tests/test_sniper_continuation.py -v
```

All 6 tests must pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sniper_continuation.py
git commit -m "test(sniper): continuation mode helpers + config sanity"
```

## Task A6: Deploy shadow to VPS

- [ ] **Step 1: Push to git**

```bash
git push origin main
```

- [ ] **Step 2: Pull on VPS and restart sniper (NOT certainty — Team B handles that)**

```bash
ssh root@72.62.78.141 "
  pkill -f 'run_sniper.py' || true
  cd /opt/polymarket-bot && git pull origin main
  nohup venv/bin/python3 -u run_sniper.py 604800 >> sniper_output.log 2>&1 &
  sleep 3
  ps aux | grep run_sniper | grep -v grep
"
```

- [ ] **Step 3: Verify shadow events appear in log within 20 minutes**

```bash
ssh root@72.62.78.141 "sleep 1200 && grep -E 'CONT_CHECK|CONT_SHADOW_FIRE|CONT_SKIP' /opt/polymarket-bot/sniper_output.log | tail -20"
```

Expected: a mix of CONT_SKIP events (most periods fail some gate), occasional CONT_SHADOW_FIRE when BTC+ETH drift aligns.

---

# Team B — Certainty Kelly Cap

## Task B1: Add hard 15% bankroll cap

**Files:**
- Modify: `run_certainty_sniper.py:60-76`

- [ ] **Step 1: Add constant + modify kelly_size**

Replace lines 60-76:

```python
# Sizing
BET_MIN              = 8.0
BET_MAX              = 40.0
MAX_BET_PCT_BANKROLL = 0.15   # Hard ceiling — guards against over-optimistic win_rate


def kelly_size(bankroll: float, win_rate: float = 0.90, avg_entry: float = 0.85) -> float:
    """Half-Kelly with hard 15% bankroll cap. Protects against mis-estimated win_rate."""
    b = (1.0 - avg_entry) / avg_entry
    kelly = (win_rate * b - (1.0 - win_rate)) / b
    size = bankroll * (kelly / 2.0)
    pct_cap = bankroll * MAX_BET_PCT_BANKROLL
    size = min(size, pct_cap)
    return min(BET_MAX, max(BET_MIN, round(size, 2)))
```

## Task B2: Tests

**Files:**
- Create: `tests/test_certainty_kelly.py`

- [ ] **Step 1: Write tests**

```python
from run_certainty_sniper import kelly_size, BET_MIN, BET_MAX, MAX_BET_PCT_BANKROLL


def test_kelly_pct_cap_binds_at_50():
    # 15% of $50 = $7.50, but BET_MIN=$8 floors it
    assert kelly_size(50.0) == BET_MIN


def test_kelly_pct_cap_binds_at_100():
    # 15% of $100 = $15, should be binding (well under BET_MAX)
    result = kelly_size(100.0)
    assert result <= 15.0
    assert result >= BET_MIN


def test_kelly_bet_max_floor_at_large_bankroll():
    # 15% of $1000 = $150, but BET_MAX=$40 caps
    assert kelly_size(1000.0) == BET_MAX


def test_kelly_with_pessimistic_win_rate_still_capped():
    # Even at 99% win_rate on $100, cap = $15
    assert kelly_size(100.0, win_rate=0.99) <= 15.0


def test_kelly_constants_sane():
    assert MAX_BET_PCT_BANKROLL == 0.15
    assert BET_MIN == 8.0
    assert BET_MAX == 40.0
```

- [ ] **Step 2: Run tests**

```bash
venv/Scripts/python.exe -m pytest tests/test_certainty_kelly.py -v
```

## Task B3: Commit and deploy

- [ ] **Step 1: Commit**

```bash
git add run_certainty_sniper.py tests/test_certainty_kelly.py
git commit -m "fix(certainty): cap Kelly at 15% of bankroll — prevent over-levering

Previously: $50 bankroll could place a $40 trade (80% of capital) on the
assumed 90% win rate. If actual win rate is 70% the sizing is 3x over.
Now hard-capped at 15% of bankroll before BET_MAX/BET_MIN clamps."
```

- [ ] **Step 2: Deploy**

```bash
git push origin main

ssh root@72.62.78.141 "
  pkill -f 'run_certainty_sniper.py' || true
  cd /opt/polymarket-bot && git pull origin main
  nohup venv/bin/python3 -u run_certainty_sniper.py 604800 >> certainty_sniper_output.log 2>&1 &
  sleep 3
  ps aux | grep run_certainty | grep -v grep
"
```

---

# Team C — Coin Tiers (DEFERRED)

**Blocking:** Must wait for Team A to merge, then rebase.

**Scope:** Prevent DOGE/XRP/BNB from solo-triggering burst mode. Allow SOL only with confirmation. BTC/ETH solo-trigger stays.

**Estimated size:** ~30 lines in run_sniper.py burst logic. Separate plan file to be written after Team A lands.

---

# Acceptance Criteria

**Team A:**
- Backtest completed with fire-rate ≤10/day AND win-rate ≥60% (else STOP)
- All 6 pytest tests pass
- `CONT_SHADOW_FIRE` or `CONT_SKIP` events appear in VPS log within 20 min of deploy
- `ENABLE_CONTINUATION_MODE=False` by default — zero live trades

**Team B:**
- `kelly_size(50.0)` returns exactly $8.00
- `kelly_size(1000.0)` returns exactly $40.00
- All 5 pytest tests pass
- Certainty sniper running on VPS with new code

**Global:**
- Existing 90s boundary sniper behavior unchanged
- Existing FAK/CLOB-depth/circuit-breaker guards all still active
- No new dependencies added

---

# Rollout / Monitoring

**Week 1 (post-deploy):**
- Collect 7 days of `CONT_SHADOW_FIRE` events + period resolutions
- Write `scripts/analyze_continuation_shadow.py` → report fire count, hypothetical win rate, estimated EV/trade

**Week 2 (go-live decision):**
- If shadow EV > $0.50/trade AND win rate ≥ 60%: flip `ENABLE_CONTINUATION_MODE = True`
- If below: retune thresholds (likely raise BTC cutoff to 0.35%) and restart 7-day shadow window
- If way below (win <50%): abandon continuation mode

**Rollback:**
```bash
ssh root@72.62.78.141 "cd /opt/polymarket-bot && git revert HEAD~1 && pkill -f run_sniper.py && nohup venv/bin/python3 -u run_sniper.py 604800 >> sniper_output.log 2>&1 &"
```

State files (bankroll, trade history) unaffected by revert.
