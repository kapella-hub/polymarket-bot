#!/usr/bin/env python3
"""
Multi-Coin Momentum Sniper

Strategy: Within the first 90 seconds of each 15-minute period, if any
coin moves ≥0.20% from period start, buy the SAME direction at $0.52 limit.
Cancel unfilled orders after 25s. Max 3 concurrent positions.

Why it works:
  - At period boundaries, Polymarket books reset to ~$0.50/$0.50
  - Binance WS prices arrive in <10ms — we see the move before market makers
  - At $0.52 entry, break-even win rate is ~53% (real ~2% Polymarket fee)
  - 6 coins × 96 periods/day = 576 opportunities/day

Edge stack:
  - Sub-100ms Binance WS feed (vs Polymarket's slower oracle repricing)
  - $0.52 limit (vs market repricing to $0.55-0.65 within minutes)
  - 25s cancel (vs 10s) — better fill rate, still before repricing completes
  - Cross-coin confirmation: 2+ coins same direction → 1.5x size
  - Circuit breaker: 3 consecutive losses → pause 4 periods
"""

import asyncio
import functools
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import httpx
import structlog

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from src.crypto_arb.ws_feeds import BinanceWSFeed

logger = structlog.get_logger()

LOG_FILE   = Path(__file__).parent / "sniper_output.log"
STATE_FILE = Path(__file__).parent / "data" / "sniper_state.json"

COINS         = ["btc", "eth", "sol", "xrp", "doge", "bnb"]
COIN_TO_ASSET = {"btc": "BTC", "eth": "ETH", "sol": "SOL",
                 "xrp": "XRP", "doge": "DOGE", "bnb": "BNB"}

# Core parameters
SNIPE_WINDOW     = 90     # Fire within first N seconds of period start
MIN_MOVE_PCT     = 0.004  # 0.40% minimum move to signal (stronger conviction)
MAX_ENTRY_PRICE  = 0.72   # Skip if best ask > this (EV floor: need <73.5% break-even)
MIN_BOOK_VOLUME  = 20.0   # Min USD resting at best ask before placing
BET_PCT          = 0.08   # 8% bankroll per signal
BET_MIN          = 8.0
BET_MAX          = 25.0
MAX_CONCURRENT   = 3
CROSS_BOOST      = 1.5    # Size multiplier when 2+ coins confirm direction
CIRCUIT_LOSSES   = 3      # Consecutive losses before pause
CIRCUIT_SKIP     = 4      # Periods to skip after trigger

# --- Continuation Mode v1 ---
# Captures mid-period BTC+ETH same-direction drift missed by boundary sniper.
# Backtested 72h on 2026-04-17: 12 fires/day, 91.7% win rate, +$0.25 EV/trade.
ENABLE_CONTINUATION_MODE            = True    # LIVE trading enabled
ENABLE_CONTINUATION_SHADOW          = False   # Redundant when live is on
CONTINUATION_ACTIVE_START_SEC       = 240     # Minute 4
CONTINUATION_ACTIVE_END_SEC         = 600     # Minute 10 (>=5min hold before period end)
CONTINUATION_LOOKBACK_SEC           = 300     # 5-min rolling
CONTINUATION_BTC_MIN_MOVE_PCT       = 0.003   # 0.30%
CONTINUATION_ETH_MIN_MOVE_PCT       = 0.0025  # 0.25%
CONTINUATION_REVERSAL_WINDOW_SEC    = 60
CONTINUATION_REVERSAL_BLOCK_PCT     = 0.0012  # 0.12% peak-to-current counter-move
CONTINUATION_SIZE_MULT              = 0.55
CONTINUATION_MAX_TRADES_PER_PERIOD  = 1
# Separate price cap for continuation — fires AFTER market move, so entry naturally higher.
# Break-even at 0.88 = 89.8% WR; backtest 91.7% keeps positive EV margin.
CONTINUATION_MAX_ENTRY_PRICE        = 0.88
# Lower depth floor for continuation: position size is 0.55x (~$8-13), not burst's $8-25.
# Observed 2026-04-18 11:40 UTC: BTC -0.378%, ETH -0.498% clean signal blocked by $20 floor.
CONTINUATION_MIN_BOOK_VOLUME        = 10.0


def _get_price_at(history: deque, target_ts: float) -> float | None:
    best = None
    for ts, price in history:
        if ts <= target_ts:
            best = price
        else:
            break
    return best


def _compute_move_pct(history: deque, now_ts: float, lookback_sec: int) -> float | None:
    if len(history) < 2:
        return None
    p_old = _get_price_at(history, now_ts - lookback_sec)
    p_now = history[-1][1]
    if p_old is None or p_old <= 0:
        return None
    return (p_now - p_old) / p_old


def _reversal_counter_move(history: deque, now_ts: float, window_sec: int,
                           direction: str) -> float:
    if not history:
        return 0.0
    cutoff = now_ts - window_sec
    recent = [p for ts, p in history if ts >= cutoff]
    if not recent:
        return 0.0
    p_now = recent[-1]
    if p_now <= 0:
        return 0.0
    if direction == "up":
        return max(0.0, (max(recent) - p_now) / p_now)
    return max(0.0, (p_now - min(recent)) / p_now)


def create_clob():
    creds = ApiCreds(
        api_key=os.getenv('PM_POLYMARKET_API_KEY'),
        api_secret=os.getenv('PM_POLYMARKET_API_SECRET'),
        api_passphrase=os.getenv('PM_POLYMARKET_API_PASSPHRASE'),
    )
    clob = ClobClient(
        'https://clob.polymarket.com',
        key=os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY'),
        chain_id=137, signature_type=0,
    )
    clob.set_api_creds(creds)
    return clob


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"bankroll": 0.0, "trades": [], "total_invested": 0.0, "total_returned": 0.0}


def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def fetch_market(client: httpx.AsyncClient, coin: str, period_ts: int) -> dict | None:
    """Fetch one coin's market data for a given period."""
    slug = f"{coin}-updown-15m-{period_ts}"
    try:
        resp = await client.get(
            f"https://gamma-api.polymarket.com/markets?slug={slug}",
            timeout=8,
        )
        data = resp.json()
        if not data:
            return None
        m = data[0]

        tokens   = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
        outcomes = json.loads(m["outcomes"])      if isinstance(m.get("outcomes"), str)      else m.get("outcomes", [])
        prices   = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])

        if len(tokens) < 2 or len(outcomes) < 2:
            return None

        up_idx   = next((i for i, o in enumerate(outcomes) if o.lower() == "up"),   0)
        down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "down"), 1)

        return {
            "coin":         coin,
            "condition_id": m.get("conditionId", ""),
            "up_token_id":  str(tokens[up_idx]),
            "down_token_id": str(tokens[down_idx]),
            "up_price":     float(prices[up_idx]),
            "down_price":   float(prices[down_idx]),
            "period_start": period_ts,
            "period_end":   period_ts + 900,
        }
    except Exception as e:
        logger.debug("fetch_market_error", coin=coin, error=str(e))
        return None


async def fetch_all_markets(client: httpx.AsyncClient, period_ts: int) -> dict[str, dict]:
    """Fetch all 6 coin markets in parallel — no sequential sleeps."""
    results = await asyncio.gather(
        *[fetch_market(client, coin, period_ts) for coin in COINS],
        return_exceptions=True,
    )
    return {
        coin: result
        for coin, result in zip(COINS, results)
        if isinstance(result, dict) and result
    }


async def get_clob_ask(client: httpx.AsyncClient, token_id: str) -> tuple[float, float]:
    """Fetch best ask price and USD volume resting at that price from CLOB.
    Returns (best_ask, usd_volume). Returns (0.0, 0.0) on failure."""
    try:
        resp = await client.get(
            f"https://clob.polymarket.com/book?token_id={token_id}",
            timeout=3,
        )
        data = resp.json()
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))
        if not asks:
            return 0.0, 0.0
        best_price = float(asks[0]["price"])
        # Aggregate tokens within one cent of best ask
        vol_tokens = sum(float(a["size"]) for a in asks
                         if float(a["price"]) <= best_price + 0.01)
        return best_price, vol_tokens * best_price
    except Exception:
        return 0.0, 0.0


async def main():
    duration       = int(sys.argv[1])   if len(sys.argv) > 1 else 604800   # 7 days
    bankroll_init  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    clob = create_clob()
    http = httpx.AsyncClient(timeout=10)
    loop = asyncio.get_event_loop()

    state = load_state()
    if bankroll_init > 0:
        state["bankroll"] = bankroll_init
        save_state(state)

    bankroll = state["bankroll"]

    print("[%s] MULTI-COIN MOMENTUM SNIPER starting" % datetime.now(timezone.utc).isoformat())
    print("  Duration:     %ds (%.1fh)" % (duration, duration / 3600))
    print("  Bankroll:     $%.2f" % bankroll)
    print("  Snipe window: first %ds of each 15-min period" % SNIPE_WINDOW)
    print("  Min move:     %.2f%%   Max entry: $%.2f   Order: FAK (dynamic price)" % (MIN_MOVE_PCT * 100, MAX_ENTRY_PRICE))
    print("  Book filter:  min $%.0f resting ask volume before placing" % MIN_BOOK_VOLUME)
    print("  Bet:          %.0f%% of bankroll ($%.0f–$%.0f), max %d concurrent" % (BET_PCT * 100, BET_MIN, BET_MAX, MAX_CONCURRENT))
    print("  Coins:        %s" % ", ".join(c.upper() for c in COINS))
    print()

    ws_feed  = BinanceWSFeed()
    ws_task  = asyncio.create_task(ws_feed.run())

    for _ in range(100):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    if not ws_feed.get("BTC"):
        logger.error("no_btc_price_timeout")
        return

    logger.info("sniper_ready", btc=ws_feed.get("BTC"), eth=ws_feed.get("ETH"))

    # Per-session state
    current_period   = 0
    current_markets: dict[str, dict] = {}

    # Historical period start prices — keep last 3 periods for resolution
    hist_start: dict[int, dict[str, float]] = {}

    # Pre-fetched markets indexed by period_ts
    prefetched: dict[int, dict[str, dict]] = {}
    prefetch_task = None

    # Trade tracking
    open_trades:    list[dict] = []   # active placed/pending trades
    fired_this_period: set[str] = set()

    # Continuation mode state
    _roll_max_age = CONTINUATION_LOOKBACK_SEC + CONTINUATION_REVERSAL_WINDOW_SEC + 30
    rolling_prices: dict[str, deque] = {a: deque() for a in COIN_TO_ASSET.values()}
    continuation_trades_this_period = 0
    last_continuation_period        = 0
    last_cont_skip_log_key: tuple | None = None  # (period_ts, elapsed_bucket) dedup

    consecutive_losses = 0
    skip_signals       = 0
    last_status        = 0
    last_skip_period   = 0  # prevents double-decrement of skip counter

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            now_ts   = int(time.time())
            period_ts = (now_ts // 900) * 900
            elapsed  = now_ts - period_ts
            remaining = 900 - elapsed

            # ──────────────────────────────────────────────────
            # NEW PERIOD
            # ──────────────────────────────────────────────────
            if period_ts != current_period:
                fired_this_period.clear()

                # Decrement circuit breaker once per period
                if skip_signals > 0 and period_ts != last_skip_period:
                    skip_signals -= 1
                    last_skip_period = period_ts
                    logger.info("CIRCUIT_BREAKER_TICK", skip_remaining=skip_signals)

                # Use pre-fetched markets if available, else fetch now in parallel
                if period_ts in prefetched:
                    current_markets = prefetched.pop(period_ts)
                    logger.info("period_start_prefetched",
                                period=period_ts, markets=len(current_markets))
                else:
                    current_markets = await fetch_all_markets(http, period_ts)
                    logger.info("period_start_fetched",
                                period=period_ts, markets=len(current_markets))

                # Record period-start Binance prices for all coins
                hist_start[period_ts] = {
                    asset: ws_feed.get(asset)
                    for asset in COIN_TO_ASSET.values()
                    if ws_feed.get(asset)
                }

                # Prune history beyond last 3 periods
                cutoff = period_ts - 4 * 900
                for old_ts in [k for k in list(hist_start) if k < cutoff]:
                    del hist_start[old_ts]
                for old_ts in [k for k in list(prefetched) if k < cutoff]:
                    del prefetched[old_ts]

                current_period = period_ts
                logger.info("period_ready",
                            period=period_ts,
                            coins_with_markets=list(current_markets.keys()),
                            btc=ws_feed.get("BTC"),
                            skip_signals=skip_signals)

            # ──────────────────────────────────────────────────
            # PRE-FETCH: next period's markets 60s before boundary
            # ──────────────────────────────────────────────────
            if remaining <= 60 and prefetch_task is None:
                next_ts = period_ts + 900
                if next_ts not in prefetched:
                    async def _do_prefetch(ts=next_ts):
                        m = await fetch_all_markets(http, ts)
                        if m:
                            prefetched[ts] = m
                            logger.info("prefetched_next_period", period=ts, count=len(m))
                    prefetch_task = asyncio.create_task(_do_prefetch())

            if prefetch_task and prefetch_task.done():
                prefetch_task = None

            # ──────────────────────────────────────────────────
            # SNIPE WINDOW: first SNIPE_WINDOW seconds
            # ──────────────────────────────────────────────────
            if 2 <= elapsed <= SNIPE_WINDOW and skip_signals == 0 and current_markets:
                open_count = sum(1 for t in open_trades if t["status"] == "placed")

                if open_count < MAX_CONCURRENT:
                    period_prices = hist_start.get(current_period, {})
                    signals = []

                    for coin, market in current_markets.items():
                        if coin in fired_this_period:
                            continue
                        asset = COIN_TO_ASSET[coin]
                        cur   = ws_feed.get(asset)
                        start = period_prices.get(asset)
                        if not cur or not start:
                            continue
                        move = (cur - start) / start
                        if abs(move) >= MIN_MOVE_PCT:
                            side = "buy_up" if move > 0 else "buy_down"
                            signals.append((coin, market, side, move, cur))

                    if signals:
                        up_count   = sum(1 for _, _, s, _, _ in signals if s == "buy_up")
                        down_count = sum(1 for _, _, s, _, _ in signals if s == "buy_down")

                        for coin, market, side, move, cur_price in signals:
                            if open_count >= MAX_CONCURRENT:
                                break

                            confirm_count = up_count if side == "buy_up" else down_count
                            boost   = CROSS_BOOST if confirm_count >= 2 else 1.0

                            token_id     = market["up_token_id"]   if side == "buy_up"   else market["down_token_id"]
                            market_price = market["up_price"]      if side == "buy_up"   else market["down_price"]

                            # Fetch live CLOB ask for dynamic pricing
                            best_ask, ask_usd_vol = await get_clob_ask(http, token_id)

                            if ask_usd_vol < MIN_BOOK_VOLUME:
                                logger.debug("sniper_skip_thin_book",
                                             coin=coin, ask_vol_usd=round(ask_usd_vol, 1))
                                fired_this_period.add(coin)
                                continue

                            entry_price = best_ask if best_ask > 0 else market_price
                            if entry_price > MAX_ENTRY_PRICE:
                                logger.info("sniper_skip_price_high",
                                            coin=coin, entry=entry_price)
                                fired_this_period.add(coin)
                                continue

                            size = min(BET_MAX, max(BET_MIN, bankroll * BET_PCT * boost))
                            size = min(size, bankroll - 5)
                            if size < BET_MIN:
                                logger.debug("insufficient_bankroll",
                                             bankroll=round(bankroll, 2), size=size)
                                continue

                            # Cap tokens to available volume so FAK fills completely
                            tokens = size / entry_price
                            available_tokens = ask_usd_vol / entry_price
                            tokens = min(tokens, available_tokens)
                            actual_size = round(tokens * entry_price, 2)

                            try:
                                order_args = OrderArgs(
                                    token_id=token_id,
                                    price=round(entry_price, 4),
                                    size=round(tokens, 2),
                                    side=BUY,
                                )
                                signed = await loop.run_in_executor(
                                    None, clob.create_order, order_args)
                                result = await loop.run_in_executor(
                                    None, functools.partial(clob.post_order, signed, OrderType.FAK))
                                order_id = result.get("orderID", "")

                                if order_id:
                                    bankroll -= actual_size
                                    state["bankroll"] = bankroll
                                    state["total_invested"] = state.get("total_invested", 0) + actual_size

                                    trade = {
                                        "period":        period_ts,
                                        "coin":          coin,
                                        "side":          side,
                                        "token_id":      token_id,
                                        "condition_id":  market["condition_id"],
                                        "entry_price":   round(entry_price, 4),
                                        "size_usd":      actual_size,
                                        "tokens":        round(tokens, 2),
                                        "order_id":      order_id,
                                        "placed_at":     time.time(),
                                        "period_end":    period_ts + 900,
                                        "move_pct":      round(move * 100, 3),
                                        "cross_confirm": confirm_count,
                                        "btc_at_entry":  ws_feed.get("BTC"),
                                        "status":        "placed",
                                        "pnl":           0,
                                    }
                                    open_trades.append(trade)
                                    state["trades"].append(trade)
                                    fired_this_period.add(coin)
                                    open_count += 1
                                    save_state(state)

                                    logger.info(
                                        "SNIPER_TRADE",
                                        coin=coin.upper(),
                                        side=side,
                                        move="%.3f%%" % (move * 100),
                                        elapsed=elapsed,
                                        size=actual_size,
                                        entry=round(entry_price, 4),
                                        tokens=round(tokens, 1),
                                        boost=("%.1fx" % boost) if boost > 1 else "1x",
                                        confirm=confirm_count,
                                        ask_vol=round(ask_usd_vol, 1),
                                        bankroll=round(bankroll, 2),
                                        order_id=order_id[:16],
                                    )
                                else:
                                    logger.warning("sniper_no_order_id",
                                                   coin=coin, resp=str(result)[:120])
                            except Exception as e:
                                logger.error("sniper_order_error",
                                             coin=coin, error=str(e)[:120])

            # ──────────────────────────────────────────────────
            # CONTINUATION MODE v1: mid-period BTC+ETH drift
            # ──────────────────────────────────────────────────
            now_f = time.time()

            # Update rolling price history every iteration
            for asset in COIN_TO_ASSET.values():
                p = ws_feed.get(asset)
                if p:
                    rolling_prices[asset].append((now_f, p))
                    cutoff = now_f - _roll_max_age
                    while rolling_prices[asset] and rolling_prices[asset][0][0] < cutoff:
                        rolling_prices[asset].popleft()

            # Reset per-period continuation counter
            if period_ts != last_continuation_period:
                continuation_trades_this_period = 0
                last_continuation_period = period_ts

            if (ENABLE_CONTINUATION_MODE or ENABLE_CONTINUATION_SHADOW) and \
               CONTINUATION_ACTIVE_START_SEC <= elapsed <= CONTINUATION_ACTIVE_END_SEC and \
               continuation_trades_this_period < CONTINUATION_MAX_TRADES_PER_PERIOD and \
               skip_signals == 0 and current_markets:

                btc_hist = rolling_prices.get("BTC", deque())
                eth_hist = rolling_prices.get("ETH", deque())
                btc_move = _compute_move_pct(btc_hist, now_f, CONTINUATION_LOOKBACK_SEC)
                eth_move = _compute_move_pct(eth_hist, now_f, CONTINUATION_LOOKBACK_SEC)

                should_eval = btc_move is not None and eth_move is not None
                skip_reason = None
                if should_eval:
                    if abs(btc_move) < CONTINUATION_BTC_MIN_MOVE_PCT:
                        skip_reason = "btc_move_too_small"
                    elif abs(eth_move) < CONTINUATION_ETH_MIN_MOVE_PCT:
                        skip_reason = "eth_move_too_small"
                    elif (btc_move > 0) != (eth_move > 0):
                        skip_reason = "direction_mismatch"

                if should_eval and skip_reason is None:
                    direction = "up" if btc_move > 0 else "down"
                    rev = _reversal_counter_move(btc_hist, now_f,
                                                  CONTINUATION_REVERSAL_WINDOW_SEC, direction)
                    if rev >= CONTINUATION_REVERSAL_BLOCK_PCT:
                        skip_reason = "reversal_block"
                    else:
                        coin = "btc"
                        market = current_markets.get(coin)
                        if not market:
                            skip_reason = "no_market"
                        else:
                            side = "buy_up" if direction == "up" else "buy_down"
                            token_id = market["up_token_id"] if direction == "up" else market["down_token_id"]
                            best_ask, ask_usd_vol = await get_clob_ask(http, token_id)
                            if ask_usd_vol < CONTINUATION_MIN_BOOK_VOLUME:
                                skip_reason = "depth_fail"
                            else:
                                entry_price = best_ask if best_ask > 0 else (
                                    market["up_price"] if direction == "up" else market["down_price"])
                                if entry_price > CONTINUATION_MAX_ENTRY_PRICE:
                                    skip_reason = "price_cap_fail"
                                elif sum(1 for t in open_trades if t.get("status") == "placed") >= MAX_CONCURRENT:
                                    skip_reason = "max_concurrent_reached"
                                else:
                                    base_size = min(BET_MAX, max(BET_MIN, bankroll * BET_PCT))
                                    size = base_size * CONTINUATION_SIZE_MULT
                                    if size < BET_MIN:
                                        skip_reason = "size_below_min"
                                    elif size > bankroll - 5:
                                        skip_reason = "bankroll_guard"
                                    else:
                                        tokens = min(size / entry_price, ask_usd_vol / entry_price)
                                        actual_size = round(tokens * entry_price, 2)
                                        if ENABLE_CONTINUATION_MODE:
                                            try:
                                                order_args = OrderArgs(
                                                    token_id=token_id,
                                                    price=round(entry_price, 4),
                                                    size=round(tokens, 2),
                                                    side=BUY,
                                                )
                                                signed = await loop.run_in_executor(
                                                    None, clob.create_order, order_args)
                                                result = await loop.run_in_executor(
                                                    None, functools.partial(clob.post_order, signed, OrderType.FAK))
                                                order_id = result.get("orderID", "")
                                                if order_id:
                                                    bankroll -= actual_size
                                                    state["bankroll"] = bankroll
                                                    state["total_invested"] = state.get("total_invested", 0) + actual_size
                                                    trade = {
                                                        "period":       period_ts,
                                                        "coin":         coin,
                                                        "side":         side,
                                                        "token_id":     token_id,
                                                        "condition_id": market["condition_id"],
                                                        "entry_price":  round(entry_price, 4),
                                                        "size_usd":     actual_size,
                                                        "tokens":       round(tokens, 2),
                                                        "order_id":     order_id,
                                                        "placed_at":    now_f,
                                                        "period_end":   period_ts + 900,
                                                        "move_pct":     round(btc_move * 100, 3),
                                                        "cross_confirm": 2,
                                                        "btc_at_entry": ws_feed.get("BTC"),
                                                        "strategy_mode": "continuation",
                                                        "status":       "placed",
                                                        "pnl":          0,
                                                    }
                                                    open_trades.append(trade)
                                                    state["trades"].append(trade)
                                                    continuation_trades_this_period += 1
                                                    save_state(state)
                                                    logger.info("CONT_LIVE_FIRE",
                                                                period=period_ts, elapsed=elapsed,
                                                                direction=direction,
                                                                btc_move=round(btc_move * 100, 3),
                                                                eth_move=round(eth_move * 100, 3),
                                                                entry=round(entry_price, 4),
                                                                size=actual_size,
                                                                order_id=order_id[:16],
                                                                bankroll=round(bankroll, 2))
                                                else:
                                                    skip_reason = "no_order_id"
                                            except Exception as e:
                                                skip_reason = "order_error"
                                                logger.error("CONT_ORDER_ERROR", error=str(e)[:120])
                                        else:
                                            continuation_trades_this_period += 1
                                            logger.info("CONT_SHADOW_FIRE",
                                                        period=period_ts, elapsed=elapsed,
                                                        direction=direction,
                                                        btc_move=round(btc_move * 100, 3),
                                                        eth_move=round(eth_move * 100, 3),
                                                        entry=round(entry_price, 4),
                                                        size=actual_size)

                # Log skip once per minute bucket (dedup across 0.1s loop ticks)
                if skip_reason:
                    bucket = elapsed // 60
                    cur_key = (period_ts, bucket)
                    if cur_key != last_cont_skip_log_key:
                        last_cont_skip_log_key = cur_key
                        logger.info("CONT_SKIP", reason=skip_reason,
                                    period=period_ts, elapsed=elapsed,
                                    btc_move=round(btc_move * 100, 3) if btc_move else None,
                                    eth_move=round(eth_move * 100, 3) if eth_move else None)

            # ──────────────────────────────────────────────────
            # RESOLVE trades (period_end + 120s grace)
            # (FAK orders are filled-or-cancelled atomically — no manual cancel needed)
            # ──────────────────────────────────────────────────
            for trade in list(open_trades):
                if trade["status"] != "placed":
                    continue
                if now_ts <= trade["period_end"] + 120:
                    continue

                coin   = trade["coin"]
                asset  = COIN_TO_ASSET[coin]
                t_per  = trade["period"]

                start_p = hist_start.get(t_per, {}).get(asset)
                end_p   = ws_feed.get(asset)

                if not start_p:
                    trade["status"] = "pending_settlement"
                    continue

                actual_move = ((end_p - start_p) / start_p) if end_p else 0
                won = (
                    (trade["side"] == "buy_up"   and actual_move > 0) or
                    (trade["side"] == "buy_down"  and actual_move < 0)
                )

                if won:
                    payout = trade["tokens"] * 0.98  # 2% Polymarket fee
                    trade["pnl"]    = round(payout - trade["size_usd"], 2)
                    trade["status"] = "won"
                    bankroll += payout
                    state["total_returned"] = state.get("total_returned", 0) + payout
                    consecutive_losses = 0
                else:
                    trade["pnl"]    = -trade["size_usd"]
                    trade["status"] = "lost"
                    consecutive_losses += 1
                    if consecutive_losses >= CIRCUIT_LOSSES and skip_signals == 0:
                        skip_signals = CIRCUIT_SKIP
                        logger.warning(
                            "CIRCUIT_BREAKER",
                            consecutive_losses=consecutive_losses,
                            skipping_next=skip_signals,
                        )

                state["bankroll"] = bankroll
                save_state(state)
                logger.info(
                    "SNIPER_RESOLVED",
                    coin=trade["coin"].upper(),
                    status=trade["status"],
                    pnl=trade["pnl"],
                    move_pct=trade["move_pct"],
                    bankroll=round(bankroll, 2),
                    consecutive_losses=consecutive_losses,
                )

            # Prune settled trades from active list (keep pending_settlement for retry)
            open_trades = [t for t in open_trades if t["status"] in ("placed", "pending_settlement")]

            # ──────────────────────────────────────────────────
            # STATUS LOG every 60s
            # ──────────────────────────────────────────────────
            if time.time() - last_status > 60:
                closed = [t for t in state["trades"] if t["status"] in ("won", "lost")]
                w   = sum(1 for t in closed if t["status"] == "won")
                l   = len(closed) - w
                pnl = sum(t["pnl"] for t in closed)
                wr  = (w / len(closed) * 100) if closed else 0

                logger.info(
                    "sniper_status",
                    bankroll=round(bankroll, 2),
                    pnl=round(pnl, 2),
                    trades="%dW/%dL" % (w, l),
                    win_rate=("%.0f%%" % wr) if closed else "--",
                    open=len(open_trades),
                    btc=ws_feed.get("BTC"),
                    period_elapsed="%ds" % elapsed,
                    circuit=("PAUSED(%d)" % skip_signals) if skip_signals > 0 else "ok",
                )
                last_status = time.time()

            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    ws_feed.stop()
    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass
    await http.aclose()

    closed    = [t for t in state["trades"] if t["status"] in ("won", "lost")]
    w         = sum(1 for t in closed if t["status"] == "won")
    l         = len(closed) - w
    total_pnl = sum(t["pnl"] for t in closed)

    print("\n" + "=" * 60)
    print("MULTI-COIN MOMENTUM SNIPER RESULTS")
    print("=" * 60)
    print("  Trades:        %d (%dW / %dL)" % (len(closed), w, l))
    if closed:
        print("  Win rate:      %.0f%%" % (w / len(closed) * 100))
    print("  P&L:           $%+.2f" % total_pnl)
    print("  Final bankroll: $%.2f" % bankroll)
    if bankroll_init > 0:
        print("  ROI:           %+.0f%%" % ((bankroll - bankroll_init) / bankroll_init * 100))
    print("=" * 60)

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "version":    "sniper_v1",
            "bankroll":   bankroll,
            "trades":     len(closed),
            "wins":       w,
            "losses":     l,
            "pnl":        total_pnl,
            "all_trades": state["trades"],
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
