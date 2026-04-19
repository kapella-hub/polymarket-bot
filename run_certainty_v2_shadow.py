#!/usr/bin/env python3
"""
Certainty V2 shadow runner.

This replaces the old "late certainty taker" idea with an earlier confirmed-drift
shadow strategy:
  - window: minutes 6-11 of the period
  - requires multi-coin confirmation
  - requires supportive Binance order-book imbalance
  - uses cheap-entry maker-style quotes, not late taker chasing
"""

import asyncio
import json
import time
from collections import deque
from pathlib import Path

import httpx
import structlog

from run_certainty_sniper import (
    COIN_TO_ASSET,
    certainty_coin_is_tradeable,
    gate3_orderbook,
    orderbook_imbalance,
)
from run_maker_shadow import fetch_market, get_book, maker_quote_price
from run_sniper import (
    PRICE_HISTORY_SEC,
    _compute_move_pct,
    _get_price_at,
    _reversal_counter_move,
    continuation_target_size,
)
from src.crypto_arb.ws_feeds import BinanceWSFeed

logger = structlog.get_logger()

STATE_FILE = Path(__file__).parent / "data" / "certainty_v2_shadow_state.json"

ACTIVE_COINS = tuple(c for c in ("btc", "eth", "sol", "bnb") if certainty_coin_is_tradeable(c))
WINDOW_START = 360
WINDOW_END = 660
CHECK_INTERVAL = 10
LOOKBACK_SEC = 240
MIN_MOVE_PCT = 0.0028
MIN_CONFIRM_MOVE_PCT = 0.0022
MIN_CONFIRMING = 2
MAX_ENTRY_PRICE = 0.84
MIN_SCORE = 1.35
MAX_REVERSAL_PCT = 0.0014
FILL_TIMEOUT_SEC = 120


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"bankroll": 100.0, "orders": [], "filled": [], "resolved": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def certainty_v2_move_threshold(elapsed: int) -> float:
    progress = min(max((elapsed - WINDOW_START) / max(WINDOW_END - WINDOW_START, 1), 0.0), 1.0)
    return MIN_MOVE_PCT - 0.0005 * progress


def certainty_v2_confirm_threshold(elapsed: int) -> float:
    progress = min(max((elapsed - WINDOW_START) / max(WINDOW_END - WINDOW_START, 1), 0.0), 1.0)
    return MIN_CONFIRM_MOVE_PCT - 0.0004 * progress


def certainty_v2_signal_score(
    move_abs: float,
    move_threshold: float,
    confirm_count: int,
    imbalance: float,
    entry_price: float,
    ask_usd_vol: float,
) -> float:
    if move_threshold <= 0:
        return 0.0
    move_score = move_abs / move_threshold
    breadth_score = max(0.0, confirm_count - 1) * 0.30
    imbalance_score = max(0.0, imbalance) * 1.5
    price_score = max(0.0, (MAX_ENTRY_PRICE - entry_price) / 0.08)
    liquidity_score = min(ask_usd_vol / 30.0, 1.0) * 0.25
    return move_score + breadth_score + imbalance_score + price_score + liquidity_score


def certainty_v2_confirm_count(moves: dict[str, float], direction: str, confirm_threshold: float) -> int:
    if direction == "up":
        return sum(1 for m in moves.values() if m >= confirm_threshold)
    return sum(1 for m in moves.values() if m <= -confirm_threshold)


async def main():
    state = load_state()
    bankroll = float(state.get("bankroll", 100.0))
    http = httpx.AsyncClient(timeout=10)
    ws_feed = BinanceWSFeed()
    ws_task = asyncio.create_task(ws_feed.run())

    for _ in range(100):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    if not ws_feed.get("BTC"):
        logger.error("certainty_v2_shadow_no_btc_price")
        return

    rolling_prices: dict[str, deque] = {asset: deque() for asset in COIN_TO_ASSET.values()}
    hist_start: dict[int, dict[str, float]] = {}
    open_quotes: list[dict] = [o for o in state.get("orders", []) if o.get("status") == "quoted"]
    filled_quotes: list[dict] = [o for o in state.get("filled", []) if o.get("status") == "filled"]
    quoted_this_period: set[tuple[int, str]] = set()
    current_period = 0
    last_check_ts = 0

    try:
        while True:
            now_ts = int(time.time())
            now_f = time.time()
            period_ts = (now_ts // 900) * 900
            elapsed = now_ts - period_ts

            for asset in COIN_TO_ASSET.values():
                p = ws_feed.get(asset)
                if p:
                    rolling_prices[asset].append((now_f, p))
                    cutoff = now_f - PRICE_HISTORY_SEC
                    while rolling_prices[asset] and rolling_prices[asset][0][0] < cutoff:
                        rolling_prices[asset].popleft()

            if period_ts != current_period:
                current_period = period_ts
                last_check_ts = 0
                quoted_this_period = set()
                hist_start[period_ts] = {
                    asset: ws_feed.get(asset)
                    for asset in COIN_TO_ASSET.values()
                    if ws_feed.get(asset)
                }
                logger.info("certainty_v2_shadow_period_ready", period=period_ts, btc=ws_feed.get("BTC"))

            in_window = WINDOW_START <= elapsed <= WINDOW_END
            if in_window and (now_ts - last_check_ts) >= CHECK_INTERVAL:
                last_check_ts = now_ts
                period_prices = hist_start.get(period_ts, {})
                all_moves = {}
                for coin in ACTIVE_COINS:
                    cur = ws_feed.get(COIN_TO_ASSET[coin])
                    start = period_prices.get(COIN_TO_ASSET[coin])
                    if cur and start and start > 0:
                        all_moves[coin] = (cur - start) / start

                move_gate = certainty_v2_move_threshold(elapsed)
                confirm_gate = certainty_v2_confirm_threshold(elapsed)

                for coin in ACTIVE_COINS:
                    if (period_ts, coin) in quoted_this_period:
                        continue
                    move = all_moves.get(coin)
                    if move is None or abs(move) < move_gate:
                        continue
                    direction = "up" if move > 0 else "down"
                    confirm_count = certainty_v2_confirm_count(all_moves, direction, confirm_gate)
                    if confirm_count < MIN_CONFIRMING:
                        continue
                    rev = _reversal_counter_move(
                        rolling_prices.get(COIN_TO_ASSET[coin], deque()),
                        now_f,
                        60,
                        direction,
                    )
                    if rev >= MAX_REVERSAL_PCT:
                        continue
                    bid_vol, ask_vol = await gate3_orderbook(http, coin)
                    imbalance = orderbook_imbalance(bid_vol, ask_vol, direction)
                    if imbalance <= 0.0:
                        continue

                    market = await fetch_market(http, coin, period_ts)
                    if not market:
                        continue
                    token_id = market["up_token_id"] if direction == "up" else market["down_token_id"]
                    best_bid, best_ask, ask_usd_vol = await get_book(http, token_id)
                    quote_price = maker_quote_price(best_bid, best_ask, MAX_ENTRY_PRICE)
                    if quote_price is None:
                        continue

                    score = certainty_v2_signal_score(
                        move_abs=abs(move),
                        move_threshold=move_gate,
                        confirm_count=confirm_count,
                        imbalance=imbalance,
                        entry_price=quote_price,
                        ask_usd_vol=ask_usd_vol,
                    )
                    if score < MIN_SCORE:
                        continue

                    size = continuation_target_size(bankroll)
                    tokens = round(size / quote_price, 2)
                    quote = {
                        "period": period_ts,
                        "coin": coin,
                        "side": "buy_up" if direction == "up" else "buy_down",
                        "token_id": token_id,
                        "quote_price": quote_price,
                        "size_usd": round(tokens * quote_price, 2),
                        "tokens": tokens,
                        "signal_score": round(score, 3),
                        "confirm_count": confirm_count,
                        "imbalance": round(imbalance, 3),
                        "quoted_at": now_f,
                        "period_end": market["period_end"],
                        "status": "quoted",
                    }
                    open_quotes.append(quote)
                    state["orders"].append(quote)
                    quoted_this_period.add((period_ts, coin))
                    save_state(state)
                    logger.info(
                        "CERTAINTY_V2_SHADOW_QUOTE",
                        period=period_ts,
                        elapsed=elapsed,
                        coin=coin.upper(),
                        side=quote["side"],
                        quote=quote_price,
                        size=quote["size_usd"],
                        score=quote["signal_score"],
                        confirm=confirm_count,
                    )

            for quote in list(open_quotes):
                age = now_f - float(quote["quoted_at"])
                if age > FILL_TIMEOUT_SEC:
                    quote["status"] = "expired"
                    save_state(state)
                    logger.info("CERTAINTY_V2_SHADOW_EXPIRED", coin=quote["coin"].upper(), quote=quote["quote_price"])
                    continue
                _, best_ask, _ = await get_book(http, quote["token_id"])
                if best_ask > 0 and best_ask <= quote["quote_price"]:
                    quote["status"] = "filled"
                    quote["filled_at"] = now_f
                    filled_quotes.append(quote)
                    state["filled"].append(quote)
                    save_state(state)
                    logger.info("CERTAINTY_V2_SHADOW_FILLED", coin=quote["coin"].upper(), fill=quote["quote_price"])

            open_quotes = [q for q in open_quotes if q["status"] == "quoted"]

            for trade in list(filled_quotes):
                if now_ts <= trade["period_end"] + 120:
                    continue
                asset = COIN_TO_ASSET[trade["coin"]]
                start_p = hist_start.get(trade["period"], {}).get(asset)
                end_p = _get_price_at(rolling_prices.get(asset, deque()), trade["period_end"])
                if not start_p or end_p is None:
                    continue
                actual_move = (end_p - start_p) / start_p
                won = (
                    (trade["side"] == "buy_up" and actual_move > 0) or
                    (trade["side"] == "buy_down" and actual_move < 0)
                )
                if won:
                    payout = trade["tokens"] * 0.98
                    trade["pnl"] = round(payout - trade["size_usd"], 2)
                    trade["status"] = "won"
                    bankroll += trade["pnl"]
                else:
                    trade["pnl"] = -trade["size_usd"]
                    trade["status"] = "lost"
                    bankroll += trade["pnl"]
                state["bankroll"] = round(bankroll, 2)
                state["resolved"].append(trade)
                save_state(state)
                logger.info(
                    "CERTAINTY_V2_SHADOW_RESOLVED",
                    coin=trade["coin"].upper(),
                    status=trade["status"],
                    pnl=trade["pnl"],
                    bankroll=round(bankroll, 2),
                )

            filled_quotes = [t for t in filled_quotes if t["status"] == "filled"]
            await asyncio.sleep(2)
    finally:
        ws_task.cancel()
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
