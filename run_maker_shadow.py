#!/usr/bin/env python3
"""
Maker-aware continuation shadow strategy.

Goal:
  Test whether a maker-style entry can preserve the surviving BTC/ETH continuation
  edge while improving entry price versus taker FAK execution.

Behavior:
  - Shadow only: no live orders are sent.
  - Watches BTC/ETH 15m markets.
  - Uses the same BTC/ETH directional continuation gate as the live sniper.
  - Quotes a hypothetical maker bid one tick inside the spread.
  - Treats the quote as filled only if the future best ask trades down to our price.
  - Resolves the filled position at period end exactly like the live continuation bot.
"""

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog

from run_sniper import (
    COIN_TO_ASSET,
    CONTINUATION_ACTIVE_END_SEC,
    CONTINUATION_ACTIVE_START_SEC,
    CONTINUATION_LOOKBACK_SEC,
    CONTINUATION_MAX_ENTRY_PRICE,
    CONTINUATION_MIN_BOOK_VOLUME,
    CONTINUATION_MIN_SCORE,
    PRICE_HISTORY_SEC,
    _compute_move_pct,
    _get_price_at,
    _reversal_counter_move,
    continuation_move_gate,
    continuation_signal_score,
    continuation_target_size,
)
from src.crypto_arb.ws_feeds import BinanceWSFeed

logger = structlog.get_logger()

LOG_FILE = Path(__file__).parent / "maker_shadow_output.log"
STATE_FILE = Path(__file__).parent / "data" / "maker_shadow_state.json"

MAKER_COINS = ("btc", "eth")
QUOTE_TICK = 0.01
MIN_SPREAD = 0.02
FILL_TIMEOUT_SEC = 90


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


def maker_quote_price(best_bid: float, best_ask: float, max_entry: float) -> float | None:
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return None
    spread = best_ask - best_bid
    if spread < MIN_SPREAD:
        return None
    quote = round(best_bid + QUOTE_TICK, 4)
    if quote >= best_ask:
        return None
    if quote > max_entry:
        return None
    return quote


async def fetch_market(client: httpx.AsyncClient, coin: str, period_ts: int) -> dict | None:
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
        tokens = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
        outcomes = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes", [])
        prices = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
        if len(tokens) < 2 or len(outcomes) < 2:
            return None
        up_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "up"), 0)
        down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "down"), 1)
        return {
            "coin": coin,
            "condition_id": m.get("conditionId", ""),
            "up_token_id": str(tokens[up_idx]),
            "down_token_id": str(tokens[down_idx]),
            "up_price": float(prices[up_idx]),
            "down_price": float(prices[down_idx]),
            "period_end": period_ts + 900,
        }
    except Exception as e:
        logger.debug("maker_shadow_market_error", coin=coin, error=str(e))
        return None


async def get_book(client: httpx.AsyncClient, token_id: str) -> tuple[float, float, float]:
    try:
        resp = await client.get(
            f"https://clob.polymarket.com/book?token_id={token_id}",
            timeout=3,
        )
        data = resp.json()
        bids = sorted(data.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))
        if not bids or not asks:
            return 0.0, 0.0, 0.0
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        ask_usd_vol = sum(float(a["size"]) for a in asks if float(a["price"]) <= best_ask + 0.01) * best_ask
        return best_bid, best_ask, ask_usd_vol
    except Exception:
        return 0.0, 0.0, 0.0


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
        logger.error("maker_shadow_no_btc_price")
        return

    rolling_prices: dict[str, deque] = {asset: deque() for asset in COIN_TO_ASSET.values()}
    hist_start: dict[int, dict[str, float]] = {}
    open_quotes: list[dict] = [o for o in state.get("orders", []) if o.get("status") == "quoted"]
    filled_quotes: list[dict] = [o for o in state.get("filled", []) if o.get("status") == "filled"]
    quoted_this_period: set[tuple[int, str]] = set()

    _roll_max_age = max(CONTINUATION_LOOKBACK_SEC + 120, PRICE_HISTORY_SEC)
    current_period = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 604800:
            now_ts = int(time.time())
            now_f = time.time()
            period_ts = (now_ts // 900) * 900
            elapsed = now_ts - period_ts

            for asset in COIN_TO_ASSET.values():
                p = ws_feed.get(asset)
                if p:
                    rolling_prices[asset].append((now_f, p))
                    cutoff = now_f - _roll_max_age
                    while rolling_prices[asset] and rolling_prices[asset][0][0] < cutoff:
                        rolling_prices[asset].popleft()

            if period_ts != current_period:
                quoted_this_period = set()
                hist_start[period_ts] = {
                    asset: ws_feed.get(asset)
                    for asset in COIN_TO_ASSET.values()
                    if ws_feed.get(asset)
                }
                current_period = period_ts
                logger.info("maker_shadow_period_ready", period=period_ts, btc=ws_feed.get("BTC"))

            if CONTINUATION_ACTIVE_START_SEC <= elapsed <= CONTINUATION_ACTIVE_END_SEC:
                btc_hist = rolling_prices.get("BTC", deque())
                eth_hist = rolling_prices.get("ETH", deque())
                btc_move = _compute_move_pct(btc_hist, now_f, CONTINUATION_LOOKBACK_SEC)
                eth_move = _compute_move_pct(eth_hist, now_f, CONTINUATION_LOOKBACK_SEC)
                if btc_move is not None and eth_move is not None and (btc_move > 0) == (eth_move > 0):
                    ok, _ = continuation_move_gate(abs(btc_move), abs(eth_move), elapsed)
                    if ok:
                        direction = "up" if btc_move > 0 else "down"
                        rev = _reversal_counter_move(btc_hist, now_f, 60, direction)
                        confirm_move = min(abs(btc_move), abs(eth_move))
                        for coin, move in (("btc", btc_move), ("eth", eth_move)):
                            if (period_ts, coin) in quoted_this_period:
                                continue
                            market = await fetch_market(http, coin, period_ts)
                            if not market:
                                continue
                            token_id = market["up_token_id"] if direction == "up" else market["down_token_id"]
                            best_bid, best_ask, ask_usd_vol = await get_book(http, token_id)
                            if ask_usd_vol < CONTINUATION_MIN_BOOK_VOLUME:
                                continue
                            quote_price = maker_quote_price(best_bid, best_ask, CONTINUATION_MAX_ENTRY_PRICE)
                            if quote_price is None:
                                continue
                            score = continuation_signal_score(
                                lead_move_abs=abs(move),
                                confirm_move_abs=confirm_move,
                                reversal=rev,
                                entry_price=quote_price,
                                ask_usd_vol=ask_usd_vol,
                            )
                            if score < CONTINUATION_MIN_SCORE:
                                continue
                            size = continuation_target_size(bankroll)
                            tokens = round(size / quote_price, 2)
                            quote = {
                                "period": period_ts,
                                "coin": coin,
                                "side": "buy_up" if direction == "up" else "buy_down",
                                "token_id": token_id,
                                "quote_price": quote_price,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "size_usd": round(tokens * quote_price, 2),
                                "tokens": tokens,
                                "signal_score": round(score, 3),
                                "quoted_at": now_f,
                                "period_end": market["period_end"],
                                "status": "quoted",
                            }
                            open_quotes.append(quote)
                            state["orders"].append(quote)
                            quoted_this_period.add((period_ts, coin))
                            save_state(state)
                            logger.info(
                                "MAKER_SHADOW_QUOTE",
                                period=period_ts,
                                elapsed=elapsed,
                                coin=coin.upper(),
                                side=quote["side"],
                                quote=quote_price,
                                best_bid=best_bid,
                                best_ask=best_ask,
                                size=quote["size_usd"],
                                score=quote["signal_score"],
                            )

            for quote in list(open_quotes):
                age = now_f - float(quote["quoted_at"])
                if age > FILL_TIMEOUT_SEC:
                    quote["status"] = "expired"
                    logger.info("MAKER_SHADOW_EXPIRED", coin=quote["coin"].upper(), quote=quote["quote_price"])
                    save_state(state)
                    continue
                _, best_ask, _ = await get_book(http, quote["token_id"])
                if best_ask > 0 and best_ask <= quote["quote_price"]:
                    quote["status"] = "filled"
                    quote["filled_at"] = now_f
                    filled_quotes.append(quote)
                    state["filled"].append(quote)
                    logger.info("MAKER_SHADOW_FILLED", coin=quote["coin"].upper(), fill=quote["quote_price"])
                    save_state(state)

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
                logger.info(
                    "MAKER_SHADOW_RESOLVED",
                    coin=trade["coin"].upper(),
                    status=trade["status"],
                    pnl=trade["pnl"],
                    bankroll=round(bankroll, 2),
                )
                save_state(state)

            filled_quotes = [t for t in filled_quotes if t["status"] == "filled"]
            await asyncio.sleep(5)
    finally:
        ws_task.cancel()
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
