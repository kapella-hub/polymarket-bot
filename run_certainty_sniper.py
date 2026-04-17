#!/usr/bin/env python3
"""
Certainty Sniper — Late-Period Triple-Confirmed Compounder

Strategy: In minutes 12-14 of each 15-min period, when a coin has moved
>0.7% AND 2+ coins confirm the direction AND Binance order book still shows
pressure, buy the winning-side token at current market price (~$0.82-0.90).
Token resolves $1 in <3 minutes. Half-Kelly compounding.

Gates:
  1. Target coin moved >=0.7% from period open
  2. >=2 coins moved >=0.5% in same direction (macro signal)
  3. Binance top-10 order book: bid depth > ask depth (UP) or vice versa
"""

import asyncio
import functools
import json
import os
import sys
import time
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

LOG_FILE   = Path(__file__).parent / "certainty_sniper_output.log"
STATE_FILE = Path(__file__).parent / "data" / "certainty_sniper_state.json"

COINS         = ["btc", "eth", "sol", "xrp", "doge", "bnb"]
COIN_TO_ASSET = {"btc": "BTC", "eth": "ETH", "sol": "SOL",
                 "xrp": "XRP", "doge": "DOGE", "bnb": "BNB"}

# Timing
CERTAINTY_START  = 720   # 12 min into period — start checking
CERTAINTY_END    = 840   # 14 min into period — stop (60s remaining in period)
CHECK_INTERVAL   = 30    # Check every 30s in window
CANCEL_AFTER     = 20    # Cancel unfilled order after 20s

# Signal thresholds
MIN_MOVE_PCT       = 0.007  # Gate 1: target coin moved >=0.7%
CONFIRM_MOVE_PCT   = 0.005  # Gate 2: confirming coins moved >=0.5%
MIN_CONFIRMING     = 2      # Gate 2: need >=2 confirming coins
MAX_ENTRY_PRICE    = 0.92   # Skip if market already priced certainty in

# Sizing
BET_MIN        = 8.0
BET_MAX        = 40.0

# Risk controls
CIRCUIT_LOSSES    = 2     # Consecutive losses before pause
CIRCUIT_SKIP      = 4     # Periods to skip after circuit triggers
DAILY_LOSS_PCT    = 0.12  # 12% daily loss → halt for 24h
MIN_BANKROLL      = 30.0  # Stop trading below this


def kelly_size(bankroll: float, win_rate: float = 0.90, avg_entry: float = 0.85) -> float:
    """Half-Kelly bet size. Returns dollars to risk."""
    b = (1.0 - avg_entry) / avg_entry  # net return per dollar at avg_entry
    kelly = (win_rate * b - (1.0 - win_rate)) / b
    size = bankroll * (kelly / 2.0)
    return min(BET_MAX, max(BET_MIN, round(size, 2)))


def gate1_move(cur: float, start: float) -> tuple[bool, float]:
    """Gate 1: coin moved >=MIN_MOVE_PCT from period open. Returns (passed, move_fraction)."""
    if not cur or not start:
        return False, 0.0
    move = (cur - start) / start
    return abs(move) >= MIN_MOVE_PCT, move


def gate2_confirm(all_moves: dict[str, float], target_direction: str) -> bool:
    """Gate 2: >=MIN_CONFIRMING coins moved >=CONFIRM_MOVE_PCT in same direction."""
    # all_moves includes the target coin (which passed Gate 1 >= MIN_MOVE_PCT > CONFIRM_MOVE_PCT),
    # so effective requirement is >= (MIN_CONFIRMING - 1) OTHER coins confirming.
    if target_direction == "up":
        confirming = sum(1 for m in all_moves.values() if m >= CONFIRM_MOVE_PCT)
    else:
        confirming = sum(1 for m in all_moves.values() if m <= -CONFIRM_MOVE_PCT)
    return confirming >= MIN_CONFIRMING


def compute_all_moves(ws_feed: BinanceWSFeed, period_prices: dict[str, float]) -> dict[str, float]:
    """Compute move fraction for each coin from period open. Returns {coin: move_fraction}."""
    moves = {}
    for coin, asset in COIN_TO_ASSET.items():
        cur   = ws_feed.get(asset)
        start = period_prices.get(asset)
        if cur and start:
            moves[coin] = (cur - start) / start
    return moves


def create_clob() -> ClobClient:
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


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning("state_file_corrupt", path=str(STATE_FILE))
    return {
        "bankroll": 0.0,
        "trades": [],
        "total_invested": 0.0,
        "total_returned": 0.0,
        "consecutive_losses": 0,
        "daily_start_bankroll": 0.0,
        "daily_start_date": "",
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def gate3_orderbook(client: httpx.AsyncClient, coin: str, direction: str) -> bool:
    """Gate 3: Binance top-10 order book confirms continued pressure in direction.
    UP move: bid volume > ask volume. DOWN move: ask volume > bid volume."""
    symbol = COIN_TO_ASSET[coin] + "USDT"
    try:
        resp = await client.get(
            f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10",
            timeout=3,
        )
        resp.raise_for_status()
        data = resp.json()
        bid_vol = sum(float(qty) for _, qty in data.get("bids", []))
        ask_vol = sum(float(qty) for _, qty in data.get("asks", []))
        if direction == "up":
            return bid_vol > ask_vol
        else:
            return ask_vol > bid_vol
    except Exception as e:
        logger.debug("orderbook_error", coin=coin, error=str(e))
        return False  # fail safe: skip if can't check


async def fetch_market_price(client: httpx.AsyncClient, coin: str, period_ts: int) -> dict | None:
    """Fetch current market prices for a coin's period — called fresh at minute 12."""
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
        tokens   = json.loads(m["clobTokenIds"])  if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
        outcomes = json.loads(m["outcomes"])       if isinstance(m.get("outcomes"), str)      else m.get("outcomes", [])
        prices   = json.loads(m["outcomePrices"])  if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
        if len(tokens) < 2 or len(outcomes) < 2:
            return None
        up_idx   = next((i for i, o in enumerate(outcomes) if o.lower() == "up"),   0)
        down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "down"), 1)
        return {
            "coin":          coin,
            "condition_id":  m.get("conditionId", ""),
            "up_token_id":   str(tokens[up_idx]),
            "down_token_id": str(tokens[down_idx]),
            "up_price":      float(prices[up_idx]),
            "down_price":    float(prices[down_idx]),
        }
    except Exception as e:
        logger.debug("fetch_market_price_error", coin=coin, error=str(e))
        return None


async def place_order(loop, clob: ClobClient, token_id: str,
                      entry_price: float, tokens: float) -> str:
    """Place a GTC buy order. Returns order_id or empty string on failure."""
    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=round(entry_price, 4),
            size=round(tokens, 2),
            side=BUY,
        )
        signed = await loop.run_in_executor(None, clob.create_order, order_args)
        result = await loop.run_in_executor(
            None, functools.partial(clob.post_order, signed, OrderType.GTC))
        return result.get("orderID", "")
    except Exception as e:
        logger.error("place_order_error", error=str(e)[:120])
        return ""


async def cancel_order(loop, clob: ClobClient, order_id: str) -> bool:
    try:
        await loop.run_in_executor(None, functools.partial(clob.cancel, order_id))
        return True
    except Exception as e:
        logger.debug("cancel_error", order_id=order_id[:16], error=str(e))
        return False
