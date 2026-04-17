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
