"""
15-Minute Crypto Market Scanner

Discovers and tracks the fast-rotating 15-minute crypto up/down markets
on Polymarket using the slug pattern: {coin}-updown-15m-{unix_timestamp}

These markets resolve every 15 minutes based on Chainlink price feeds.
New markets are created automatically at each 15-minute boundary.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()

COINS = ["btc", "eth", "sol", "xrp", "doge", "bnb"]

COIN_TO_BINANCE = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
    "doge": "DOGEUSDT",
    "bnb": "BNBUSDT",
}

COIN_TO_ASSET = {
    "btc": "BTC", "eth": "ETH", "sol": "SOL",
    "xrp": "XRP", "doge": "DOGE", "bnb": "BNB",
}


@dataclass
class FastMarket:
    """A 15-minute crypto up/down market."""
    coin: str  # "btc", "eth", etc.
    asset: str  # "BTC", "ETH", etc.
    slug: str
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    volume: float
    period_start: int  # Unix timestamp of period start
    period_end: int  # Unix timestamp of period end (start + 900)


@dataclass
class FastSignal:
    """An arbitrage signal on a 15-minute market."""
    market: FastMarket
    side: str  # "buy_up" or "buy_down"
    token_id: str
    entry_price: float
    confidence: float
    binance_price: float
    binance_change_pct: float  # Price change since period start
    seconds_remaining: int
    timestamp: datetime


class FastMarketScanner:
    """Discovers 15-minute crypto markets using slug pattern."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10)
        self._current_markets: dict[str, FastMarket] = {}

    async def scan_current_period(self) -> list[FastMarket]:
        """Fetch markets for the current 15-minute period."""
        ts = int(time.time() // 900) * 900
        markets = []

        for coin in COINS:
            slug = f"{coin}-updown-15m-{ts}"
            try:
                resp = await self._client.get(
                    f"https://gamma-api.polymarket.com/markets?slug={slug}"
                )
                data = resp.json()
                if not data:
                    continue

                m = data[0]
                parsed = self._parse(m, coin, ts)
                if parsed:
                    markets.append(parsed)
                    self._current_markets[coin] = parsed

                await asyncio.sleep(0.3)  # Rate limit
            except Exception as e:
                logger.debug("fast_scan_error", coin=coin, error=str(e))

        return markets

    async def scan_next_period(self) -> list[FastMarket]:
        """Pre-fetch markets for the next 15-minute period."""
        ts = (int(time.time() // 900) + 1) * 900
        markets = []

        for coin in COINS:
            slug = f"{coin}-updown-15m-{ts}"
            try:
                resp = await self._client.get(
                    f"https://gamma-api.polymarket.com/markets?slug={slug}"
                )
                data = resp.json()
                if data:
                    parsed = self._parse(data[0], coin, ts)
                    if parsed:
                        markets.append(parsed)
                await asyncio.sleep(0.3)
            except Exception:
                pass

        return markets

    def _parse(self, raw: dict, coin: str, period_ts: int) -> Optional[FastMarket]:
        """Parse a Gamma API market response."""
        try:
            tokens = json.loads(raw["clobTokenIds"]) if isinstance(raw.get("clobTokenIds"), str) else raw.get("clobTokenIds", [])
            outcomes = json.loads(raw["outcomes"]) if isinstance(raw.get("outcomes"), str) else raw.get("outcomes", [])
            prices = json.loads(raw["outcomePrices"]) if isinstance(raw.get("outcomePrices"), str) else raw.get("outcomePrices", [])

            if len(tokens) < 2 or len(outcomes) < 2:
                return None

            up_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "up"), 0)
            down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "down"), 1)

            return FastMarket(
                coin=coin,
                asset=COIN_TO_ASSET[coin],
                slug=f"{coin}-updown-15m-{period_ts}",
                condition_id=raw.get("conditionId", ""),
                question=raw.get("question", ""),
                up_token_id=str(tokens[up_idx]),
                down_token_id=str(tokens[down_idx]),
                up_price=float(prices[up_idx]),
                down_price=float(prices[down_idx]),
                best_bid=float(raw["bestBid"]) if raw.get("bestBid") else None,
                best_ask=float(raw["bestAsk"]) if raw.get("bestAsk") else None,
                volume=float(raw.get("volumeNum", 0)),
                period_start=period_ts,
                period_end=period_ts + 900,
            )
        except Exception as e:
            logger.debug("fast_parse_error", coin=coin, error=str(e))
            return None

    def get(self, coin: str) -> Optional[FastMarket]:
        return self._current_markets.get(coin)

    async def close(self):
        await self._client.aclose()


class FastArbDetector:
    """Detects latency arb on 15-minute markets.

    Strategy: Compare real-time Binance price change to the market's
    Up/Down pricing. If Binance shows a clear directional move but
    the market hasn't repriced yet, buy the winning side.

    Key constraints:
    - 10% maker+taker fee (need >55% win rate for profitability)
    - Markets resolve based on price at END vs START of 15-min window
    - Best signals come late in the window (10-14 min in) when the
      direction is more certain but the market may still be stale
    """

    def __init__(
        self,
        min_move_pct: float = 0.003,  # 0.3% minimum price move to signal
        min_seconds_elapsed: int = 300,  # Wait at least 5 min into period
        max_entry_price: float = 0.65,  # Don't pay more than $0.65 for a token
        min_edge_after_fees: float = 0.05,  # 5% minimum edge after 10% fees
    ):
        self.min_move_pct = min_move_pct
        self.min_seconds_elapsed = min_seconds_elapsed
        self.max_entry_price = max_entry_price
        self.min_edge_after_fees = min_edge_after_fees
        self._period_start_prices: dict[str, float] = {}

    def record_start_price(self, asset: str, price: float, period_ts: int):
        """Record the Binance price at the start of a period."""
        self._period_start_prices[f"{asset}_{period_ts}"] = price

    def detect(
        self,
        market: FastMarket,
        binance_price: float,
        now: Optional[datetime] = None,
    ) -> Optional[FastSignal]:
        """Check if there's an arb opportunity on this market."""
        now = now or datetime.now(timezone.utc)
        now_ts = int(now.timestamp())

        # How far into the period are we?
        elapsed = now_ts - market.period_start
        remaining = market.period_end - now_ts

        if elapsed < self.min_seconds_elapsed:
            return None  # Too early — direction unclear
        if remaining < 15:
            return None  # Too late — can't get filled

        # Get start-of-period price
        start_key = f"{market.asset}_{market.period_start}"
        start_price = self._period_start_prices.get(start_key)
        if start_price is None or start_price <= 0:
            return None  # Don't know the start price

        # Calculate price change
        change_pct = (binance_price - start_price) / start_price

        if abs(change_pct) < self.min_move_pct:
            return None  # Not enough movement

        # Determine direction and check if market is stale
        if change_pct > 0:
            # Price went UP — "Up" should win
            side = "buy_up"
            token_id = market.up_token_id
            entry_price = market.up_price
            # Confidence scales with move size and time elapsed
            confidence = min(0.95, 0.6 + abs(change_pct) * 20 + (elapsed / 900) * 0.2)
        else:
            # Price went DOWN — "Down" should win
            side = "buy_down"
            token_id = market.down_token_id
            entry_price = market.down_price
            confidence = min(0.95, 0.6 + abs(change_pct) * 20 + (elapsed / 900) * 0.2)

        # Check if entry price is attractive enough
        if entry_price > self.max_entry_price:
            return None  # Market already repriced — no edge

        # Calculate edge after fees (10% round-trip)
        fee_rate = 0.10
        gross_return = (1.0 / entry_price) - 1.0  # e.g., buy at 0.45 -> 122% gross
        net_return = gross_return - fee_rate
        if net_return < self.min_edge_after_fees:
            return None  # Not enough edge after fees

        return FastSignal(
            market=market,
            side=side,
            token_id=token_id,
            entry_price=entry_price,
            confidence=confidence,
            binance_price=binance_price,
            binance_change_pct=change_pct,
            seconds_remaining=remaining,
            timestamp=now,
        )
