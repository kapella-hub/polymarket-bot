"""Polymarket CLOB exchange adapter."""

import asyncio
import time
from typing import Optional

import structlog
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from src.config import settings
from src.exchange.base import (
    ExchangeAdapter,
    OrderBook,
    OrderBookEntry,
    OrderResult,
    TradeRecord,
)

logger = structlog.get_logger()


class RateLimiter:
    """Token bucket rate limiter.

    Polymarket allows 3500 POST-order/10s burst, 36000/10min sustained.
    We use conservative defaults well below those limits.
    """

    def __init__(self, rate: float = 100.0, burst: int = 200):
        self._rate = rate  # Tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


class PolymarketAdapter(ExchangeAdapter):
    """Polymarket CLOB API adapter wrapping py-clob-client."""

    def __init__(self):
        self._client: Optional[ClobClient] = None
        self._limiter = RateLimiter()
        self._connected = False

    async def connect(self) -> None:
        """Initialize ClobClient and set API credentials."""
        self._client = await asyncio.to_thread(
            ClobClient,
            settings.polymarket_host,
            key=settings.polymarket_wallet_private_key,
            chain_id=settings.polymarket_chain_id,
        )
        # Derive or set API credentials for authenticated endpoints
        if settings.polymarket_api_key:
            await asyncio.to_thread(
                self._client.set_api_creds,
                {
                    "apiKey": settings.polymarket_api_key,
                    "secret": settings.polymarket_api_secret,
                    "passphrase": settings.polymarket_api_passphrase,
                },
            )
        else:
            creds = await asyncio.to_thread(self._client.create_or_derive_api_creds)
            await asyncio.to_thread(self._client.set_api_creds, creds)

        self._connected = True
        logger.info("polymarket_connected", host=settings.polymarket_host)

    def _ensure_connected(self) -> None:
        if not self._connected or self._client is None:
            raise RuntimeError("PolymarketAdapter not connected. Call connect() first.")

    async def _retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Retry with exponential backoff."""
        for attempt in range(max_retries):
            try:
                await self._limiter.acquire()
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "exchange_call_failed",
                        func=func.__name__,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    raise
                wait = 2**attempt
                logger.warning(
                    "exchange_call_retry",
                    func=func.__name__,
                    attempt=attempt + 1,
                    wait=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)

    async def get_order_book(self, token_id: str) -> OrderBook:
        self._ensure_connected()
        try:
            raw = await self._retry(self._client.get_order_book, token_id)
        except Exception as e:
            if "404" in str(e) or "No orderbook" in str(e):
                # Market has no order book (delisted or ultra-thin) — return empty
                return OrderBook(bids=[], asks=[], timestamp=time.time())
            raise

        bids = [
            OrderBookEntry(price=float(b["price"]), size=float(b["size"]))
            for b in raw.get("bids", [])
        ]
        asks = [
            OrderBookEntry(price=float(a["price"]), size=float(a["size"]))
            for a in raw.get("asks", [])
        ]
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(bids=bids, asks=asks, timestamp=time.time())

    async def get_midpoint(self, token_id: str) -> float:
        self._ensure_connected()
        result = await self._retry(self._client.get_midpoint, token_id)
        return float(result)

    async def get_spread(self, token_id: str) -> float:
        self._ensure_connected()
        result = await self._retry(self._client.get_spread, token_id)
        return float(result)

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> OrderResult:
        """Place a GTC limit order on Polymarket."""
        self._ensure_connected()

        from py_clob_client.order_builder.constants import BUY, SELL

        clob_side = BUY if side.lower() == "buy" else SELL

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side,
            )
            signed_order = await self._retry(self._client.create_order, order_args)
            result = await self._retry(
                self._client.post_order, signed_order, OrderType.GTC
            )

            order_id = result.get("orderID", result.get("id", ""))
            logger.info(
                "order_placed",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                order_id=order_id,
            )
            return OrderResult(order_id=order_id, success=True)

        except Exception as e:
            logger.error(
                "order_failed",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                error=str(e),
            )
            return OrderResult(order_id="", success=False, message=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        try:
            await self._retry(self._client.cancel, order_id)
            logger.info("order_cancelled", order_id=order_id)
            return True
        except Exception as e:
            logger.error("cancel_failed", order_id=order_id, error=str(e))
            return False

    async def cancel_all(self) -> int:
        self._ensure_connected()
        try:
            result = await self._retry(self._client.cancel_all)
            count = len(result) if isinstance(result, list) else 0
            logger.info("all_orders_cancelled", count=count)
            return count
        except Exception as e:
            logger.error("cancel_all_failed", error=str(e))
            return 0

    async def get_open_orders(self) -> list[dict]:
        self._ensure_connected()
        try:
            return await self._retry(self._client.get_orders)
        except Exception:
            return []

    async def get_trades(self) -> list[TradeRecord]:
        self._ensure_connected()
        try:
            raw_trades = await self._retry(self._client.get_trades)
            return [
                TradeRecord(
                    trade_id=str(t.get("id", "")),
                    order_id=str(t.get("orderID", t.get("order_id", ""))),
                    token_id=str(t.get("asset_id", t.get("token_id", ""))),
                    side=t.get("side", "").lower(),
                    price=float(t.get("price", 0)),
                    size=float(t.get("size", 0)),
                    fee=float(t.get("fee_rate_bps", 0)) / 10000 * float(t.get("size", 0)) * float(t.get("price", 0)),
                    timestamp=float(t.get("created_at", 0)),
                )
                for t in (raw_trades if isinstance(raw_trades, list) else [])
            ]
        except Exception as e:
            logger.error("get_trades_failed", error=str(e))
            return []
