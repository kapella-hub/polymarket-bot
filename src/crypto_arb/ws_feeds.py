"""
WebSocket price feeds for sub-second latency.

Replaces REST polling with persistent WebSocket streams from Binance.
Price updates arrive in ~10ms instead of every 2000ms.
"""

import asyncio
import json
import time
from typing import Callable, Optional

import structlog
import websockets

logger = structlog.get_logger()


class BinanceWSFeed:
    """Real-time price feed via Binance WebSocket.

    Subscribes to miniTicker streams for all tracked coins.
    Pushes price updates via callback within ~10ms of exchange update.
    """

    WS_URL = "wss://stream.binance.com:9443/ws"

    SYMBOLS = {
        "btcusdt": "BTC",
        "ethusdt": "ETH",
        "solusdt": "SOL",
        "xrpusdt": "XRP",
        "dogeusdt": "DOGE",
        "bnbusdt": "BNB",
    }

    def __init__(self, on_price: Optional[Callable] = None):
        self._prices: dict[str, float] = {}
        self._last_update: dict[str, float] = {}
        self._on_price = on_price  # Callback: (asset, price, timestamp) -> None
        self._ws = None
        self._running = False
        self._connect_count = 0

    @property
    def prices(self) -> dict[str, float]:
        return dict(self._prices)

    def get(self, asset: str) -> Optional[float]:
        return self._prices.get(asset)

    def age_ms(self, asset: str) -> float:
        """Milliseconds since last update for an asset."""
        last = self._last_update.get(asset, 0)
        return (time.time() - last) * 1000 if last else float("inf")

    async def run(self):
        """Connect and stream prices. Auto-reconnects on failure."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                if not self._running:
                    break
                self._connect_count += 1
                wait = min(30, 2 ** min(self._connect_count, 5))
                logger.warning("binance_ws_reconnecting", error=str(e), wait=wait)
                await asyncio.sleep(wait)

    async def _connect_and_stream(self):
        """Single connection lifecycle."""
        # Subscribe to all coin miniTickers via combined stream
        streams = [f"{sym}@miniTicker" for sym in self.SYMBOLS.keys()]
        url = f"{self.WS_URL}/{'/'.join(streams)}"

        # Use the combined stream URL format
        combined_url = "wss://stream.binance.com:9443/stream?streams=" + "/".join(streams)

        async with websockets.connect(combined_url, ping_interval=20) as ws:
            self._ws = ws
            self._connect_count = 0
            logger.info("binance_ws_connected", symbols=len(self.SYMBOLS))

            async for msg in ws:
                if not self._running:
                    break

                try:
                    data = json.loads(msg)
                    # Combined stream wraps in {"stream": "...", "data": {...}}
                    payload = data.get("data", data)
                    self._process_ticker(payload)
                except Exception as e:
                    logger.debug("binance_ws_parse_error", error=str(e))

    def _process_ticker(self, data: dict):
        """Process a miniTicker message."""
        symbol = data.get("s", "").lower()
        asset = self.SYMBOLS.get(symbol)
        if not asset:
            return

        price = float(data.get("c", 0))  # "c" = close/current price
        if price <= 0:
            return

        self._prices[asset] = price
        self._last_update[asset] = time.time()

        if self._on_price:
            self._on_price(asset, price, time.time())

    def stop(self):
        self._running = False
        if self._ws:
            asyncio.ensure_future(self._ws.close())


class PolymarketWSFeed:
    """Real-time orderbook updates from Polymarket WebSocket.

    Subscribes to price changes for specific token IDs.
    Used to detect when Polymarket reprices (our arb window is closing).
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, on_price: Optional[Callable] = None):
        self._token_prices: dict[str, float] = {}
        self._on_price = on_price
        self._ws = None
        self._running = False
        self._subscribed_tokens: set[str] = set()

    async def run(self):
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                if not self._running:
                    break
                logger.warning("polymarket_ws_reconnecting", error=str(e))
                await asyncio.sleep(5)

    async def _connect_and_stream(self):
        async with websockets.connect(self.WS_URL, ping_interval=20) as ws:
            self._ws = ws
            logger.info("polymarket_ws_connected")

            # Re-subscribe to any pending tokens
            for token_id in self._subscribed_tokens:
                await self._subscribe(ws, token_id)

            async for msg in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(msg)
                    self._process_message(data)
                except Exception:
                    pass

    def _process_message(self, data: dict):
        """Process price update messages."""
        # Polymarket WS sends various message types
        msg_type = data.get("type", "")
        if msg_type in ("price_change", "last_trade_price"):
            token_id = data.get("asset_id", data.get("token_id", ""))
            price = float(data.get("price", 0))
            if token_id and price > 0:
                self._token_prices[token_id] = price
                if self._on_price:
                    self._on_price(token_id, price, time.time())

    async def subscribe(self, token_id: str):
        """Subscribe to price updates for a token."""
        self._subscribed_tokens.add(token_id)
        if self._ws:
            await self._subscribe(self._ws, token_id)

    async def _subscribe(self, ws, token_id: str):
        """Send subscription message."""
        try:
            await ws.send(json.dumps({
                "type": "subscribe",
                "channel": "price",
                "assets_ids": [token_id],
            }))
        except Exception:
            pass

    def get_price(self, token_id: str) -> Optional[float]:
        return self._token_prices.get(token_id)

    def stop(self):
        self._running = False
        if self._ws:
            asyncio.ensure_future(self._ws.close())
