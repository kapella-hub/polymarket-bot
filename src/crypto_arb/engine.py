"""
Crypto Latency Arbitrage Engine

Exploits the price lag between centralized exchanges (Binance) and
Polymarket crypto prediction markets.

Strategy:
1. Poll Binance for real-time BTC/ETH/SOL prices every 1-2 seconds
2. Fetch Polymarket crypto markets (e.g., "BTC above $72k on April 9?")
3. When Binance price crosses a threshold, the outcome is known/highly likely
   BEFORE Polymarket reprices the market
4. Buy the winning outcome at the stale (cheap) price
5. Collect $1.00 at resolution

Edge: pure execution speed. No prediction needed — we're trading on
information that already exists but hasn't been priced in yet.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    ABOVE = "above"  # "Will BTC be above $X?"
    BELOW = "below"  # "Will BTC dip to $X?"


@dataclass
class CryptoMarket:
    """A Polymarket crypto threshold market."""
    condition_id: str
    question: str
    asset: str  # "BTC", "ETH", "SOL"
    threshold: float  # The price threshold (e.g., 72000)
    direction: Direction  # above or below
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    end_date: Optional[datetime]
    volume: float


@dataclass
class Signal:
    """A detected latency arbitrage opportunity."""
    market: CryptoMarket
    side: str  # "buy_yes" or "buy_no"
    entry_price: float  # Price we'd pay
    expected_payout: float  # $1.00 if correct
    edge: float  # expected_payout - entry_price
    confidence: float  # How certain we are the threshold is crossed
    binance_price: float  # Current Binance price
    timestamp: datetime


@dataclass
class TradeLog:
    """Record of a paper/live trade."""
    signal: Signal
    size_usd: float
    token_qty: float
    entry_time: datetime
    status: str = "open"  # "open", "won", "lost"
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Binance Price Feed
# ---------------------------------------------------------------------------

class BinanceFeed:
    """Fast price polling from Binance REST API."""

    SYMBOLS = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
        "DOGE": "DOGEUSDT",
        "BNB": "BNBUSDT",
    }

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=5)
        self._prices: dict[str, float] = {}
        self._last_update: float = 0

    async def poll(self) -> dict[str, float]:
        """Get current prices for all tracked assets."""
        symbols = list(self.SYMBOLS.values())
        try:
            quoted = [f'"{s}"' for s in symbols]
            symbols_param = f'[{",".join(quoted)}]'
            resp = await self._client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbols": symbols_param},
            )
            data = resp.json()
            for item in data:
                symbol = item["symbol"]
                price = float(item["price"])
                # Reverse lookup: BTCUSDT -> BTC
                for asset, sym in self.SYMBOLS.items():
                    if sym == symbol:
                        self._prices[asset] = price
                        break
            self._last_update = time.time()
        except Exception as e:
            logger.warning("binance_poll_error", error=str(e))

        return self._prices

    def get(self, asset: str) -> Optional[float]:
        return self._prices.get(asset)

    @property
    def age_seconds(self) -> float:
        return time.time() - self._last_update if self._last_update else float("inf")

    async def close(self):
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Polymarket Crypto Market Scanner
# ---------------------------------------------------------------------------

class PolymarketCryptoScanner:
    """Fetches and parses crypto prediction markets from Polymarket."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=15, follow_redirects=True)
        self._markets: list[CryptoMarket] = []

    async def scan(self) -> list[CryptoMarket]:
        """Fetch all active crypto markets."""
        try:
            resp = await self._client.get("https://polymarket.com/api/crypto/markets")
            data = resp.json()
            events = data.get("events", [])

            markets = []
            for event in events:
                for m in event.get("markets", []):
                    parsed = self._parse_market(m, event)
                    if parsed:
                        markets.append(parsed)

            self._markets = markets
            return markets
        except Exception as e:
            logger.error("crypto_scan_error", error=str(e))
            return self._markets

    def _parse_market(self, raw: dict, event: dict) -> Optional[CryptoMarket]:
        """Parse a raw market response into CryptoMarket."""
        try:
            question = raw.get("question", "")
            q_lower = question.lower()

            # Detect asset
            asset = None
            for a in ["Bitcoin", "Ethereum", "Solana", "XRP", "Dogecoin", "BNB"]:
                if a.lower() in q_lower:
                    asset = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
                             "xrp": "XRP", "dogecoin": "DOGE", "bnb": "BNB"}[a.lower()]
                    break
            if not asset:
                return None

            # Detect direction and threshold
            direction = None
            threshold = None
            group_title = raw.get("groupItemTitle", "")

            if "above" in q_lower:
                direction = Direction.ABOVE
            elif "dip to" in q_lower or "below" in q_lower or "reach" in q_lower:
                # "reach" with upward arrow = above, "dip" = touches-below
                if "↓" in group_title or "dip" in q_lower:
                    direction = Direction.BELOW
                else:
                    direction = Direction.ABOVE

            # Extract threshold from question: "$72,000" or "$72000"
            import re
            price_match = re.search(r'\$([0-9,.]+)', question)
            if price_match:
                threshold = float(price_match.group(1).replace(",", ""))
            elif group_title:
                clean = group_title.replace("↑", "").replace("↓", "").replace(",", "").strip()
                try:
                    threshold = float(clean)
                except ValueError:
                    pass

            if direction is None or threshold is None or threshold <= 0:
                return None

            # Parse tokens
            import json as jsonmod
            clob_tokens = raw.get("clobTokenIds", [])
            outcomes = raw.get("outcomes", [])
            prices = raw.get("outcomePrices", [])

            if isinstance(clob_tokens, str):
                clob_tokens = jsonmod.loads(clob_tokens)
            if isinstance(outcomes, str):
                outcomes = jsonmod.loads(outcomes)
            if isinstance(prices, str):
                prices = jsonmod.loads(prices)

            if len(clob_tokens) < 2 or len(outcomes) < 2:
                return None

            # YES is always first outcome
            yes_idx = 0
            no_idx = 1
            for i, o in enumerate(outcomes):
                if o.lower() == "yes":
                    yes_idx = i
                elif o.lower() == "no":
                    no_idx = i

            yes_price = float(prices[yes_idx]) if yes_idx < len(prices) else 0
            no_price = float(prices[no_idx]) if no_idx < len(prices) else 0

            end_date = None
            end_str = raw.get("endDate")
            if end_str:
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            return CryptoMarket(
                condition_id=raw.get("conditionId", ""),
                question=question,
                asset=asset,
                threshold=threshold,
                direction=direction,
                yes_token_id=str(clob_tokens[yes_idx]),
                no_token_id=str(clob_tokens[no_idx]),
                yes_price=yes_price,
                no_price=no_price,
                best_bid=float(raw["bestBid"]) if raw.get("bestBid") else None,
                best_ask=float(raw["bestAsk"]) if raw.get("bestAsk") else None,
                end_date=end_date,
                volume=float(raw.get("volumeNum", 0)),
            )
        except Exception as e:
            logger.debug("crypto_market_parse_error", error=str(e))
            return None

    async def close(self):
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Arbitrage Signal Detector
# ---------------------------------------------------------------------------

class ArbDetector:
    """Detects latency arbitrage opportunities by comparing Binance prices
    to Polymarket market thresholds."""

    def __init__(
        self,
        min_edge: float = 0.30,  # Minimum 30% expected return to trade
        min_confidence: float = 0.85,  # Only trade when >85% confident
        buffer_pct: float = 0.005,  # 0.5% buffer above threshold for confidence
    ):
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.buffer_pct = buffer_pct

    def scan(
        self,
        markets: list[CryptoMarket],
        prices: dict[str, float],
    ) -> list[Signal]:
        """Scan all markets for arbitrage signals."""
        signals = []
        now = datetime.now(timezone.utc)

        for market in markets:
            sig = self._check_market(market, prices, now)
            if sig:
                signals.append(sig)

        return signals

    def _check_market(
        self,
        market: CryptoMarket,
        prices: dict[str, float],
        now: datetime,
    ) -> Optional[Signal]:
        """Check if a single market has an arb opportunity."""
        binance_price = prices.get(market.asset)
        if binance_price is None:
            return None

        # Skip markets that already expired
        if market.end_date and now > market.end_date:
            return None

        # Determine if the outcome is known/highly likely
        if market.direction == Direction.ABOVE:
            # "Will BTC be above $72,000?" — YES if binance > threshold
            distance_pct = (binance_price - market.threshold) / market.threshold

            if distance_pct > self.buffer_pct:
                # Price is ABOVE threshold — YES should be ~$1.00
                confidence = min(0.99, 0.85 + distance_pct * 5)
                if market.yes_price < (1.0 - self.min_edge):
                    # YES is underpriced — BUY YES
                    entry = market.best_ask if market.best_ask else market.yes_price
                    if entry >= 0.95:  # Already priced in
                        return None
                    return Signal(
                        market=market,
                        side="buy_yes",
                        entry_price=entry,
                        expected_payout=1.0,
                        edge=1.0 - entry,
                        confidence=confidence,
                        binance_price=binance_price,
                        timestamp=now,
                    )

            elif distance_pct < -self.buffer_pct:
                # Price is BELOW threshold — NO should be ~$1.00
                confidence = min(0.99, 0.85 + abs(distance_pct) * 5)

                # But only if there's enough time remaining OR end is very close
                # Close to expiry with price below = high confidence NO
                if market.end_date:
                    hours_left = (market.end_date - now).total_seconds() / 3600
                    # Need bigger buffer if more time left (price could recover)
                    if hours_left > 24 and abs(distance_pct) < 0.05:
                        return None  # Too much time, price could recover
                    if hours_left > 72:
                        return None  # Way too far out

                if market.no_price < (1.0 - self.min_edge):
                    entry = 1.0 - (market.best_bid if market.best_bid else market.yes_price)
                    if entry >= 0.95:
                        return None
                    return Signal(
                        market=market,
                        side="buy_no",
                        entry_price=entry,
                        expected_payout=1.0,
                        edge=1.0 - entry,
                        confidence=confidence,
                        binance_price=binance_price,
                        timestamp=now,
                    )

        elif market.direction == Direction.BELOW:
            # "Will BTC dip to $60,000?" — YES if binance ever touches below
            # This is different — it's a "touch" market, not "settle above"
            distance_pct = (binance_price - market.threshold) / market.threshold

            if distance_pct < 0:
                # Price is BELOW threshold — YES has been triggered
                confidence = 0.99
                if market.yes_price < (1.0 - self.min_edge):
                    entry = market.best_ask if market.best_ask else market.yes_price
                    if entry >= 0.95:
                        return None
                    return Signal(
                        market=market,
                        side="buy_yes",
                        entry_price=entry,
                        expected_payout=1.0,
                        edge=1.0 - entry,
                        confidence=confidence,
                        binance_price=binance_price,
                        timestamp=now,
                    )

        return None


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTrader:
    """Simulates trades and tracks P&L."""

    def __init__(self, bankroll: float = 1000.0, max_per_trade: float = 200.0):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.max_per_trade = max_per_trade
        self.trades: list[TradeLog] = []
        self.open_positions: dict[str, TradeLog] = {}  # condition_id -> trade

    def execute(self, signal: Signal) -> Optional[TradeLog]:
        """Paper-execute a signal."""
        # Skip if we already have a position in this market
        if signal.market.condition_id in self.open_positions:
            return None

        # Size: fraction of bankroll, capped
        size = min(
            self.max_per_trade,
            self.bankroll * 0.15,  # Max 15% per trade
            self.bankroll,
        )
        if size < 5.0:
            return None

        token_qty = size / signal.entry_price
        self.bankroll -= size

        trade = TradeLog(
            signal=signal,
            size_usd=size,
            token_qty=token_qty,
            entry_time=signal.timestamp,
        )
        self.trades.append(trade)
        self.open_positions[signal.market.condition_id] = trade

        logger.info(
            "paper_trade",
            market=signal.market.question[:60],
            side=signal.side,
            entry=f"${signal.entry_price:.4f}",
            size=f"${size:.2f}",
            edge=f"{signal.edge:.1%}",
            confidence=f"{signal.confidence:.1%}",
            binance=f"${signal.binance_price:,.2f}",
            threshold=f"${signal.market.threshold:,.0f}",
        )
        return trade

    def mark_resolved(self, condition_id: str, outcome_yes: bool) -> Optional[float]:
        """Mark a position as resolved and calculate P&L."""
        trade = self.open_positions.pop(condition_id, None)
        if not trade:
            return None

        won = (trade.signal.side == "buy_yes" and outcome_yes) or \
              (trade.signal.side == "buy_no" and not outcome_yes)

        if won:
            payout = trade.token_qty * 1.0  # $1 per token
            trade.pnl = payout - trade.size_usd
            trade.status = "won"
            self.bankroll += payout
        else:
            trade.pnl = -trade.size_usd
            trade.status = "lost"

        logger.info(
            "paper_resolved",
            market=trade.signal.market.question[:60],
            status=trade.status,
            pnl=f"${trade.pnl:+.2f}",
            bankroll=f"${self.bankroll:.2f}",
        )
        return trade.pnl

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.status != "open")

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.status != "open"]
        if not closed:
            return 0
        return sum(1 for t in closed if t.status == "won") / len(closed)

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.status != "open"]
        return {
            "total_trades": len(self.trades),
            "open": len(self.open_positions),
            "closed": len(closed),
            "won": sum(1 for t in closed if t.status == "won"),
            "lost": sum(1 for t in closed if t.status == "lost"),
            "win_rate": f"{self.win_rate:.1%}",
            "total_pnl": f"${self.total_pnl:+.2f}",
            "bankroll": f"${self.bankroll:.2f}",
            "roi": f"{(self.bankroll / self.initial_bankroll - 1) * 100:+.1f}%",
        }


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

class CryptoArbEngine:
    """Main engine: polls prices, scans markets, detects arbs, trades."""

    def __init__(
        self,
        mode: str = "paper",  # "paper" or "live"
        bankroll: float = 1000.0,
        poll_interval: float = 2.0,  # seconds between Binance polls
        scan_interval: float = 30.0,  # seconds between market scans
    ):
        self.mode = mode
        self.feed = BinanceFeed()
        self.scanner = PolymarketCryptoScanner()
        self.detector = ArbDetector()
        self.trader = PaperTrader(bankroll=bankroll)
        self.poll_interval = poll_interval
        self.scan_interval = scan_interval
        self._markets: list[CryptoMarket] = []
        self._running = False

    async def run(self, duration_seconds: float = 300):
        """Run the arb engine for a specified duration."""
        self._running = True
        start = time.time()

        logger.info(
            "crypto_arb_starting",
            mode=self.mode,
            bankroll=f"${self.trader.bankroll:.2f}",
            duration=f"{duration_seconds}s",
        )

        # Initial market scan
        self._markets = await self.scanner.scan()
        logger.info("crypto_markets_loaded", count=len(self._markets))

        last_scan = time.time()
        cycle = 0

        while self._running and (time.time() - start) < duration_seconds:
            cycle += 1

            # Poll Binance
            prices = await self.feed.poll()

            # Periodic market rescan
            if time.time() - last_scan > self.scan_interval:
                self._markets = await self.scanner.scan()
                last_scan = time.time()

            # Detect arb signals
            signals = self.detector.scan(self._markets, prices)

            for sig in signals:
                if self.mode == "paper":
                    self.trader.execute(sig)
                elif self.mode == "live":
                    # TODO: actual order placement via CLOB
                    pass

            # Log status periodically
            if cycle % 30 == 0:  # Every ~60 seconds
                self._log_status(prices, signals)

            await asyncio.sleep(self.poll_interval)

        # Final summary
        logger.info("crypto_arb_stopped", summary=self.trader.summary())
        await self.feed.close()
        await self.scanner.close()

        return self.trader.summary()

    def stop(self):
        self._running = False

    def _log_status(self, prices: dict, signals: list):
        asset_str = " | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items()) if a in ("BTC", "ETH", "SOL"))
        logger.info(
            "crypto_arb_cycle",
            prices=asset_str,
            markets=len(self._markets),
            signals=len(signals),
            open_positions=len(self.trader.open_positions),
            bankroll=f"${self.trader.bankroll:.2f}",
            pnl=f"${self.trader.total_pnl:+.2f}",
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def main():
    """Run the crypto arb engine in paper mode."""
    import sys

    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0

    engine = CryptoArbEngine(
        mode="paper",
        bankroll=bankroll,
        poll_interval=2.0,
        scan_interval=30.0,
    )

    print(f"Starting crypto arb engine (paper mode, ${bankroll:.0f} bankroll, {duration}s)")
    print("Press Ctrl+C to stop\n")

    try:
        result = await engine.run(duration_seconds=duration)
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")
    except KeyboardInterrupt:
        engine.stop()
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
