"""Smart money alpha: track profitable Polymarket wallets and follow their trades.

Uses public Polymarket APIs to:
1. Identify consistently profitable traders from the leaderboard
2. Monitor their current positions
3. Generate alpha signals when smart money enters a market

No auth needed — leaderboard, profiles, and positions are all public endpoints.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

from src.alpha.base import AlphaOutput, AlphaSource
from src.db.models import Market

logger = structlog.get_logger()

POLYMARKET_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
PROFILE_API = "https://polymarket.com/api"


@dataclass
class SmartTrader:
    """A consistently profitable Polymarket trader."""

    address: str
    pnl: float = 0.0
    volume: float = 0.0
    markets_traded: int = 0
    win_rate: Optional[float] = None
    rank: int = 0


@dataclass
class SmartMoneyPosition:
    """A position held by a smart money wallet."""

    trader_address: str
    market_id: str  # condition_id
    token_id: str
    outcome: str
    size: float
    avg_price: float
    trader_pnl: float  # Trader's overall PnL (credibility signal)


class SmartMoneyTracker:
    """Tracks top Polymarket traders and their positions.

    Refreshes the leaderboard periodically and monitors
    top traders' positions for alpha signals.
    """

    def __init__(
        self,
        top_n: int = 25,
        min_pnl: float = 50_000,
        refresh_interval: int = 3600,  # 1 hour
    ):
        self._top_n = top_n
        self._min_pnl = min_pnl
        self._refresh_interval = refresh_interval
        self._client = httpx.AsyncClient(timeout=20, follow_redirects=True)

        # Cache
        self._traders: list[SmartTrader] = []
        self._positions: dict[str, list[SmartMoneyPosition]] = {}  # market_id -> positions
        self._last_refresh: float = 0

    async def close(self) -> None:
        await self._client.aclose()

    async def refresh_if_stale(self) -> None:
        """Refresh leaderboard and positions if cache is stale."""
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval and self._traders:
            return

        await self._refresh_leaderboard()
        await self._refresh_positions()
        self._last_refresh = now

    async def get_smart_money_for_market(
        self, market_id: str
    ) -> list[SmartMoneyPosition]:
        """Get smart money positions for a specific market."""
        await self.refresh_if_stale()
        return self._positions.get(market_id, [])

    async def _refresh_leaderboard(self) -> None:
        """Fetch top traders from Polymarket leaderboard."""
        try:
            # Polymarket leaderboard API
            resp = await self._client.get(
                f"{GAMMA_API}/leaderboard",
                params={"limit": self._top_n * 2, "window": "all"},
            )

            if resp.status_code != 200:
                # Fallback: try the CLOB API leaderboard
                resp = await self._client.get(
                    f"{POLYMARKET_API}/leaderboard",
                    params={"limit": self._top_n * 2},
                )

            if resp.status_code != 200:
                logger.warning("leaderboard_fetch_failed", status=resp.status_code)
                return

            data = resp.json()
            traders = []

            # Handle different response formats
            entries = data if isinstance(data, list) else data.get("data", data.get("results", []))

            for entry in entries:
                pnl = float(entry.get("pnl", entry.get("profit", entry.get("totalPnl", 0))))
                if pnl < self._min_pnl:
                    continue

                traders.append(SmartTrader(
                    address=entry.get("address", entry.get("userAddress", entry.get("wallet", ""))),
                    pnl=pnl,
                    volume=float(entry.get("volume", entry.get("totalVolume", 0))),
                    markets_traded=int(entry.get("marketsTraded", entry.get("numMarkets", 0))),
                    rank=len(traders) + 1,
                ))

                if len(traders) >= self._top_n:
                    break

            self._traders = traders
            logger.info(
                "leaderboard_refreshed",
                top_traders=len(traders),
                top_pnl=f"${traders[0].pnl:,.0f}" if traders else "N/A",
            )

        except Exception as e:
            logger.warning("leaderboard_error", error=str(e))

    async def _refresh_positions(self) -> None:
        """Fetch current positions for all tracked traders."""
        self._positions.clear()

        for trader in self._traders[:10]:  # Top 10 to limit API calls
            try:
                positions = await self._fetch_trader_positions(trader)
                for pos in positions:
                    if pos.market_id not in self._positions:
                        self._positions[pos.market_id] = []
                    self._positions[pos.market_id].append(pos)
            except Exception as e:
                logger.debug(
                    "trader_positions_error",
                    address=trader.address[:10],
                    error=str(e),
                )

            await asyncio.sleep(0.5)  # Rate limit courtesy

        total_positions = sum(len(v) for v in self._positions.values())
        logger.info(
            "positions_refreshed",
            traders=len(self._traders[:10]),
            markets_with_positions=len(self._positions),
            total_positions=total_positions,
        )

    async def _fetch_trader_positions(
        self, trader: SmartTrader
    ) -> list[SmartMoneyPosition]:
        """Fetch positions for a single trader."""
        positions = []

        try:
            resp = await self._client.get(
                f"{PROFILE_API}/profile/{trader.address}/positions",
                params={"limit": 50, "status": "open"},
            )

            if resp.status_code != 200:
                # Fallback: try CLOB API
                resp = await self._client.get(
                    f"{POLYMARKET_API}/positions",
                    params={"user": trader.address},
                )

            if resp.status_code != 200:
                return []

            data = resp.json()
            entries = data if isinstance(data, list) else data.get("data", data.get("positions", []))

            for entry in entries:
                market_id = entry.get("conditionId", entry.get("market_id", entry.get("condition_id", "")))
                if not market_id:
                    continue

                size = float(entry.get("size", entry.get("amount", 0)))
                if size <= 0:
                    continue

                positions.append(SmartMoneyPosition(
                    trader_address=trader.address,
                    market_id=market_id,
                    token_id=entry.get("tokenId", entry.get("asset_id", "")),
                    outcome=entry.get("outcome", entry.get("title", "Unknown")),
                    size=size,
                    avg_price=float(entry.get("avgPrice", entry.get("avg_price", 0))),
                    trader_pnl=trader.pnl,
                ))

        except Exception as e:
            logger.debug("fetch_positions_error", error=str(e))

        return positions


class SmartMoneyAlpha(AlphaSource):
    """Alpha source: smart money wallet positioning.

    When multiple profitable traders hold positions in a market,
    that's a directional signal weighted by their profitability.
    """

    name = "smart_money"

    def __init__(self, tracker: Optional[SmartMoneyTracker] = None):
        self._tracker = tracker or SmartMoneyTracker()

    async def compute(
        self,
        market: Market,
        context: dict,
    ) -> Optional[AlphaOutput]:
        positions = await self._tracker.get_smart_money_for_market(market.id)

        if not positions:
            return None

        # Aggregate: what direction are smart traders leaning?
        yes_weight = 0.0
        no_weight = 0.0

        for pos in positions:
            # Weight by trader PnL (more profitable = stronger signal)
            credibility = min(pos.trader_pnl / 100_000, 3.0)  # Cap at 3x

            if pos.outcome.lower() in ("yes", "true", "1"):
                yes_weight += pos.size * credibility
            else:
                no_weight += pos.size * credibility

        total = yes_weight + no_weight
        if total == 0:
            return None

        # Smart money consensus: >0 = bullish on YES, <0 = bullish on NO
        consensus = (yes_weight - no_weight) / total  # -1 to +1

        # Convert to edge: scale by confidence factor
        market_price = market.best_bid or market.last_price or 0.5
        edge = consensus * 0.10  # Max 10% edge from smart money alone

        confidence = min(0.5, len(positions) * 0.1)  # More traders = more confident

        return AlphaOutput(
            edge=edge,
            confidence=confidence,
            notes=f"{len(positions)} smart traders, consensus={consensus:+.2f}",
            meta={
                "yes_weight": yes_weight,
                "no_weight": no_weight,
                "num_traders": len(positions),
                "top_trader_pnl": max(p.trader_pnl for p in positions),
            },
        )
