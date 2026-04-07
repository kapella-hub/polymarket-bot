"""Order book microstructure alpha."""

from typing import Optional

from src.alpha.base import AlphaOutput, AlphaSource
from src.db.models import Market
from src.exchange.base import OrderBook


class OrderBookAlpha(AlphaSource):
    """Alpha derived from order book microstructure.

    Signals:
    - Bid-ask imbalance: if bids >> asks, buyers are more aggressive (bullish)
    - Depth asymmetry: total bid depth vs ask depth near midpoint
    - Spread width: wider spreads suggest uncertainty
    """

    name = "orderbook"

    def __init__(self, depth_levels: int = 5):
        self._depth_levels = depth_levels

    async def compute(
        self,
        market: Market,
        context: dict,
    ) -> Optional[AlphaOutput]:
        book: Optional[OrderBook] = context.get("order_book")
        if book is None or not book.bids or not book.asks:
            return None

        midpoint = book.midpoint
        if midpoint is None:
            return None

        # Compute bid-ask imbalance (top N levels)
        bid_depth = sum(
            b.size for b in book.bids[: self._depth_levels]
        )
        ask_depth = sum(
            a.size for a in book.asks[: self._depth_levels]
        )
        total_depth = bid_depth + ask_depth
        if total_depth == 0:
            return None

        # Imbalance: +1 = all bids, -1 = all asks, 0 = balanced
        imbalance = (bid_depth - ask_depth) / total_depth

        # Spread as a fraction of midpoint
        spread = book.spread or 0
        spread_pct = spread / midpoint if midpoint > 0 else 0

        # Convert imbalance to directional edge estimate
        # Positive imbalance (more bids) suggests YES is underpriced
        # Scale: 0.02 edge per unit of imbalance (conservative)
        edge = imbalance * 0.02

        # Confidence inversely related to spread (wide spread = uncertain)
        # and proportional to total depth (more depth = more reliable)
        confidence = min(0.3, max(0.05, (1.0 - spread_pct * 5) * 0.3))

        return AlphaOutput(
            edge=edge,
            confidence=confidence,
            notes=f"imbalance={imbalance:+.2f} spread={spread:.4f}",
            meta={
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "imbalance": imbalance,
                "spread": spread,
                "midpoint": midpoint,
            },
        )
