"""LLM alpha: reads probability estimates from the signal store."""

from typing import Optional

from src.alpha.base import AlphaOutput, AlphaSource
from src.db.models import Market, MarketSignal


class LLMAlpha(AlphaSource):
    """Alpha derived from Claude CLI probability estimates.

    Reads the latest fresh signal from the signal store.
    Edge = LLM probability - current market price.
    """

    name = "llm"

    async def compute(
        self,
        market: Market,
        context: dict,
    ) -> Optional[AlphaOutput]:
        signal: Optional[MarketSignal] = context.get("signal")
        if signal is None:
            return None

        # Use midpoint for fair value reference
        if market.best_bid is not None and market.best_ask is not None and market.best_bid > 0 and market.best_ask > 0:
            market_price = (market.best_bid + market.best_ask) / 2
        elif market.last_price is not None and market.last_price > 0:
            market_price = market.last_price
        else:
            market_price = market.best_bid if market.best_bid is not None and market.best_bid > 0 else None
        if market_price is None:
            return None

        edge = signal.probability - market_price

        return AlphaOutput(
            edge=edge,
            confidence=signal.confidence,
            notes=signal.reasoning or "",
            meta={
                "llm_probability": signal.probability,
                "market_price": market_price,
                "key_factors": signal.key_factors,
            },
        )
