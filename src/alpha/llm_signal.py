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

        # Current market price (YES outcome)
        market_price = market.best_bid or market.last_price
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
