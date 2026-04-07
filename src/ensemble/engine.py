"""Ensemble engine: combines alpha signals into trade decisions."""

from typing import Optional

import structlog

from src.alpha.base import AlphaOutput, AlphaSource
from src.config import settings
from src.db.models import Market, StrategyMode
from src.ensemble.strategies import TradeDecision

logger = structlog.get_logger()


class EnsembleEngine:
    """Combines multiple alpha sources into trade decisions.

    Uses confidence-weighted averaging to combine alpha outputs.
    Applies minimum edge threshold before generating trade decisions.
    """

    def __init__(self, alphas: list[AlphaSource]):
        self._alphas = alphas

    async def evaluate(
        self,
        market: Market,
        context: dict,
    ) -> Optional[TradeDecision]:
        """Run all alphas and combine into a trade decision.

        Returns TradeDecision if combined edge exceeds threshold, else None.
        """
        outputs: list[tuple[AlphaSource, AlphaOutput]] = []

        for alpha in self._alphas:
            try:
                result = await alpha.compute(market, context)
                if result is not None:
                    outputs.append((alpha, result))
            except Exception as e:
                logger.warning(
                    "alpha_error",
                    alpha=alpha.name,
                    market_id=market.id,
                    error=str(e),
                )

        if not outputs:
            return None

        # Confidence-weighted average of edges
        total_weight = sum(o.confidence for _, o in outputs)
        if total_weight == 0:
            return None

        combined_edge = sum(o.edge * o.confidence for _, o in outputs) / total_weight
        combined_confidence = total_weight / len(outputs)  # Average confidence

        # Minimum edge threshold
        if abs(combined_edge) < settings.min_edge_threshold:
            return None

        # Determine trade direction
        if combined_edge > 0:
            # YES is underpriced — buy YES
            side = "buy"
            token_id = market.clob_token_id_yes
        else:
            # YES is overpriced — buy NO (equivalent to selling YES)
            side = "buy"
            token_id = market.clob_token_id_no
            combined_edge = abs(combined_edge)  # Flip to positive for sizing

        # Kelly criterion position sizing
        market_price = market.best_bid or market.last_price or 0.5
        if side == "buy" and token_id == market.clob_token_id_no:
            market_price = 1.0 - market_price  # NO token price

        odds = (1.0 / market_price - 1.0) if market_price > 0 and market_price < 1 else 1.0
        kelly = combined_edge / odds if odds > 0 else 0
        kelly_sized = kelly * settings.kelly_fraction  # Fractional Kelly

        # Convert to USDC size (capped by per-market limit)
        # Bankroll is not tracked here — risk controller will cap it
        suggested_size = min(
            kelly_sized * 10000,  # Placeholder bankroll scaling
            settings.max_position_per_market_usd,
        )
        suggested_size = max(suggested_size, 0)

        notes_parts = [f"{a.name}:{o.edge:+.3f}@{o.confidence:.2f}" for a, o in outputs]

        logger.info(
            "ensemble_decision",
            market_id=market.id,
            combined_edge=f"{combined_edge:+.3f}",
            confidence=f"{combined_confidence:.2f}",
            side=side,
            token=token_id[:16],
            kelly=f"{kelly:.4f}",
            size=f"${suggested_size:.2f}",
            alphas=", ".join(notes_parts),
        )

        return TradeDecision(
            market_id=market.id,
            strategy=StrategyMode.INFORMATION,
            side=side,
            token_id=token_id,
            edge=combined_edge,
            confidence=combined_confidence,
            suggested_size=suggested_size,
            notes="; ".join(notes_parts),
        )
