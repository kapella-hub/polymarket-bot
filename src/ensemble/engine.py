"""Ensemble engine: combines alpha signals into trade decisions."""

from datetime import datetime, timezone
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

        # Adaptive edge threshold: lower for near-resolution or high-confidence
        threshold = settings.min_edge_threshold
        if market.end_date is not None:
            now = datetime.now(timezone.utc)
            days_to_end = (market.end_date - now).total_seconds() / 86400
            if 1 <= days_to_end <= 14:
                # Up to 50% threshold reduction near resolution
                time_factor = max(0.5, days_to_end / 14)
                threshold *= time_factor
        if combined_confidence > 0.7:
            threshold *= 0.8  # 20% reduction for high confidence
        threshold = max(threshold, 0.02)  # Hard floor at 2 cents

        if abs(combined_edge) < threshold:
            return None

        # --- Bayesian Combo Strategy Filter ---
        if settings.strategy_filter_enabled:
            filtered = self._bayesian_filter(
                market, context, combined_edge, combined_confidence,
            )
            if filtered:
                logger.info(
                    "strategy_filtered",
                    market_id=market.id,
                    reason=filtered,
                    edge=f"{combined_edge:+.3f}",
                    confidence=f"{combined_confidence:.2f}",
                )
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

        # Kelly criterion position sizing: f* = (p*b - q) / b
        market_price = market.best_bid or market.last_price or market.best_ask
        if market_price is None:
            return None
        if side == "buy" and token_id == market.clob_token_id_no:
            market_price = 1.0 - market_price  # NO token price

        if 0 < market_price < 1:
            b = (1.0 / market_price) - 1.0  # decimal odds
            prob = market_price + combined_edge  # estimated true prob
            prob = max(0.01, min(0.99, prob))
            q = 1.0 - prob
            kelly = (prob * b - q) / b if b > 0 else 0
            kelly = max(kelly, 0)
        else:
            kelly = 0
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

    @staticmethod
    def _bayesian_filter(
        market: Market,
        context: dict,
        combined_edge: float,
        combined_confidence: float,
    ) -> str:
        """Bayesian combo filter: require multiple signals to align before trading.

        Returns empty string if trade passes, or rejection reason if filtered.
        Scoring: edge size, confidence, directional conviction, category.
        Must meet min_bayesian_score (default 2) to trade.
        """
        # Category blacklist — hard reject
        blacklist = {
            c.strip()
            for c in settings.category_blacklist.split(",")
            if c.strip()
        }
        if market.category and market.category in blacklist:
            return f"category_blacklisted:{market.category}"

        abs_edge = abs(combined_edge)

        # Edge * confidence product gate
        if abs_edge * combined_confidence < settings.min_edge_confidence_product:
            return f"edge_conf_product:{abs_edge * combined_confidence:.3f}<{settings.min_edge_confidence_product}"

        # Volume-adjusted edge: higher volume markets need more edge
        vol_scale = min(market.volume / 2_000_000, 1.0)
        required_edge = settings.volume_edge_base + settings.volume_edge_scale * vol_scale
        if abs_edge < required_edge:
            return f"volume_adj_edge:{abs_edge:.3f}<{required_edge:.3f}"

        # Bayesian score: accumulate evidence from independent signals
        score = 0

        # Signal 1: meaningful edge
        if abs_edge > 0.05:
            score += 1
        if abs_edge > 0.12:
            score += 1

        # Signal 2: confidence
        if combined_confidence >= 0.80:
            score += 1

        # Signal 3: directional conviction from LLM signal
        signal = context.get("signal")
        if signal and hasattr(signal, "probability"):
            prob = signal.probability
            thresh = settings.min_conviction_prob
            if prob < thresh or prob > (1.0 - thresh):
                score += 1

        if score < settings.min_bayesian_score:
            return f"bayesian_score:{score}<{settings.min_bayesian_score}"

        return ""
