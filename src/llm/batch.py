"""Batch scheduler for LLM market evaluations."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from src.config import settings
from src.db.database import async_session
from src.db.repositories import MarketRepository, PositionRepository, SignalRepository
from src.enrichment.cross_platform import CrossPlatformScraper
from src.llm.claude_runner import ClaudeRunner
from src.llm.power_prompt import build_power_prompt
from src.markets.models import MarketInfo, Outcome

logger = structlog.get_logger()


class BatchScheduler:
    """Schedules and runs batched LLM market evaluations.

    Prioritization:
    1. Markets with open positions + stale signal (risk)
    2. New markets never evaluated (opportunity)
    3. Markets with high volume + stale signal
    4. Markets with fresh signals (skip)
    """

    def __init__(
        self,
        runner: Optional[ClaudeRunner] = None,
        scraper: Optional[CrossPlatformScraper] = None,
        nexus=None,
    ):
        self._runner = runner or ClaudeRunner()
        self._scraper = scraper or CrossPlatformScraper()
        self._nexus = nexus

    async def run_batch(self) -> int:
        """Run one evaluation batch. Returns count evaluated."""
        async with async_session() as session:
            market_repo = MarketRepository(session)
            signal_repo = SignalRepository(session)
            position_repo = PositionRepository(session)

            markets = await market_repo.get_active()
            if not markets:
                logger.info("batch_skip_no_markets")
                return 0

            market_ids = [m.id for m in markets]
            needing_ids = await signal_repo.get_markets_needing_eval(
                market_ids, settings.llm_signal_ttl_seconds
            )
            positions = await position_repo.get_all()
            position_market_ids = {p.market_id for p in positions}

        prioritized = self._prioritize(markets, needing_ids, position_market_ids)
        batch = prioritized[: settings.llm_batch_size]

        if not batch:
            logger.info("batch_skip_all_fresh")
            return 0

        logger.info(
            "batch_starting",
            total_candidates=len(prioritized),
            batch_size=len(batch),
        )

        count = 0
        for db_market in batch:
            market_info = self._db_to_info(db_market)
            if await self._run_single(market_info):
                count += 1

        logger.info("batch_complete", evaluated=count, attempted=len(batch))
        return count

    def _prioritize(self, markets, needing_ids: list[str], position_ids: set[str]):
        """Sort markets by evaluation priority."""
        needing_set = set(needing_ids)

        def sort_key(m):
            needs = m.id in needing_set
            has_pos = m.id in position_ids
            # Lower = higher priority
            if needs and has_pos:
                tier = 0  # Position at risk with stale signal
            elif needs:
                tier = 1  # Needs evaluation
            else:
                tier = 10  # Fresh signal
            return (tier, -m.volume)

        return sorted(markets, key=sort_key)

    async def _run_single(self, market: MarketInfo) -> bool:
        """Evaluate one market via power prompt with cross-platform intel + Cortex calibration."""
        logger.info(
            "evaluating_market",
            market_id=market.id,
            question=market.question[:80],
        )

        # 1. Recall past calibration from NexusCortex
        calibration_note = ""
        if self._nexus:
            cal = await self._nexus.recall_calibration(market.question, market.category)
            if cal:
                calibration_note = cal
                logger.info("calibration_recalled", market_id=market.id, preview=cal[:100])

        # 2. Gather cross-platform intelligence (independent signals)
        cross_platform = await self._scraper.gather(market.question, market.category)

        # 3. Build power prompt with all context + calibration
        prompt = build_power_prompt(market, cross_platform, calibration_note)

        result = await self._runner.evaluate(prompt)
        if result is None:
            logger.warning("market_eval_failed", market_id=market.id)
            return False

        market_price = market.yes_price or 0.5
        edge = result.probability - market_price
        now = datetime.now(timezone.utc)

        signal_data = {
            "market_id": market.id,
            "probability": result.probability,
            "confidence": result.confidence,
            "edge_over_market": edge,
            "reasoning": result.reasoning,
            "key_factors": result.key_factors,
            "market_price_at_eval": market_price,
            "evaluated_at": now,
            "expires_at": now + timedelta(seconds=settings.llm_signal_ttl_seconds),
        }

        async with async_session() as session:
            repo = SignalRepository(session)
            await repo.write_signal(signal_data)
            await session.commit()

        logger.info(
            "signal_stored",
            market_id=market.id,
            probability=result.probability,
            market_price=market_price,
            edge=f"{edge:+.3f}",
            confidence=result.confidence,
        )

        # 4. Learn this evaluation in NexusCortex for future calibration
        if self._nexus:
            await self._nexus.learn_evaluation(
                market_id=market.id,
                question=market.question,
                estimated_prob=result.probability,
                market_price=market_price,
                confidence=result.confidence,
                category=market.category,
            )

        return True

    @staticmethod
    def _db_to_info(db_market) -> MarketInfo:
        """Convert DB Market row to MarketInfo domain model."""
        return MarketInfo(
            id=db_market.id,
            question=db_market.question,
            category=db_market.category,
            end_date=db_market.end_date,
            volume=db_market.volume,
            liquidity=db_market.liquidity,
            best_bid=db_market.best_bid,
            best_ask=db_market.best_ask,
            outcomes=[
                Outcome(
                    name=db_market.outcome_yes,
                    clob_token_id=db_market.clob_token_id_yes,
                    price=db_market.best_bid,
                ),
                Outcome(
                    name=db_market.outcome_no,
                    clob_token_id=db_market.clob_token_id_no,
                    price=None,
                ),
            ],
            description=db_market.description,
            resolution_source=db_market.resolution_source,
            tags=db_market.tags,
        )
