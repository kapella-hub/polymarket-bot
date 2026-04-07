"""Execution engine: Shadow/Paper/Live modes."""

from datetime import datetime, timezone
from typing import Optional

import structlog

from src.config import ExecutionMode, settings
from src.db.database import async_session
from src.db.models import InvalidationReason, OrderSide, StrategyMode
from src.db.repositories import PositionRepository, TradeRepository
from src.ensemble.strategies import TradeDecision
from src.exchange.base import ExchangeAdapter
from src.execution.intent import IntentManager
from src.risk.controller import RiskCheck, RiskController

logger = structlog.get_logger()


class ExecutionEngine:
    """Processes trade decisions through risk checks and order placement.

    Modes:
    - SHADOW: Log decisions only, track hypothetical P&L
    - PAPER: Simulate fills against real order book data
    - LIVE: Submit real orders to the exchange
    """

    def __init__(
        self,
        exchange: Optional[ExchangeAdapter],
        risk: RiskController,
        intents: Optional[IntentManager] = None,
    ):
        self._exchange = exchange
        self._risk = risk
        self._intents = intents or IntentManager()
        self._mode = settings.execution_mode

    async def process(self, decision: TradeDecision) -> bool:
        """Process a trade decision through the full pipeline.

        Returns True if the trade was executed (or logged in shadow mode).
        """
        # Risk check
        check = await self._risk.check(decision)
        if not check.allowed:
            logger.info(
                "trade_blocked_by_risk",
                market_id=decision.market_id,
                reason=check.reason,
            )
            return False

        # Adjust size if risk controller capped it
        size = check.adjusted_size or decision.suggested_size

        # Create intent
        intent_id = await self._intents.create_from_decision(decision)

        if self._mode == ExecutionMode.SHADOW:
            return await self._shadow_execute(intent_id, decision, size)
        elif self._mode == ExecutionMode.PAPER:
            return await self._paper_execute(intent_id, decision, size)
        elif self._mode == ExecutionMode.LIVE:
            return await self._live_execute(intent_id, decision, size)

        return False

    async def _shadow_execute(
        self, intent_id: int, decision: TradeDecision, size: float
    ) -> bool:
        """Shadow mode: log the intended trade, no execution."""
        logger.info(
            "shadow_trade",
            intent_id=intent_id,
            market_id=decision.market_id,
            side=decision.side,
            edge=f"{decision.edge:+.3f}",
            size=f"${size:.2f}",
            strategy=decision.strategy.value,
        )
        # Mark as executed for tracking (hypothetical)
        await self._intents.execute(
            intent_id,
            exchange_order_id="shadow",
            filled_price=0.0,
            filled_size=size,
        )
        return True

    async def _paper_execute(
        self, intent_id: int, decision: TradeDecision, size: float
    ) -> bool:
        """Paper mode: simulate fill against real order book."""
        # Get current order book for realistic fill simulation
        fill_price = 0.5  # Default
        if self._exchange:
            try:
                book = await self._exchange.get_order_book(decision.token_id)
                if decision.side == "buy" and book.best_ask is not None:
                    fill_price = book.best_ask
                elif decision.side == "sell" and book.best_bid is not None:
                    fill_price = book.best_bid
            except Exception:
                pass

        logger.info(
            "paper_trade",
            intent_id=intent_id,
            market_id=decision.market_id,
            side=decision.side,
            fill_price=fill_price,
            size=f"${size:.2f}",
        )

        await self._intents.execute(
            intent_id,
            exchange_order_id=f"paper-{intent_id}",
            filled_price=fill_price,
            filled_size=size,
        )

        # Update position tracking
        async with async_session() as session:
            pos_repo = PositionRepository(session)
            # Determine outcome name from token_id vs market
            outcome = "Yes"  # Default, would need market lookup for accuracy
            await pos_repo.upsert_from_fill(
                market_id=decision.market_id,
                clob_token_id=decision.token_id,
                outcome=outcome,
                side=decision.side,
                price=fill_price,
                size=size / fill_price if fill_price > 0 else size,
            )
            await session.commit()

        return True

    async def _live_execute(
        self, intent_id: int, decision: TradeDecision, size: float
    ) -> bool:
        """Live mode: submit real order to exchange."""
        if not self._exchange:
            logger.error("live_execute_no_exchange")
            await self._intents.invalidate(
                intent_id, InvalidationReason.RISK_LIMIT
            )
            return False

        # Get current price for limit order
        try:
            book = await self._exchange.get_order_book(decision.token_id)
        except Exception as e:
            logger.error("live_book_fetch_failed", error=str(e))
            await self._intents.invalidate(
                intent_id, InvalidationReason.SPREAD_WIDENED
            )
            return False

        # Set limit price: slightly better than best available
        if decision.side == "buy":
            price = book.best_ask if book.best_ask else 0.5
        else:
            price = book.best_bid if book.best_bid else 0.5

        # Arm the intent with the price
        await self._intents.arm(intent_id, price)

        # Calculate token quantity from USDC size
        token_qty = size / price if price > 0 else 0
        if token_qty <= 0:
            await self._intents.invalidate(
                intent_id, InvalidationReason.RISK_LIMIT
            )
            return False

        # Place order
        result = await self._exchange.place_limit_order(
            token_id=decision.token_id,
            side=decision.side,
            price=price,
            size=token_qty,
        )

        if result.success:
            await self._intents.execute(
                intent_id,
                exchange_order_id=result.order_id,
                filled_price=price,
                filled_size=size,
            )

            # Record trade
            async with async_session() as session:
                trade_repo = TradeRepository(session)
                pos_repo = PositionRepository(session)

                await trade_repo.record({
                    "market_id": decision.market_id,
                    "clob_token_id": decision.token_id,
                    "intent_id": intent_id,
                    "side": OrderSide.BUY if decision.side == "buy" else OrderSide.SELL,
                    "price": price,
                    "size": token_qty,
                    "fee": 0.0,  # TODO: compute from exchange response
                    "strategy": decision.strategy,
                    "exchange_order_id": result.order_id,
                    "executed_at": datetime.now(timezone.utc),
                })

                await pos_repo.upsert_from_fill(
                    market_id=decision.market_id,
                    clob_token_id=decision.token_id,
                    outcome="Yes",
                    side=decision.side,
                    price=price,
                    size=token_qty,
                )
                await session.commit()

            logger.info(
                "live_trade_executed",
                intent_id=intent_id,
                order_id=result.order_id,
                price=price,
                size=token_qty,
            )
            return True
        else:
            logger.error(
                "live_trade_failed",
                intent_id=intent_id,
                error=result.message,
            )
            await self._intents.invalidate(
                intent_id, InvalidationReason.RISK_LIMIT
            )
            return False
