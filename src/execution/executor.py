"""Execution engine: Shadow/Paper/Live modes."""

from datetime import datetime, timezone
from typing import Optional

import structlog

from src.config import ExecutionMode, settings
from src.db.database import async_session
from src.db.models import InvalidationReason, OrderSide, StrategyMode
from src.db.repositories import MarketRepository, PositionRepository, TradeRepository
from src.ensemble.strategies import TradeDecision
from src.exchange.base import ExchangeAdapter, TradeRecord
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
        nexus=None,
    ):
        self._exchange = exchange
        self._risk = risk
        self._intents = intents or IntentManager()
        self._nexus = nexus
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
            if self._nexus:
                await self._nexus.event_risk_triggered(
                    check.reason, f"market:{decision.market_id[:16]}"
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

    async def _resolve_outcome_name(self, decision: TradeDecision) -> str:
        async with async_session() as session:
            market_repo = MarketRepository(session)
            market = await market_repo.get_by_id(decision.market_id)

        if market is None:
            return "Unknown"
        if decision.token_id == market.clob_token_id_yes:
            return market.outcome_yes
        if decision.token_id == market.clob_token_id_no:
            return market.outcome_no
        return "Unknown"

    async def _get_fill_for_order(
        self,
        order_id: str,
    ) -> tuple[float, float, float]:
        if not self._exchange:
            return 0.0, 0.0, 0.0

        try:
            trades = await self._exchange.get_trades()
        except Exception:
            return 0.0, 0.0, 0.0

        matched: list[TradeRecord] = [t for t in trades if t.order_id == order_id]
        if not matched:
            return 0.0, 0.0, 0.0

        filled_size = sum(t.size for t in matched)
        if filled_size <= 0:
            return 0.0, 0.0, 0.0

        total_notional = sum(t.price * t.size for t in matched)
        total_fee = sum(t.fee for t in matched)
        avg_price = total_notional / filled_size if filled_size > 0 else 0.0
        return filled_size, avg_price, total_fee

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
        if not await self._intents.arm(intent_id, 0.0):
            return False
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
        fill_price = None
        if self._exchange:
            try:
                book = await self._exchange.get_order_book(decision.token_id)
                if decision.side == "buy" and book.best_ask is not None:
                    fill_price = book.best_ask
                elif decision.side == "sell" and book.best_bid is not None:
                    fill_price = book.best_bid
            except Exception:
                pass

        if fill_price is None or fill_price <= 0:
            logger.warning(
                "paper_no_fill_price",
                intent_id=intent_id,
                market_id=decision.market_id,
            )
            await self._intents.invalidate(
                intent_id, InvalidationReason.SPREAD_WIDENED
            )
            return False

        logger.info(
            "paper_trade",
            intent_id=intent_id,
            market_id=decision.market_id,
            side=decision.side,
            fill_price=fill_price,
            size=f"${size:.2f}",
        )

        token_qty = size / fill_price if fill_price > 0 else 0
        if token_qty <= 0:
            await self._intents.invalidate(
                intent_id, InvalidationReason.RISK_LIMIT
            )
            return False

        if not await self._intents.arm(intent_id, fill_price):
            return False
        await self._intents.execute(
            intent_id,
            exchange_order_id=f"paper-{intent_id}",
            filled_price=fill_price,
            filled_size=token_qty,
        )

        # Update position tracking
        async with async_session() as session:
            pos_repo = PositionRepository(session)
            outcome = await self._resolve_outcome_name(decision)
            realized_pnl = await pos_repo.upsert_from_fill(
                market_id=decision.market_id,
                clob_token_id=decision.token_id,
                outcome=outcome,
                side=decision.side,
                price=fill_price,
                size=token_qty,
            )
            await session.commit()

        # Track P&L for daily loss limit
        self._risk.record_fill(realized_pnl)

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

        # Set limit price from order book — reject if no liquidity
        if decision.side == "buy":
            price = book.best_ask
        else:
            price = book.best_bid

        if price is None or price <= 0:
            logger.warning(
                "live_no_liquidity",
                intent_id=intent_id,
                market_id=decision.market_id,
                side=decision.side,
            )
            await self._intents.invalidate(
                intent_id, InvalidationReason.SPREAD_WIDENED
            )
            return False

        # Arm the intent with the price
        if not await self._intents.arm(intent_id, price):
            return False

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
            filled_qty, filled_price, fee_paid = await self._get_fill_for_order(
                result.order_id
            )
            if filled_qty <= 0 or filled_price <= 0:
                cancelled = await self._exchange.cancel_order(result.order_id)
                if cancelled:
                    await self._intents.invalidate(
                        intent_id, InvalidationReason.SPREAD_WIDENED
                    )
                else:
                    logger.warning(
                        "live_unconfirmed_order_left_open",
                        intent_id=intent_id,
                        order_id=result.order_id,
                    )
                return False

            await self._intents.execute(
                intent_id,
                exchange_order_id=result.order_id,
                filled_price=filled_price,
                filled_size=filled_qty,
            )

            # Record trade
            async with async_session() as session:
                trade_repo = TradeRepository(session)
                pos_repo = PositionRepository(session)
                outcome = await self._resolve_outcome_name(decision)

                await trade_repo.record({
                    "market_id": decision.market_id,
                    "clob_token_id": decision.token_id,
                    "intent_id": intent_id,
                    "side": OrderSide.BUY if decision.side == "buy" else OrderSide.SELL,
                    "price": filled_price,
                    "size": filled_qty,
                    "fee": fee_paid,
                    "strategy": decision.strategy,
                    "exchange_order_id": result.order_id,
                    "executed_at": datetime.now(timezone.utc),
                })

                realized_pnl = await pos_repo.upsert_from_fill(
                    market_id=decision.market_id,
                    clob_token_id=decision.token_id,
                    outcome=outcome,
                    side=decision.side,
                    price=filled_price,
                    size=filled_qty,
                )
                await session.commit()

            self._risk.record_fill(realized_pnl - fee_paid)

            logger.info(
                "live_trade_executed",
                intent_id=intent_id,
                order_id=result.order_id,
                price=filled_price,
                size=filled_qty,
            )

            # Push to NexusStack
            if self._nexus:
                await self._nexus.event_trade_executed(
                    decision.market_id,
                    decision.side,
                    filled_qty * filled_price,
                    filled_price,
                    decision.edge,
                )
                await self._nexus.learn_trade(
                    market_id=decision.market_id,
                    question="",  # Would need market question here
                    side=decision.side,
                    size=filled_qty * filled_price,
                    price=filled_price,
                    strategy=decision.strategy.value,
                    edge=decision.edge,
                )
                await self._nexus.broadcast_position_update(
                    decision.market_id, "Yes", token_qty, size,
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
