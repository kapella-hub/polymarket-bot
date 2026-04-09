"""Order intent state machine for tracking order lifecycle."""

from datetime import datetime, timezone
from typing import Optional

import structlog

from src.db.database import async_session
from src.db.models import IntentState, InvalidationReason, OrderSide, StrategyMode
from src.db.repositories import IntentRepository
from src.ensemble.strategies import TradeDecision

logger = structlog.get_logger()

# Valid state transitions
_TRANSITIONS: dict[IntentState, set[IntentState]] = {
    IntentState.CREATED: {IntentState.ARMED, IntentState.CANCELLED, IntentState.INVALIDATED},
    IntentState.ARMED: {IntentState.EXECUTED, IntentState.EXPIRED, IntentState.CANCELLED, IntentState.INVALIDATED},
    IntentState.EXECUTED: set(),  # Terminal
    IntentState.EXPIRED: set(),   # Terminal
    IntentState.CANCELLED: set(), # Terminal
    IntentState.INVALIDATED: set(),  # Terminal
}


class IntentManager:
    """Manages the lifecycle of order intents.

    Flow: CREATED -> ARMED -> EXECUTED / EXPIRED / CANCELLED / INVALIDATED
    """

    async def create_from_decision(
        self,
        decision: TradeDecision,
        signal_id: Optional[int] = None,
    ) -> int:
        """Create a new order intent from an ensemble decision."""
        intent_data = {
            "market_id": decision.market_id,
            "clob_token_id": decision.token_id,
            "side": OrderSide.BUY if decision.side == "buy" else OrderSide.SELL,
            "price": 0.0,  # Set during arming
            "size": decision.suggested_size,
            "strategy": decision.strategy,
            "signal_id": signal_id,
            "edge_at_creation": decision.edge,
            "confidence_at_creation": decision.confidence,
            "state": IntentState.CREATED,
        }

        async with async_session() as session:
            repo = IntentRepository(session)
            intent_id = await repo.create(intent_data)
            await session.commit()

        logger.info(
            "intent_created",
            intent_id=intent_id,
            market_id=decision.market_id,
            side=decision.side,
            edge=f"{decision.edge:+.3f}",
        )
        return intent_id

    async def transition(
        self,
        intent_id: int,
        new_state: IntentState,
        reason: Optional[InvalidationReason] = None,
        **extra_fields,
    ) -> bool:
        """Transition an intent to a new state with validation."""
        async with async_session() as session:
            repo = IntentRepository(session)

            # Read current state and validate transition
            current_state = await repo.get_state(intent_id)
            if current_state is None:
                logger.error("intent_not_found", intent_id=intent_id)
                return False

            valid_next = _TRANSITIONS.get(current_state, set())
            if new_state not in valid_next:
                logger.error(
                    "invalid_intent_transition",
                    intent_id=intent_id,
                    current=current_state.value,
                    attempted=new_state.value,
                    valid=sorted(s.value for s in valid_next),
                )
                return False

            await repo.update_state(intent_id, new_state, reason, **extra_fields)
            await session.commit()

        logger.info(
            "intent_transition",
            intent_id=intent_id,
            new_state=new_state.value,
            reason=reason.value if reason else None,
        )
        return True

    async def arm(self, intent_id: int, price: float) -> bool:
        """Arm an intent with a specific price (ready to execute)."""
        return await self.transition(
            intent_id, IntentState.ARMED, price=price
        )

    async def execute(
        self,
        intent_id: int,
        exchange_order_id: str,
        filled_price: float,
        filled_size: float,
    ) -> bool:
        """Mark intent as executed with fill details."""
        return await self.transition(
            intent_id,
            IntentState.EXECUTED,
            exchange_order_id=exchange_order_id,
            filled_price=filled_price,
            filled_size=filled_size,
            fill_timestamp=datetime.now(timezone.utc),
        )

    async def invalidate(
        self,
        intent_id: int,
        reason: InvalidationReason,
    ) -> bool:
        """Invalidate an intent with a reason code."""
        return await self.transition(intent_id, IntentState.INVALIDATED, reason)
